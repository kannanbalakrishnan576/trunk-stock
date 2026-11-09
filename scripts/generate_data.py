"""Generate the synthetic CSVs in data/.

Standard library only. Deterministic (seeded). Every number in data/ comes
from this file -- nothing here is real service data.

The story the data is built to tell:
  * The problem code is known at dispatch ("no heat", "no cooling", ...).
  * Each problem code has a known list of candidate parts.
  * Which of those parts a job actually consumes is not fixed. The mix moves
    with the weather, with the start of a season, with an ageing install
    base and with a bad supplier batch.
  * A van-stock rule set from last year's attach rate will lag those moves.

Run:  python scripts/generate_data.py
"""
import csv
import datetime as dt
import math
import os
import random

random.seed(20261109)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

HIST_START = dt.date(2023, 11, 13)          # a Monday
HIST_WEEKS = 156                            # three years of history
HORIZON = 12                                # forecast weeks (weather only)
WEEKS = [HIST_START + dt.timedelta(weeks=i) for i in range(HIST_WEEKS + HORIZON)]
LAST_HIST = WEEKS[HIST_WEEKS - 1]

REGIONS = {
    #  name        mean F, swing F, peak day-of-year (hottest)
    "Desert":  (75.0, 19.0, 200),
    "Lakes":   (50.0, 24.0, 200),
    "Coastal": (66.0, 13.0, 205),
}

PROBLEMS = [
    ("P01", "No cooling", "Outdoor unit runs or hums but the house does not cool"),
    ("P02", "No heat", "Gas furnace calls for heat but does not light or stay lit"),
    ("P03", "Blower not running", "Thermostat calls, indoor blower does not move air"),
    ("P04", "Water at indoor unit", "Water on the floor or in the pan at the air handler"),
]

# part_id, problem, name, unit cost USD, trunk slots, base share (of jobs)
PARTS = [
    ("CAP-45-5",  "P01", "Dual run capacitor 45/5 uF",      14,  1, 0.30),
    ("CON-30A",   "P01", "Contactor 30A 24V",               19,  1, 0.17),
    ("FM-1/4HP",  "P01", "Condenser fan motor 1/4 hp",      128, 3, 0.11),
    ("HSK-01",    "P01", "Compressor hard-start kit",       42,  1, 0.06),
    ("R410-25",   "P01", "Refrigerant R-410A 25 lb",        210, 4, 0.13),
    ("TXV-3T",    "P01", "TXV 3-ton",                       96,  1, 0.04),
    ("IGN-HSI",   "P02", "Hot surface igniter",             34,  1, 0.28),
    ("FS-01",     "P02", "Flame sensor",                    12,  1, 0.21),
    ("IND-MTR",   "P02", "Inducer motor assembly",          185, 3, 0.10),
    ("PS-0.55",   "P02", "Pressure switch 0.55 in. wc",     28,  1, 0.12),
    ("CB-50A",    "P02", "Furnace control board",           165, 2, 0.07),
    ("GV-24V",    "P02", "Gas valve 24V",                   140, 2, 0.05),
    ("BM-1/2HP",  "P03", "Blower motor 1/2 hp PSC",         155, 4, 0.24),
    ("CAP-10",    "P03", "Blower capacitor 10 uF",          9,   1, 0.30),
    ("ECM-MOD",   "P03", "ECM motor module",                260, 2, 0.12),
    ("CB-50A",    "P03", "Furnace control board",           165, 2, 0.14),
    ("BLR-RLY",   "P03", "Fan relay 24V",                   16,  1, 0.08),
    ("CP-120",    "P04", "Condensate pump 120V",            72,  2, 0.30),
    ("FLT-SW",    "P04", "Float safety switch",             18,  1, 0.22),
    ("DRN-KIT",   "P04", "Drain line and trap kit",         15,  2, 0.25),
    ("EVAP-PAN",  "P04", "Secondary drain pan",             38,  3, 0.08),
]

# Jobs per week per (region, problem) at mean conditions.
BASE_JOBS = {
    ("Desert", "P01"): 70, ("Desert", "P02"): 14, ("Desert", "P03"): 22, ("Desert", "P04"): 26,
    ("Lakes", "P01"): 30, ("Lakes", "P02"): 58, ("Lakes", "P03"): 26, ("Lakes", "P04"): 16,
    ("Coastal", "P01"): 44, ("Coastal", "P02"): 22, ("Coastal", "P03"): 24, ("Coastal", "P04"): 36,
}


def temp_for(region, day, noise=True):
    mean, swing, peak = REGIONS[region]
    doy = day.timetuple().tm_yday
    t = mean + swing * math.cos(2 * math.pi * (doy - peak) / 365.25)
    if noise:
        t += random.gauss(0, 3.2)
    return t


def demand_multiplier(problem, temp):
    """How many jobs of this kind the weather produces."""
    if problem == "P01":   # cooling failures climb steeply with heat
        return max(0.12, 1 + 0.055 * (temp - 70))
    if problem == "P02":   # heating failures climb with cold
        return max(0.08, 1 + 0.05 * (55 - temp))
    if problem == "P03":
        return 1 + 0.012 * abs(temp - 65)
    if problem == "P04":   # condensate follows cooling run hours
        return max(0.35, 1 + 0.03 * (temp - 68))
    return 1.0


def weeks_since(day, start):
    return (day - start).days / 7.0


def share_weights(region, problem, day, temp, temp_prev):
    """Unnormalised weights for each candidate part, plus 'none'."""
    w = {}
    for pid, prob, _name, _cost, _slots, base in PARTS:
        if prob != problem:
            continue
        m = 1.0
        # Heat stress: run capacitors and contactors fail in heat waves.
        if pid in ("CAP-45-5", "CON-30A"):
            m *= 1 + 0.035 * max(0.0, temp - 85)
        if pid == "R410-25":
            m *= 1 + 0.02 * max(0.0, temp - 80)
        # Season start: igniters crack on the first cold weeks of the year,
        # flame sensors foul over the season and peak later.
        if pid == "IGN-HSI":
            first_cold = max(0.0, temp_prev - temp) * max(0.0, 60 - temp) / 40
            m *= 1 + 0.6 * first_cold
        if pid == "FS-01":
            m *= 1 + 0.025 * max(0.0, 40 - temp)
        # Drift 1 -- Lakes: a cohort of furnaces installed in 2014 ages into
        # control-board failures from spring 2026. Static attach rate lags it.
        if pid == "CB-50A" and region == "Lakes":
            ramp = weeks_since(day, dt.date(2026, 3, 2))
            if ramp > 0:
                m *= 1 + min(3.6, 0.11 * ramp)
        # Drift 2 -- Desert: a bad batch of run capacitors installed in
        # spring 2026 starts failing in July and is still elevated.
        if pid == "CAP-45-5" and region == "Desert":
            since = weeks_since(day, dt.date(2026, 7, 6))
            if since > 0:
                m *= 1 + 1.4 * math.exp(-since / 30.0)
        # Drift 3 -- Coastal: a run of new ECM blowers from one build year
        # starts losing modules in 2026.
        if pid == "ECM-MOD" and region == "Coastal":
            ramp = weeks_since(day, dt.date(2026, 1, 5))
            if ramp > 0:
                m *= 1 + min(2.2, 0.05 * ramp)
        if pid == "CP-120" and region == "Coastal":
            m *= 1 + 0.02 * max(0.0, temp - 70)
        w[pid] = base * m
    # Share of jobs fixed without a part (adjustment, cleaning, reset).
    w["NONE"] = 0.12 if problem != "P04" else 0.15
    return w


def draw(weights):
    r = random.random() * sum(weights.values())
    for k, v in weights.items():
        r -= v
        if r <= 0:
            return k
    return k


def poisson(lam):
    # Knuth for small lambda, normal approximation above.
    if lam > 40:
        return max(0, int(round(random.gauss(lam, math.sqrt(lam)))))
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= random.random()
        if p <= L:
            return k
        k += 1


def write(name, header, rows):
    with open(os.path.join(DATA, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------- reference
write("problem_codes.csv", ["problem_code", "name", "description"], PROBLEMS)

seen, parts_rows, cand_rows = set(), [], []
for pid, prob, name, cost, slots, base in PARTS:
    if pid not in seen:
        seen.add(pid)
        parts_rows.append([pid, name, cost, slots])
    cand_rows.append([prob, pid])
write("parts.csv", ["part_id", "name", "unit_cost_usd", "trunk_slots"], parts_rows)
write("problem_parts.csv", ["problem_code", "part_id"], cand_rows)

# ---------------------------------------------------------------- weather
weather = {}
wrows = []
for region in REGIONS:
    for i, week in enumerate(WEEKS):
        t = temp_for(region, week + dt.timedelta(days=3))
        weather[(region, week)] = t
        kind = "observed" if i < HIST_WEEKS else "forecast"
        wrows.append([week.isoformat(), region, round(t, 1), kind])
write("weather_weekly.csv", ["week", "region", "avg_temp_f", "kind"], wrows)

# ---------------------------------------------------------------- history
jobs_rows, use_rows = [], []
for region in REGIONS:
    for problem, _n, _d in PROBLEMS:
        for i, week in enumerate(WEEKS[:HIST_WEEKS]):
            temp = weather[(region, week)]
            prev = weather[(region, WEEKS[i - 1])] if i else temp
            lam = BASE_JOBS[(region, problem)] * demand_multiplier(problem, temp)
            jobs = poisson(lam)
            weights = share_weights(region, problem, week, temp, prev)
            counts = {k: 0 for k in weights}
            for _ in range(jobs):
                counts[draw(weights)] += 1
            jobs_rows.append([week.isoformat(), region, problem, jobs])
            for pid, n in counts.items():
                if pid != "NONE":
                    use_rows.append([week.isoformat(), region, problem, pid, n])
write("jobs_weekly.csv", ["week", "region", "problem_code", "jobs"], jobs_rows)
write("part_usage_weekly.csv",
      ["week", "region", "problem_code", "part_id", "uses"], use_rows)

# ---------------------------------------------------------------- dispatches
first = LAST_HIST + dt.timedelta(weeks=1)          # Monday 9 Nov 2026
techs = [("T-104", "Desert"), ("T-117", "Desert"), ("T-201", "Lakes"),
         ("T-233", "Lakes"), ("T-310", "Coastal"), ("T-342", "Coastal")]
mix = {"Desert": ["P01", "P01", "P03", "P04", "P02"],
       "Lakes": ["P02", "P02", "P02", "P03", "P01"],
       "Coastal": ["P04", "P01", "P03", "P02", "P04"]}
visits, n = [], 1
for day_off in range(3):
    day = first + dt.timedelta(days=day_off)
    for tech, region in techs:
        for slot in range(random.choice([3, 4])):
            problem = random.choice(mix[region])
            hour = 8 + slot * 2
            age = random.choice([3, 5, 7, 9, 11, 12, 14, 16])
            visits.append([f"V{n:04d}", day.isoformat(), f"{hour:02d}:00", tech,
                           region, problem, age])
            n += 1
write("visits.csv", ["visit_id", "date", "window_start", "tech_id", "region",
                     "problem_code", "asset_age_years"], visits)

print(f"history {WEEKS[0]} .. {LAST_HIST} ({HIST_WEEKS} weeks), "
      f"weather forecast to {WEEKS[-1]}, {len(visits)} visits")

# ---------------------------------------------------------------- install base
# What the asset register knows ahead of time: how many installed units are
# inside a known failure window, week by week, including the forecast weeks.
# Install dates are recorded, so this is a known-future covariate.
#   Lakes   P02  furnaces installed 2013-15 entering the 11-13 year window
#   Desert  P01  outdoor units fitted with the spring-2026 capacitor batch,
#                counted while they are within 6 months of their first summer
#   Coastal P03  ECM blowers from the 2023 build year passing three years old
# Every other (region, problem) has a flat, noisy register count.
rng = random.Random(7)
ib_rows = []
for region in REGIONS:
    for problem, _n, _d in PROBLEMS:
        for week in WEEKS:
            base = 1000 + rng.gauss(0, 8)
            extra = 0.0
            if (region, problem) == ("Lakes", "P02"):
                extra = 2600 * min(1.0, max(0.0, weeks_since(week, dt.date(2026, 2, 16)) / 34))
            if (region, problem) == ("Desert", "P01"):
                s = weeks_since(week, dt.date(2026, 6, 22))
                extra = 0 if s < 0 else 1500 * math.exp(-s / 32)
            if (region, problem) == ("Coastal", "P03"):
                extra = 1100 * min(1.0, max(0.0, weeks_since(week, dt.date(2025, 12, 22)) / 44))
            ib_rows.append([week.isoformat(), region, problem, round(base + extra)])
write("install_base_weekly.csv", ["week", "region", "problem_code", "units_at_risk"], ib_rows)

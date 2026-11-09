"""Shared logic: load the CSVs, turn forecasts into part probabilities,
pick the trunk list, and score a pick against what actually happened.

Standard library only, so the baseline, the backtest and the recommender
run anywhere Python 3.9+ runs. Only forecast_timesfm.py needs extra packages.
"""
import csv
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")


def read(name):
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


def write(name, header, rows):
    with open(os.path.join(DATA, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load():
    """Everything the other scripts need, keyed for quick lookup."""
    parts = {r["part_id"]: r for r in read("parts.csv")}
    candidates = defaultdict(list)
    for r in read("problem_parts.csv"):
        candidates[r["problem_code"]].append(r["part_id"])
    jobs = {(r["week"], r["region"], r["problem_code"]): int(r["jobs"])
            for r in read("jobs_weekly.csv")}
    uses = {(r["week"], r["region"], r["problem_code"], r["part_id"]): int(r["uses"])
            for r in read("part_usage_weekly.csv")}
    weather = {(r["week"], r["region"]): float(r["avg_temp_f"])
               for r in read("weather_weekly.csv")}
    weeks = sorted({k[0] for k in jobs})
    regions = sorted({k[1] for k in jobs})
    return dict(parts=parts, candidates=candidates, jobs=jobs, uses=uses,
                weather=weather, weeks=weeks, regions=regions,
                problems=sorted(candidates))


def trailing_share(d, region, problem, end_idx, n_weeks):
    """Attach rate per part over the n weeks ending at weeks[end_idx - 1].

    This is how van stock is usually set: count what each problem consumed
    over a window, divide by the jobs.
    """
    weeks = d["weeks"][max(0, end_idx - n_weeks):end_idx]
    total = sum(d["jobs"][(w, region, problem)] for w in weeks) or 1
    return {p: sum(d["uses"].get((w, region, problem, p), 0) for w in weeks) / total
            for p in d["candidates"][problem]}


def forecast_probs(rows):
    """forecast.csv rows -> {(run, week, region, problem): {part: (p10, p50, p90)}}

    Each value is a forecast attach rate: the probability that a job with
    this problem code, in this region and week, consumes this part.
    """
    out = defaultdict(dict)
    for r in rows:
        key = (r["run"], r["week"], r["region"], r["problem_code"])
        out[key][r["series"]] = tuple(min(1.0, max(0.0, float(r[c])))
                                      for c in ("p10", "p50", "p90"))
    return dict(out)


def top_k(probs, k):
    """Rank candidate parts by probability and keep k. probs: {part: p}."""
    return [p for p, _ in sorted(probs.items(), key=lambda kv: -kv[1])[:k]]


def coverage(d, week, region, problem, picked):
    """(part jobs covered, part jobs) for one week, given the trunk list.

    A job that needed a part is covered when that part was in the trunk.
    Jobs fixed with no part are left out -- the trunk did not matter to them.
    """
    used = {p: d["uses"].get((week, region, problem, p), 0)
            for p in d["candidates"][problem]}
    return sum(used[p] for p in picked), sum(used.values())

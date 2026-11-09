"""Print the trunk list for each upcoming visit in visits.csv.

  python recommend.py            top 3 per visit
  python recommend.py -k 5       top 5
  python recommend.py --tech T-201 --date 2026-11-09   one technician's day

The day view merges the visits: one part covers every visit that wants it,
so a four-visit route rarely needs four times the stock.
Writes data/trunk_lists.csv. Standard library only.
"""
import argparse
import datetime as dt

import trunk


def monday(date_str):
    d = dt.date.fromisoformat(date_str)
    return (d - dt.timedelta(days=d.weekday())).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--tech")
    ap.add_argument("--date")
    a = ap.parse_args()

    d = trunk.load()
    fp = trunk.forecast_probs(trunk.read("forecast.csv"))
    visits = trunk.read("visits.csv")
    if a.tech:
        visits = [v for v in visits if v["tech_id"] == a.tech]
    if a.date:
        visits = [v for v in visits if v["date"] == a.date]

    rows, day = [], {}
    for v in visits:
        probs = fp[("production", monday(v["date"]), v["region"], v["problem_code"])]
        ranked = sorted(probs.items(), key=lambda kv: -kv[1][1])[:a.k]
        cover = sum(q[1] for _p, q in ranked) / (sum(q[1] for q in probs.values()) or 1)
        print(f"{v['visit_id']} {v['date']} {v['window_start']} {v['tech_id']} "
              f"{v['region']:8} {v['problem_code']}  covers {cover:.0%} of part jobs")
        for i, (part, (lo, mid, hi)) in enumerate(ranked, 1):
            name = d["parts"][part]["name"]
            print(f"    {i}. {part:9} {name:32} {mid:5.0%}  ({lo:.0%}-{hi:.0%})")
            rows.append([v["visit_id"], i, part, round(lo, 4), round(mid, 4), round(hi, 4)])
            day.setdefault((v["tech_id"], v["date"]), set()).add(part)

    if a.tech and a.date:
        parts = sorted(day.get((a.tech, a.date), []))
        slots = sum(int(d["parts"][p]["trunk_slots"]) for p in parts)
        print(f"\n{a.tech} on {a.date}: {len(parts)} distinct parts, {slots} trunk slots")
        for p in parts:
            print(f"    {p:9} {d['parts'][p]['name']}")

    if not a.tech and not a.date:
        trunk.write("trunk_lists.csv", ["visit_id", "rank", "part_id", "p10", "p50", "p90"], rows)


if __name__ == "__main__":
    main()

"""Score ways of choosing the trunk list against 26 weeks that happened.

Rolling origin: for every week in the holdout, each method sees only the
weeks before it and picks next week's trunk list.

  annual    attach rate over the previous 52 weeks -- how van stock is
            usually set
  recent    attach rate over the previous 8 weeks -- the obvious fix
  timesfm   TimesFM 3.0 one-week-ahead forecast (forecast.csv, run =
            rolling), with temperature and install-base units at risk as
            known-future covariates
  tfm_bare / tfm_weather   the same model with no covariates, and with
            temperature only -- present if those forecast files exist

Every method sees the same history and nothing after its cutoff. For every
week, region and problem code in the holdout, each method picks its top k
parts; the score is the share of jobs needing a part whose part was in the
trunk.

Writes data/backtest.csv, backtest_cells.csv, backtest_weekly.csv and
calibration.csv. Standard library only.
"""
import os
from collections import defaultdict

import trunk

HOLDOUT = 26
KS = range(1, 7)


def main():
    d = trunk.load()
    idx = {w: i for i, w in enumerate(d["weeks"])}
    holdout = d["weeks"][-HOLDOUT:]

    methods = {
        "annual": lambda w, r, p: trunk.trailing_share(d, r, p, idx[w], 52),
        "recent": lambda w, r, p: trunk.trailing_share(d, r, p, idx[w], 8),
    }
    for name, label in (("forecast_none.csv", "tfm_bare"),
                        ("forecast_weather.csv", "tfm_weather"),
                        ("forecast.csv", "timesfm")):
        if os.path.exists(os.path.join(trunk.DATA, name)):
            fp = trunk.forecast_probs(trunk.read(name))
            methods[label] = (lambda fp: lambda w, r, p: {
                part: q[1] for part, q in fp[("rolling", w, r, p)].items()})(fp)

    totals = defaultdict(lambda: [0, 0])
    cells = defaultdict(lambda: [0, 0])
    weekly = []
    for m, probs_for in methods.items():
        for w in holdout:
            for r in d["regions"]:
                for p in d["problems"]:
                    ranked = trunk.top_k(probs_for(w, r, p), 99)
                    for k in KS:
                        got, need = trunk.coverage(d, w, r, p, ranked[:k])
                        totals[(m, k)][0] += got
                        totals[(m, k)][1] += need
                        if k == 3:
                            cells[(m, r, p)][0] += got
                            cells[(m, r, p)][1] += need
                            weekly.append([m, w, r, p, got, need])

    trunk.write("backtest.csv", ["method", "k", "covered", "part_jobs", "coverage"],
                [[m, k, a, b, round(a / b, 4)] for (m, k), (a, b) in sorted(totals.items())])
    trunk.write("backtest_weekly.csv",
                ["method", "week", "region", "problem_code", "covered", "part_jobs"], weekly)
    trunk.write("backtest_cells.csv",
                ["method", "region", "problem_code", "covered", "part_jobs", "coverage_k3"],
                [[m, r, p, a, b, round(a / b, 4)] for (m, r, p), (a, b) in sorted(cells.items())])

    print(f"holdout {holdout[0]} .. {holdout[-1]}\n")
    print("k   " + "".join(f"{m:>13}" for m in methods))
    for k in KS:
        print(f"{k}   " + "".join(f"{totals[(m, k)][0] / totals[(m, k)][1]:>13.1%}"
                                  for m in methods))
    # Calibration of the TimesFM 80% interval (p10..p90) against the actual
    # 4-week attach rate it was forecasting. Only part-weeks with at least 20
    # jobs behind them are counted, so the actual rate means something.
    if os.path.exists(os.path.join(trunk.DATA, "forecast.csv")):
        fp = trunk.forecast_probs(trunk.read("forecast.csv"))
        inside = total = 0
        for w in holdout:
            ws = d["weeks"][idx[w] - 3:idx[w] + 1]
            for r in d["regions"]:
                for p in d["problems"]:
                    jobs = sum(d["jobs"][(x, r, p)] for x in ws)
                    if jobs < 20:
                        continue
                    for part, (lo, _mid, hi) in fp[("rolling", w, r, p)].items():
                        act = sum(d["uses"].get((x, r, p, part), 0) for x in ws) / jobs
                        inside += lo <= act <= hi
                        total += 1
        print(f"\nTimesFM 80% interval held the actual attach rate "
              f"{inside}/{total} times ({inside / total:.0%})")
        trunk.write("calibration.csv", ["interval", "inside", "total", "share"],
                    [["p10-p90", inside, total, round(inside / total, 4)]])

    print("\nk=3 by region and problem")
    for r in d["regions"]:
        for p in d["problems"]:
            print(f"  {r:8} {p}  " + "".join(
                f"{cells[(m, r, p)][0] / cells[(m, r, p)][1]:>10.0%}" for m in methods))


if __name__ == "__main__":
    main()

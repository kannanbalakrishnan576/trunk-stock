"""Forecast parts consumption with Google's TimesFM 3.0, locally.

One multivariate call per (region, problem code):
  targets               the attach rate of every candidate part -- the share
                        of jobs that consumed it, over a rolling 4 weeks
  past-future covariates  average temperature (observed, then forecast)
                          units at risk from the install base (known ahead,
                          because install dates are recorded)

Forecast the mix, not the volume. The trunk question is "which part will
this job need", which is a share. Forecasting each part's count and dividing
by a jobs forecast ties every part to the seasonality of job volume and
buries the change in mix -- the thing that matters. An earlier version of
this script did exactly that and lost to an 8-week average.

TimesFM 3.0 forecasts the targets jointly, so a heat wave that lifts
capacitor failures and a cold snap that lifts igniters are read against the
same temperature series and against each other.

Why the install-base covariate: a zero-shot model reads a level shift it has
never seen before as noise and forecasts a return to the old level. Without
it, the Lakes control-board rise is forecast to fade. The model cannot know
that a cohort of furnaces is ageing into its failure window. The asset
register does.

Two runs are written to data/forecast.csv (or forecast_<covariates>.csv
when --covariates is not the default):
  rolling     for each of the last 26 weeks of history, a forecast of that
              week made from the weeks before it -- the way it would run in
              production, re-forecast every week for next week's trunks.
              backtest.py scores these against what happened.
  production  cut off at the end of history, forecasting the next 12 weeks
              -- the weeks the upcoming visits in visits.csv fall in

Needs:  pip install "timesfm[mlx]"     (Apple silicon)
    or  pip install "timesfm[torch]"   (anything else; pass --backend torch)

The TimesFM 3.0 weights are under a non-commercial licence. Use them for a
demo like this one. For production, the same model is licensed through
BigQuery's AI.FORECAST -- see bigquery/forecast.sql.
"""
import argparse

import numpy as np

import trunk

HORIZON = 12        # production: the next 12 weeks
ROLLING = 26        # backtest: the last 26 weeks, one week ahead each
WINDOW = 4          # weeks in the rolling attach rate
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
P10, P50, P90 = 0, 4, 8


def attach_rates(d, region, problem, part, cut):
    """Rolling WINDOW-week attach rate of one part, for weeks[:cut]."""
    out = []
    for i in range(cut):
        ws = d["weeks"][max(0, i - WINDOW + 1):i + 1]
        jobs = sum(d["jobs"][(w, region, problem)] for w in ws)
        used = sum(d["uses"].get((w, region, problem, part), 0) for w in ws)
        out.append(used / jobs if jobs else 0.0)
    return out


def load_model(backend):
    if backend == "mlx":
        from timesfm3.mlx import TimesFM3Forecaster
        return TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    return TimesFM3Evaluator(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--covariates", default="weather,register",
                    help="comma list from weather,register; 'none' for neither")
    args = ap.parse_args()

    d = trunk.load()
    use = set() if args.covariates == "none" else set(args.covariates.split(","))
    register = {(r["week"], r["region"], r["problem_code"]): float(r["units_at_risk"])
                for r in trunk.read("install_base_weekly.csv")}
    all_weeks = sorted({w for (w, _r) in d["weather"]})
    hist = d["weeks"]
    model = load_model(args.backend)
    runs = [("rolling", cut, 1) for cut in range(len(hist) - ROLLING, len(hist))]
    runs.append(("production", len(hist), HORIZON))

    rows = []
    for run, cut, horizon in runs:
        batch, keys, covs = [], [], []
        for region in d["regions"]:
            for problem in d["problems"]:
                series = d["candidates"][problem]
                target = np.array([attach_rates(d, region, problem, s, cut)
                                   for s in series], dtype=np.float32)
                span = all_weeks[:cut + horizon]
                cov = []
                if "weather" in use:
                    cov.append([d["weather"][(w, region)] / 100 for w in span])
                if "register" in use:
                    cov.append([register[(w, region, problem)] / 1000 for w in span])
                batch.append(target)
                covs.append(np.array(cov, dtype=np.float32) if cov else None)
                keys.append((region, problem, series))

        outs = model.predict_batch(
            contexts=batch, horizon=horizon,
            past_future_covariates=covs,
            return_quantiles=True, make_positive=True)

        for (region, problem, series), out in zip(keys, outs):
            q = np.asarray(out.quantiles)              # (variates, horizon, 9)
            for v, s in enumerate(series):
                for h in range(horizon):
                    week = all_weeks[cut + h]
                    rows.append([run, week, region, problem, s,
                                 round(float(q[v, h, P10]), 3),
                                 round(float(q[v, h, P50]), 3),
                                 round(float(q[v, h, P90]), 3)])

    name = ("forecast.csv" if args.covariates == "weather,register"
            else f"forecast_{args.covariates.replace(',', '_')}.csv")
    trunk.write(name, ["run", "week", "region", "problem_code", "series",
                       "p10", "p50", "p90"], rows)
    print(f"wrote data/{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()

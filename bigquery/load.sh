#!/usr/bin/env bash
# Load the CSVs into a BigQuery dataset called `trunk` in your default project.
# Needs the bq CLI, authenticated (gcloud auth login). Small: under 1 MB.
set -euo pipefail
cd "$(dirname "$0")/.."

bq --location=US mk --force --dataset trunk

for t in jobs_weekly part_usage_weekly weather_weekly install_base_weekly parts problem_parts problem_codes; do
  bq load --replace --autodetect --source_format=CSV --skip_leading_rows=1 \
    "trunk.${t}" "data/${t}.csv"
done
echo "loaded. next: bq query --use_legacy_sql=false < bigquery/forecast_P02.sql"

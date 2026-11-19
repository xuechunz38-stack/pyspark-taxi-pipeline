#!/usr/bin/env bash
# Download NYC TLC Yellow Taxi trip records (2023-01 .. 2024-12, ~1.2 GB Parquet)
# plus the Taxi Zone lookup table used for the join.
# Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
set -euo pipefail
cd "$(dirname "$0")/../data"
mkdir -p yellow
BASE="https://d37ci6vzurychx.cloudfront.net"

for y in 2023 2024; do
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    f="yellow_tripdata_${y}-${m}.parquet"
    [ -s "yellow/$f" ] && { echo "skip $f"; continue; }
    echo "downloading $f"
    curl -fsSL -o "yellow/$f" "$BASE/trip-data/$f"
  done
done

[ -s taxi_zone_lookup.csv ] || curl -fsSL -o taxi_zone_lookup.csv "$BASE/misc/taxi_zone_lookup.csv"
du -sh yellow taxi_zone_lookup.csv

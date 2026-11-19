# PySpark Data Processing Pipeline — NYC Yellow Taxi 2023–2024

A PySpark pipeline that loads ~78 million NYC taxi trips (24 monthly Parquet files, ~1.2 GB),
cleans and enriches them, aggregates them with the DataFrame API and Spark SQL, analyzes the
physical execution plans, benchmarks caching, writes partitioned Parquet output, and trains a small MLlib model.

| File | Contents |
|---|---|
| `pyspark_pipeline.ipynb` / `pyspark_pipeline.html` | The full pipeline with outputs |
| `scripts/download_data.sh` | Downloads the dataset |
| `scripts/build_notebook.py` | Generates the notebook source |
| `screenshots/` | `.explain()` output, Spark UI jobs / SQL query details, successful run |

## Dataset

* **NYC TLC Yellow Taxi Trip Records**, January 2023 – December 2024 (24 Parquet files, ~1.2 GB compressed, ~78M rows).
  Source: [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
  Each row is one trip: pickup/dropoff timestamps and zone IDs, distance, passenger count, payment type, fare, tip, total.
* **Taxi Zone Lookup** (`taxi_zone_lookup.csv`, 265 rows): maps `LocationID` → Borough / Zone. Used for the joins.

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
./scripts/download_data.sh
jupyter nbconvert --to notebook --execute --inplace pyspark_pipeline.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to html pyspark_pipeline.ipynb
```
Requires Java 11/17. While the notebook runs, the Spark UI is at http://localhost:4040.

## Pipeline

1. **Load** – each monthly file is read with an explicit projection of 10 columns and type casts
   (the TLC files have small schema drifts across months), then unioned with `unionByName`.
2. **Filter early** – time window, `0 < trip_distance < 100`, `0 < fare < 500`, `passenger_count > 0`, applied immediately after the scan;
   a 4th filter on derived `duration_min` (1–180 min).
3. **`withColumn`** – `duration_min`, `pickup_hour`, `pickup_dow`, `year_month`, `tip_pct`, `payment` (label), `speed_mph`.
4. **Join** – broadcast joins with the zone table for pickup and dropoff zones.
5. **`groupBy`** – borough × hour (count, avg fare, avg miles, median duration, avg tip %, revenue) and monthly trends.
6. **SQL** – (1) top-10 revenue pickup zones with share of total (window over aggregate),
   (2) tipping by payment type × daypart, (3) busiest airport-bound route each month (`ROW_NUMBER` window).
7. **Write** – cleaned fact table as Parquet partitioned by `year_month`; aggregates as Parquet/CSV.
8. **MLlib** – GBT regression predicting `fare_amount` from distance, duration, time, and borough.

## Performance analysis

All timings are from one run on a laptop (Apple M1, 8 cores, 8 GB RAM, Spark 3.5.3 `local[*]`, driver memory 6 GB, 64 shuffle partitions, AQE on).

| Measurement | Result |
|---|---|
| Raw rows / rows after cleaning | 79,479,946 → 70,948,650 (89.3% kept) |
| Defining transformations vs. running the action | **88 ms** vs. **10.6 s** |
| Pickup join as SortMergeJoin (filters after) vs. broadcast join (filters first) | **41.9 s** vs. **12.9 s** (3.2× faster) |
| 3 aggregations without cache vs. with `.cache()` | **109.9 s** vs. **19.4 s** (5.7× faster; filling the cache took 73.5 s once) |
| Read back one month of the partitioned output | `PartitionFilters: [year_month = 2024-06]`; only 1 of 24 folders read, count in 0.2 s |
| GBT fare model (2024-06, 20% sample) | RMSE **$5.87**, R² **0.897** |

**How Spark optimized the query.** Catalyst rewrote the plan before anything ran. For each of the 24 monthly files it merged my
separate `.filter()` calls into a single `Filter` node right on top of the scan. That includes the `duration_min` filter: even though it
uses a column derived with `withColumn`, the optimizer inlined the `unix_timestamp` expression and moved it below both joins
(see `Filter (2)` in [06](screenshots/06_explain_formatted_physical_plan.png)). **Column pruning:** `ReadSchema` lists only the
9 columns that query needs out of the 19 in the files. `payment_type` was dropped too, because `borough_hour` never uses it.
**Predicate pushdown:** `PushedFilters` holds the date range plus the `trip_distance` and `fare_amount` bounds, so the Parquet
reader can skip row groups using their min/max statistics. Both zone lookups ran as `BroadcastHashJoin`, with a 265-row table
broadcast in about 2 ms and *no* shuffle of the 71M-row side. The `groupBy` became partial `ObjectHashAggregate` → `Exchange hashpartitioning`
→ final aggregate. Because of that map-side pre-aggregation, only 10,202 rows were shuffled instead of 71M
([03](screenshots/03_query5_borough_hour_broadcast_join_detail.png)). AQE then merged the 64 shuffle partitions down to 10.

**Where filters were *not* pushed down, and other bottlenecks.** `passenger_count > 0` shows up only as `IsNotNull(passenger_count)`
in `PushedFilters`. I cast the column to `int` to fix schema drift between months, and a predicate on a cast expression cannot be pushed into the
Parquet reader. Filtering before the cast, or casting only in the final output, would let Spark push it down. The worst bottleneck was
`percentile_approx` (the median duration). It forces `ObjectHashAggregate`, which fell back to sort-based aggregation in 63 tasks and
**spilled 1.8 GB** to disk, so `borough_hour` took 203 s while the SQL queries took 37–70 s each. The writes were the other
bottleneck (about 21 minutes in total). Nothing was cached, so every `.write` recomputed the whole lineage from the raw Parquet files.
`repartition("year_month")` also shuffled the full table (555 s), and writing `borough_hour` repeated the expensive percentile
aggregation (492 s).

**What I did to optimize, and what the experiments show.** (1) *Filter early + broadcast*: the comparison in section 5 joins first with
broadcasting disabled. Catalyst still pushed `trip_distance > 0` under the join, but it had to use a **SortMergeJoin**, which shuffled 77.6M
rows (1.18 GB), sorted them, and spilled 1 GB ([04](screenshots/04_query12_sortmergejoin_bad_detail.png)). The filter-first +
`F.broadcast()` version is 3.2× faster. (2) *Column pruning*: each file is read with an explicit 10-column projection.
(3) *Shuffle tuning*: 64 shuffle partitions instead of the default 200, with AQE coalescing on top. (4) *Partitioned output*: the clean
table is written with `partitionBy("year_month")`, so later month-level queries read 1/24 of the data. (5) *Caching (bonus)*: after
`.cache()`, the plan reads from `InMemoryTableScan` instead of rescanning 24 Parquet files
([05](screenshots/05_query18_cached_inmemory_detail.png)). Three repeated aggregations dropped from 109.9 s to 19.4 s, so the one-time
73.5 s cost is paid back the first time the queries run. Next step: persist `enriched` (or write `trips_clean` first and aggregate from it)
before the final writes, which would remove most of the 21 minutes spent recomputing.

## Key findings

1. **Airports dominate revenue.** JFK and LaGuardia are only 2 of 265 zones but account for **23.1%** of all yellow-cab revenue
   in 2023–24 ($305M and $170M). JFK trips average 16.2 miles, compared with about 2–3 miles for Midtown pickups.
2. **The busiest airport route never changed.** Times Square/Theatre District → LaGuardia was the #1 airport-bound route in **all 24
   months** (3–5.5k trips per month), peaking in October each year.
3. **Credit-card riders tip about 25% of the fare, and 94–96% of them tip.** Evening trips tip the most (26.4%). Cash tips are not recorded in
   the TLC data (always 0), so cash trips (~16% of the total) cannot be compared.
4. **Traffic is getting slower.** Average trip speed fell from 12.3 mph (Jan 2023) to 10.1 mph (Dec 2024), and each 2024 month was slower
   than the same month in 2023. Average fares stayed flat at around $18–21.
5. **Bronx pickups are long and rarely tipped.** Morning Bronx trips average about 10 miles and 40 minutes, with tips under 1% of the fare.
   The likely reason is cash payment and trips into Manhattan.
6. **Fare is mostly distance.** In the GBT model, `trip_distance` has 0.785 feature importance and `duration_min` 0.155. Time of day,
   day of week, and passenger count add almost nothing (R² = 0.897).

## Screenshots

| # | Screenshot | Shows |
|---|---|---|
| 01 | [Spark UI – Jobs](screenshots/01_spark_ui_jobs.png) | All jobs of the run completed successfully |
| 02 | [Spark UI – SQL / DataFrame](screenshots/02_spark_ui_sql_queries.png) | Every query with its duration |
| 03 | [Query details – borough × hour](screenshots/03_query5_borough_hour_broadcast_join_detail.png) ([full DAG](screenshots/03b_query5_full_dag_24file_union.png)) | Filter per scan, 2× BroadcastHashJoin, partial aggregation, AQE coalescing |
| 04 | [Query details – un-optimized join](screenshots/04_query12_sortmergejoin_bad_detail.png) | SortMergeJoin: 77.6M-row shuffle + sort spill |
| 05 | [Query details – cached query](screenshots/05_query18_cached_inmemory_detail.png) | `InMemoryTableScan` replacing the Parquet scan |
| 06 | [`.explain("formatted")`](screenshots/06_explain_formatted_physical_plan.png) | Physical plan with PushedFilters, ReadSchema, BroadcastExchange |
| 07 | [Actions vs. transformations](screenshots/07_actions_vs_transformations.png) | 88 ms (lazy) vs. 10.6 s (action) |
| 08 | [Spark UI – Stages](screenshots/08_spark_ui_stages.png) | Stage-level input / shuffle metrics |
| 09 | [Successful run](screenshots/09_successful_run.png) | Join benchmark, cache benchmark, writes, partition pruning, MLlib results |

The Spark UI screenshots come from the Spark History Server replaying this run's event log (`spark.eventLog.enabled=true`).

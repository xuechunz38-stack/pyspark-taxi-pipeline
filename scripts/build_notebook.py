"""Builds pyspark_pipeline.ipynb from the cell list below (keeps the notebook diff-friendly)."""
import json, pathlib

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md("""
# Week 13 Mini Assignment — PySpark Data Processing Pipeline
**Dataset:** NYC TLC Yellow Taxi trip records, Jan 2023 – Dec 2024 (24 Parquet files, ~1.2 GB compressed, ~78M rows)
+ Taxi Zone lookup table (265 zones) for joins.

Sections
1. Setup & loading
2. Actions vs. transformations (lazy evaluation demo)
3. Transformation pipeline (early filters, `withColumn`, broadcast join, `groupBy`)
4. Spark SQL queries
5. Execution plans (`.explain()`) & optimization
6. Caching benchmark
7. Writing results (partitioned Parquet)
8. MLlib (optional) — fare regression
""")

md("## 1. Setup & loading")
code("""
import time, glob, os
from functools import reduce
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.types import DoubleType, IntegerType, LongType

os.makedirs("spark-events", exist_ok=True)
spark = (SparkSession.builder
         .config("spark.eventLog.enabled", "true")          # lets the Spark History Server replay the UI later
         .config("spark.eventLog.dir", "spark-events")
         .appName("NYC-Taxi-PySpark-Pipeline")
         .master("local[*]")
         .config("spark.driver.memory", "6g")
         .config("spark.sql.shuffle.partitions", "64")      # default 200 is too many for one machine
         .config("spark.sql.adaptive.enabled", "true")       # AQE: coalesce shuffle partitions, skew joins
         .config("spark.sql.autoBroadcastJoinThreshold", str(10 * 1024 * 1024))
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
print("Spark", spark.version, "| UI:", spark.sparkContext.uiWebUrl)
""")
code("""
DATA = "data"
files = sorted(glob.glob(f"{DATA}/yellow/yellow_tripdata_*.parquet"))
total_gb = sum(os.path.getsize(f) for f in files) / 1024**3
print(f"{len(files)} files, {total_gb:.2f} GB on disk")

# The monthly TLC files are not perfectly schema-consistent (e.g. INT32 vs INT64, airport_fee casing),
# so each file is read with an explicit projection + cast and the results are unioned by name.
# Selecting only the needed columns here is also our column-pruning step: Parquet is columnar,
# so the other columns are never read from disk.
COLS = {
    "tpep_pickup_datetime": None, "tpep_dropoff_datetime": None,
    "passenger_count": IntegerType(), "trip_distance": DoubleType(),
    "PULocationID": IntegerType(), "DOLocationID": IntegerType(),
    "payment_type": IntegerType(), "fare_amount": DoubleType(),
    "tip_amount": DoubleType(), "total_amount": DoubleType(),
}
def load(path):
    df = spark.read.parquet(path)
    return df.select([F.col(c).cast(t).alias(c) if t else F.col(c) for c, t in COLS.items()])

trips_raw = reduce(lambda a, b: a.unionByName(b), [load(f) for f in files])

zones = (spark.read.option("header", True).option("inferSchema", True)
         .csv(f"{DATA}/taxi_zone_lookup.csv"))
trips_raw.printSchema()
zones.show(5)
""")
code("""
t = time.time()
n_raw = trips_raw.count()
print(f"raw rows: {n_raw:,}  ({time.time()-t:.1f}s)")
""")

md("""
## 2. Actions vs. Transformations
Transformations (`filter`, `withColumn`, `select`, `groupBy`, `join`) only build a logical plan and return instantly.
Actions (`count`, `show`, `collect`, `write`) trigger a Spark job that actually reads and processes data.
""")
code("""
t0 = time.time()
lazy_df = (trips_raw
           .filter(F.col("trip_distance") > 10)
           .withColumn("fare_per_mile", F.col("fare_amount") / F.col("trip_distance"))
           .groupBy("PULocationID").agg(F.avg("fare_per_mile").alias("avg_fpm")))
t_transform = time.time() - t0
print(f"Transformations defined in {t_transform*1000:.1f} ms — no job ran (check Spark UI: no new job)")
print(type(lazy_df))

t0 = time.time()
result = lazy_df.orderBy(F.desc("avg_fpm")).limit(5).collect()   # ACTION
t_action = time.time() - t0
print(f"Action (collect) took {t_action:.2f} s — this is when Spark actually read the Parquet files")
for r in result: print(r)
""")

md("""
## 3. Transformation pipeline
Filters are applied **first** (right after the projection) so that every later step — derived columns,
the join and the shuffle for `groupBy` — operates on fewer rows. Simple comparison filters on Parquet
columns are pushed into the scan (`PushedFilters` in the plan).
""")
code("""
START, END = "2023-01-01", "2025-01-01"

trips = (trips_raw
    # --- filter 1: valid time window (files contain a few mis-dated records) ---
    .filter((F.col("tpep_pickup_datetime") >= START) & (F.col("tpep_pickup_datetime") < END))
    # --- filter 2: physically plausible trips ---
    .filter((F.col("trip_distance") > 0) & (F.col("trip_distance") < 100))
    # --- filter 3: valid money / passengers ---
    .filter((F.col("fare_amount") > 0) & (F.col("fare_amount") < 500) & (F.col("passenger_count") > 0))
    # --- column transformations ---
    .withColumn("duration_min",
                (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0)
    .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    .withColumn("pickup_dow", F.dayofweek("tpep_pickup_datetime"))
    .withColumn("year_month", F.date_format("tpep_pickup_datetime", "yyyy-MM"))
    .withColumn("tip_pct", F.when(F.col("fare_amount") > 0, F.col("tip_amount") / F.col("fare_amount") * 100))
    .withColumn("payment", F.when(F.col("payment_type") == 1, "Credit card")
                             .when(F.col("payment_type") == 2, "Cash")
                             .otherwise("Other"))
    # filter 4 depends on a derived column, so it must come after withColumn
    .filter((F.col("duration_min") > 1) & (F.col("duration_min") < 180))
    .withColumn("speed_mph", F.col("trip_distance") / (F.col("duration_min") / 60.0))
)

# --- join: zones is tiny (265 rows) -> broadcast join, no shuffle of the 78M-row side ---
pu = zones.select(F.col("LocationID").alias("PULocationID"),
                  F.col("Borough").alias("pu_borough"), F.col("Zone").alias("pu_zone"))
do = zones.select(F.col("LocationID").alias("DOLocationID"),
                  F.col("Borough").alias("do_borough"), F.col("Zone").alias("do_zone"))
enriched = (trips
            .join(F.broadcast(pu), "PULocationID", "left")
            .join(F.broadcast(do), "DOLocationID", "left"))
enriched.select("tpep_pickup_datetime", "pu_zone", "do_zone", "trip_distance",
                "duration_min", "speed_mph", "tip_pct", "payment").show(5, truncate=False)
""")
code("""
# --- groupBy with multiple aggregations ---
t = time.time()
borough_hour = (enriched
    .groupBy("pu_borough", "pickup_hour")
    .agg(F.count("*").alias("trips"),
         F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
         F.round(F.avg("trip_distance"), 2).alias("avg_miles"),
         F.round(F.expr("percentile_approx(duration_min, 0.5)"), 1).alias("median_min"),
         F.round(F.avg("tip_pct"), 2).alias("avg_tip_pct"),
         F.round(F.sum("total_amount") / 1e6, 2).alias("revenue_musd"))
    .orderBy("pu_borough", "pickup_hour"))
borough_hour.show(24)
print(f"{time.time()-t:.1f}s")
""")
code("""
monthly = (enriched.groupBy("year_month")
           .agg(F.count("*").alias("trips"),
                F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
                F.round(F.avg("speed_mph"), 2).alias("avg_speed_mph"))
           .orderBy("year_month"))
monthly.show(24)
""")

md("## 4. Spark SQL queries")
code("""
enriched.createOrReplaceTempView("trips")
zones.createOrReplaceTempView("zones")

# SQL 1: top 10 pickup zones by revenue, with share of total
top_zones = spark.sql('''
    SELECT pu_borough, pu_zone,
           COUNT(*)                         AS trips,
           ROUND(SUM(total_amount)/1e6, 2)  AS revenue_musd,
           ROUND(AVG(trip_distance), 2)     AS avg_miles,
           ROUND(100 * SUM(total_amount) / SUM(SUM(total_amount)) OVER (), 2) AS revenue_share_pct
    FROM trips
    GROUP BY pu_borough, pu_zone
    ORDER BY revenue_musd DESC
    LIMIT 10
''')
top_zones.show(truncate=False)
""")
code("""
# SQL 2: tipping behaviour by payment type and time of day (cash tips are not recorded by TLC)
tips = spark.sql('''
    SELECT payment,
           CASE WHEN pickup_hour BETWEEN 6 AND 11 THEN '1 morning'
                WHEN pickup_hour BETWEEN 12 AND 16 THEN '2 afternoon'
                WHEN pickup_hour BETWEEN 17 AND 21 THEN '3 evening'
                ELSE '4 night' END                       AS daypart,
           COUNT(*)                                     AS trips,
           ROUND(AVG(tip_pct), 2)                       AS avg_tip_pct,
           ROUND(AVG(CASE WHEN tip_amount > 0 THEN 1 ELSE 0 END) * 100, 1) AS pct_trips_tipped
    FROM trips
    WHERE payment IN ('Credit card', 'Cash')
    GROUP BY payment, daypart
    ORDER BY payment, daypart
''')
tips.show(truncate=False)
""")
code("""
# SQL 3: window function — busiest airport-bound route each month
routes = spark.sql('''
    WITH r AS (
        SELECT year_month, pu_zone, do_zone, COUNT(*) AS trips
        FROM trips
        WHERE do_zone IN ('JFK Airport', 'LaGuardia Airport', 'Newark Airport')
        GROUP BY year_month, pu_zone, do_zone
    )
    SELECT year_month, pu_zone, do_zone, trips
    FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY year_month ORDER BY trips DESC) AS rk FROM r)
    WHERE rk = 1
    ORDER BY year_month
''')
routes.show(24, truncate=False)
""")

md("""
## 5. Execution plans & optimization
`explain("formatted")` prints the physical plan. Things to look for:
* **`PushedFilters`** in each `Scan parquet` node → predicates evaluated inside the Parquet reader (row-group skipping via min/max stats).
* **`ReadSchema`** containing only 10 columns → column pruning.
* **`BroadcastHashJoin`** (not `SortMergeJoin`) → no shuffle for the zone join.
* **`HashAggregate` → `Exchange hashpartitioning` → `HashAggregate`** → partial (map-side) aggregation before the shuffle.
* **`AdaptiveSparkPlan`** → AQE re-optimizes at runtime (coalesces shuffle partitions).
""")
code("""
borough_hour.explain("formatted")
""")
code("""
top_zones.explain()   # SQL query: same Catalyst optimizer, same kind of physical plan
""")
code("""
# Comparison: a *badly ordered* pipeline — join (without broadcast hint and with broadcasting disabled)
# BEFORE filtering. Catalyst still pushes simple filters below the join, but the join strategy
# becomes SortMergeJoin, which shuffles the whole trips table.
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
bad = (trips_raw.join(pu, "PULocationID")
       .withColumn("duration_min",
                   (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0)
       .filter((F.col("duration_min") > 1) & (F.col("trip_distance") > 0))
       .groupBy("pu_borough").agg(F.count("*").alias("trips")))
bad.explain()
t = time.time(); bad.collect(); t_bad = time.time() - t
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(10 * 1024 * 1024))

good = (trips_raw.filter((F.col("trip_distance") > 0))
        .withColumn("duration_min",
                    (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0)
        .filter(F.col("duration_min") > 1)
        .join(F.broadcast(pu), "PULocationID")
        .groupBy("pu_borough").agg(F.count("*").alias("trips")))
t = time.time(); good.collect(); t_good = time.time() - t
print(f"SortMergeJoin version: {t_bad:.1f}s   |   filter-first + broadcast version: {t_good:.1f}s")
""")

md("## 6. Caching benchmark (bonus)")
code("""
def run_queries(df):
    t = time.time()
    df.groupBy("pu_borough").count().collect()
    df.groupBy("payment").agg(F.avg("tip_pct")).collect()
    df.groupBy("pickup_hour").agg(F.avg("speed_mph")).collect()
    return time.time() - t

slim = enriched.select("pu_borough", "payment", "tip_pct", "pickup_hour", "speed_mph")
t_nocache = run_queries(slim)

slim.cache()
t = time.time(); slim.count(); t_materialize = time.time() - t   # first action fills the cache
t_cache = run_queries(slim)
print(f"3 queries without cache: {t_nocache:.1f}s")
print(f"cache materialization  : {t_materialize:.1f}s (one-time)")
print(f"3 queries with cache   : {t_cache:.1f}s  -> {t_nocache/t_cache:.1f}x faster")
slim.groupBy("pu_borough").count().explain()   # shows InMemoryTableScan instead of Scan parquet
slim.unpersist()
""")

md("## 7. Write results")
code("""
OUT = "output"
t = time.time()
# Partition the cleaned fact table by month: downstream queries filtering on year_month only read
# the matching directories (partition pruning). repartition by the same key -> ~1 file per partition.
(enriched.repartition("year_month")
    .write.mode("overwrite").partitionBy("year_month")
    .parquet(f"{OUT}/trips_clean"))
borough_hour.coalesce(1).write.mode("overwrite").parquet(f"{OUT}/borough_hour")
top_zones.coalesce(1).write.mode("overwrite").parquet(f"{OUT}/top_zones")
tips.coalesce(1).write.mode("overwrite").parquet(f"{OUT}/tips_by_payment_daypart")
routes.coalesce(1).write.mode("overwrite").parquet(f"{OUT}/top_airport_route_by_month")
monthly.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT}/monthly_summary_csv")
print(f"written in {time.time()-t:.1f}s")

# verify + show partition pruning on read-back
back = spark.read.parquet(f"{OUT}/trips_clean").filter(F.col("year_month") == "2024-06")
print("rows in 2024-06:", back.count())
back.explain()   # look for PartitionFilters: [isnotnull(year_month), (year_month = 2024-06)]
""")

md("## 8. MLlib (optional) — predict fare from trip features")
code("""
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

ml_df = (enriched.filter(F.col("year_month") == "2024-06")      # one month is plenty for a demo
         .select("fare_amount", "trip_distance", "duration_min", "pickup_hour", "pickup_dow",
                 "passenger_count", F.coalesce("pu_borough", F.lit("Unknown")).alias("pu_borough"))
         .sample(0.2, seed=42))
train, test = ml_df.randomSplit([0.8, 0.2], seed=42)

pipe = Pipeline(stages=[
    StringIndexer(inputCol="pu_borough", outputCol="borough_idx", handleInvalid="keep"),
    OneHotEncoder(inputCols=["borough_idx"], outputCols=["borough_vec"]),
    VectorAssembler(inputCols=["trip_distance", "duration_min", "pickup_hour", "pickup_dow",
                               "passenger_count", "borough_vec"], outputCol="features"),
    GBTRegressor(featuresCol="features", labelCol="fare_amount", maxIter=30, maxDepth=5, seed=42),
])
t = time.time()
model = pipe.fit(train)
pred = model.transform(test)
ev = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction")
print(f"train rows {train.count():,} | fit {time.time()-t:.1f}s")
print(f"RMSE = {ev.evaluate(pred, {ev.metricName: 'rmse'}):.2f} USD,  R2 = {ev.evaluate(pred, {ev.metricName: 'r2'}):.3f}")
fi = model.stages[-1].featureImportances
names = ["trip_distance", "duration_min", "pickup_hour", "pickup_dow", "passenger_count"]
print({n: round(float(fi[i]), 3) for i, n in enumerate(names)})
""")
code("""
# Keep the session alive while taking Spark UI screenshots (http://localhost:4040), then stop it.
# spark.stop()
""")

nb = {
    "cells": [
        {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
         **({"outputs": [], "execution_count": None} if t == "code" else {})}
        for t, s in cells
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = pathlib.Path(__file__).resolve().parent.parent / "pyspark_pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)

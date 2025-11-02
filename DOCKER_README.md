# IoT-23 Pipeline - Docker Setup

## Quick Start

### 1. Run Full Pipeline (Stage 1 + Stage 2)
This runs Stage 1 first, then Stage 2 in the same container.
```bash
docker compose run full_pipeline
````

### 2. Run Only Stage 2 (manually, whenever you want)

This skips Stage 1 and just runs Stage 2.

```bash
docker compose run stage2_only
```

### 3. Rebuild / clean run

If you changed code or config:

```bash
docker compose down
docker compose build --no-cache
docker compose up    # (optional: attaches logs if you want to watch)
```

## File Structure

```text
project/
├── data/                    # Mount your raw CSV files here
│   └── raw/
│       └── original/        # Put IoT-23 CSVs here
├── outputs/                 # All outputs get written here
│   ├── stage1_feasibility/
│   └── stage2_prepared/
│       ├── engineered_data.parquet
│       ├── splits/
│       └── train/test/val parquet files
├── analysis/
│   ├── stage1/
│   │   └── run_stage1.py
│   └── stage2/
│       ├── feature_engineering.py
│       ├── device_aggregation.py
│       ├── train_test_split.py
│       └── run_stage2.py
├── config.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## What Actually Runs

### `docker compose run full_pipeline`

1. **Stage 1** (PySpark / analysis.stage1.run_stage1):

   * Data quality analysis
   * Binary / multiclass / malware-family feasibility
   * Graph structure exploration
   * Unsupervised / autoencoder / clustering / GAN feasibility
   * Writes summaries and reports to `outputs/stage1_feasibility/`

2. **Stage 2** (PySpark / analysis.stage2.run_stage2):

   * Feature engineering at scale (Spark, can scale to very large traffic logs)
   * Device-level aggregation
   * Train/val/test splitting
   * Writes parquet outputs to `outputs/stage2_prepared/`

### `docker compose run stage2_only`

* Only runs Stage 2:

  * Uses the engineered feature pipeline
  * Builds parquet datasets + splits for training
  * Useful when you've already run Stage 1 and just want to regenerate training data

You control when to re-run Stage 2. Stage 1 does not auto-run unless you call `full_pipeline`.

## Configuration

Edit `config.py` to change:

* `DATA_SAMPLE_FRACTION` — how much of the data to sample (0.0–1.0).
  Example: `1.0` = full data, `0.1` = 10%.
* `TEST_MODE` — toggle test / dev mode logic in your pipeline.
* `USE_STRATIFIED_SAMPLING` — choose stratified sampling vs random for the Spark read.

You can also override some runtime behavior through environment variables in `docker-compose.yml`, for example:

```yaml
environment:
  - DATA_SAMPLE_FRACTION=1.0
  - SPARK_DRIVER_MEMORY=8g
  - SPARK_EXECUTOR_MEMORY=8g
```

## Memory Settings

By default we give Spark 8g driver / 8g executor in `docker-compose.yml`.
If you hit OOM, bump memory like this:

```yaml
environment:
  - SPARK_DRIVER_MEMORY=16g
  - SPARK_EXECUTOR_MEMORY=16g
```

Then rebuild:

```bash
docker compose down
docker compose build --no-cache
```

## Outputs to Check

After it runs, you should see:

* `outputs/stage1_feasibility/summary.json`

  * High-level findings from Stage 1

* `outputs/stage2_prepared/engineered_data.parquet`

  * Feature-engineered Spark dataset

* `outputs/stage2_prepared/splits/train_set.parquet`

* `outputs/stage2_prepared/splits/val_set.parquet`

* `outputs/stage2_prepared/splits/test_set.parquet`

  * Final train/val/test splits ready for modeling

## Troubleshooting

**It hangs on "Encoding categorical features..."**

* You’re good now: Stage 2 uses Spark with controlled shuffle partitions and skips insane high-cardinality columns, so that step should complete instead of freezing.

**Out of memory:**

```bash
# Increase Spark memory in docker-compose.yml
SPARK_DRIVER_MEMORY=16g
SPARK_EXECUTOR_MEMORY=16g
docker compose down
docker compose build --no-cache
```

**Imports or module not found:**

```bash
# Rebuild the image from scratch to make sure code changes are picked up
docker compose build --no-cache
```

**I just want to rerun Stage 2 with new data but NOT rerun Stage 1:**

```bash
docker compose run stage2_only
```

**I want to run the whole pipeline again from scratch:**

```bash
docker compose run full_pipeline
```

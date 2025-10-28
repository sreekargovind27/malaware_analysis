# Docker Setup for IoT-23 Analysis

Run Stage 1 analysis in Docker with pre-configured Java 11 and PySpark.

## Prerequisites

- Docker installed
- Docker Compose installed (optional but recommended)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 2: Using Docker directly

```bash
# Build image
docker build -t iot23-analysis .

# Run Stage 1 analysis
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/stage1_feasibility:/app/stage1_feasibility \
           iot23-analysis
```

## What Gets Mounted

The Docker setup mounts these directories as volumes:

- `./outputs` → `/app/outputs` (all pipeline outputs)
  - `stage1_feasibility/` - Stage 1 JSON reports
  - `stage2_prepared/` - Engineered features & splits
  - `stage2_quality/` - Validation reports
  - `models_trained/` - Trained models
  - `results/` - Evaluation results
  - `logs/` - Pipeline logs

## Output

After running, check `/app/stage1_feasibility` for:
- `summary.json`
- `data_quality.json`
- `binary_feasibility.json`
- `multiclass_feasibility.json`
- `malware_family_feasibility.json`
- `graph_feasibility.json`
- `autoencoder_feasibility.json`
- `clustering_feasibility.json`
- `gan_candidates.json`

## Troubleshooting

**No data found error?**
- Make sure `data/raw_test/` exists with CSV files
- Check `Config.RAW_DIR_ORIGINAL` in `config.py`

**Permission issues?**
```bash
sudo chown -R $USER:$USER stage1_feasibility/
```

**Rebuild after code changes:**
```bash
docker-compose up --build
```

## Environment

- **Python**: 3.9
- **Java**: OpenJDK 11
- **PySpark**: 3.5.0+
- **OS**: Debian-based Linux

## Running Other Scripts

To run different analysis scripts:

```bash
# Modify docker-compose.yml command, or:
docker-compose run stage1-analysis python your_script.py
```

## Clean Up

```bash
# Remove containers
docker-compose down

# Remove images
docker rmi iot23-analysis

# Remove volumes (careful - deletes data!)
docker-compose down -v
```

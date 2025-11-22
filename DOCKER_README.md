## Memory Settings

**For testing (10% sample):** Default 8GB is fine

**For full 40GB dataset:** Update docker-compose.yml:

```yaml
environment:
  - SPARK_DRIVER_MEMORY=16g
  - SPARK_EXECUTOR_MEMORY=16g
```

## Quick Start Examples

```bash
# Test with 10% sample
docker compose run -e DATA_SAMPLE_FRACTION=0.1 full_pipeline

# Run full dataset
docker compose run full_pipeline

# Run only Stage 1
docker compose run stage1_only

# Run only Stage 2
docker compose run stage2_only
```
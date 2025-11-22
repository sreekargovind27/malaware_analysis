# Official Spark image with Python pre-installed
FROM apache/spark:3.5.1-scala2.12-java17-python3-ubuntu

USER root
WORKDIR /app

# Copy and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY config.py .
COPY run_full_pipeline.py .
COPY analysis/ ./analysis/
COPY scripts/ ./scripts/

# Create directories
RUN mkdir -p outputs/stage1_feasibility \
             outputs/stage2_prepared \
             outputs/stage2_prepared/splits \
             outputs/stage2_quality \
             outputs/models_trained \
             outputs/results \
             outputs/logs

# Environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# IMPORTANT:
# We do NOT force stage2 here.
# We leave entrypoint empty so docker-compose can decide what to run.
ENTRYPOINT []
CMD ["python3", "run_full_pipeline.py"]
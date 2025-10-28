# Official Apache Spark image - NO Jupyter, just Spark + Python
FROM apache/spark:3.5.1-scala2.12-java17-python3-ubuntu

# Switch to root
USER root

WORKDIR /app

# Install Python if not available (and pip)
RUN apt-get update && apt-get install -y python3 python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Copy and install additional Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY config.py .
COPY analysis/ ./analysis/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Create directories
RUN mkdir -p outputs/stage1_feasibility \
             outputs/stage2_prepared \
             outputs/stage2_prepared/splits \
             outputs/stage2_quality \
             outputs/models_trained \
             outputs/results \
             outputs/logs

# Fix permissions for spark user
RUN chown -R spark:spark /app

# Environment variables
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3
ENV PYTHONPATH=/app:$PYTHONPATH

# Run as regular user (security)
USER spark

# Use full path to python3
CMD ["/usr/bin/python3", "analysis/stage1/run_stage1.py"]
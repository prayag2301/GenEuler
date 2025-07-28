# Use the official PyTorch+CUDA runtime image so torch.cuda is available
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install only the extra system deps you need
# install netcat for the “wait‐for‐tcp” loop
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      postgresql-client \
      build-essential \
      cmake \
      python3-dev \
      netcat && \
    rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /app

# Copy & install your Python requirements (no torch/torchvision/tb)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# Copy your SAC agent code and set PYTHONPATH
COPY src/ ./src/
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# **Tell PyTorch to split large CUDA allocations into ≤128 MiB blocks:**
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Expose FastAPI port
EXPOSE 8008

# Pass through your DB envs and enable CUDA in config.py
ENV POSTGRES_HOST=sac_db \
    POSTGRES_DB=sac_db \
    POSTGRES_USER=postgres \
    POSTGRES_PASSWORD=berlin2000 \
    USE_CUDA=1

# Block until sac_db:5432 resolves & accepts connections
ENTRYPOINT ["sh","-c","echo \"Waiting for ${POSTGRES_HOST}:5432…\"; \
  until nc -z ${POSTGRES_HOST} 5432; do sleep 1; done; \
  echo \"DB is ready—starting Uvicorn\"; \
  exec uvicorn src.api:app --host 0.0.0.0 --port 8008 --reload"]

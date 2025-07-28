# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set non-interactive environment variable
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    wget \
    git \
    libgl1-mesa-glx \
    curl \
    postgresql-client \
    && apt-get clean

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Set PATH to include conda
ENV PATH="/opt/conda/bin:$PATH"
ENV MAMBA_ROOT_PREFIX="/opt/conda"

# Install mamba for fast dependency solving
RUN conda install -n base -c conda-forge mamba --yes

# Create a conda environment for reward_to service
RUN mamba create -n reward-env -c conda-forge python=3.10 fastapi uvicorn psycopg2-binary numpy requests --yes

# Set working directory
WORKDIR /app

# Copy requirements.txt if available and install dependencies
COPY requirements.txt .
RUN conda run -n reward-env pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY src/ ./src/

# Ensure the src directory is a package
RUN touch ./src/__init__.py

# Set PYTHONPATH so that the src folder is in the module search path
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Expose the API port (e.g., 8006)
EXPOSE 8006

# Set environment variables for database and TO service connection
ENV POSTGRES_HOST=reward_topology_db
ENV POSTGRES_DB=reward_topology_db
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=berlin2000

# Activate the conda environment and run Uvicorn
ENTRYPOINT ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "reward-env", "uvicorn"]
CMD ["src.api:app", "--host", "0.0.0.0", "--port", "8006", "--reload"]

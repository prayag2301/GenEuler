# Use Ubuntu as the base image
FROM ubuntu:22.04

# Set environment variables for non-interactive installation
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

# Set Conda environment variables
ENV PATH="/opt/conda/bin:$PATH"
ENV MAMBA_ROOT_PREFIX="/opt/conda"

# Install mamba for faster dependency solving
RUN conda install -n base -c conda-forge mamba --yes

# Create a conda environment for reward_aggregator
RUN mamba create -n reward-env -c conda-forge python=3.10 fastapi uvicorn psycopg2-binary numpy requests --yes

# Set working directory and copy requirements
WORKDIR /app
COPY requirements.txt .
RUN conda run -n reward-env pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Ensure the src directory is a package
RUN touch ./src/__init__.py

# Set PYTHONPATH so that the src folder is in the module search path
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Expose the API port (for example, 8004)
EXPOSE 8004

# Set environment variables for the reward database connection
ENV POSTGRES_HOST=reward_db
ENV POSTGRES_DB=reward_db
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=berlin2000

# Activate the conda environment and run Uvicorn
ENTRYPOINT ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "reward-env", "uvicorn"]
CMD ["src.api:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]

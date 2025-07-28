# Use Ubuntu as the base image
FROM ubuntu:22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    wget \
    git \
    libgl1-mesa-glx \
    libgmp-dev \
    libmpfr-dev \
    libboost-all-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libhdf5-dev \
    curl \
    gmsh \
    postgresql-client \ 
    && apt-get clean

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Set Conda environment variables
ENV PATH="/opt/conda/bin:$PATH"
ENV MAMBA_ROOT_PREFIX="/opt/conda"

# Install `mamba` for better dependency solving
RUN conda install -n base -c conda-forge mamba --yes

# Create a Conda environment with FEniCSx and dependencies
RUN mamba create -n fenicsx-env -c conda-forge \
    python=3.10 \
    numpy \
    scipy \
    mpi4py \
    petsc4py \
    h5py \
    fenics-dolfinx \
    pyvista \
    fastapi \
    uvicorn \
    meshio \ 
    psycopg2 \ 
    --yes

# Set Conda environment as default
SHELL ["/bin/bash", "-c"]

# Activate Conda environment and install remaining dependencies via pip
WORKDIR /app
COPY requirements.txt .
RUN conda run -n fenicsx-env pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Ensure assets directory exists for storing topology results
RUN mkdir -p /app/src/assets/

# Set PYTHONPATH to include the src directory
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Expose API port
EXPOSE 8002

# Add environment variables for database connection
ENV POSTGRES_HOST=topology_db
ENV POSTGRES_DB=topology_db
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=berlin2000

# Activate Conda environment and run Uvicorn
ENTRYPOINT ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "fenicsx-env", "uvicorn"]
CMD ["src.api:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]

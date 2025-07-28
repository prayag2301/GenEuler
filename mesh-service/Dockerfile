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
    libhdf5-dev \
    libgmp-dev \
    libmpfr-dev \
    libboost-all-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    curl \
    gmsh \
    postgresql-client \
    && apt-get clean

# Install Python dependencies that do not require Conda (if any)
RUN pip3 install --no-cache-dir numpy gmsh h5py

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Set Conda environment variables
ENV PATH="/opt/conda/bin:$PATH"
ENV MAMBA_ROOT_PREFIX="/opt/conda"

# Install mamba for faster dependency solving
RUN conda install -n base -c conda-forge mamba --yes

# Create a Conda environment with FEniCSx and necessary packages.
# Note: We remove PyVista here since our current code no longer requires it.
RUN mamba create -n fenicsx-env -c conda-forge \
    python=3.10 \
    numpy \
    scipy \
    mpi4py \
    petsc4py \
    h5py \
    fenics-dolfinx \
    fastapi \
    uvicorn \
    gmsh \
    tetgen \
    psycopg2 \
    --yes

# Copy requirements.txt into the container
WORKDIR /app
COPY requirements.txt .

# Install additional Python packages using pip within the Conda environment.
# (These can include packages not available or already installed via Conda.)
RUN conda run -n fenicsx-env pip install --no-cache-dir -r requirements.txt

# Copy application source code (assume your code is in the src/ folder)
COPY src/ ./src/

# Set PYTHONPATH to include the src directory
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Expose the API port
EXPOSE 8003

# Add environment variables for database connection
ENV POSTGRES_HOST=mesh_db
ENV POSTGRES_DB=mesh_db
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=berlin2000

# Activate Conda environment and run Uvicorn
# ENTRYPOINT ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "fenicsx-env", "uvicorn"]
# CMD ["src.api:app", "--host", "0.0.0.0", "--port", "8003"]

ENTRYPOINT ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "fenicsx-env", "uvicorn"]
CMD ["src.api:app", "--host", "0.0.0.0", "--port", "8003"]


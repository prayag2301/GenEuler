# Use Ubuntu as the base image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

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
    libhdf5-mpi-dev \
    curl \
    gmsh \
    postgresql-client \
    && apt-get clean

# Install basic Python packages (avoid installing h5py here)
RUN pip install numpy meshio gmsh

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"
ENV MAMBA_ROOT_PREFIX="/opt/conda"

# Set strict channel priority to enforce conda-forge packages
RUN conda config --set channel_priority strict

# Install mamba from conda-forge for faster dependency resolution
RUN conda install -n base -c conda-forge mamba --yes

# Create a Conda environment with FEniCSx and MPI-enabled dependencies, using openmpi
RUN mamba create -n fenicsx-env -c conda-forge \
    python=3.10 \
    numpy \
    scipy \
    mpi4py \
    openmpi \
    petsc4py \
    "h5py[mpi]" \
    fenics-dolfinx \
    pyvista \
    fastapi \
    uvicorn \
    psycopg2 \
    hdf5 \
    --yes

# (Optional) Print configuration to debug h5py settings
RUN conda run -n fenicsx-env python -c "import h5py; print('MPI support:', h5py.get_config().mpi)"

SHELL ["/bin/bash", "-c"]

WORKDIR /app
COPY requirements.txt .
# Ensure requirements.txt does not install h5py again
RUN sed -i '/h5py/d' requirements.txt
RUN conda run -n fenicsx-env pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONPATH="/app/src:$PYTHONPATH"
EXPOSE 8001

ENV POSTGRES_HOST=fem_db
ENV POSTGRES_DB=fem_db
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=berlin2000

ENTRYPOINT ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "fenicsx-env", "uvicorn"]
CMD ["src.api:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
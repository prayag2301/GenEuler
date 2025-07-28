"""
topology_solver.py

Handles PDE-based or SIMP-based topology optimization in FEniCSx.
"""

import numpy as np
import dolfinx
from dolfinx import mesh, fem
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import petsc
from dolfinx.io import XDMFFile
import basix
from basix.ufl import blocked_element
import json
import os
import requests
import h5py
from db import get_latest_fem_iteration, save_topology_results

ASSET_DIR = "/app/src/assets/"

class TopologySolver:
    def __init__(self, mesh, volume_fraction=0.5, penalization=3.0, filter_radius=1.5):
        """
        Initializes the topology optimization problem.
        :param mesh: The FEniCSx mesh from the FEM service (loaded from XDMF).
        :param volume_fraction: Target volume fraction (how much material remains).
        :param penalization: SIMP penalty factor (typically 3.0).
        :param filter_radius: Filter size for smoothing material density (not yet used here).
        """
        self.mesh = mesh
        self.volume_fraction = volume_fraction
        self.penalization = penalization
        self.filter_radius = filter_radius

        # Define a scalar density function (1 means full material, 0 means void).
        P1 = fem.functionspace(self.mesh, ("CG", 1))
        self.density = fem.Function(P1)
        # Initialize density to 1 (fully solid).
        self.density.interpolate(lambda x: np.ones(x.shape[1]))

    def optimize_structure(self, stress_field):
        """
        Runs SIMP-based topology optimization with an input 'stress_field'.
        Typically, this could be an actual stress or displacement function.
        :param stress_field: A dolfinx.fem.Function or UFL expression.
        :return: Optimized density array (1D NumPy).
        """
        # Local references
        penal = self.penalization
        density = self.density

        # Penalize the density field (SIMP approach)
        rho_p = density**penal

        # Example objective: minimize ∫ (stress_field · stress_field * rho_p) dx
        strain_energy = ufl.inner(stress_field, stress_field) * rho_p * ufl.dx
        J = fem.assemble_scalar(fem.form(strain_energy))

        # Volume constraint = ∫ density dx - volume_fraction * ∫ 1 dx
        vol_constraint_expr = density - self.volume_fraction
        vol_constraint = vol_constraint_expr * ufl.dx
        V_val = fem.assemble_scalar(fem.form(vol_constraint))

        # --- Option 2: Scale the objective term ---
        scale_factor = 1000.0  # adjust scale factor as required
        update_factor = 0.2   # using the original factor 0.2
        density.x.array[:] = np.maximum(0.01, np.minimum(1.0, density.x.array[:] - update_factor * scale_factor * J / V_val))

        # Get iteration number from FEM database (do not add 1 here if your get_latest_fem_iteration already returns desired value)
        iteration = get_latest_fem_iteration()

        # Build file paths directly in the assets folder
        xdmf_filename = os.path.join(ASSET_DIR, f"topology_density_{iteration}.xdmf")
        h5_filename   = os.path.join(ASSET_DIR, f"topology_density_{iteration}.h5")
        json_filename = os.path.join(ASSET_DIR, f"topology_results_{iteration}.json")

        # Save results to the database: now pass the JSON file path as the density_results_json
        save_topology_results(
            iteration=iteration,
            volume_fraction=self.volume_fraction,
            penalization=self.penalization,
            filter_radius=self.filter_radius,
            density_json=json_filename,          # This column now holds the path
            xdmf_path=xdmf_filename,
            h5_path=h5_filename,
        )

        print(f"✅ Topology optimization results for iteration {iteration} stored in database.")

        return density.x.array

    
    def _send_updated_density_to_fem(self, density_data):
        """
        Sends the updated density field to the FEM service for a second FEM iteration.
        """
        FEM_SERVICE_URL = "http://fem_service:8001/second_fem"
        
        try:
            response = requests.post(FEM_SERVICE_URL, json=density_data)
            if response.status_code == 200:
                print("✅ Successfully triggered second FEM analysis.")
            else:
                print(f"❌ Failed to trigger second FEM analysis: {response.text}")
        except Exception as e:
            print(f"❌ Error in sending updated density to FEM: {e}")


def load_mesh_and_displacement(xdmf_path, h5_path):
    """
    Loads the mesh from XDMF and the displacement function from HDF5.
    """
    print(f"🔵 Loading mesh from {xdmf_path}...", flush=True)

    # ✅ Load the mesh from XDMF file
    with XDMFFile(MPI.COMM_WORLD, xdmf_path, "r") as xdmf:
        loaded_mesh = xdmf.read_mesh(name="mesh")

    print(f"🔵 Loading displacement function from {h5_path}...", flush=True)

    # ✅ Define the function space for the displacement
    basix_element = basix.create_element(
        family=basix.ElementFamily.P,
        celltype=basix.CellType.tetrahedron,
        degree=1,
        lagrange_variant=basix.LagrangeVariant.equispaced
    )
    ufl_element = basix.ufl.wrap_element(basix_element)
    vector_element = blocked_element(ufl_element, (3,))
    V = fem.functionspace(loaded_mesh, vector_element)
    disp_fun = fem.Function(V)

    # ✅ Read displacement values from the correct HDF5 dataset
    with h5py.File(h5_path, "r") as h5_file:
        dataset_path = "/Function/f/0"  # ✅ Correct dataset path from XDMF
        if dataset_path not in h5_file:
            raise RuntimeError(f"❌ '{dataset_path}' dataset not found in {h5_path}!")
        # Flatten the displacement array (from shape (n,3) to (n*3,))
        disp_array = np.array(h5_file[dataset_path]).flatten()
        
        if disp_array.shape[0] != disp_fun.x.array.shape[0]:
            raise RuntimeError("❌ Mismatch in displacement data and function space DOFs!")
        
        # ✅ Assign flattened values to the function
        disp_fun.x.array[:] = disp_array

    print("✅ Successfully loaded displacement function from HDF5.")

    return loaded_mesh, disp_fun

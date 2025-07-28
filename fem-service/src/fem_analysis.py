import petsc4py
petsc4py.init()

import os
import json
import numpy as np
import dolfinx
from dolfinx import mesh, fem
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import basix
from basix.ufl import blocked_element
from dolfinx.io import XDMFFile
import h5py
import requests
from db import get_db_connection, get_latest_iteration
import dolfinx.fem.petsc
from dolfinx.fem.petsc import set_bc

# Global nondimensionalization parameters.
L_char = 0.005                # Characteristic length: 5 mm.
E_dim = 210e6                 # SI Young's modulus (Pa).
# (Note: we no longer apply any extra force scaling.)

# After nondimensionalization, the mesh and boundary points are scaled by L_char.
BOUNDING_BOX = {"x": [-0.05, 0.05], "y": [0.0, 0.13], "z": [-0.13, 0.13]}

# Nondimensionalize the attachment point.
ATTACHMENT_POSITION = np.array([0.0, 0.13, -0.11]) / L_char  # becomes [0.0, 26.0, -22.0].

# Set a tolerance suitable for the nondimensional mesh.
TOLERANCE = 1e-2

MESH_SERVICE_URL = "http://mesh_service:8003"
SHARED_ASSETS_DIR = "/app/src/assets/"

class FEMAnalysis:
    def __init__(self, updated_density=None):
        self.xdmf_path, self.h5_path = self._fetch_mesh_paths()
        self.mesh, self.disp_fun = self._load_mesh()
        self.V = self._setup_function_space()
        self.boundary_markers = self._setup_boundaries()
        self.updated_density = updated_density
        print(f"📌 Using Mesh:\n  XDMF: {self.xdmf_path}\n  H5: {self.h5_path}", flush=True)

    def _fetch_mesh_paths(self):
        print("🔵 Fetching latest mesh iteration from Mesh Service...", flush=True)
        response = requests.get(f"{MESH_SERVICE_URL}/latest_mesh_iteration")
        latest_iteration = response.json().get("iteration")
        response = requests.get(f"{MESH_SERVICE_URL}/get_mesh_info/{latest_iteration}")
        mesh_data = response.json()
        xdmf_path = mesh_data.get("xdmf_path")
        h5_path = mesh_data.get("h5_path")
        if not xdmf_path or not h5_path:
            raise RuntimeError("❌ Mesh paths missing!")
        print(f"✅ Using mesh files: {xdmf_path}, {h5_path}", flush=True)
        return xdmf_path, h5_path

    def _load_mesh(self):
        print(f"🔵 Loading mesh from {self.xdmf_path}...", flush=True)
        with XDMFFile(MPI.COMM_WORLD, self.xdmf_path, "r") as xdmf:
            loaded_mesh = xdmf.read_mesh(name="mesh")
        # First, convert from mm to m, then nondimensionalize using L_char.
        print("🔵 Converting mesh coordinates from mm to m and nondimensionalizing...", flush=True)
        loaded_mesh.geometry.x[:] /= 1000.0
        loaded_mesh.geometry.x[:] /= L_char
        
        # === Diagnostic: Compute and print cell volumes ===
        tdim = loaded_mesh.topology.dim
        # Force creation of cell-to-node connectivity.
        loaded_mesh.topology.create_connectivity(tdim, 0)
        # Retrieve the connectivity array *and reshape it*
        # For tetrahedral cells we expect 4 nodes per cell.
        cells = loaded_mesh.topology.connectivity(tdim, 0).array.reshape((-1, 4))  
        coords = loaded_mesh.geometry.x
        volumes = []
        for cell in cells:
            # cell is now an array of 4 indices.
            x0, x1, x2, x3 = coords[cell[0]], coords[cell[1]], coords[cell[2]], coords[cell[3]]
            # Compute volume of tetrahedron: |det([x1-x0, x2-x0, x3-x0])| / 6.
            vol = np.abs(np.linalg.det(np.column_stack((x1 - x0, x2 - x0, x3 - x0)))) / 6.0
            volumes.append(vol)
        volumes = np.array(volumes)
        print(f"🛠 Mesh cell volume diagnostics: min volume = {np.min(volumes):.3e}, max volume = {np.max(volumes):.3e}", flush=True)
        # ======================================================

        print(f"🔵 Loading displacement function from {self.h5_path}...", flush=True)
        V_temp = fem.functionspace(loaded_mesh, ("CG", 1, (3,)))
        total_dofs = V_temp.dofmap.index_map.size_global * V_temp.dofmap.index_map_bs
        expected = 3 * loaded_mesh.geometry.x.shape[0]
        print(f"DEBUG: V_temp has {total_dofs} DOFs (expected {expected}).", flush=True)
        disp_fun = fem.Function(V_temp)
        comm = MPI.COMM_WORLD
        if comm.rank == 0:
            with h5py.File(self.h5_path, "r") as h5_file:
                func_group = h5_file.get("Function")
                f_group = func_group.get("f")
                dataset_keys = list(f_group.keys())
                disp_array = np.array(f_group[dataset_keys[0]]).flatten()
        else:
            disp_array = None
        disp_array = comm.bcast(disp_array, root=0)
        if disp_array.shape[0] != total_dofs:
            raise RuntimeError("❌ Mismatch in displacement data and function space DOFs!")
        disp_fun.x.array[:] = disp_array.reshape(-1)
        print("✅ Displacement function loaded.", flush=True)
        return loaded_mesh, disp_fun

    def _setup_boundaries(self):
        print("🔵 Setting up boundary conditions...", flush=True)
        # Floor boundary: facets where the nondimensional y is minimal.
        floor_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh, self.mesh.topology.dim - 1,
            lambda x: np.isclose(x[1], np.min(self.mesh.geometry.x[:,1]), atol=1e-2)
        )
        print(f"👉 Floor boundary: {len(floor_facets)} facets found.", flush=True)
        floor_markers = np.full(len(floor_facets), 1, dtype=np.int32)
    
        # Attachment boundary: here we use a more restrictive criterion.
        # In the nondimensional mesh the x‑coordinates range roughly from –10 to 10.
        # If we want a small (e.g. localized) load near the center, we might choose:
        attachment_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh, self.mesh.topology.dim - 1,
            lambda x: np.abs(x[0]) < 0.5  # tighter than before (0.5 nondimensional units).
        )
        print(f"👉 Attachment boundary: {len(attachment_facets)} facets found.", flush=True)
        attachment_markers = np.full(len(attachment_facets), 2, dtype=np.int32)
    
        all_facets = np.concatenate((floor_facets, attachment_facets))
        all_markers = np.concatenate((floor_markers, attachment_markers))
        meshtags = dolfinx.mesh.meshtags(self.mesh, self.mesh.topology.dim - 1, all_facets, all_markers)
        return meshtags

    def _setup_function_space(self):
        print("🔵 Setting up function space...", flush=True)
        # Use the tuple notation to create a vector function space with 3 components.
        V = fem.functionspace(self.mesh, ("CG", 1, (3,)))
        block_size = V.dofmap.index_map_bs
        total_dofs = V.dofmap.index_map.size_global * block_size
        expected = 3 * self.mesh.geometry.x.shape[0]
        if total_dofs != expected:
            print(f"⚠️ Unexpected DOF count: {total_dofs} vs expected {expected}.", flush=True)
        else:
            print(f"✅ Vector function space has the expected {expected} DOFs.", flush=True)
        return V

    def _apply_forces(self, u, v):
        print("🔵 Applying forces to FEM model...", flush=True)
        # --- Body force due to gravity (keep in nondimensional units) ---
        # we nondimensionalize force density f = ρ g → f' = (ρ g L_char) / E_dim
        rho = 7850      # kg/m³
        g   = 9.81      # m/s²
        f_nd = rho * g * L_char / E_dim
        gravity_vec = ufl.as_vector([0.0, -f_nd, 0.0])
        grav_const  = fem.Constant(self.mesh, PETSc.ScalarType(gravity_vec))
        body_force  = ufl.dot(grav_const, v) * ufl.dx
    
        # --- Attachment force as a Neumann term ---
        # Now, the ATTACHMENT_POSITION is already nondimensional.
        attachment_point = ATTACHMENT_POSITION  # nondimensional coordinates.
        tol = 0.1   # Use a tight tolerance (0.5 nondimensional units)
        print("🔍 Checking attachment point proximity...", flush=True)
        all_pts = self.mesh.geometry.x
        close_pts = [p for p in all_pts if np.linalg.norm(p - attachment_point) < tol]
        if len(close_pts) == 0:
            raise ValueError("❌ No mesh points found near the attachment point!")
        print(f"✅ Found {len(close_pts)} points near the attachment.", flush=True)
        print("Attachment nodes (first 5):", np.array(all_pts)[np.where(np.linalg.norm(all_pts - attachment_point, axis=1) < tol)][:5])
    
        ds_attach = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.boundary_markers)
        # Define the attachment force (no extra scaling now).
        # point‐load magnitude originally 800 N → nondim traction T' = T·L_char²/E_dim
        force_vec = np.array([0.0, -800.0 * L_char**2 / E_dim, 0.0], dtype=np.float64)
        force_const = fem.Constant(self.mesh, PETSc.ScalarType(ufl.as_vector(force_vec)))
        F_attach   = ufl.dot(force_const, v) * ds_attach(2)
        print(f"✅ Applying attachment force {force_vec} over marker 2.", flush=True)
    
        # Store diagnostic info.
        self._attachment_dofs = fem.locate_dofs_geometrical(
            self.V, lambda x: np.linalg.norm(x.T - attachment_point, axis=1) < tol
        )
        print(f"[DEBUG] Number of attachment DOFs: {len(self._attachment_dofs)}", flush=True)
    
        return body_force + F_attach
    
    def run_analysis(self, updated_density=None):
        try:
            iteration = get_latest_iteration() + 1
            print(f"📌 Running FEM analysis for iteration {iteration}.", flush=True)
    
            num_cells = self.mesh.topology.index_map(self.mesh.topology.dim).size_global
            if num_cells == 0:
                raise ValueError("❌ No cells in mesh! Mesh generation failed.")
    
            nodes = self.mesh.geometry.x
            print(f"[DEBUG] Nondimensional mesh has {nodes.shape[0]} nodes.")
            print(f"[DEBUG] Mesh Y-range (nondim): min={np.min(nodes[:,1]):.3e}, max={np.max(nodes[:,1]):.3e}")
    
            # ——— Nondimensional material properties ———
            # We choose stress‐scale σ₀ = E_dim, length‐scale L_char,
            # so our nondim Young’s modulus is unity:
            E_nd = 1.0
            nu = 0.3
            mu_nd     = 1.0 / (2.0 * (1.0 + nu))
            lmbda_nd  = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            print(f"[DEBUG] Nondimensional Material properties: E={E_nd:.3e}, nu={nu}, mu={mu_nd:.3e}, lambda={lmbda_nd:.3e}", flush=True)
    
            # Get the vector function space (already nondimensional).
            self.V = fem.functionspace(self.mesh, ("CG", 1, (3,)))
            total_dofs = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
            expected = 3 * self.mesh.geometry.x.shape[0]
            if total_dofs != expected:
                print(f"⚠️ Unexpected DOF count: {total_dofs} vs expected {expected}.", flush=True)
            else:
                print(f"✅ Vector function space has the expected {expected} DOFs.", flush=True)
    
            def epsilon(u):
                return ufl.sym(ufl.grad(u))
    
            def sigma(u):
                return lmbda_nd * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2 * mu_nd * epsilon(u)
    
            U_trial = ufl.TrialFunction(self.V)
            V_test = ufl.TestFunction(self.V)
            a = ufl.inner(sigma(U_trial), epsilon(V_test)) * ufl.dx
    
            L_term = self._apply_forces(U_trial, V_test)
    
            print("🔵 Assembling the system...", flush=True)
            A = dolfinx.fem.petsc.assemble_matrix(fem.form(a))
            A.assemble()
            # A.view()
            b = dolfinx.fem.petsc.assemble_vector(fem.form(L_term))
            b_local = b.getArray()
            print(f"[DEBUG] Norm of load vector b (before BC): {np.linalg.norm(b_local):.3e}", flush=True)
    
            # Apply Dirichlet BC on the floor.
            # Apply Dirichlet BC (fix displacement on the floor).
            # Fix the floor as before:
            zero_vec = fem.Constant(self.mesh, np.array([0.0, 0.0, 0.0], dtype=np.float64))
            floor_dofs = fem.locate_dofs_geometrical(
                self.V,
                lambda x: np.isclose(x[1], np.min(self.mesh.geometry.x[:,1]), atol=1e-6)
            )
            bc_floor = fem.dirichletbc(zero_vec, floor_dofs, self.V)

            # Additional constraint: pin one node completely to remove rigid body motion.
            def fixed_node(x):
                # Select the node(s) that have minimum x and y coordinates.
                return np.logical_and(
                    np.isclose(x[0], np.min(x[0]), atol=1e-8),
                    np.isclose(x[1], np.min(x[1]), atol=1e-8)
                )

            fixed_dofs = fem.locate_dofs_geometrical(self.V, fixed_node)
            bc_fixed = fem.dirichletbc(zero_vec, fixed_dofs, self.V)

            # Apply both boundary conditions.
            dolfinx.fem.petsc.set_bc(b, [bc_floor, bc_fixed])

            b_after = b.getArray()
            print(f"[DEBUG] Norm of load vector b (after BC): {np.linalg.norm(b_after):.3e}", flush=True)
    
            # Choose a solver & preconditioner
            PETSc.Options().setValue("ksp_type", "gmres")
            PETSc.Options().setValue("pc_type", "gamg")
            # Set a reasonable tolerance and iteration cap
            # (either via Options or directly on ksp)
            ksp = PETSc.KSP().create(MPI.COMM_WORLD)
            ksp.setOperators(A)
            # Direct API call avoids overflow
            ksp.setTolerances(rtol=1e-6, max_it=10000)
            ksp.setErrorIfNotConverged(True)
            ksp.setFromOptions()
    
            x = A.createVecRight()
            ksp.solve(b, x)
            sol_array = x.getArray().copy()
    
            u_sol = fem.Function(self.V)
            u_sol.x.array[:] = sol_array
    
            print(f"[DEBUG] Displacement vector (first 10 entries): {u_sol.x.array[:10]}", flush=True)
            print(f"[DEBUG] Min displacement: {np.min(u_sol.x.array):.12e}, Max displacement: {np.max(u_sol.x.array):.12e}", flush=True)
            print(f"[DEBUG] Norm of displacement: {np.linalg.norm(u_sol.x.array):.12e}", flush=True)
    
            print("✅ FEM Solver succeeded!", flush=True)
    
            os.makedirs(SHARED_ASSETS_DIR, exist_ok=True)
            xdmf_filename = os.path.join(SHARED_ASSETS_DIR, f"fem_results_{iteration}.xdmf")
            h5_filename = os.path.join(SHARED_ASSETS_DIR, f"fem_results_{iteration}.h5")
            json_filename = os.path.join(SHARED_ASSETS_DIR, f"fem_results_{iteration}.json")
            with XDMFFile(MPI.COMM_WORLD, xdmf_filename, "w") as xdmf_file:
                xdmf_file.write_mesh(self.mesh)
                xdmf_file.write_function(u_sol, t=0.0)
            displacement_data = {"iteration": iteration, "displacement": u_sol.x.array.tolist()}
            with open(json_filename, "w") as f_out:
                json.dump(displacement_data, f_out, indent=4)
            print(f"✅ FEM results stored (iteration {iteration}).", flush=True)
            return {"iteration": iteration, "xdmf_file": xdmf_filename,
                    "h5_file": h5_filename, "json_file": json_filename}
    
        except Exception as e:
            print(f"❌ FEM Analysis Error: {e}", flush=True)
            raise RuntimeError(e)
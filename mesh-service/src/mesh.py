import os
import sys
import json
import numpy as np
import subprocess
from scipy.spatial import KDTree, ConvexHull, Delaunay
import logging
import h5py
from dolfinx.io import XDMFFile
from mpi4py import MPI
from db import get_db_connection, get_latest_iteration, store_mesh_metadata
import gmsh
from typing import List, Set
from pathlib import Path

from dolfinx.mesh import create_mesh
from dolfinx.fem import coordinate_element  # Correct import location
from dolfinx.cpp.mesh import CellType
from petsc4py import PETSc
import tetgen
import gc


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Directories
MESH_DIR = "/app/src/assets/"
os.makedirs(MESH_DIR, exist_ok=True)
ENV_STATE_DIR = os.path.join(MESH_DIR, "env_states")
os.makedirs(ENV_STATE_DIR, exist_ok=True)

# Constants for bounding box and attachment point
BOUNDING_BOX = {
    "x": [-50, 50],
    "y": [0, 130],
    "z": [-130, 130]
}
ATTACHMENT_POSITION = [0, 130, -110]
TOLERANCE = 5  # Minimum allowed distance from the attachment point

########################################################################
# Custom MSH File I/O (without Meshio)
########################################################################

def write_msh_file(filename, points, cells, surface=None):
    """
    Write a mesh to a MSH 2.2 ASCII file.
    - points:   (N×3) array of node coordinates
    - cells:    (M×4) array of tetrahedra (0-based indices)
    - surface:  optional (K×3) array of triangles (0-based indices)
    """
    with open(filename, 'w') as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        # --- nodes ---
        f.write("$Nodes\n")
        f.write(f"{len(points)}\n")
        for i, (x,y,z) in enumerate(points, start=1):
            f.write(f"{i} {x} {y} {z}\n")
        f.write("$EndNodes\n")
        # --- elements (first surfaces if any, then tets) ---
        n_surf = len(surface) if surface is not None else 0
        n_tet  = len(cells)
        f.write("$Elements\n")
        f.write(f"{n_surf + n_tet}\n")
        eid = 1
        # write triangles as element type 2
        if surface is not None:
            for tri in surface:
                a,b,c = tri
                # type 2 = triangle, 0 tags
                f.write(f"{eid} 2 0 {a+1} {b+1} {c+1}\n")
                eid += 1
        # write tetrahedra as element type 4
        for tet in cells:
            a,b,c,d = tet
            f.write(f"{eid} 4 0 {a+1} {b+1} {c+1} {d+1}\n")
            eid += 1
        f.write("$EndElements\n")
    logger.info(f"Custom MSH file written to {filename}.")

    

def read_msh_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    points = []
    cells = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "$Nodes":
            i += 1
            n_nodes = int(lines[i].strip())
            i += 1
            for _ in range(n_nodes):
                parts = lines[i].strip().split()
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                i += 1
        elif line == "$Elements":
            i += 1
            n_elems = int(lines[i].strip())
            i += 1
            for _ in range(n_elems):
                parts = lines[i].strip().split()
                elem_type = int(parts[1])
                if elem_type == 4:                              # only tetrahedra
                    n_tags    = int(parts[2])
                    # node IDs start at parts[3 + n_tags]
                    node_ids  = parts[3 + n_tags : 3 + n_tags + 4]
                    cell      = [int(x) - 1 for x in node_ids]
                    cells.append(cell)
                # otherwise skip triangles/quads/etc.
                i += 1
        else:
            i += 1
    return np.array(points), np.array(cells, dtype=int)

########################################################################
# Mesh Environment and Core Mesh Operations (Using gmsh via subprocess)
########################################################################

class MeshEnvironment:
    def __init__(self, mesh_file=None):
        """
        Initialize the mesh environment.
        If mesh_file exists, load its points and tetrahedral cells;
        otherwise, initialize with empty arrays.
        """
        if mesh_file and os.path.exists(mesh_file):
            try:
                points, cells = read_msh_file(mesh_file)
                self.points = points  # shape (n_points, 3)
                self.cells = {"tetra": cells}
                logger.info(f"Loaded mesh from {mesh_file} with {len(self.points)} points.")
            except Exception as e:
                logger.error(f"Failed to load mesh from {mesh_file}: {e}")
                self.points = np.empty((0, 3))
                self.cells = {"tetra": np.empty((0, 4), dtype=int)}
        else:
            self.points = np.empty((0, 3))
            self.cells = {"tetra": np.empty((0, 4), dtype=int)}

    def get_state(self):
        nodes = self.points.tolist()
        connectivity = {"tetra": self.cells["tetra"].tolist()} if self.cells["tetra"].size else {}
        return {"nodes": nodes, "connectivity": connectivity}

    def save(self, mesh_file):
        """
        Save the current mesh state to a MSH file using our custom writer.
        """
        write_msh_file(mesh_file, self.points, self.cells["tetra"])
        logger.info(f"Mesh saved to {mesh_file}.")

    def _is_inside(self, vertex):
        """
        Return True only if strictly inside the box (not on any face)
        and outside the attachment-radius.
        """
        x, y, z = vertex
        # strictly inside
        inside = (BOUNDING_BOX["x"][0] < x < BOUNDING_BOX["x"][1] and
                  BOUNDING_BOX["y"][0] < y < BOUNDING_BOX["y"][1] and
                  BOUNDING_BOX["z"][0] < z < BOUNDING_BOX["z"][1])
        far_enough = np.linalg.norm(np.array(vertex) - np.array(ATTACHMENT_POSITION)) >= TOLERANCE
        return inside and far_enough

    ########################################################################
    # NEW: Boundary Protection Helper Methods
    ########################################################################

    def get_boundary_nodes(self, tol=1e-6) -> List[int]:
        """
        Return indices of all vertices lying on any outer face of the bounding box.
        """
        idxs = []
        x0, x1 = BOUNDING_BOX["x"]
        y0, y1 = BOUNDING_BOX["y"]
        z0, z1 = BOUNDING_BOX["z"]
        for i, (x, y, z) in enumerate(self.points):
            if (abs(x - x0) < tol or abs(x - x1) < tol or
                abs(y - y0) < tol or abs(y - y1) < tol or
                abs(z - z0) < tol or abs(z - z1) < tol):
                idxs.append(i)
        return idxs

    def get_floor_nodes(self, tol_fraction=0.01):
        """
        Returns indices of nodes with y-values within tol_fraction of the minimum y (floor nodes).
        """
        if self.points.shape[0] == 0:
            return []
        y_vals = self.points[:, 1]
        min_y = np.min(y_vals)
        tol = tol_fraction * (np.max(y_vals) - np.min(y_vals))
        floor_nodes = np.where(y_vals <= (min_y + tol))[0]
        return floor_nodes.tolist()

    def get_attachment_nodes(self, radius=5.0):
        """
        Returns indices of nodes within a given radius of the attachment point.
        """
        if self.points.shape[0] == 0:
            return []
        distances = np.linalg.norm(self.points - np.array(ATTACHMENT_POSITION), axis=1)
        attachment_nodes = np.where(distances <= radius)[0]
        return attachment_nodes.tolist()

    def get_protected_nodes(self) -> Set[int]:
        """Returns a set of node indices to protect (floor, attachment, and boundary)."""
        floor    = set(self.get_floor_nodes())
        attach   = set(self.get_attachment_nodes())
        boundary = set(self.get_boundary_nodes())
        return floor | attach | boundary

    ########################################################################
    # NEW: Mesh Quality Diagnostics and Smoothing
    ########################################################################

    def compute_tetra_volume(self, v0, v1, v2, v3):
        """
        Compute the volume of a tetrahedron with vertices v0, v1, v2, v3.
        Volume = |det([v1-v0, v2-v0, v3-v0])| / 6.
        """
        mat = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
        return abs(np.linalg.det(mat)) / 6.0

    def repair_mesh_quality(self, volume_threshold=1e-6, smoothing_iterations=5):
        """
        Evaluate mesh quality by computing tetrahedral volumes.
        If any tetrahedron has volume below volume_threshold, perform Laplacian smoothing
        on vertices in degenerate cells (excluding protected nodes).
        """
        vols = []
        for cell in self.cells["tetra"]:
            v0 = self.points[cell[0]]
            v1 = self.points[cell[1]]
            v2 = self.points[cell[2]]
            v3 = self.points[cell[3]]
            vols.append(self.compute_tetra_volume(v0, v1, v2, v3))
        vols = np.array(vols)
        min_vol = np.min(vols)
        logger.debug(f"Before smoothing: min cell volume = {min_vol:e}")

        if min_vol >= volume_threshold:
            logger.debug("Mesh quality acceptable; no smoothing needed.")
            return

        n_points = self.points.shape[0]
        neighbors = {i: set() for i in range(n_points)}
        for cell in self.cells["tetra"]:
            for i in cell:
                for j in cell:
                    if i != j:
                        neighbors[i].add(j)

        for it in range(smoothing_iterations):
            new_points = self.points.copy()
            degenerate_vertices = set()
            for cell in self.cells["tetra"]:
                v0 = self.points[cell[0]]
                v1 = self.points[cell[1]]
                v2 = self.points[cell[2]]
                v3 = self.points[cell[3]]
                vol = self.compute_tetra_volume(v0, v1, v2, v3)
                if vol < volume_threshold:
                    degenerate_vertices.update(cell.tolist())
            logger.debug(f"Iteration {it}: {len(degenerate_vertices)} vertices in degenerate elements.")

            # Exclude protected nodes from smoothing.
            protected = self.get_protected_nodes()
            unsmoothed = degenerate_vertices - protected
            logger.debug(f"Vertices to smooth (after protection): {unsmoothed}")

            if not unsmoothed:
                logger.debug("No unsmoothed degenerate vertices found; stopping smoothing.")
                break

            for i in unsmoothed:
                nb = list(neighbors[i])
                if nb:
                    new_points[i] = np.mean(self.points[nb], axis=0)
            self.points = new_points

            vols = []
            for cell in self.cells["tetra"]:
                v0 = self.points[cell[0]]
                v1 = self.points[cell[1]]
                v2 = self.points[cell[2]]
                v3 = self.points[cell[3]]
                vols.append(self.compute_tetra_volume(v0, v1, v2, v3))
            vols = np.array(vols)
            min_vol = np.min(vols)
            logger.debug(f"After smoothing iteration {it+1}: min cell volume = {min_vol:e}")
            if min_vol >= volume_threshold:
                logger.debug("Mesh quality improved; stopping smoothing iterations.")
                break

    ########################################################################
    # NEW: Enforce a Flat Floor Boundary with Slight Dispersion
    ########################################################################
    def enforce_floor_boundary(self):
        if self.points.shape[0] == 0:
            return
        y_vals = self.points[:, 1]
        min_y = np.min(y_vals)
        if min_y < 0:
            min_y = 0.0
        floor_nodes = self.get_floor_nodes(tol_fraction=0.01)
        if not floor_nodes:
            quantile_val = np.percentile(y_vals, 5)
            floor_nodes = np.where(y_vals <= quantile_val)[0].tolist()
            logger.warning(f"Fallback: Selected {len(floor_nodes)} floor nodes using bottom 5-percentile.")
        # Instead of snapping to min_y exactly, allow a random dispersion within ±1e-4
        for idx in floor_nodes:
            self.points[idx, 1] = min_y + np.random.uniform(-1e-4, 1e-4)
        self.points[:, 1] = np.maximum(self.points[:, 1], 0.0)
        logger.debug(f"Enforced floor boundary on {len(floor_nodes)} nodes to y>=0.")

    ########################################################################
    # End of NEW: Mesh Quality and Floor Enforcement
    ########################################################################


    def add_material(self, new_vertices, new_faces=None):
        """
        Add new material by:
          0) merging in the new vertices so Gmsh will see them,
          1) filtering & appending new vertices,
          2) extracting the outer shell facets (plus any new patch faces),
          3) tetrahedralizing (local refinement) via Gmsh while preserving that shell,
          4) cleaning up degenerate/skinny tets,
          5) smoothing and enforcing floor boundary.
        """
        # 0) original point count
        orig_n = self.points.shape[0]

        # 1) filter & append new vertices
        pts = np.array([v for v in new_vertices if self._is_inside(v)], dtype=np.float64)
        if not pts.size:
            return {"status": "success", "message": "No new vertices inside bounding box."}
        self.points = pts if orig_n == 0 else np.vstack([self.points, pts])

        # 2a) extract boundary facets from the old tetrahedra
        old_tets    = self.cells.get("tetra", np.empty((0,4), dtype=int))
        shell_faces = extract_boundary_faces(old_tets)  # (M,3) array

        # 2b) append any new patch faces (rebased into global indices)
        facets = shell_faces.tolist()
        if new_faces:
            for f in new_faces:
                if len(f) == 3:
                    facets.append([f[0], f[1], f[2]])
                elif len(f) == 4:
                    # split quad → two tris
                    a, b, c, d = f
                    facets.append([a, b, c])
                    facets.append([a, c, d])
                else:
                    raise ValueError(f"Unexpected patch‐face length {len(f)}")
                
        max_idx = self.points.shape[0] - 1
        good = []
        for tri in facets:
            if all(0 <= v <= max_idx for v in tri):
                good.append(tri)
            else:
                logger.warning(f"Dropping invalid facet {tri}")
        facets = good

        # contiguous int32 array of triangles to pass as "surface"
        facets_arr = np.ascontiguousarray(np.array(facets, dtype=np.int32))

        # 3) write a temp MSH that includes your tetrahedra + those boundary triangles
        tmp_msh = os.path.join(MESH_DIR, "convex_hull_tmp.msh")
        # ── DROP the 'surface' argument: only write the volume tets ──
        write_msh_file(tmp_msh,
                       self.points,
                       self.cells["tetra"])
        
        try:
            gmsh.clear()
        except:
            pass
        safe_gmsh_initialize()
        gmsh.open(tmp_msh)

        # 4) initialize & load into Gmsh
        safe_gmsh_initialize()
        gmsh.open(tmp_msh)

        # 5) build a Distance‐Threshold sizing field around newly added nodes
        _, all_coords, _ = gmsh.model.mesh.getNodes()
        coords_arr = np.array(all_coords).reshape(-1, 3)
        tree = KDTree(coords_arr)
        new_tags = [tree.query(v)[1] + 1 for v in self.points[orig_n:]]  # 1‐based

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", new_tags)

        R, LcMin, LcMax = 10.0, 3.75, 4.25
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", LcMin)
        gmsh.model.mesh.field.setNumber(2, "LcMax", LcMax)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(2, "DistMax", R)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        # 6) remesh in 3D
        gmsh.model.mesh.generate(3)

        # 7) pull back the refined node coords and tetra connectivity
        _, coords_out, _ = gmsh.model.mesh.getNodes()
        etypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=3)
        tet_block = list(etypes).index(4)
        flat_conn = elemNodeTags[tet_block]
        elems_out = np.array(flat_conn, dtype=np.int32).reshape(-1, 4) - 1

        gmsh.finalize()

        # 8) overwrite your in‐memory mesh
        self.points         = np.array(coords_out).reshape(-1, 3)
        self.cells["tetra"] = np.ascontiguousarray(elems_out, dtype=np.int32)

        # 9a) remove degenerate tets
        cleaned = remove_degenerate_cells(self, volume_threshold=1e-8)
        cleaned.topology.create_connectivity(cleaned.topology.dim, 0)
        self.cells["tetra"] = cleaned.topology.connectivity(
            cleaned.topology.dim, 0).array.reshape(-1, 4)

        # 9b) remove skinny tets
        cleaned2 = remove_skinny_cells(self, aspect_ratio_threshold=15.0)
        cleaned2.topology.create_connectivity(cleaned2.topology.dim, 0)
        self.cells["tetra"] = cleaned2.topology.connectivity(
            cleaned2.topology.dim, 0).array.reshape(-1, 4)

        # 10) final smoothing & enforce flat floor
        self.repair_mesh_quality(volume_threshold=1e-6, smoothing_iterations=5)
        self.enforce_floor_boundary()

        return {
            "status":  "success",
            "message": "Material added, shell preserved, remeshed, cleaned, smoothed, and floor enforced."
        }




    def remove_material(self, vertex_indices):
        """
        Remove vertices (and corresponding cells) from the mesh.
        Protected nodes (floor and attachment) are preserved.
        """
        if self.points.size == 0:
            return {"status": "error", "message": "No vertices to remove."}
        # filter out any protected indices so the shell stays intact
        protected = self.get_protected_nodes()
        to_remove = [i for i in vertex_indices if i not in protected]

        # now apply the usual removal logic on to_remove
        n_points = self.points.shape[0]
        mask = np.ones(n_points, dtype=bool)
        mask[to_remove] = False
        new_points = self.points[mask]
        new_index = -np.ones(n_points, dtype=int)
        new_index[mask] = np.arange(new_points.shape[0])
        new_cells_list = []
        for cell in self.cells["tetra"]:
            if np.all(new_index[cell] != -1):
                new_cells_list.append(new_index[cell])
        new_cells = np.array(new_cells_list, dtype=int) if new_cells_list else np.empty((0, 4), dtype=int)
        self.points = new_points
        self.cells["tetra"] = new_cells
        return {"status": "success", "message": "Material removed successfully, protected nodes preserved."}

########################################################################
# Helper Functions for Conversion and Mapping (Using gmsh instead of meshio)
########################################################################

def extract_boundary_faces(cells: np.ndarray) -> np.ndarray:
    """
    Given an (M,4) array of tetrahedra (0-based indices), return all
    triangular faces that belong to exactly one tetra (the boundary).
    """
    face_count: dict = {}
    # build counts of each face
    for tet in cells:
        i, j, k, l = tet
        for face in [(i, j, k), (i, j, l), (i, k, l), (j, k, l)]:
            key = tuple(sorted(face))
            face_count[key] = face_count.get(key, 0) + 1
    # keep only those faces seen exactly once
    boundary_faces = [list(face) for face, cnt in face_count.items() if cnt == 1]
    return np.array(boundary_faces, dtype=int)

def compute_mesh_mapping(raw_vertices, refined_vertices):
    tree = KDTree(np.array(raw_vertices))
    mapping = [int(tree.query(rv)[1]) for rv in refined_vertices]
    return mapping

def generate_gmsh_mesh_internal(vertices, mesh_filename):
    vertices_filename = mesh_filename.replace(".msh", "_vertices.npy")
    np.save(vertices_filename, vertices)
    subprocess.run(["python3", "src/mesh_subprocess.py", vertices_filename, mesh_filename], check=True)
    logger.info(f"Custom MSH file written to {mesh_filename}.")
    return mesh_filename

import subprocess
def convert_mesh(mesh_filename: str) -> tuple:
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mesh_convert_subprocess.py"))
    logger.info(f"Using conversion script at: {script_path}")
    env = os.environ.copy()
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    cmd = [
        "mpirun",
        "-n", "1",
        "python3",
        script_path,
        mesh_filename
    ]
    logger.info(f"Running conversion with command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        logger.error(f"Conversion failed with code {result.returncode}.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        raise Exception(f"Conversion failed with code {result.returncode}")
    output = result.stdout.strip()
    logger.debug(f"Subprocess output: '{output}'")
    try:
        xdmf_filename, h5_filename = output.split(",")
    except Exception as e:
        logger.error(f"Error parsing conversion output: '{output}'. Exception: {e}")
        raise Exception("Unable to parse conversion output.")
    logger.info(f"✅ Converted {mesh_filename} → XDMF: {xdmf_filename}, HDF5: {h5_filename}")
    return xdmf_filename, h5_filename

def convert_mesh_to_json(mesh_filename):
    full_path = mesh_filename if os.path.isabs(mesh_filename) else os.path.join(MESH_DIR, mesh_filename)
    pts, cells = read_msh_file(full_path)
    connectivity = {"tetra": cells.tolist()}
    return {"nodes": pts.tolist(), "connectivity": connectivity}

def generate_mesh_and_mapping(iteration: int = None):
    if get_latest_iteration() == 0:
        logger.info("No existing mesh found. Initializing baseline mesh.")
        baseline_info = initialize_baseline_mesh()
        return baseline_info

    if iteration is None:
        current_state_iter = get_latest_iteration()
        state_file = os.path.join(ENV_STATE_DIR, f"environment_state_{current_state_iter}.json")
    else:
        state_file = os.path.join(ENV_STATE_DIR, f"environment_state_{iteration}.json")
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"Raw environment state file {state_file} not found.")
    with open(state_file, "r") as f:
        state_data = json.load(f)
    raw_vertices = state_data.get("structure_vertices") or state_data.get("vertices") or []
    if not raw_vertices:
        raise ValueError("No vertices found in the raw environment state.")
    raw_vertices = np.array(raw_vertices, dtype=np.float64)
    new_iteration = get_latest_iteration() + 1
    new_mesh_filename = os.path.join(MESH_DIR, f"convex_hull_{new_iteration}.msh")
    vertices_npy_file = new_mesh_filename.replace(".msh", "_vertices.npy")
    np.save(vertices_npy_file, raw_vertices)
    generate_gmsh_mesh_internal(raw_vertices, new_mesh_filename)
    xdmf_filename, h5_filename = convert_mesh(new_mesh_filename)
    
    # ─── CLEANUP: remove any tiny/degenerate tets before proceeding ───
    from dolfinx.io import XDMFFile
    # 1) load back into DolfinX
    with XDMFFile(MPI.COMM_WORLD, xdmf_filename, "r") as xdmf:
        dolfin_mesh = xdmf.read_mesh(name="mesh")
    # 2) prune cells below volume threshold
    cleaned = remove_degenerate_cells(dolfin_mesh, volume_threshold=1e-8)
    # 3) overwrite the .msh with the cleaned connectivity
    coords = cleaned.geometry.x
    cleaned.topology.create_connectivity(cleaned.topology.dim, 0)
    cells = cleaned.topology.connectivity(cleaned.topology.dim, 0).array.reshape((-1, 4))
    write_msh_file(new_mesh_filename, coords, cells)
    logger.info("Cleaned degenerate cells; mesh rewritten.")
    # ───────────────────────────────────────────────────────────────
    # ─── CLEANUP: remove any remaining SKINNY cells by aspect ratio ───
    cleaned2 = remove_skinny_cells(cleaned, aspect_ratio_threshold=15.0)
    coords2 = cleaned2.geometry.x
    cleaned2.topology.create_connectivity(cleaned2.topology.dim, 0)
    cells2 = cleaned2.topology.connectivity(cleaned2.topology.dim,0).array.reshape((-1,4))
    write_msh_file(new_mesh_filename, coords2, cells2)
    logger.info("Removed skinny cells; mesh rewritten again.")
    # ───────────────────────────────────────────────────────────────
    
    refined_mesh_data = convert_mesh_to_json(os.path.basename(new_mesh_filename))
    mesh_json_filename = new_mesh_filename.replace(".msh", ".json")
    with open(mesh_json_filename, "w") as f:
        json.dump(refined_mesh_data, f)
    logger.info(f"Mesh JSON representation saved to {mesh_json_filename}")
    refined_vertices = np.array(refined_mesh_data["nodes"])
    mapping = compute_mesh_mapping(raw_vertices, refined_vertices)
    mapping_filename = os.path.join(MESH_DIR, f"mesh_mapping_{new_iteration}.json")
    with open(mapping_filename, "w") as f:
        json.dump(mapping, f)
    logger.info(f"Mesh mapping saved to {mapping_filename}")
    store_mesh_metadata(new_iteration, new_mesh_filename, xdmf_filename, h5_filename, mapping_filename, mesh_json_filename)
    return {
        "mesh_iteration": new_iteration,
        "mesh_path": new_mesh_filename,
        "xdmf_path": xdmf_filename,
        "h5_path": h5_filename,
        "mapping_file": mapping_filename,
        "mesh_json": mesh_json_filename
    }

def update_mesh_from_actions(actions: dict, iteration: int = None):
    from db import get_latest_iteration, store_mesh_metadata
    # 1) figure out the next iteration number
    #    prefer explicit `iteration` arg, else bump latest in DB
    if iteration is None:
        iteration = get_latest_iteration() + 1

    # 2) build new .msh filename
    new_mesh_filename = Path(MESH_DIR) / f"convex_hull_{iteration}.msh"
    prev_mesh_filename = Path(MESH_DIR) / f"convex_hull_{iteration-1}.msh"

    # 3) start **from the previous mesh**, not an empty one
    mesh_env = MeshEnvironment(
        str(prev_mesh_filename) if prev_mesh_filename.exists() else None
    )

    # 1. Process removals **only** if it won't delete everything
    if "remove_indices" in actions and actions["remove_indices"]:
        n_before = mesh_env.points.shape[0]
        n_remove = len(actions["remove_indices"])
        # leave at least 4 points for a valid tetra-mesh
        if n_before - n_remove >= 4:
            logger.debug("Removing material from mesh before addition.")
            mesh_env.remove_material(actions["remove_indices"])
            logger.debug(f"Mesh after removal has {mesh_env.points.shape[0]} vertices.")
        else:
            logger.warning(f"Skipping removal of {n_remove} pts: only {n_before} available.")

    # 2. Process additions
    if "add_vertices" in actions and actions["add_vertices"]:
        add_vertices = actions["add_vertices"]
        add_faces    = actions.get("add_faces", [])
        logger.debug(f"Adding {len(add_vertices)} vertices and {len(add_faces)} faces to the mesh.")
        # pass both lists into add_material
        res = mesh_env.add_material(add_vertices, add_faces)
        if res.get("status") == "error":
            return res

        # 1) Remove vanishing‐volume tets
        cleaned = remove_degenerate_cells(mesh_env, volume_threshold=1e-6)
        cleaned.topology.create_connectivity(cleaned.topology.dim, 0)
        coords = cleaned.geometry.x.copy()
        cells  = cleaned.topology.connectivity(cleaned.topology.dim, 0).array.reshape(-1, 4)
        mesh_env.points         = coords
        mesh_env.cells["tetra"] = cells

        # 2) Remove very skinny tets (aspect_ratio > 10)
        cleaned2 = remove_skinny_cells(cleaned, aspect_ratio_threshold=10.0)
        cleaned2.topology.create_connectivity(cleaned2.topology.dim, 0)
        coords2 = cleaned2.geometry.x.copy()
        cells2  = cleaned2.topology.connectivity(cleaned2.topology.dim, 0).array.reshape(-1, 4)
        mesh_env.points         = coords2
        mesh_env.cells["tetra"] = cells2

    # 3) Write out the clean MSH → single convert to XDMF/H5
    write_msh_file(str(new_mesh_filename),
                   mesh_env.points,
                   mesh_env.cells["tetra"])
    xdmf_filename, h5_filename = convert_mesh(str(new_mesh_filename))

    # ─── EXTRA PASS: read that XDMF into a DolfinX Mesh ──────────────────────
    from dolfinx.io import XDMFFile
    from mpi4py import MPI
    with XDMFFile(MPI.COMM_WORLD, xdmf_filename, "r") as xdmf:
        dolfin_mesh = xdmf.read_mesh(name="mesh")

    # 4) remove degenerate cells on the DolfinX Mesh
    cleaned_mesh = remove_degenerate_cells(dolfin_mesh,
                                           volume_threshold=1e-8)

    # 5) extract cleaned geometry & connectivity
    tdim = cleaned_mesh.topology.dim
    cleaned_mesh.topology.create_connectivity(tdim, 0)
    coords = cleaned_mesh.geometry.x.copy()
    cells  = cleaned_mesh.topology.connectivity(tdim, 0).array.reshape(-1, 4)

    # 6) overwrite the MSH with the cleaned mesh
    write_msh_file(str(new_mesh_filename), coords, cells)
    logger.info(f"Degenerate‐cleaned mesh written to {new_mesh_filename}")

    # 7) re‐convert that cleaned MSH
    xdmf_filename, h5_filename = convert_mesh(str(new_mesh_filename))

    # 8) continue with JSON export, mapping, DB store…
    refined = convert_mesh_to_json(os.path.basename(new_mesh_filename))
    # Save using the canonical convex_hull_{iteration}.json name so that the
    # pre-processing service can locate it.
    convex_json_path = Path(MESH_DIR) / f"convex_hull_{iteration}.json"
    with open(convex_json_path, "w") as f:
        json.dump(refined, f)
    logger.info(f"Convex hull JSON saved to {convex_json_path}")

    # Also keep the mesh_data_{iteration}.json for backward compatibility
    mesh_json_path = Path(MESH_DIR) / f"mesh_data_{iteration}.json"
    with open(mesh_json_path, "w") as f:
        json.dump(refined, f)
    logger.info(f"Mesh JSON data saved to {mesh_json_path}")
    mapping = compute_mesh_mapping(np.array(refined["nodes"]), np.array(refined["nodes"]))
    mapping_path = Path(MESH_DIR) / f"mesh_mapping_{iteration}.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)
    logger.info(f"Mesh mapping data saved to {mapping_path}")
    store_mesh_metadata(
        iteration,
        str(new_mesh_filename),
        xdmf_filename,
        h5_filename,
        str(mapping_path),
        str(convex_json_path),
    )

    return {
        "mesh_iteration": iteration,
        "mesh_path": str(new_mesh_filename),
        "xdmf_path": xdmf_filename,
        "h5_path": h5_filename,
        "mapping_file": str(mapping_path),
        "mesh_json": str(convex_json_path),
    }

def initialize_baseline_mesh():
    from db import get_latest_iteration, store_mesh_metadata
    import json, os, numpy as np, gmsh

    if get_latest_iteration() > 0:
        logger.info("Baseline mesh already exists; skipping initialization.")
        return

    # ------------------------------------------------------------
    # 1) Set up Gmsh and define a box matching your bounding box
    # ------------------------------------------------------------
    safe_gmsh_initialize()
    # ── Force MSH v2.2 ASCII output so our simple parser works ──
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.model.add("baseline_box")

    # ─── Pick an element size h _first_ ─────────
    h = 4  # global element size for the mesh
    # ─── Now inject the attachment point at that size ─────────

    x0, x1 = BOUNDING_BOX["x"]
    y0, y1 = BOUNDING_BOX["y"]
    z0, z1 = BOUNDING_BOX["z"]

    # create the eight corner points
    pts = []
    for X in (x0, x1):
        for Y in (y0, y1):
            for Z in (z0, z1):
                pts.append(gmsh.model.geo.addPoint(X, Y, Z, h))
    # name them for convenience
    p000, p001, p010, p011, p100, p101, p110, p111 = pts

    # build the edges
    lines = [
        gmsh.model.geo.addLine(p000, p100),
        gmsh.model.geo.addLine(p100, p110),
        gmsh.model.geo.addLine(p110, p010),
        gmsh.model.geo.addLine(p010, p000),

        gmsh.model.geo.addLine(p001, p101),
        gmsh.model.geo.addLine(p101, p111),
        gmsh.model.geo.addLine(p111, p011),
        gmsh.model.geo.addLine(p011, p001),

        gmsh.model.geo.addLine(p000, p001),
        gmsh.model.geo.addLine(p100, p101),
        gmsh.model.geo.addLine(p110, p111),
        gmsh.model.geo.addLine(p010, p011),
    ]

    # make curve loops for each face
    def face_loop(ids):
        return gmsh.model.geo.addCurveLoop(ids)

    loops = [
        face_loop([lines[0], lines[9],  -lines[4], -lines[8]]),   # x0-face
        face_loop([lines[1], lines[10], -lines[5], -lines[9]]),   # x1-face
        face_loop([lines[2], lines[11], -lines[6], -lines[10]]),  # y1-face
        face_loop([lines[3], lines[8],  -lines[7], -lines[11]]),  # y0-face
        face_loop([lines[0], lines[1], lines[2], lines[3]]),      # z0-face
        face_loop([lines[4], lines[5], lines[6], lines[7]]),      # z1-face
    ]

    # make surfaces
    surfaces = [gmsh.model.geo.addPlaneSurface([l]) for l in loops]

    # create volume
    sl = gmsh.model.geo.addSurfaceLoop(surfaces)
    gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()
    
    # ——————————————————————————————
    # 1.5) REFINE AROUND ATTACHMENT
    # ——————————————————————————————
    attach_h   = h * 0.2        # 5× finer near attachment
    outer_h    = h              # default global element size
    cx, cy, cz = ATTACHMENT_POSITION
    gmsh.model.mesh.field.add("Ball", 1)
    gmsh.model.mesh.field.setNumber(1, "Radius",   2.0 * TOLERANCE)
    gmsh.model.mesh.field.setNumber(1, "VIn",      attach_h)
    gmsh.model.mesh.field.setNumber(1, "VOut",     outer_h)
    gmsh.model.mesh.field.setNumber(1, "XCenter",  cx)
    gmsh.model.mesh.field.setNumber(1, "YCenter",  cy)
    gmsh.model.mesh.field.setNumber(1, "ZCenter",  cz)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    # ------------------------------------------------------------
    # 2) Generate a 3D mesh
    # ------------------------------------------------------------
    gmsh.model.mesh.generate(3)

    # ------------------------------------------------------------
    # 3) Write out .msh, finalize Gmsh
    # ------------------------------------------------------------
    baseline_mesh_filename = os.path.join(MESH_DIR, "convex_hull_1.msh")
    gmsh.write(baseline_mesh_filename)
    gmsh.finalize()
    logger.info(f"Baseline mesh created and saved to {baseline_mesh_filename}.")

    # ------------------------------------------------------------
    # 4) Convert to XDMF/H5 and JSON, store metadata
    # ------------------------------------------------------------
    xdmf, h5 = convert_mesh(baseline_mesh_filename)
    mesh_json = baseline_mesh_filename.replace(".msh", ".json")
    with open(mesh_json, "w") as f:
        json.dump(convert_mesh_to_json(os.path.basename(baseline_mesh_filename)), f)

    # The mapping is trivial here: every initial vertex maps to itself
    pts_data = convert_mesh_to_json(os.path.basename(baseline_mesh_filename))["nodes"]
    mapping = list(range(len(pts_data)))
    mapping_file = os.path.join(MESH_DIR, "mesh_mapping_1.json")
    with open(mapping_file, "w") as f:
        json.dump(mapping, f)

    store_mesh_metadata(
        1,
        baseline_mesh_filename,
        xdmf,
        h5,
        mapping_file,
        mesh_json
    )

    return {
        "mesh_iteration": 1,
        "mesh_path":      baseline_mesh_filename,
        "xdmf_path":      xdmf,
        "h5_path":        h5,
        "mapping_file":   mapping_file,
        "mesh_json":      mesh_json
    }

if __name__ == "__main__":
    if len(sys.argv) == 3:
        vertices_filename = sys.argv[1]
        mesh_filename = sys.argv[2]
        try:
            vertices = np.load(vertices_filename)
            logger.info(f"Loaded vertices from {vertices_filename}, shape: {vertices.shape}")
            generate_gmsh_mesh_internal(vertices, mesh_filename)
            logger.info(f"Successfully generated mesh at {mesh_filename}")
        except Exception as e:
            logger.error(f"Error in mesh generation: {e}")
            sys.exit(1)
    else:
        print("Usage: python mesh.py <vertices_filename> <mesh_filename>")

def remove_degenerate_cells(mesh_obj, volume_threshold=1e-8):
    """
    Remove tetrahedral cells from a mesh (or MeshEnvironment) whose volume is below
    a given threshold, and return a new DolfinX Mesh built from the remaining cells.
    
    Parameters:
      mesh_obj : Either a DolfinX Mesh (with attributes 'topology' and 'geometry')
                 or a MeshEnvironment instance (with attributes 'points' and 'cells').
      volume_threshold : float
          Tetrahedral cells with volume below this threshold are removed.
          
    Returns:
      new_mesh : a DolfinX Mesh containing only nondegenerate cells.
      
    Raises:
      RuntimeError if all cells are removed or if the resulting mesh has no boundary facets.
    """
    import numpy as np
    from dolfinx.mesh import create_mesh
    from dolfinx.fem import coordinate_element
    from dolfinx.cpp.mesh import CellType
    from mpi4py import MPI
    from petsc4py import PETSc

    comm = MPI.COMM_WORLD

    # Attempt to extract geometry and connectivity depending on the input type.
    try:
        # Assume mesh_obj is a MeshEnvironment.
        coords = np.array(mesh_obj.points)
        cells = np.array(mesh_obj.cells["tetra"], dtype=np.int32).reshape((-1, 4))
    except Exception:
        # Otherwise, assume it's a full DolfinX Mesh.
        mesh_obj.topology.create_connectivity(mesh_obj.topology.dim, 0)
        coords = mesh_obj.geometry.x.copy()
        cells = mesh_obj.topology.connectivity(mesh_obj.topology.dim, 0).array.reshape((-1, 4))
    
    def tetra_volume(c):
        v0, v1, v2, v3 = coords[c[0]], coords[c[1]], coords[c[2]], coords[c[3]]
        mat = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
        return abs(np.linalg.det(mat)) / 6.0

    initial_cell_count = cells.shape[0]
    valid_cells = []
    for cell in cells:
        vol = tetra_volume(cell)
        if vol >= volume_threshold:
            valid_cells.append(cell)
        else:
            PETSc.Sys.Print(f"Removed degenerate cell {cell} with volume {vol:e}\n")
    valid_cells = np.array(valid_cells, dtype=np.int32)
    remaining_cell_count = valid_cells.shape[0]
    PETSc.Sys.Print(f"Initial cell count: {initial_cell_count}, after cleaning: {remaining_cell_count}\n")
    if remaining_cell_count == 0:
        raise RuntimeError("All cells are degenerate; unable to create a valid mesh.")

    # Create a coordinate element for a tetrahedral mesh.
    ce = coordinate_element(CellType.tetrahedron, 1)
    new_mesh = create_mesh(comm, valid_cells, coords, ce)
    
    # OPTIONAL: Ensure the new mesh has boundary facets.
    new_mesh.topology.create_connectivity(new_mesh.topology.dim - 1, 0)
    facet_count = new_mesh.topology.index_map(new_mesh.topology.dim - 1).size_global
    PETSc.Sys.Print(f"Boundary facet count: {facet_count}\n")
    if facet_count == 0:
        raise RuntimeError("The cleaned mesh has no boundary facets; the domain is invalid.")
    
    return new_mesh

def remove_skinny_cells(mesh_obj, aspect_ratio_threshold: float = 15.0):
    """
    Remove tetrahedral cells whose max_edge_length / min_edge_length
    exceeds aspect_ratio_threshold.
    """
    import numpy as np
    from dolfinx.mesh import create_mesh
    from dolfinx.fem import coordinate_element
    from dolfinx.cpp.mesh import CellType
    from mpi4py import MPI
    from petsc4py import PETSc

    comm = MPI.COMM_WORLD
    # Extract coords & cells
    try:
        coords = np.array(mesh_obj.points)
        cells  = np.array(mesh_obj.cells["tetra"], dtype=np.int32).reshape((-1,4))
    except AttributeError:
        mesh_obj.topology.create_connectivity(mesh_obj.topology.dim, 0)
        coords = mesh_obj.geometry.x.copy()
        cells  = mesh_obj.topology.connectivity(mesh_obj.topology.dim,0).array.reshape((-1,4))

    def aspect(c):
        p = coords[c]
        L = [np.linalg.norm(p[i]-p[j]) for i in range(4) for j in range(i+1,4)]
        return max(L)/min(L)

    valid, removed = [], 0
    for c in cells:
        if aspect(c) <= aspect_ratio_threshold:
            valid.append(c)
        else:
            removed += 1
            PETSc.Sys.Print(f"Removed skinny cell {c}, aspect={aspect(c):.1f}\n")
    valid = np.array(valid, dtype=np.int32)
    PETSc.Sys.Print(f"remove_skinny_cells: kept {len(valid)}/{len(cells)} cells\n")

    ce = coordinate_element(CellType.tetrahedron, 1)
    return create_mesh(comm, valid, coords, ce)

logger = logging.getLogger(__name__)

def safe_gmsh_initialize():
    """
    Call gmsh.initialize() but quietly ignore the
    “signal only works in main thread” error if it arises.
    """
    try:
        gmsh.initialize()
    except Exception as e:
        msg = str(e)
        # This is the exact text Python gives you
        if "signal only works in main thread" in msg:
            logger.warning(
                "Gmsh.initialize(): ignored signal‐handler error "
                "in non-main thread — continuing without registering signals"
            )
        else:
            # Some other genuinely bad error: re-raise it
            raise
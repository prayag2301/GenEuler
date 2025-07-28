# utils.py
import os
import json
import numpy as np
import torch
from torch_geometric.data import Data
from config import DATA_PATHS, BOUNDING_BOX, ATTACHMENT_POSITION, DEVICE
from scipy.spatial import KDTree
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def vertex_inside(vertex, bounding_box):
    x, y, z = vertex
    return (bounding_box["x"][0] <= x <= bounding_box["x"][1] and
            bounding_box["y"][0] <= y <= bounding_box["y"][1] and
            bounding_box["z"][0] <= z <= bounding_box["z"][1])


def convert_np_types(obj):
    """
    Recursively convert NumPy types to native Python types so that JSON can be serialized.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(v) for v in obj]
    else:
        return obj

def load_json_file(file_path):
    """
    Load and return a JSON file from the given file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

def get_file_path(iteration, file_type="merged_data"):
    """
    Construct the file path for a given iteration and file type.
    file_type can be "merged_data", "mesh_data".
    """
    filename = ""
    if file_type == "merged_data":
        filename = f"merged_data_{iteration}.json"
    elif file_type == "mesh_data":
                filename = f"mesh_data_{iteration}.json"
    else:
        raise ValueError("Invalid file_type provided.")
    return os.path.join(DATA_PATHS[file_type], filename)

def build_graph(merged_data, mesh_data, device=DEVICE):
    """
    Build a PyG Data object (graph) from the merged dynamic data and mesh connectivity.
    IMPORTANT: We trim the connectivity so that only edges with node indices
    less than the number of nodes in the merged data are included.
    """
    node_features = np.array(merged_data["node_features"], dtype=np.float32)
    num_nodes = node_features.shape[0]
    # convert NumPy float32 array to tensor (preserves dtype)
    x = torch.from_numpy(node_features).to(device)
    
    edge_list = []
    # Get connectivity for tetrahedral cells (adjust if you have other cell types)
    tetra_cells = mesh_data["connectivity"].get("tetra", [])
    for cell in tetra_cells:
        # Only consider indices that are valid given the trimmed node features.
        valid_cell = [idx for idx in cell if idx < num_nodes]
        # Only add edges if there are at least two valid nodes.
        if len(valid_cell) < 2:
            continue
        # Create a complete graph among the valid nodes (bidirectional edges)
        for i in range(len(valid_cell)):
            for j in range(i + 1, len(valid_cell)):
                edge_list.append([valid_cell[i], valid_cell[j]])
                edge_list.append([valid_cell[j], valid_cell[i]])
    
    # Turn the Python list into a tensor and then *filter out* any oob entries.
    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        e = torch.tensor(edge_list, dtype=torch.long, device=device)  # shape (E,2)
        e = e.t().contiguous()                                         # shape (2,E)
        # mask out any edges whose endpoints are >= num_nodes
        valid = (e[0] >= 0) & (e[0] < num_nodes) & (e[1] >= 0) & (e[1] < num_nodes)
        edge_index = e[:, valid]

    # ─── SAFETY GUARD ──────────────────────────────────────────
    # remove any edges that still reference nodes ≥ num_nodes
    N = num_nodes
    mask = (edge_index[0] < N) & (edge_index[1] < N)
    edge_index = edge_index[:, mask]
        
    if edge_index.numel() > 0:
        max_index = torch.max(edge_index)
        print(f"[DEBUG] Maximum index in edge_index: {max_index} (num_nodes: {num_nodes})")
    
    data = Data(x=x, edge_index=edge_index)

    # expose per‑vector rewards (broadcast if only a single scalar)
    ir = merged_data.get("individual_rewards", {})
    for key, attr in [("combined", "combined_reward"),
                      ("fem",      "fem_reward"),
                      ("topology", "topology_reward")]:
        arr = np.array(ir.get(key, [0.0]), dtype=np.float32)
        if arr.size == 1:
            arr = np.full((num_nodes,), float(arr), dtype=np.float32)
        data[attr] = torch.tensor(arr.reshape(-1,1), device=device)

    return data

def normalize_features(x):
    """
    Normalize node features.
    If x is a numpy array, normalize along axis=0.
    If x is a torch tensor, normalize along dim=0.
    """
    if isinstance(x, np.ndarray):
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True) + 1e-6
        return (x - mean) / std
    try:
        import torch
        if isinstance(x, torch.Tensor):
            mean = torch.mean(x, dim=0, keepdim=True)
            std = torch.std(x, dim=0, keepdim=True) + 1e-6
            return (x - mean) / std
    except ImportError:
        pass
    raise TypeError("x must be a numpy array or a torch tensor")

def filter_vertices_within_bounding_box(node_indices,
                                        merged_data,
                                        bounding_box=BOUNDING_BOX,
                                        attachment_position=ATTACHMENT_POSITION,
                                        tol=5.0):
    """
    Given node_indices and merged_data["node_features"], return only those
    strictly inside the bbox (i.e. not on the outer shell) and ≥ tol from attachment.
    """
    filtered = []
    bound_eps = 1e-6   # tiny tolerance to avoid float‐precision hits on the shell
    for idx in node_indices:
        x, y, z = merged_data["node_features"][idx][:3]

        # Strictly inside (no ≤ on any face)
        if (bounding_box["x"][0] + bound_eps < x < bounding_box["x"][1] - bound_eps and
            bounding_box["y"][0] + bound_eps < y < bounding_box["y"][1] - bound_eps and
            bounding_box["z"][0] + bound_eps < z < bounding_box["z"][1] - bound_eps):

            # Same attachment‐point filter as before
            if np.linalg.norm(np.array([x, y, z]) - np.array(attachment_position)) >= tol:
                filtered.append(idx)

    return filtered


def generate_vertices_for_nodes(node_indices, merged_data, mesh_data):
    """
    Generate new vertex positions for material addition.
    For each node flagged for addition, the function retrieves the node’s coordinates 
    from the merged_data’s node_features (assuming the first three features are the spatial coordinates)
    and adds a small random offset.
    """
    vertices = []
    for idx in node_indices:
        base_coord = np.array(merged_data["node_features"][idx][:3])
        offset = np.random.uniform(-0.05, 0.05, size=3)
        new_vertex = (base_coord + offset).tolist()
        vertices.append(new_vertex)
    return vertices

# def generate_faces_for_nodes(candidate_indices, merged_data, mesh_data):
#     """
#     For each candidate new vertex (given by its index in the merged data),
#     generate a tetrahedral cell that connects the new vertex to three nearest
#     static vertices from the current mesh.
    
#     The new connectivity indices are generated relative to the new vertices only.
#     """
#     faces = []
#     static_nodes = np.array(mesh_data["nodes"])

#     # Do not use the static count here; new vertices will get reindexed later.
#     for new_local_idx, global_idx in enumerate(candidate_indices):
#         # Get the candidate vertex position (from merged_data).
#         candidate_vertex = np.array(merged_data["node_features"][global_idx][:3])
#         # Find three nearest static nodes.
#         distances = np.linalg.norm(static_nodes - candidate_vertex, axis=1)
#         nearest_static = distances.argsort()[:3].tolist()  # These indices refer to static mesh nodes.
#         # Create a face: first element is the candidate new vertex's local index (0, 1, 2, ...)
#         # and the other three are the nearest static vertex indices.
#         face = [new_local_idx] + nearest_static
#         faces.append(face)
#     return faces

def generate_faces_for_nodes(candidate_indices, merged_data, mesh_data):
    faces = []
    static_nodes = np.array(mesh_data["nodes"])  # Ensure this is the current mesh nodes.
    for new_local_idx, global_idx in enumerate(candidate_indices):
        candidate_vertex = np.array(merged_data["node_features"][global_idx][:3])
        distances = np.linalg.norm(static_nodes - candidate_vertex, axis=1)
        nearest_static = distances.argsort()[:3].tolist()
        # Validate that the static indices are within proper range
        if max(nearest_static) >= len(static_nodes):
            raise ValueError("Generated static index out of range.")
        face = [new_local_idx] + nearest_static
        faces.append(face)
    return faces

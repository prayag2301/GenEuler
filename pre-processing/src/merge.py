import os
import json
import numpy as np

# Shared assets directory
ASSETS_DIR = "/app/src/assets"

def load_json_file(filename, subdir=""):
    """
    Load a JSON file from ASSETS_DIR, optionally from a subdirectory.
    """
    file_path = os.path.join(ASSETS_DIR, subdir, filename) if subdir else os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(file_path):
        # Provide a clearer error message when expected data is missing
        msg = f"File not found: {file_path}. Current working directory: {os.getcwd()}"
        print(f"❌ {msg}")
        raise FileNotFoundError(msg)
    # Optionally log a success message without dumping full contents.
    print(f"✅ Loaded JSON file: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

def load_reward_files(reward_files_dict):
    """
    Load reward files.
    Returns a dictionary mapping reward keys to NumPy arrays.
    """
    rewards = {}
    for key, filename in reward_files_dict.items():
        data = load_json_file(filename)
        if "per_vector_rewards" in data:
            rewards[key] = np.array(data["per_vector_rewards"])
        elif "combined_per_vector_rewards" in data:
            rewards[key] = np.array(data["combined_per_vector_rewards"])
        else:
            print(f"Warning: Expected reward key not found in {filename}")
    return rewards

def load_simulation_files(simulation_files_dict):
    """
    Load simulation output files.
    For "fem_results", expects a flat list "displacement" reshaped into (N,3).
    For "topology_results", expects a key like "density".
    """
    simulations = {}
    for key, filename in simulation_files_dict.items():
        data = load_json_file(filename)
        if key == "fem_results" and "displacement" in data:
            displacement_flat = data["displacement"]
            if len(displacement_flat) % 3 != 0:
                raise ValueError("Displacement data length is not a multiple of 3.")
            simulations[key] = np.array(displacement_flat).reshape(-1, 3)
        elif key == "topology_results" and "density" in data:
            simulations[key] = np.array(data["density"])
        else:
            print(f"Warning: Expected simulation key not found or unhandled for {filename}")
    return simulations

def merge_all_data(mesh_json_filename, reward_files_dict, simulation_files_dict):
    """
    Merge dynamic data from reward and simulation files with the static mesh.
    Pads any shorter arrays with zeros so that we always have one feature‐row
    per mesh node.
    """
    # Load the static mesh JSON file.
    mesh_data = load_json_file(mesh_json_filename)
    nodes = np.array(mesh_data["nodes"])  # Shape (N, 3)
    N = nodes.shape[0]                     # total number of mesh nodes

    # Load dynamic data.
    rewards     = load_reward_files(reward_files_dict)      # dict of arrays
    simulations = load_simulation_files(simulation_files_dict)

    # ─── PAD OR TRIM each array to length N ───────────────────────
    def pad_or_trim(arr, name):
        arr = np.asarray(arr)
        L   = arr.shape[0]
        if L < N:
            # pad with zeros
            if arr.ndim == 1:
                pad = np.zeros((N - L,), dtype=arr.dtype)
                return np.concatenate([arr, pad])
            else:
                pad = np.zeros((N - L, arr.shape[1]), dtype=arr.dtype)
                return np.vstack([arr, pad])
        elif L > N:
            # trim down
            print(f"Warning: truncating '{name}' from {L} to {N} entries")
            return arr[:N]
        else:
            return arr

    for key in list(rewards):
        rewards[key] = pad_or_trim(rewards[key], key)

    for key in list(simulations):
        simulations[key] = pad_or_trim(simulations[key], key)

    # ─── Build the feature matrix ────────────────────────────────
    # start with static spatial coordinates.
    feature_list = [nodes]  # (N, 3)

    # then each reward as a (N,1) column
    for key, arr in rewards.items():
        arr = np.asarray(arr).reshape(N, -1)
        feature_list.append(arr)

    # then each simulation output
    for key, arr in simulations.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(N, 1)
        feature_list.append(arr)

    node_features = np.hstack(feature_list)

    merged_dynamic = {
        "individual_rewards": {k: v.tolist() for k, v in rewards.items()},
        "simulation_outputs": {k: v.tolist() for k, v in simulations.items()},
        "node_features":      node_features.tolist()
    }
    return merged_dynamic, mesh_data

# Uncomment below for testing the merge functionality independently.
# if __name__ == "__main__":
#     iteration = 1
#     mesh_json_filename = f"convex_hull_{iteration}.json"
#     reward_files = {
#         "combined": f"reward_combined_result_{iteration}.json",
#         "topology": f"reward_topology_results_{iteration}.json",
#         "fem": f"reward_fem_results_{iteration}.json"
#     }
#     simulation_files = {
#         "fem_results": f"fem_results_{iteration}.json",
#         "topology_results": f"topology_results_{iteration}.json"
#     }
#     merged_dynamic, mesh_data = merge_all_data(mesh_json_filename, reward_files, simulation_files)
#     merged = {"merged_dynamic": merged_dynamic, "mesh_data": mesh_data}
#     print(json.dumps(merged, indent=4))

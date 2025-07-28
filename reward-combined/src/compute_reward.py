import os
import json
import numpy as np

# Helper function to load a JSON file
def load_json_file(file_path):
    if not os.path.exists(file_path):
        raise Exception(f"File not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

def fetch_fem_per_vector_rewards(iteration):
    # The FEM reward service stores its results as "reward_fem_results_{iteration}.json"
    fem_json_path = f"/app/src/assets/reward_fem_results_{iteration}.json"
    data = load_json_file(fem_json_path)
    # We expect the FEM service to have stored its per–vector rewards under the key "per_vector_rewards"
    fem_rewards = data.get("per_vector_rewards")
    if fem_rewards is None:
        raise Exception("FEM per-vector rewards not found in JSON file")
    return fem_rewards

def fetch_topo_per_vector_rewards(iteration):
    # The topology reward service stores its results as "reward_topology_results_{iteration}.json"
    topo_json_path = f"/app/src/assets/reward_topology_results_{iteration}.json"
    data = load_json_file(topo_json_path)
    # We expect the topology service to have stored its per–vector rewards under the key "per_vector_rewards"
    topo_rewards = data.get("per_vector_rewards")
    if topo_rewards is None:
        raise Exception("Topology per-vector rewards not found in JSON file")
    return topo_rewards

def compute_combined_per_vector_rewards(iteration):
    fem_rewards = fetch_fem_per_vector_rewards(iteration)
    topo_rewards = fetch_topo_per_vector_rewards(iteration)
    
    if len(fem_rewards) != len(topo_rewards):
        raise Exception("Mismatch in number of vectors between FEM and Topology rewards")
    
    # Combine rewards vector-by-vector (here we simply add the two rewards)
    combined_rewards = [f + t for f, t in zip(fem_rewards, topo_rewards)]
    
    stats = {
        "mean_reward": float(np.mean(combined_rewards)),
        "std_reward": float(np.std(combined_rewards))
    }
    
    breakdown = {
        "combined_per_vector_rewards": combined_rewards,
        "reward_stats": stats
    }
    
    return combined_rewards, breakdown

def compute_total_reward(iteration):
    # Compute the per–vector combined rewards and summary statistics.
    combined_rewards, breakdown = compute_combined_per_vector_rewards(iteration)
    # For an overall scalar reward (if needed) we use the mean of the combined per–vector rewards.
    total_reward = breakdown["reward_stats"]["mean_reward"]
    return total_reward, breakdown

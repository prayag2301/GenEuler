import numpy as np
import json

# keep your hyper-params
TARGET_DENSITY = 0.5
TOLERANCE      = 0.02
BETA           = 10.0  # controls the “peak sharpness”

def compute_per_vector_topology_rewards(topology_json_path):
    with open(topology_json_path, "r") as f:
        topo = json.load(f)
    dens = np.array(topo.get("density", []), dtype=float)
    if dens.size == 0:
        raise ValueError("No density data found in the topology file.")

    # 1) raw gaussian‐shaped reward
    raw = np.exp(-BETA * (dens - TARGET_DENSITY)**2)

    # 2) z-score normalize
    mu, sigma = raw.mean(), raw.std() + 1e-8
    norm = (raw - mu) / sigma

    # 3) clip
    norm = np.clip(norm, -3.0, +3.0)

    stats = {
        "mean_reward": float(norm.mean()),
        "std_reward":  float(norm.std())
    }
    return norm.tolist(), stats

def compute_total_topology_reward(iteration):
    per_vec, stats = compute_per_vector_topology_rewards(
        f"/app/src/assets/topology_results_{iteration}.json")
    total_reward = stats["mean_reward"]
    return total_reward, {
      "topology_stats": stats,
      "per_vector_rewards": per_vec
    }    
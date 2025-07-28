import numpy as np, json, os, requests

# Hyperparameters — tune these!
D_REF       = 0.1      # raise the “no‐penalty” threshold to catch small flexes
MAX_PENALTY = 1.0      # (you can leave this, or bump higher if you still want to clip)
ALPHA       = 100.0    # reduce the penalty scale so raw_pen stays below 1.0

FEM_SERVICE_URL = "http://fem_service:8001"

def reward_for_vector(displacement_vector):
    magnitude = np.linalg.norm(displacement_vector)
    # A quadratic penalty that starts penalizing as the magnitude increases above the reference
    # return - ALPHA * max(0, (magnitude - D_REF))**2
    # Always penalize quadratically (no threshold)
    return - ALPHA * (magnitude)**2

def compute_per_vector_rewards(fem_json_path):
    # 1) load raw displacement field
    with open(fem_json_path, "r") as f:
        fem_data = json.load(f)
    disp = np.array(fem_data.get("displacement", []))
    if disp.size == 0:
        raise RuntimeError("No displacement data found.")
    disp = disp.reshape(-1, 3)

    # 2) compute per-node deflection norm and quadratic penalty
    norms = np.linalg.norm(disp, axis=1)               # absolute displacement per node
    raw_pen = ALPHA * (norms ** 2)
    # Optionally: clip *above* some very large value if you want to guard against Inf,
    # but leave it uncapped on [0, ∞) so the penalty is meaningful
    # raw_pen = np.minimum(raw_pen, 1e6)  

    # 3) map into [0,1]: 1.0 = no deflection, 0.0 = worst penalty
    per_node_reward = - raw_pen  

    # 4) build stats from norms & penalty
    stats = {
        "mean_deflection":    float(norms.mean()),
        "max_deflection":     float(norms.max()),
        "penalty_mean":       float(raw_pen.mean()),
        "penalty_std":        float(raw_pen.std())
    }

    return per_node_reward.tolist(), stats

def compute_total_reward(iteration):
    # 1) fetch path from FEM service
    resp = requests.get(f"{FEM_SERVICE_URL}/fem_results/{iteration}")
    if resp.status_code != 200:
        raise RuntimeError(f"FEM service error: {resp.text}")
    fem_results = resp.json()
    fem_json = fem_results.get("json_path")
    if not fem_json or not os.path.exists(fem_json):
        raise RuntimeError(f"FEM JSON not found at {fem_json}")

    # 2) compute per-node rewards and stats
    per_vec, stats = compute_per_vector_rewards(fem_json)

    # 3) overall total reward is the mean of our [0,1] per-node scores
    total = float(np.mean(per_vec))

    # ─── (Optional) tiny bonus proportional to *number of nodes* in your new mesh,
    # which encourages growth if you still see stagnation around the attachment.
    # bonus_per_node = 1e-3
    # total += bonus_per_node * len(per_vec)

    breakdown = {
        "fem_stats":          stats,
        "per_vector_rewards": per_vec
    }
    return total, breakdown

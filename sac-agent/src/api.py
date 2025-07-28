import os
import json
import torch
from fastapi import FastAPI, HTTPException, Query
from config import DEVICE, DATA_PATHS, SAC_CONFIG, GNN_CONFIG, BOUNDING_BOX, ATTACHMENT_POSITION, TOLERANCE, MIN_ALLOWED_VERTICES
from utils import load_json_file, get_file_path, build_graph, normalize_features, filter_vertices_within_bounding_box, generate_vertices_for_nodes, generate_faces_for_nodes, vertex_inside
from agent import SACAgent
from db import initialize_db, get_latest_epoch, save_training_result, get_db_connection
import requests
import numpy as np
import time
import threading
import logging
import math
from fastapi.responses import JSONResponse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

logger = logging.getLogger(__name__)
app = FastAPI()

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize SAC agent.
agent = SACAgent(
    input_dim=GNN_CONFIG["input_dim"],
    action_dim=3,            # <— change this to 3
    sac_config=SAC_CONFIG,
    device=DEVICE
)

# now make sure the networks actually live on that device:
agent.actor.to(DEVICE)
agent.critic1.to(DEVICE)
agent.critic2.to(DEVICE)

CURRENT_EPOCH = 0

# --------------------------------------------
# Simplified Initialization using startup events
# --------------------------------------------

@app.on_event("startup")
def init_db():
    # Simplest layout — only initialize the database.
    initialize_db()
    logger.info("✅ Database initialized successfully.")

@app.on_event("startup")
def load_checkpoint():
    # Load the latest checkpoint after DB initialization.
    global CURRENT_EPOCH
    CURRENT_EPOCH = get_latest_epoch()
    checkpoint_path = os.path.join(checkpoint_dir, f"sac_checkpoint_epoch_{CURRENT_EPOCH}.pt")
    if os.path.exists(checkpoint_path):
        agent.actor.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        logger.info(f"✅ Loaded checkpoint from epoch {CURRENT_EPOCH}")
    else:
        logger.info("⚠️ No checkpoint found; using randomly initialized weights.")

@app.on_event("startup")
def warmup_cuda():
    """
    Force CUDA (and cuBLAS) to initialize on the main thread
    so later calls in worker threads won’t crash.
    """
    # DEVICE is a string like "cuda" or "cpu"
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        # Allocate a small matrix and do a multiply → invokes cuBLAS on main thread
        a = torch.rand((2,2), device=DEVICE)
        b = torch.rand((2,2), device=DEVICE)
        _ = torch.mm(a, b)           # this is a cuBLAS GEMM call
        torch.cuda.synchronize()     # wait for the BLAS call to finish
        logger.info(f"✅ CUDA/cuBLAS fully warmed up on {DEVICE}")

# ------------------------------------------------
# Rest of your API endpoints
# ------------------------------------------------

# Set the environment API base URL using the Docker service name.
mesh_api_base = os.getenv("MESH_SERVICE_API", "http://mesh_service:8003")
PREPROCESSING_API = "http://pre_processing:8007/latest_iteration"

def load_mesh_mapping_safe(iteration: int, num_nodes: int) -> list:
    mapping_filename = os.path.join(DATA_PATHS["mesh_data"], f"mesh_mapping_{iteration}.json")
    if not os.path.exists(mapping_filename):
        raise FileNotFoundError(f"Mesh mapping file not found: {mapping_filename}")
    mapping = json.load(open(mapping_filename, "r"))
    if len(mapping) != num_nodes:
        raise ValueError(f"Mapping length ({len(mapping)}) does not match number of nodes ({num_nodes})")
    return mapping

def sanitize(obj):
    """
    Recursively walk obj and replace NaN/Inf floats with 0.0.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0.0
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    else:
        return obj

@app.get("/get_action")
def get_action(iteration: int = Query(...)):
    global CURRENT_EPOCH

    try:
        # FREE UP ANY UNUSED CUDA MEMORY ───────────────────────────────
        torch.cuda.empty_cache()
        
        # ─── 0) reload latest weights if checkpoint advanced ─────────────────
        latest_epoch = get_latest_epoch()
        if latest_epoch != CURRENT_EPOCH:
            CURRENT_EPOCH = latest_epoch
            ckpt = os.path.join(checkpoint_dir,
                                 f"sac_checkpoint_epoch_{CURRENT_EPOCH}.pt")
            if os.path.exists(ckpt):
                agent.actor.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                logger.info(f"🔄 Reloaded actor weights from epoch {CURRENT_EPOCH}")
            else:
                logger.warning(f"No checkpoint at {ckpt}; using existing weights.")

        # ─── 1) build graph from merged + mesh JSON ───────────────────────────
        merged = load_json_file(get_file_path(iteration, "merged_data"))
        mesh   = load_json_file(get_file_path(iteration, "mesh_data"))
        # build_graph will now place both x and edge_index on the same DEVICE
        graph  = build_graph(merged, mesh, DEVICE)
        # (just to be 100% sure) move the entire PyG Data onto DEVICE
        graph  = graph.to(DEVICE)
        graph.x = normalize_features(graph.x)
        graph.x = torch.nan_to_num(graph.x, nan=0.0, posinf=1.0, neginf=-1.0)
        N = graph.x.shape[0]

        # ─── 1b) get per-node topology density (may be missing) ─────────────
        raw_d = merged.get("density", [])
        densities = np.array(raw_d, dtype=float)
        if densities.shape[0] != N:
            densities = np.zeros(N, dtype=float)
        density_thr = SAC_CONFIG.get("density_threshold", 0.5)

        # ─── 2) actor forward pass → get raw actor outputs ────────────────────
        torch.cuda.empty_cache()
        agent.actor.eval()
        with torch.no_grad():
            raw_add, removal_score = agent.get_action(graph)
        agent.actor.train()

        # ─── 3) normalize addition offsets into [0,1]^3 ───────────────────────
        norm_add = torch.sigmoid(raw_add)  # (N,3) in [0,1]
        # clamp any NaN/Inf → valid numbers
        norm_add = torch.nan_to_num(norm_add, nan=0.5, posinf=1.0, neginf=0.0)
        norm_np  = norm_add.cpu().numpy()                  # (N,3) numpy
        # likewise for removal scores
        rem_np   = removal_score.cpu().numpy().flatten()   # (N,)
        rem_np   = np.nan_to_num(rem_np, nan=0.0, posinf=1.0, neginf=0.0)

        # for training / replay: use the normalized‐coords tensor as your “action”
        add_offset = norm_add                            # torch.Tensor (N,3)
        add_np     = norm_np                             # numpy for summary

        # pull out the true, un-normalized node positions
        raw_pos = np.array([feat[:3] for feat in merged.get("node_features", [])])
        # build min/max vectors for the bbox
        mins = np.array([BOUNDING_BOX["x"][0],
                         BOUNDING_BOX["y"][0],
                         BOUNDING_BOX["z"][0]])
        maxs = np.array([BOUNDING_BOX["x"][1],
                         BOUNDING_BOX["y"][1],
                         BOUNDING_BOX["z"][1]])

        # ─── 3a) pick top-K actor proposals by confidence (ignore density) ────
        norms = np.linalg.norm(norm_np, axis=1)
        eps   = SAC_CONFIG.get("min_add_norm", 1e-3)
        K     = SAC_CONFIG.get("max_add_candidates", 150)

        # take the K largest norms
        add_idxs = np.argsort(-norms)[:K].tolist()

        # if for some reason there are fewer than K, pad with next best
        if len(add_idxs) < K:
            more = np.argsort(-norms)[len(add_idxs):K]
            add_idxs += [int(i) for i in more]

        # ─── 3b) ensure we have at least K candidates ─────────────────────────
        if len(add_idxs) < K:
            # pad with next best norms
            more = np.argsort(-norms)[len(add_idxs):K]
            add_idxs += [int(i) for i in more]

        # ─── 3c) build actor-proposed points *always* inside the bbox ─────────────
        cand_new = []
        for i in add_idxs:
            # map [0,1]^3 → real‐world bbox
            pt  = mins + norm_np[i] * (maxs - mins)
            off = pt - raw_pos[i]
            # enforce your minimum‐move norm
            if np.linalg.norm(off) < eps:
                continue
            cand_new.append(pt.tolist())

        # ─── 3d) fallback: pad until we have MIN_PAD points (even if actor gave a few)
        MIN_PAD = SAC_CONFIG.get("min_fallback_count", 10)
        if len(cand_new) < MIN_PAD:
            R = MIN_PAD
            tries = 0
            max_tries = R * 10
            seen = set()
            # keep adding random, deduped points until MIN_PAD reached
            while len(cand_new) < R and tries < max_tries:
                 tries += 1
                 pt = np.random.uniform(
                     [BOUNDING_BOX["x"][0], BOUNDING_BOX["y"][0], BOUNDING_BOX["z"][0]],
                     [BOUNDING_BOX["x"][1], BOUNDING_BOX["y"][1], BOUNDING_BOX["z"][1]]
                 )
                 # skip attachment region
                 if np.linalg.norm(pt - np.array(ATTACHMENT_POSITION)) < TOLERANCE:
                     continue
                # dedupe via coarse rounding
                 key = tuple(np.round(pt, 10))
                 if key in seen:
                     continue
                 seen.add(key)
                 cand_new.append(pt.tolist())

        # ─── 4) pick removals (throttle to at most 0.1% of nodes) ─────────────
        rem_thr = rem_np.mean() - 0.5 * rem_np.std()
        # only remove nodes whose score is *significantly* low
        rem_idxs = [i for i, s in enumerate(rem_np) if s < rem_thr]
        # NO forced fallback removal
        # clamp removals to at most a tiny fraction (or zero, if you prefer)
        MAX_REM = SAC_CONFIG.get("max_removals", 0)  # set to 0 by default
        rem_idxs = rem_idxs[:MAX_REM]

        # ─── 5) push to replay & train ────────────────────────────────────────
        torch.cuda.empty_cache()
        crew       = merged.get("individual_rewards", {}).get("combined", [])
        fem_reward = float(crew[0]) if crew else 0.0
        # base reward from mesh‐service…
        base_reward = fem_reward
        # bonus proportional to how many vertices we actually added this turn
        add_bonus_coeff = SAC_CONFIG.get("addition_reward_coeff", 0.1)
        add_frac = len(cand_new) / max(1, N)
        bonus = add_bonus_coeff * add_frac
        reward = base_reward + bonus
        # keep within [–1,1]
        reward = max(-1.0, min(1.0, reward))
        done       = False
        agent.replay_buffer.push(graph, add_offset, reward, graph, done)

        # track the last loss so we can return it
        total_loss = None
        for _ in range(SAC_CONFIG.get("updates_per_step", 1)):
            if len(agent.replay_buffer) < agent.replay_buffer.batch_size:
                break
            batch = agent.replay_buffer.sample()
            total_loss, actor_loss, critic_loss, ent = agent.train_step(batch)
            CURRENT_EPOCH += 1
            ckpt = os.path.join(checkpoint_dir, f"sac_checkpoint_epoch_{CURRENT_EPOCH}.pt")
            torch.save(agent.actor.state_dict(), ckpt)
            save_training_result(CURRENT_EPOCH, critic_loss, actor_loss, ckpt)
            logger.info(f"Epoch {CURRENT_EPOCH} trained (loss={total_loss:.4f})")

        # ─── 6) dedupe actor proposals ───────────────────
        seen_pts = set()
        unique_cand = []
        for pt in cand_new:
            key = tuple(np.round(pt, 10))
            if key not in seen_pts:
                seen_pts.add(key)
                unique_cand.append(pt)

        # ─── 7) if after dedupe we still have too few, pad with random uniques ───
        MIN_UNIQUE = SAC_CONFIG.get("min_fallback_count", 10)
        seen2 = set(tuple(np.round(p,6)) for p in unique_cand)
        tries = 0
        while len(unique_cand) < MIN_UNIQUE and tries < MIN_UNIQUE*10:
            tries += 1
            pt = np.random.uniform(
                [BOUNDING_BOX["x"][0], BOUNDING_BOX["y"][0], BOUNDING_BOX["z"][0]],
                [BOUNDING_BOX["x"][1], BOUNDING_BOX["y"][1], BOUNDING_BOX["z"][1]]
            )
            if np.linalg.norm(pt - np.array(ATTACHMENT_POSITION)) < TOLERANCE:
                continue
            key = tuple(np.round(pt,10))
            if key in seen2:
                continue
            seen2.add(key)
            unique_cand.append(pt.tolist())
            
        logger.info(f"After padding, sending {len(unique_cand)} unique additions")

        cand_new = unique_cand

        faces   = generate_faces_for_nodes(add_idxs, merged, mesh)
        payload = {
            "iteration":      iteration,
            "add_vertices":   cand_new,
            "add_faces":      faces,
            "remove_indices": rem_idxs
        }
        # convert any numpy types → native Python so json.dumps will succeed
        from utils import convert_np_types
        safe_payload = convert_np_types(payload)
        resp = requests.post(f"{mesh_api_base}/update_mesh", json=safe_payload, timeout=60000)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # if mesh service rejects due to ill-conditioned mesh, just skip this update
            if resp.status_code == 400:
                logger.warning(f"Mesh service rejected update (400): {resp.text!r} – skipping additions/removals this iteration.")
            else:
                # some other failure is unexpected: re-raise
                raise

        # ─── 7) return summary ────────────────────────────────────────────────
        # build the return payload (including epoch & loss)
        out = {
            "iteration":         iteration,
            "addition_offsets":  add_np.tolist(),
            "removal_scores":    rem_np.tolist(),
            "num_new_additions": len(cand_new),
            "num_removals":      len(rem_idxs)
        }
        # attach the epoch and loss from this call
        out["epoch"] = CURRENT_EPOCH
        out["loss"]  = float(total_loss) if total_loss is not None else None

        json_out = os.path.join(DATA_PATHS["merged_data"], f"sac_agent_actions_{iteration}.json")
        with open(json_out, "w") as f:
            json.dump(out, f, indent=2)

        return {"status":"success", **sanitize(out)}

    except Exception as e:
        logger.exception("error in get_action")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_checkpoint")
def save_checkpoint():
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = int(time.time())
        checkpoint_filename = f"sac_checkpoint_{timestamp}.pt"
        checkpoint_path_new = os.path.join(checkpoint_dir, checkpoint_filename)
        torch.save(agent.actor.state_dict(), checkpoint_path_new)
        print(f"✅ Checkpoint saved at {checkpoint_path_new}")
        return {"status": "success", "message": f"Checkpoint saved at {checkpoint_path_new}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/reset_agent")
def reset_agent():
    try:
        checkpoint_path = os.path.join(checkpoint_dir, f"sac_checkpoint_epoch_{CURRENT_EPOCH}.pt")
        if os.path.exists(checkpoint_path):
            agent.actor.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            print(f"✅ SAC agent reset using checkpoint from epoch {CURRENT_EPOCH}")
            return {"status": "success", "message": f"SAC agent reset using checkpoint from epoch {CURRENT_EPOCH}"}
        else:
            return {"status": "error", "message": "No checkpoint found to reset the agent."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
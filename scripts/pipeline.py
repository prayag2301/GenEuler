#!/usr/bin/env python3
"""
gen_euler_pipeline_flow.py

This Prefect flow orchestrates the full GenEuler pipeline. The steps are:
  1. (If iteration==1) Add initial random material within the bounding box.
  2. Generate Mesh.
  3. Run FEM Analysis.
  4. Run Topology Optimization.
  5. Compute Reward for FEM.
  6. Compute Reward for Topology.
  7. Compute Combined Reward.
  8. Merge dynamic data.
  9. Run the SAC Agent task to determine node-level adjustments and trigger the mesh update.
  
The flow loops indefinitely until manually stopped.
Each task waits until its predecessor finishes.
"""

import os
import json
import numpy as np
import time
from prefect import flow, task, get_run_logger
import requests
import subprocess
from itertools import count

# ----------------------------------------------------------------------
# Service endpoint configuration.
# Note: We no longer use SHARED_ENV_API; all environment-related calls now point to mesh_service.
MESH_SERVICE_API = os.getenv("MESH_SERVICE_API", "http://localhost:8003")
FEM_SERVICE_API = os.getenv("FEM_SERVICE_API", "http://localhost:8001")
TOPOLOGY_SERVICE_API = os.getenv("TOPOLOGY_SERVICE_API", "http://localhost:8002")
REWARD_FEM_API = os.getenv("REWARD_FEM_API", "http://localhost:8005")
REWARD_TOPOLOGY_API = os.getenv("REWARD_TOPOLOGY_API", "http://localhost:8006")
REWARD_COMBINED_API = os.getenv("REWARD_COMBINED_API", "http://localhost:8004")
PRE_PROCESSING_API = os.getenv("PRE_PROCESSING_API", "http://localhost:8007")
SAC_AGENT_API = os.getenv("SAC_AGENT_API", "http://localhost:8008")

SERVICE_ENDPOINTS = {
    "update_mesh": f"{MESH_SERVICE_API}/update_mesh",  # NEW: consolidated endpoint for both adding and removing material.
    "generate_mesh": f"{MESH_SERVICE_API}/generate_mesh",
    "run_fem": f"{FEM_SERVICE_API}/run_fem",
    "optimize_topology": f"{TOPOLOGY_SERVICE_API}/optimize",
    "compute_reward_fem": f"{REWARD_FEM_API}/compute_reward",
    "compute_reward_topology": f"{REWARD_TOPOLOGY_API}/compute_reward",
    "compute_reward_combined": f"{REWARD_COMBINED_API}/compute_reward",
    "merge_data": f"{PRE_PROCESSING_API}/merge_data",
    "sac_agent_get_action": f"{SAC_AGENT_API}/get_action"
}

# Global timeout (in seconds) for long-running requests.
LONG_TIMEOUT = 60000  # 100 minutes

# ----------------------------------------------------------------------
# Prefect Tasks.
@task(retries=3, retry_delay_seconds=60)
def add_material_task(iteration: int) -> dict:
    logger = get_run_logger()
    if iteration == 1:
        logger.info("Iteration 1: triggering mesh generation (initial material addition).")
        # For iteration 1, call the generate_mesh endpoint directly.
        response = requests.post(SERVICE_ENDPOINTS["generate_mesh"], json={}, timeout=LONG_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Generate Mesh Response (from add_material_task): {json.dumps(result)}")
        return result
    else:
        logger.info("Not iteration 1; skipping add_material_task.")
        return {"status": "skipped", "message": "Initial material addition is only performed at iteration 1."}

@task(retries=2, retry_delay_seconds=60)
def run_fem_task() -> dict:
    logger = get_run_logger()
    response = requests.get(SERVICE_ENDPOINTS["run_fem"], timeout=LONG_TIMEOUT)
    response.raise_for_status()
    result = response.json()
    logger.info(f"Run FEM Response: {json.dumps(result)}")
    return result

@task(retries=2, retry_delay_seconds=60)
def optimize_topology_task() -> dict:
    logger = get_run_logger()
    response = requests.post(SERVICE_ENDPOINTS["optimize_topology"], json={}, timeout=30)
    response.raise_for_status()
    result = response.json()
    logger.info(f"Optimize Topology Response: {json.dumps(result)}")
    return result

@task(retries=2, retry_delay_seconds=60)
def compute_reward_fem_task() -> dict:
    logger = get_run_logger()
    response = requests.get(SERVICE_ENDPOINTS["compute_reward_fem"], timeout=30)
    response.raise_for_status()
    result = response.json()
    logger.info(f"Compute Reward FEM Response: {json.dumps(result)}")
    return result

@task(retries=2, retry_delay_seconds=60)
def compute_reward_topology_task() -> dict:
    logger = get_run_logger()
    response = requests.get(SERVICE_ENDPOINTS["compute_reward_topology"], timeout=30)
    response.raise_for_status()
    result = response.json()
    logger.info(f"Compute Reward Topology Response: {json.dumps(result)}")
    return result

@task(retries=2, retry_delay_seconds=60)
def compute_reward_combined_task() -> dict:
    logger = get_run_logger()
    response = requests.get(SERVICE_ENDPOINTS["compute_reward_combined"], timeout=30)
    response.raise_for_status()
    result = response.json()
    logger.info(f"Compute Reward Combined Response: {json.dumps(result)}")
    return result

@task(retries=2, retry_delay_seconds=60)
def merge_data_task() -> dict:
    logger = get_run_logger()
    response = requests.get(SERVICE_ENDPOINTS["merge_data"], timeout=30)
    response.raise_for_status()
    result = response.json()
    logger.info(f"Merge Data Response: {json.dumps(result)}")
    return result

@task(retries=3, retry_delay_seconds=60)
def sac_agent_task(iteration: int) -> dict:
    """
    Calls the SAC agent API endpoint with the iteration number.
    The SAC agent API itself handles loading the merged data, building the graph,
    computing node-level actions, and triggering the mesh update via /update_mesh.
    """
    logger = get_run_logger()
    params = {"iteration": iteration}
    response = requests.get(SERVICE_ENDPOINTS["sac_agent_get_action"], params=params, timeout=LONG_TIMEOUT)
    response.raise_for_status()
    result = response.json()
    # use the true counts returned by the SAC agent
    summary = {
        "iteration":       iteration,
        "num_additions":   result.get("num_new_additions", 0),
        "num_removals":    result.get("num_removals", 0),
        "add_head":        result.get("addition_offsets", [])[:3],
        "rem_head":        result.get("removal_scores", [])[:3],
    }
    logger.info(f"SAC Agent Response summary: {json.dumps(summary)}")
    return result

# ----------------------------------------------------------------------
# Define the Prefect flow.
@flow(name="GenEuler Pipeline", persist_result=True)
def gen_euler_pipeline(iteration: int = 1):
    logger = get_run_logger()
    logger.info(f"Starting pipeline iteration {iteration}")

    # For the first iteration, add initial material via update_mesh.
    add_material_future = add_material_task.submit(iteration)
    add_material_result = add_material_future.result()
    
    fem_future = run_fem_task.submit()
    fem_result = fem_future.result()
    
    topology_future = optimize_topology_task.submit()
    topology_result = topology_future.result()
    
    reward_fem_future = compute_reward_fem_task.submit()
    reward_fem = reward_fem_future.result()
    
    reward_topology_future = compute_reward_topology_task.submit()
    reward_topology = reward_topology_future.result()
    
    reward_combined_future = compute_reward_combined_task.submit()
    reward_combined = reward_combined_future.result()
    
    merge_data_future = merge_data_task.submit()
    merge_data_result = merge_data_future.result()
    
    # Increase the wait time after merging data before calling the SAC agent.
    logger.info("Waiting 5 seconds before triggering SAC Agent...")
    time.sleep(5)
    
    sac_future = sac_agent_task.submit(iteration)
    sac_result = sac_future.result()
    
    # Additional wait after the SAC agent call, if desired.
    logger.info("Waiting an additional 30 seconds after SAC Agent update...")
    time.sleep(15)
    
    # build a concise SAC summary instead of the full ±1 lists
    sac_summary = {
        "epoch": sac_result.get("epoch"),
        "loss":  sac_result.get("loss"),
        "num_additions": sac_result["num_new_additions"],
        "num_removals":  sac_result["num_removals"],
        "add_head":      sac_result["addition_offsets"][:3],
        "rem_head":      sac_result["removal_scores"][:3],
    }
    return {
        "iteration": iteration,
        "add_material": add_material_result,
        "fem": fem_result,
        "topology": topology_result,
        "reward_fem": reward_fem,
        "reward_topology": reward_topology,
        "reward_combined": reward_combined,
        "merged_data": merge_data_result,
        "sac_agent": sac_summary,
    }

# ----------------------------------------------------------------------
# Main loop: continuously run the pipeline until manually stopped.
if __name__ == "__main__":
    import os, json, time
    os.makedirs("results_json", exist_ok=True)

    # starts at 1, automatically increments
    for iteration in count(1):
        try:
            result = gen_euler_pipeline(iteration=iteration)
            print(f"Pipeline Result (iter {iteration}):", json.dumps(result, indent=4))

            out_file = f"results_json/pipeline_result_{iteration}.json"
            with open(out_file, "w") as fp:
                json.dump(result, fp, indent=4)

        except Exception as e:
            print(f"Error in iteration {iteration}: {e}")
        time.sleep(1)
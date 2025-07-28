from fastapi import FastAPI, HTTPException, Query
from .merge import merge_all_data, load_json_file
from .db import initialize_db, save_merged_data, get_latest_merged_iteration
import os

app = FastAPI()

@app.on_event("startup")
def startup_event():
    initialize_db()

@app.get("/merge_data")
def api_merge_data(iteration: int = Query(None, description="Iteration number for merging")):
    try:
        # Determine iteration number.
        if iteration is None:
            iteration = get_latest_merged_iteration() + 1

        # Use the convex hull JSON file as the static mesh.
        mesh_json_filename = f"convex_hull_{iteration}.json"

        reward_files = {
            "combined": f"reward_combined_result_{iteration}.json",
            "topology": f"reward_topology_results_{iteration}.json",
            "fem": f"reward_fem_results_{iteration}.json"
        }
        simulation_files = {
            "fem_results": f"fem_results_{iteration}.json",
            "topology_results": f"topology_results_{iteration}.json"
        }
        # Merge dynamic data with the static mesh (loaded from the convex hull JSON file).
        merged_dynamic, mesh_data = merge_all_data(mesh_json_filename, reward_files, simulation_files)

        # Optionally, compute an aggregate metric (e.g., mean of combined rewards).
        combined_rewards = merged_dynamic.get("individual_rewards", {}).get("combined", [])
        if combined_rewards:
            total_reward = float(sum(combined_rewards) / len(combined_rewards))
        else:
            total_reward = 0.0

        # Save merged dynamic data, static mesh data, and environment data into the DB.
        save_merged_data(iteration, merged_dynamic, mesh_data)

        # Return a concise success response with file paths.
        response = {
            "status": "success",
            "iteration": iteration,
            "merged_json_path": f"/app/src/assets/merged_data_{iteration}.json",
            "mesh_json_path": f"/app/src/assets/mesh_data_{iteration}.json"
        }
        return response
    except FileNotFoundError as e:
        # Provide a clearer 404 error when expected input files are missing
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latest_iteration")
def latest_iteration():
    try:
        iteration = get_latest_merged_iteration()
        return {"iteration": iteration}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException, Query
from .db import initialize_db, save_reward_result, get_latest_reward_iteration
from compute_reward import compute_total_topology_reward, compute_per_vector_topology_rewards

app = FastAPI()

@app.on_event("startup")
def startup_event():
    initialize_db()
    
@app.get("/compute_reward")
def api_compute_reward(iteration: int = Query(None, description="Iteration number for reward computation")):
    try:
        # If no iteration is provided, use the next iteration after the latest saved reward.
        if iteration is None:
            iteration = get_latest_reward_iteration() + 1
        
        # Compute overall reward
        total_reward, breakdown = compute_total_topology_reward(iteration)
        
        # Compute per-vector rewards from the topology JSON file
        topology_json_path = f"/app/src/assets/topology_results_{iteration}.json"
        per_vector_rewards, reward_stats = compute_per_vector_topology_rewards(topology_json_path)
        
        # save the result (the breakdown is stored as a JSON file and the path is inserted into the DB)
        save_reward_result(iteration, total_reward, breakdown)
        
        return {
            "status": "success",
            "iteration": iteration,
            "total_reward": total_reward,
            "reward_breakdown_path": f"/app/src/assets/reward_topology_results_{iteration}.json"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
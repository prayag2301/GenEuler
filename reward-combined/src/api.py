from fastapi import FastAPI, HTTPException, Query
from .compute_reward import compute_total_reward
from .db import initialize_db, save_reward_result, get_latest_reward_iteration

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
        
        total_reward, breakdown = compute_total_reward(iteration)
        
        # Save the detailed combined per-vector rewards breakdown to a JSON file,
        # and store the file path along with the scalar total reward in the database.
        save_reward_result(iteration, total_reward, breakdown)
        
        return {
            "status": "success",
            "iteration": iteration,
            "total_reward": total_reward,
            "reward_breakdown_path": f"/app/src/assets/reward_combined_result_{iteration}.json"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

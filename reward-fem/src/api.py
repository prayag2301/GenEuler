from fastapi import FastAPI, HTTPException, Query
from .compute_reward import compute_total_reward, compute_per_vector_rewards
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

        # Compute overall FEM reward by fetching FEM results from the FEM service.
        total_reward, breakdown = compute_total_reward(iteration)
        # Compute per–vector rewards from the FEM JSON file.
        fem_json_path = f"/app/src/assets/fem_results_{iteration}.json"
        per_vector_rewards, reward_stats = compute_per_vector_rewards(fem_json_path)
        breakdown["per_vector_rewards"] = per_vector_rewards
        breakdown["reward_stats"] = reward_stats

        # Save the combined result.
        save_reward_result(iteration, total_reward, breakdown)

        return {
            "status": "success",
            "iteration": iteration,
            "total_reward": total_reward,
            "reward_breakdown_path": f"/app/src/assets/reward_fem_results_{iteration}.json"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


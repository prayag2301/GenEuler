from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
import json
from fem_analysis import FEMAnalysis
from db import get_db_connection, initialize_db, get_latest_iteration, save_fem_results

app = FastAPI()

@app.on_event("startup")
def init_db():
    initialize_db()

@app.get("/run_fem")
def run_fem_analysis():
    """
    Run FEM analysis using the latest mesh from the mesh service.
    This endpoint instantiates FEMAnalysis, which fetches mesh metadata from the mesh service,
    loads the mesh from the shared assets, and runs the FEM simulation.
    """
    try:
        fem_analyzer = FEMAnalysis()
        results = fem_analyzer.run_analysis()
        save_fem_results(results)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latest_fem_iteration")
def latest_fem_iteration():
    """
    Return the latest FEM iteration from the FEM database.
    """
    iteration = get_latest_iteration()
    return {"status": "success", "iteration": iteration}

@app.get("/fem_results/{iteration}")
def get_fem_results(iteration: int):
    """
    Fetch FEM results (file paths) for a given iteration.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT xdmf_path, h5_path, json_path FROM fem_data WHERE iteration = %s", (iteration,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return {"status": "success", "xdmf_path": row[0], "h5_path": row[1], "json_path": row[2]}
    else:
        raise HTTPException(status_code=404, detail="FEM results not found for the given iteration")

@app.get("/download_fem_json/{iteration}")
def download_fem_json(iteration: int):
    """
    Download the FEM JSON results for a specific iteration.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT json_path FROM fem_data WHERE iteration = %s", (iteration,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row and os.path.exists(row[0]):
        return FileResponse(row[0], media_type="application/json", filename=f"fem_results_{iteration}.json")
    else:
        raise HTTPException(status_code=404, detail="FEM JSON file not found for the given iteration")

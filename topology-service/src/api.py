from fastapi import FastAPI, HTTPException
import requests
import numpy as np
import os
import json
from fastapi.responses import FileResponse
from topology_solver import TopologySolver, load_mesh_and_displacement
from db import get_latest_fem_iteration, save_topology_results, get_db_connection

app = FastAPI()

# Service URLs
FEM_SERVICE_URL = "http://fem_service:8001"
MESH_SERVICE_URL = "http://mesh_service:8003"

# Directory to store topology results
ASSETS_DIR = "/app/src/assets/"
os.makedirs(ASSETS_DIR, exist_ok=True)

@app.post("/optimize")
def run_topology_optimization():
    """
    Runs topology optimization by fetching data from FEM Service.
    """
    print("🔵 Fetching latest FEM results from API...")
    iteration = get_latest_fem_iteration()
    
    # ✅ Fetch FEM results
    fem_response = requests.get(f"{FEM_SERVICE_URL}/fem_results/{iteration}")
    if (fem_response.status_code != 200):
        raise HTTPException(status_code=500, detail=f"❌ FEM service error: {fem_response.text}")

    fem_data = fem_response.json()
    xdmf_path = fem_data.get("xdmf_path")
    h5_path = fem_data.get("h5_path")

    print(f"✅ FEM XDMF Path: {xdmf_path}")
    print(f"✅ FEM H5 Path: {h5_path}")

    if (not xdmf_path or not h5_path):
        raise HTTPException(status_code=500, detail="❌ FEM results missing 'xdmf_file' or 'h5_file'.")

    # ✅ Ensure files exist in `topology_service`
    xdmf_local = f"/app/src/assets/{os.path.basename(xdmf_path)}"
    h5_local = f"/app/src/assets/{os.path.basename(h5_path)}"

    # ✅ Download the files from `fem_service`
    if (not os.path.exists(xdmf_local)):
        print(f"🔵 Downloading {xdmf_path} to {xdmf_local}...")
        xdmf_content = requests.get(f"{FEM_SERVICE_URL}/download_xdmf/{iteration}")
        with open(xdmf_local, "wb") as f:
            f.write(xdmf_content.content)

    if (not os.path.exists(h5_local)):
        print(f"🔵 Downloading {h5_path} to {h5_local}...")
        h5_content = requests.get(f"{FEM_SERVICE_URL}/download_h5/{iteration}")
        with open(h5_local, "wb") as f:
            f.write(h5_content.content)

    print("✅ FEM files successfully downloaded.")

    # ✅ Load mesh and displacement
    mesh, disp_fun = load_mesh_and_displacement(xdmf_local, h5_local)

    # ✅ Run topology optimization
    solver = TopologySolver(mesh=mesh)
    optimized_density = solver.optimize_structure(disp_fun)

    # Get the (latest) iteration number (here we assume get_latest_fem_iteration() returns the new iteration)
    iteration = get_latest_fem_iteration()

    # Build file paths directly in the assets folder with the iteration number appended:
    density_json_path = os.path.join(ASSETS_DIR, f"topology_results_{iteration}.json")
    xdmf_filename    = os.path.join(ASSETS_DIR, f"topology_density_{iteration}.xdmf")
    h5_filename      = os.path.join(ASSETS_DIR, f"topology_density_{iteration}.h5")

    # ✅ Store JSON results to file
    density_data = {"iteration": iteration, "density": optimized_density.tolist()}
    with open(density_json_path, "w") as f:
        json.dump(density_data, f, indent=4)
    print(f"✅ Stored density JSON in {density_json_path}")

    print(f"🔵 Saving topology results for iteration {iteration}...")
    # ✅ Save results to database – now we pass the file path for the density results JSON
    save_topology_results(
        iteration=iteration,
        volume_fraction=solver.volume_fraction,
        penalization=solver.penalization,
        filter_radius=solver.filter_radius,
        density_json=density_json_path,  # file path instead of dumped JSON string
        xdmf_path=xdmf_filename,
        h5_path=h5_filename
    )
    print(f"✅ Topology results for iteration {iteration} saved in database.")

    return {
        "status": "success",
        "iteration": iteration,
        "density_json": density_json_path,
        "xdmf_path": xdmf_filename,
        "h5_path": h5_filename
    }

@app.get("/latest_topology_iteration")
def latest_topology_iteration():
    """
    Returns the latest topology iteration number.
    """
    iteration = get_latest_fem_iteration()
    return {"status": "success", "iteration": iteration}

@app.get("/get_topology_by_iteration/{iteration}")
def get_topology_by_iteration(iteration: int):
    """
    Fetch the topology results for a specific iteration.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT density_json, xdmf_path, h5_path FROM topology_data WHERE iteration = %s", (iteration,))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result:
        return {
            "status": "success",
            "density_json": result[0],
            "xdmf_path": result[1],
            "h5_path": result[2]
        }
    else:
        return {"status": "error", "message": "Topology results not found for the given iteration"}

@app.get("/download_topology_json/{iteration}")
def download_topology_json(iteration: int):
    """
    Returns the file path for the topology JSON results for a specific iteration.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT density_json FROM topology_data WHERE iteration = %s", (iteration,))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result:
        return {"status": "success", "density_json": result[0]}
    else:
        return {"status": "error", "message": "JSON results not found for the given iteration"}

@app.get("/download_topology_xdmf/{iteration}")
def download_topology_xdmf(iteration: int):
    """
    Fetch the topology density XDMF file for a specific iteration.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT xdmf_path FROM topology_data WHERE iteration = %s", (iteration,))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result and os.path.exists(result[0]):
        return FileResponse(result[0], media_type="application/octet-stream", filename=f"topology_density_{iteration}.xdmf")
    else:
        return {"status": "error", "message": "XDMF file not found for the given iteration"}

@app.get("/download_topology_h5/{iteration}")
def download_topology_h5(iteration: int):
    """
    Fetch the topology density H5 file for a specific iteration.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT h5_path FROM topology_data WHERE iteration = %s", (iteration,))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result and os.path.exists(result[0]):
        return FileResponse(result[0], media_type="application/octet-stream", filename=f"topology_density_{iteration}.h5")
    else:
        return {"status": "error", "message": "H5 file not found for the given iteration"}

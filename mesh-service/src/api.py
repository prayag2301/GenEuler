from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
import os
import json
import logging
from db import get_db_connection, initialize_db, get_latest_iteration
# Import functions from the updated mesh module (which now uses Gmsh instead of Meshio)
from mesh import (
    generate_mesh_and_mapping,
    update_mesh_from_actions,  # Updated to use Gmsh-based logic
    convert_mesh_to_json,
    MESH_DIR,
    initialize_baseline_mesh,
    MeshEnvironment
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
os.makedirs(MESH_DIR, exist_ok=True)

@app.on_event("startup")
def startup_event():
    initialize_db()

@app.get("/latest_mesh_iteration")
def latest_mesh_iteration():
    iteration = get_latest_iteration()
    return {"status": "success", "iteration": iteration}

@app.post("/generate_mesh")
def api_generate_mesh(iteration: int = Query(None, description="Optional raw state iteration number to use for baseline generation")):
    try:
        if get_latest_iteration() == 0:
            logger.info("No mesh exists yet. Initializing baseline mesh.")
            baseline_info = initialize_baseline_mesh()
            return {"status": "success", **baseline_info}
        else:
            result = generate_mesh_and_mapping(iteration)
            return {"status": "success", **result}
    except Exception as e:
        logger.error(f"Error generating mesh: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_mesh")
async def update_mesh(request: Request):
    actions = await request.json()
    # now includes both add_vertices & add_faces
    result = update_mesh_from_actions(actions)
    # if the helper signals an error, forward it as a 400
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return {"status": "success", **result}

@app.get("/download_mesh/{iteration}")
def download_mesh(iteration: int):
    mesh_filename = os.path.join(MESH_DIR, f"convex_hull_{iteration}.msh")
    if os.path.exists(mesh_filename):
        return FileResponse(mesh_filename, media_type="application/octet-stream", filename=f"convex_hull_{iteration}.msh")
    else:
        raise HTTPException(status_code=404, detail="Mesh file not found.")

@app.get("/download_xdmf/{iteration}")
def download_xdmf(iteration: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT xdmf_path FROM mesh_data WHERE iteration = %s", (iteration,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result and os.path.exists(result[0]):
        return FileResponse(result[0], media_type="application/octet-stream", filename=f"convex_hull_{iteration}.xdmf")
    else:
        raise HTTPException(status_code=404, detail="XDMF file not found.")

@app.get("/download_h5/{iteration}")
def download_h5(iteration: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT h5_path FROM mesh_data WHERE iteration = %s", (iteration,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result and os.path.exists(result[0]):
        return FileResponse(result[0], media_type="application/octet-stream", filename=f"convex_hull_{iteration}.h5")
    else:
        raise HTTPException(status_code=404, detail="H5 file not found.")

@app.get("/get_mesh_state/{iteration}")
def get_mesh_state(iteration: int):
    mesh_filename = f"convex_hull_{iteration}.msh"
    try:
        mesh_state = convert_mesh_to_json(mesh_filename)
        return {"status": "success", "mesh_state": mesh_state}
    except Exception as e:
        logger.error(f"Error converting mesh to JSON: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_mesh_info/{iteration}")
def get_mesh_info(iteration: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT iteration, mesh_path, xdmf_path, h5_path, mapping_path, mesh_json_path
        FROM mesh_data
        WHERE iteration = %s;
    """, (iteration,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return {
            "status": "success",
            "iteration": row[0],
            "mesh_path": row[1],
            "xdmf_path": row[2],
            "h5_path": row[3],
            "mapping_file": row[4],
            "mesh_json": row[5]
        }
    else:
        raise HTTPException(status_code=404, detail="Mesh info not found for the given iteration.")

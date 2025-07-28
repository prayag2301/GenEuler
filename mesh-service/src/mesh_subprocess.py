import sys
import numpy as np
import gmsh
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_mesh(vertices_filename, mesh_filename):
    # Load vertices from the provided .npy file
    vertices = np.load(vertices_filename)
    
    # IMPORTANT: Ensure that gmsh functions run in this subprocess's main thread
    gmsh.initialize()
    gmsh.model.add("mesh")
    
    # Add points to the model (using 1-based indexing)
    for i, v in enumerate(vertices):
        gmsh.model.geo.addPoint(v[0], v[1], v[2], 1.0, i + 1)
    
    gmsh.model.geo.synchronize()
    
    # Generate 3D mesh (using built-in Delaunay 3D algorithm)
    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_filename)
    gmsh.finalize()
    
    logger.info(f"Mesh generated and written to {mesh_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mesh_subprocess.py <vertices_filename> <mesh_filename>")
        sys.exit(1)
    generate_mesh(sys.argv[1], sys.argv[2])

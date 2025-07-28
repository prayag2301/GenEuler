#!/usr/bin/env python3
"""
mesh_convert_subprocess.py

This script converts a Gmsh-generated .msh file into a DolfinX Mesh and then writes
it to an XDMF file (and also generates a dummy displacement HDF5 file).

It reads the mesh using gmsh, extracts the nodes and tetrahedral elements,
builds a mapping between the Gmsh node tags and a 0-based contiguous indexing scheme,
and then uses dolfinx.mesh.create_mesh to create a DolfinX mesh.
"""

import sys
import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx.fem import coordinate_element
from dolfinx.cpp.mesh import CellType
from dolfinx.mesh import create_mesh
from dolfinx.io import XDMFFile
import h5py

def convert_mesh(mesh_filename):
    # We force serial execution with COMM_SELF.
    comm = MPI.COMM_SELF

    # Initialize gmsh and open the .msh file
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(mesh_filename)
    except Exception as e:
        print(f"Error during gmsh.open: {e}", flush=True)
        gmsh.finalize()
        sys.exit(1)

    # --- Extract Nodes ---
    # gmsh.model.mesh.getNodes() returns three items:
    #   nodeTags: a list of node tags (which may not be sequential)
    #   nodeCoords: a flat list of coordinates (x0,y0,z0, x1,y1,z1, ...)
    #   parametricCoords (which we ignore here)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    nodeTags = np.array(nodeTags, dtype=int)
    # Reshape nodeCoords into an array of shape (n_nodes, 3)
    nodes = np.array(nodeCoords, dtype=float).reshape((-1, 3))
    
    # Build an explicit mapping from each node tag to its 0-based index
    tag_to_index = {tag: idx for idx, tag in enumerate(nodeTags)}

    # --- Extract Tetrahedral Elements ---
    # Get all elements from the mesh
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements()
    
    # Look for tetrahedral elements (Gmsh element type 4 in .msh 2.2 format)
    tetra_index = None
    for i, et in enumerate(elemTypes):
        if et == 4:
            tetra_index = i
            break
    if tetra_index is None:
        gmsh.finalize()
        print("Error: No tetrahedral elements found in the mesh.", flush=True)
        sys.exit(1)

    # Extract the tetrahedral element node tags, and reshape to a (n_cells, 4) array.
    # Note: These are the original node tags from gmsh.
    raw_elem = np.array(elemNodeTags[tetra_index], dtype=int).reshape((-1, 4))
    # Use the mapping to convert each node tag to a contiguous index (0-based)
    # This ensures that if node tags are not sequential the connectivity remains valid.
    tetra_cells = np.vectorize(lambda tag: tag_to_index[tag])(raw_elem)

    # Finalize gmsh as we no longer need it
    gmsh.finalize()

    # --- Create the Dolfinx Mesh ---
    # Create a Dolfinx mesh using the extracted connectivity (tetra_cells) and nodes.
    mesh = create_mesh(comm, tetra_cells, nodes, coordinate_element(CellType.tetrahedron, 1))

    # --- Write the Mesh to an XDMF File ---
    xdmf_filename = mesh_filename.replace(".msh", "_mesh.xdmf")
    with XDMFFile(comm, xdmf_filename, "w") as xdmf_file:
        xdmf_file.write_mesh(mesh)

    # --- Create a Dummy Displacement HDF5 File ---
    h5_filename = mesh_filename.replace(".msh", "_disp.h5")
    n_points = nodes.shape[0]
    # In this example, we create a displacement field of zeros.
    displacement_data = np.zeros((n_points, 3), dtype=np.float64)
    with h5py.File(h5_filename, "w") as h5_file:
        func_group = h5_file.create_group("Function")
        f_group = func_group.create_group("f")
        f_group.create_dataset("0", data=displacement_data)

    # Print the results to stdout in a comma-separated format.
    print(f"{xdmf_filename},{h5_filename}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mesh_convert_subprocess.py <mesh_filename>", flush=True)
        sys.exit(1)
    convert_mesh(sys.argv[1])

#!/usr/bin/env python3
# """
# mesh_convert_subprocess.py

# This script converts a Gmsh-generated .msh file into a DolfinX mesh.
# It creates a mapping from the Gmsh node tags to a 0-based contiguous
# indexing scheme and then uses dolfinx.mesh.create_mesh to create the mesh.
# """

# import sys
# import numpy as np
# import gmsh
# from mpi4py import MPI
# from dolfinx.mesh import create_mesh
# from dolfinx.fem import coordinate_element
# from dolfinx.cpp.mesh import CellType
# from dolfinx.io import XDMFFile
# import h5py

# def convert_mesh(mesh_filename):
#     # Use MPI.COMM_SELF for serial execution in the conversion process.
#     comm = MPI.COMM_SELF

#     # Initialize gmsh and open the mesh file.
#     gmsh.initialize()
#     gmsh.option.setNumber("General.Terminal", 0)
#     gmsh.open(mesh_filename)

#     # --- Extract Nodes ---
#     nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
#     nodeTags = np.array(nodeTags, dtype=int)
#     # Reshape the coordinates into (n_nodes, 3)
#     nodes = np.array(nodeCoords, dtype=float).reshape((-1, 3))

#     # Build a mapping from the Gmsh node tags to new contiguous indices.
#     tag_to_index = {tag: idx for idx, tag in enumerate(nodeTags)}
    
#     # --- Extract Tetrahedral Elements ---
#     elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements()
#     tetra_index = None
#     for i, et in enumerate(elemTypes):
#         if et == 4:
#             tetra_index = i
#             break
#     if tetra_index is None:
#         gmsh.finalize()
#         raise RuntimeError("Error: No tetrahedral elements found in the mesh.")
    
#     # Get the raw element connectivity (node tags) and reshape to (-1, 4)
#     raw_elem = np.array(elemNodeTags[tetra_index], dtype=int).reshape((-1, 4))
#     # Re-index connectivity using our mapping
#     tetra_cells = np.vectorize(lambda tag: tag_to_index[tag])(raw_elem)

#     gmsh.finalize()  # Close gmsh now that we have the data.

#     # --- Create the DolfinX Mesh ---
#     # IMPORTANT: Before creating the mesh, make sure that the cell-to-vertex connectivity exists.
#     # Determine topological dimension:
#     tdim = 3  # For tetrahedral meshes, tdim is 3.
#     # (If you prefer, you can also use: tdim = loaded_mesh.topology.dim)
#     # Call create_connectivity to ensure connectivity from cells (dimension 3) to vertices (dimension 0) exists:
#     # (Note: This call is not needed later because create_mesh will set up geometry from the provided points.)
#     # However, if later diagnostics or methods need the connectivity, it must be created.
#     # In our simple conversion, this extra call can be added before (or after) create_mesh if desired.
    
#     # Now create the mesh (the function create_mesh does not itself require connectivity to have been created,
#     # but if you later want to query cell connectivity you must create it).
#     mesh = create_mesh(comm, tetra_cells, nodes, coordinate_element(CellType.tetrahedron, 1))

#     # --- Write the Mesh to an XDMF File (which references a mesh HDF5 file) ---
#     xdmf_filename = mesh_filename.replace(".msh", "_mesh.xdmf")
#     with XDMFFile(comm, xdmf_filename, "w") as xdmf_file:
#         xdmf_file.write_mesh(mesh)
#     # The XDMF file will reference a mesh HDF5 file generated alongside.
#     h5_filename = mesh_filename.replace(".msh", "_mesh.h5")

#     # --- Create a Dummy Displacement HDF5 File ---
#     h5_disp_filename = mesh_filename.replace(".msh", "_disp.h5")
#     n_points = nodes.shape[0]
#     displacement_data = np.zeros((n_points, 3), dtype=np.float64)
#     with h5py.File(h5_disp_filename, "w") as h5_file:
#         func_group = h5_file.create_group("Function")
#         f_group = func_group.create_group("f")
#         f_group.create_dataset("0", data=displacement_data)

#     # Print only the XDMF and mesh H5 filenames in a comma-separated format.
#     print(f"{xdmf_filename},{h5_filename}", flush=True)

#     return mesh, xdmf_filename, h5_filename

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python mesh_convert_subprocess.py <mesh_filename>", flush=True)
#         sys.exit(1)
#     convert_mesh(sys.argv[1])

import psycopg2
import os

DB_NAME = os.getenv("POSTGRES_DB", "mesh_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "berlin2000")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")  # Will use mesh_db inside Docker

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    """
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)

def initialize_db():
    """
    Create the necessary database table if it does not exist.
    
    The table 'mesh_data' stores:
      - iteration: the mesh iteration number (unique)
      - mesh_path: the path to the .msh file in the shared assets directory
      - xdmf_path: the path to the converted XDMF file
      - h5_path: the path to the H5 file used for displacement data
      - mapping_path: the path to the JSON file that contains the mapping
      - mesh_json_path: the path of the converted mesh to JSON
      - timestamp: automatically set to the current time
    """
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS mesh_data (
            id SERIAL PRIMARY KEY,
            iteration INT UNIQUE NOT NULL,
            mesh_path TEXT NOT NULL,
            xdmf_path TEXT NOT NULL,
            h5_path TEXT NOT NULL,
            mapping_path TEXT NOT NULL,
            mesh_json_path TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW()
        );
    """)

    conn.commit()
    cur.close()
    conn.close()

    print("✅ Database initialized successfully.")

def get_latest_iteration():
    """
    Fetch the latest mesh iteration from the database.
    
    Returns:
        int: The highest iteration number stored in the table, or 0 if none exists.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT iteration FROM mesh_data ORDER BY iteration DESC LIMIT 1;")
    row = cur.fetchone()
    latest_iteration = row[0] if row else 0

    cur.close()
    conn.close()

    return latest_iteration if latest_iteration else 0

def store_mesh_metadata(iteration, mesh_path, xdmf_path, h5_path, mapping_path, mesh_json_path):
    """
    Store metadata for a mesh iteration into the database.
    
    Parameters:
      - iteration (int): The new mesh iteration number.
      - mesh_path (str): The file path to the generated .msh file.
      - xdmf_path (str): The file path to the converted XDMF file.
      - h5_path (str): The file path to the H5 file for displacement data.
      - mapping_path (str): The file path to the JSON mapping file.
      
    All file paths should be absolute paths pointing to the shared assets folder.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO mesh_data (iteration, mesh_path, xdmf_path, h5_path, mapping_path, mesh_json_path)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (iteration, mesh_path, xdmf_path, h5_path, mapping_path, mesh_json_path),
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Stored mesh iteration {iteration} in database.")

# Uncomment the following if you want to initialize the database on script execution.
# if __name__ == "__main__":
#     initialize_db()

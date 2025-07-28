import os
import psycopg2
import psycopg2.extras
import json
import numpy as np

# Database connection parameters for the merged data service
DB_NAME = os.getenv("POSTGRES_DB", "merged_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "berlin2000")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")

# Directory where JSON files will be stored.
JSON_SAVE_DIR = "/app/src/assets"

def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST
    )

def initialize_db():
    conn = get_db_connection()
    cur = conn.cursor()
    # Create a table that stores the iteration, and separate JSON file paths for:
    # - merged dynamic data,
    # - static mesh data,
    cur.execute("""
        CREATE TABLE IF NOT EXISTS merged_data (
            id SERIAL PRIMARY KEY,
            iteration INT UNIQUE NOT NULL,
            json_path_merged TEXT NOT NULL,
            json_path_mesh TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Merged Data DB initialized successfully.")

def save_merged_data(iteration, merged_dynamic, mesh_data):
    """
    Save the merged dynamic data and static mesh data state as separate JSON files,
    and record their paths in the database.
    """
    merged_converted = convert_np_types(merged_dynamic)
    mesh_converted = convert_np_types(mesh_data)
    
    json_filename_merged = f"merged_data_{iteration}.json"
    json_filename_mesh = f"mesh_data_{iteration}.json"
    
    json_path_merged = os.path.join(JSON_SAVE_DIR, json_filename_merged)
    json_path_mesh = os.path.join(JSON_SAVE_DIR, json_filename_mesh)
    
    with open(json_path_merged, "w") as f:
        json.dump(merged_converted, f, indent=4)
    with open(json_path_mesh, "w") as f:
        json.dump(mesh_converted, f, indent=4)
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO merged_data (iteration, json_path_merged, json_path_mesh)
        VALUES (%s, %s, %s)
    """, (iteration, json_path_merged, json_path_mesh))
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Merged data saved for iteration {iteration}.\nMerged JSON: {json_path_merged}\nMesh JSON: {json_path_mesh}\n")

def get_latest_merged_iteration():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT iteration FROM merged_data ORDER BY iteration DESC LIMIT 1;")
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else 0

def convert_np_types(obj):
    """
    Recursively convert NumPy types to native Python types so that JSON can serialize them.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(v) for v in obj]
    else:
        return obj
import os
import psycopg2
import psycopg2.extras
import json
import numpy as np

# Reward database connection parameters
DB_NAME = os.getenv("POSTGRES_DB", "reward_fem_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "berlin2000")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")

def get_db_connection():
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)

def initialize_db():
    conn = get_db_connection()
    cur = conn.cursor()
    # Update the table to store the JSON file path as TEXT
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reward_fem_data (
            id SERIAL PRIMARY KEY,
            iteration INT UNIQUE NOT NULL,
            total_reward FLOAT NOT NULL,
            json_path TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Reward DB initialized successfully.")
    
def save_reward_result(iteration, total_reward, breakdown):
    # Convert any NumPy types in the breakdown dictionary to native Python types
    breakdown_converted = convert_np_types(breakdown)
    # Define the JSON file path – here we use the assets folder (adjust if needed)
    json_file_path = f"/app/src/assets/reward_fem_results_{iteration}.json"
    
    # Write the breakdown dictionary to the JSON file
    with open(json_file_path, "w") as f:
        json.dump(breakdown_converted, f, indent=4)
    
    conn = get_db_connection()
    cur = conn.cursor()
    # Now insert the file path (a plain string) into the database.
    cur.execute("""
        INSERT INTO reward_fem_data (iteration, total_reward, json_path)
        VALUES (%s, %s, %s)
    """, (iteration, float(total_reward), json_file_path))
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Reward result saved for iteration {iteration} with JSON at {json_file_path}.")

def get_latest_reward_iteration():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT iteration FROM reward_fem_data ORDER BY iteration DESC LIMIT 1;")
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else 0

def convert_np_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(v) for v in obj]
    else:
        return obj

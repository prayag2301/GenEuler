import psycopg2
import os

DB_NAME = os.getenv("POSTGRES_DB", "fem_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "berlin2000")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")  # Will use fem_db inside Docker

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    """
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)

def initialize_db():
    """
    Create the fem_data table if it doesn't exist.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fem_data (
            id SERIAL PRIMARY KEY,
            iteration INT UNIQUE NOT NULL,
            xdmf_path TEXT NOT NULL,
            h5_path TEXT NOT NULL,
            json_path TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_fem_results(results):
    """
    Save FEM results file paths into the PostgreSQL database.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # ✅ Fetch latest iteration number
    cur.execute("SELECT iteration FROM fem_data ORDER BY iteration DESC LIMIT 1;")
    latest_iteration = cur.fetchone()
    iteration = (latest_iteration[0] + 1) if latest_iteration else 1

    cur.execute("""
        INSERT INTO fem_data (iteration, xdmf_path, h5_path, json_path)
        VALUES (%s, %s, %s, %s)
    """, (iteration, results["xdmf_file"], results["h5_file"], results["json_file"]))
    
    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ FEM results stored in database under iteration {iteration}.")
    
def get_latest_iteration():
    """
    Fetch the latest mesh iteration from the database.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT iteration FROM fem_data ORDER BY iteration DESC LIMIT 1")
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result[0] if result else 0

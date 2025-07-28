import psycopg2
import os

# Load environment variables for database connection
DB_NAME = os.getenv("POSTGRES_DB", "topology_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "berlin2000")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")  # Uses 'topology_db' inside Docker

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database.
    """
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)

def initialize_db():
    """
    Creates the required database table if it doesn't already exist.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Create a table for topology optimization results
    cur.execute("""
        CREATE TABLE IF NOT EXISTS topology_data (
            id SERIAL PRIMARY KEY,
            iteration INT UNIQUE NOT NULL,
            volume_fraction FLOAT NOT NULL,
            penalization FLOAT NOT NULL,
            filter_radius FLOAT NOT NULL,
            density_json TEXT NOT NULL,
            xdmf_path TEXT NOT NULL,
            h5_path TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW()
        );
    """)

    conn.commit()
    cur.close()
    conn.close()

    print("✅ Topology Database initialized successfully.")

# Ensure DB initializes on import
initialize_db()

def get_latest_fem_iteration():
    """
    Fetch the latest FEM iteration from fem_db.
    """
    conn = psycopg2.connect(
        dbname="fem_db",  # ✅ Connect explicitly to fem_db
        user="postgres",
        password="berlin2000",
        host="fem_db"
    )
    cur = conn.cursor()
    cur.execute("SELECT iteration FROM fem_data ORDER BY iteration DESC LIMIT 1;")
    row = cur.fetchone()
    cur.close()
    conn.close()

    return row[0] if row else 0

def save_topology_results(iteration, volume_fraction, penalization, filter_radius, density_json, xdmf_path, h5_path):
    """
    Saves the topology optimization results to the database, ensuring no duplicate iterations.
    The density_results_json now stores the path to the JSON file.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Check if the iteration already exists
    cur.execute("SELECT 1 FROM topology_data WHERE iteration = %s", (iteration,))
    exists = cur.fetchone()

    if exists:
        print(f"⚠️ Iteration {iteration} already exists in topology_data. Updating instead of inserting.")
        cur.execute("""
            UPDATE topology_data
            SET volume_fraction = %s, penalization = %s, filter_radius = %s,
                density_json = %s, xdmf_path = %s, h5_path = %s
            WHERE iteration = %s;
        """, (volume_fraction, penalization, filter_radius, density_json, xdmf_path, h5_path, iteration))
    else:
        print(f"✅ Inserting new topology results for iteration {iteration}.")
        cur.execute("""
            INSERT INTO topology_data (iteration, volume_fraction, penalization, filter_radius, density_json, xdmf_path, h5_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """, (iteration, volume_fraction, penalization, filter_radius, density_json, xdmf_path, h5_path))

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Topology results saved successfully for iteration {iteration}.")


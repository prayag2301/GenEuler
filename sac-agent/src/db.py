# db.py
import os
import psycopg2
import psycopg2.extras

DB_NAME = os.getenv("POSTGRES_DB", "sac_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "berlin2000")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    """
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)

def initialize_db():
    """
    Create the sac_training table if it doesn't exist.
    This table logs training loss and model checkpoint paths per epoch.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sac_training (
            id SERIAL PRIMARY KEY,
            epoch INT UNIQUE NOT NULL,
            critic_loss FLOAT NOT NULL,
            actor_loss FLOAT NOT NULL,
            checkpoint_path TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ SAC Training DB initialized successfully.")

def save_training_result(epoch, critic_loss, actor_loss, checkpoint_path):
    """
    Save or update the training result for a given epoch.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO sac_training (epoch, critic_loss, actor_loss, checkpoint_path)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (epoch) DO UPDATE SET
            critic_loss    = EXCLUDED.critic_loss,
            actor_loss     = EXCLUDED.actor_loss,
            checkpoint_path = EXCLUDED.checkpoint_path;
    """, (epoch, float(critic_loss), float(actor_loss), checkpoint_path))
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Training result saved for epoch {epoch} (critic={critic_loss:.4f}, actor={actor_loss:.4f}) at {checkpoint_path}.")

def get_latest_epoch():
    """
    Retrieve the latest (highest) epoch number from the sac_training table.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT epoch FROM sac_training ORDER BY epoch DESC LIMIT 1;")
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else 0

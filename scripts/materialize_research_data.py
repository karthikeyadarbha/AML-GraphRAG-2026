import logging
from pathlib import Path
import duckdb

# Configure logging for clear run-time status.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standardized environment setup
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# The target DuckDB database file
DB_PATH = PROCESSED_DIR / "argus_research.db"

def materialize_duckdb_environment():
    """
    Reads the SOTA generated data from data/raw/ and materializes it into
    a highly optimized, zero-IPC DuckDB database for GraphRAG traversal.
    """
    logger.info(f"Initializing DuckDB materialization at: {DB_PATH}")
    
    # Connect to the persistent DuckDB file
    con = duckdb.connect(str(DB_PATH))

    try:
        # --- 1. Load the synthetic transaction ledger ---
        ledger_path = RAW_DATA_DIR / "synthetic_ledger.csv"
        logger.info(f"Materializing raw_ledger from {ledger_path}...")
        # read_csv_auto automatically infers schema types (including BOOLEAN and TIMESTAMP)
        if not ledger_path.exists():
            raise FileNotFoundError(f"Missing required file: {ledger_path}")
        con.execute(f"CREATE OR REPLACE TABLE raw_ledger AS SELECT * FROM read_csv_auto('{ledger_path}')")

        # --- 2. Load the KYC profiles ---
        kyc_path = RAW_DATA_DIR / "kyc_profiles.json"
        logger.info(f"Materializing kyc_profiles from {kyc_path}...")
        if not kyc_path.exists():
            raise FileNotFoundError(f"Missing required file: {kyc_path}")
        con.execute(f"CREATE OR REPLACE TABLE kyc_profiles AS SELECT * FROM read_json_auto('{kyc_path}')")

        # --- 3. Load the adverse media documents ---
        adverse_path = RAW_DATA_DIR / "adverse_media.json"
        logger.info(f"Materializing adverse_media from {adverse_path}...")
        if not adverse_path.exists():
            raise FileNotFoundError(f"Missing required file: {adverse_path}")
        con.execute(f"CREATE OR REPLACE TABLE adverse_media AS SELECT * FROM read_json_auto('{adverse_path}')")

        # --- Validation ---
        tables = con.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        
        logger.info(f"SUCCESS: DuckDB successfully materialized the following tables: {table_names}")

    except Exception as e:
        logger.error(f"Failed to materialize DuckDB tables: {e}")
    finally:
        # Always ensure the connection is closed to flush to disk
        con.close()
        logger.info("Database connection closed. Zero-IPC Database is ready for Step 3.")

if __name__ == "__main__":
    materialize_duckdb_environment()
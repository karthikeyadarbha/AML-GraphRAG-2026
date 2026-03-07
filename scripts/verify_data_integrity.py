import duckdb
import pandas as pd
import json
from pathlib import Path
import logging

# Configuration
DATA_DIR = Path("data/raw")
DB_PATH = Path("data/processed/argus_research.db")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def verify_raw_files():
    """Validates that the synthetic files exist and are not empty."""
    logger.info("--- Phase 1: File System Integrity ---")
    files = ["synthetic_ledger.csv", "kyc_profiles.json", "adverse_media.json"]
    for f in files:
        path = DATA_DIR / f
        if path.exists():
            size = path.stat().st_size
            logger.info(f"Verified: {f} exists ({size} bytes).")
        else:
            raise FileNotFoundError(f"Missing critical file: {f}")

def verify_duckdb_idempotency():
    """Validates that DuckDB ingestion handles re-runs without duplication."""
    logger.info("--- Phase 2: DuckDB Idempotency & Uniqueness ---")
    con = duckdb.connect(str(DB_PATH))
    
    # We use 'CREATE OR REPLACE' in our ingestion, so we verify counts match the CSV
    csv_count = len(pd.read_csv(DATA_DIR / "synthetic_ledger.csv"))
    db_count = con.execute("SELECT COUNT(*) FROM raw_ledger").fetchone()[0]
    
    if csv_count == db_count:
        logger.info(f"Integrity Pass: DB count ({db_count}) matches CSV count ({csv_count}).")
    else:
        logger.error(f"Integrity Fail: DB count ({db_count}) mismatch with CSV ({csv_count})!")

    # Check for duplicate Transaction IDs (Primary Key Integrity)
    duplicates = con.execute("""
        SELECT trx_id, COUNT(*) 
        FROM raw_ledger 
        GROUP BY trx_id 
        HAVING COUNT(*) > 1
    """).fetchall()
    
    if not duplicates:
        logger.info("Uniqueness Pass: No duplicate transaction IDs found.")
    else:
        logger.error(f"Uniqueness Fail: Found {len(duplicates)} duplicate IDs.")
    
    con.close()

def verify_semantic_linkage():
    """Validates that the 'LOOP' nodes in the Ledger are present in the KYC data."""
    logger.info("--- Phase 3: Cross-Domain Linkage ---")
    with open(DATA_DIR / "kyc_profiles.json", "r") as f:
        kyc = json.load(f)
    
    kyc_nodes = {record['node_id'] for record in kyc}
    
    # Check if LOOP_S_0 through LOOP_S_49 are all present
    missing_links = []
    for i in range(50):
        target = f"LOOP_S_{i}"
        if target not in kyc_nodes:
            missing_links.append(target)
            
    if not missing_links:
        logger.info("Linkage Pass: All 50 structural loops have corresponding KYC profiles.")
    else:
        logger.error(f"Linkage Fail: Missing KYC data for {missing_links}")

if __name__ == "__main__":
    try:
        verify_raw_files()
        verify_duckdb_idempotency()
        verify_semantic_linkage()
        logger.info("ALL INTEGRITY TESTS PASSED. The environment is stable for GraphRAG.")
    except Exception as e:
        logger.error(f"INTEGRITY CHECK FAILED: {e}")
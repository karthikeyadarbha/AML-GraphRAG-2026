import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# --- RESEARCH REPRODUCIBILITY SEED ---
# This constant is the "Anchor" for the experiment's ground truth.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 1. Standardized environment setup
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Faker with a fixed seed for deterministic semantic synthesis.
FAKER = Faker()
FAKER.seed_instance(RANDOM_SEED)

# Configure logging for clear run-time status.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_unified_dataset(num_benign: int = 10000, num_loops: int = 50) -> None:
    """
    Create a linked dataset where structural anomalies
    in the ledger correspond to semantic clues in the KYC/News corpus.
    """
    logger.info(
        "Initializing Materialization: %s transactions, %s loops.",
        num_benign,
        num_loops,
    )

    # --- PART A: STRUCTURAL LEDGER GENERATION ---
    ledger_data = {
        "trx_id": [f"TXN_{i}" for i in range(num_benign)],
        "source_id": [f"CUST_{np.random.randint(1, 1000)}" for _ in range(num_benign)],
        "target_id": [f"CUST_{np.random.randint(1, 1000)}" for _ in range(num_benign)],
        "amount": np.random.uniform(10, 5000, num_benign),
        "timestamp": pd.date_range(start="2026-01-01", periods=num_benign, freq="min"),
    }
    ledger_df = pd.DataFrame(ledger_data)

    # --- PART B: SEMANTIC CORPUS & LOOP INJECTION ---
    kyc_records = []
    news_records = []

    for i in range(num_loops):
        s, m, t = f"LOOP_S_{i}", f"LOOP_M_{i}", f"LOOP_T_{i}"
        ts = pd.Timestamp("2026-02-01") + pd.Timedelta(hours=i)

        # 1. Create Ledger Edges (Structural Evidence)
        loop_txns = [
            {"trx_id": f"L_A_{i}", "source_id": s, "target_id": m, "amount": 1000.0, "timestamp": ts},
            {"trx_id": f"L_B_{i}", "source_id": m, "target_id": t, "amount": 950.0,  "timestamp": ts + pd.Timedelta(seconds=30)},
            {"trx_id": f"L_C_{i}", "source_id": t, "target_id": s, "amount": 915.0,  "timestamp": ts + pd.Timedelta(seconds=60)},
        ]
        ledger_df = pd.concat([ledger_df, pd.DataFrame(loop_txns)], ignore_index=True)

        # 2. Create Semantic Evidence (Investigative Evidence)
        kyc_records.append({
            "node_id": s,
            "entity_name": FAKER.company(),
            "jurisdiction": random.choice(["Panama", "Cyprus", "Cayman Islands", "Seychelles"]),
            "investigator_notes": f"High frequency circular transfers detected. Linked to beneficial owner {FAKER.name()}."
        })

        news_records.append({
            "related_node": s,
            "source": "Global Fin-Watch",
            "article_snippet": f"Investigation into {FAKER.word()} networks reveals systematic tax evasion in {random.choice(['Eastern Europe', 'Bermuda'])}."
        })

    # --- PART C: NOISE INJECTION (Benign Entities) ---
    for i in range(500):
        node = f"CUST_{i}"
        kyc_records.append({
            "node_id": node,
            "entity_name": FAKER.name(),
            "jurisdiction": "United Kingdom",
            "investigator_notes": "Retail account. Low risk profile.",
        })

    # --- PART D: FILE PERSISTENCE ---
    ledger_path = DATA_DIR / "synthetic_ledger.csv"
    kyc_path = DATA_DIR / "kyc_profiles.json"
    news_path = DATA_DIR / "adverse_media.json"

    ledger_df.to_csv(ledger_path, index=False)
    with kyc_path.open("w", encoding="utf-8") as kyc_file:
        json.dump(kyc_records, kyc_file, indent=4)
    with news_path.open("w", encoding="utf-8") as news_file:
        json.dump(news_records, news_file, indent=4)

    logger.info("SUCCESS: Structural and Semantic evidence materialized in %s", DATA_DIR)


if __name__ == "__main__":
    generate_unified_dataset()
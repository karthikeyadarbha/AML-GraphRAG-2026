import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List

logger = logging.getLogger(__name__)

def generate_benign_haystack(num_accounts: int = 5000, num_txns: int = 100000) -> pd.DataFrame:
    """
    Phase 1: Generates the benign macroeconomic baseline (The Haystack).
    Utilizes a log-normal distribution to mimic real-world financial transaction sizes,
    ensuring statistical realism for peer-review validation.
    """
    logger.info(f"Phase 1: Generating {num_txns:,} benign transactions across {num_accounts:,} accounts...")
    
    # Set seed for reproducible academic baseline
    np.random.seed(42)
    
    accounts: List[str] = [f"ACC_{i}" for i in range(num_accounts)]
    sources: np.ndarray = np.random.choice(accounts, num_txns)
    targets: np.ndarray = np.random.choice(accounts, num_txns)
    
    # Enforce topological realism: Prevent 0-hop self-loops (A -> A)
    mask = sources == targets
    while mask.any():
        targets[mask] = np.random.choice(accounts, mask.sum())
        mask = sources == targets

    # Log-normal distribution prevents negative values and models wealth disparity
    amounts: np.ndarray = np.clip(
        np.random.lognormal(mean=np.log(500), sigma=0.8, size=num_txns), 
        a_min=10.0, 
        a_max=None
    )
    
    # Stochastic temporal generation (Transactions occurring in Jan 2026)
    start_date = datetime(2026, 1, 1)
    timestamps = [start_date + timedelta(minutes=int(m)) for m in np.random.randint(0, 43200, num_txns)]

    benign_df = pd.DataFrame({
        'Transaction_ID': [f"TXN_BEN_{i}" for i in range(num_txns)],
        'Source_Account': sources,
        'Target_Account': targets,
        'Transfer_Weight': np.round(amounts, 2),
        'Timestamp': timestamps,
        'Is_Synthetic_Fraud': False  # Target label for recall metrics
    })
    
    logger.info("Phase 1: Benign ABM baseline generated successfully.")
    return benign_df
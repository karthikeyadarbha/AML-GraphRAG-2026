import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def inject_illicit_typologies(benign_df: pd.DataFrame, num_cycles: int = 25) -> pd.DataFrame:
    """
    Phase 2: Parametric topological injection.
    Injects circular multi-hop networks (AML Layering) while strictly enforcing 
    a Principal Value Retention (PVR) of 85% to 95%.
    """
    logger.info(f"Phase 2: Injecting {num_cycles} strict AML loops into the ledger...")
    np.random.seed(42)
    
    illicit_txns = []
    base_time = datetime(2026, 1, 15)
    
    for cycle_id in range(num_cycles):
        # Dynamically allocate 4 to 6 hops to challenge the SQL depth traversal
        hops: int = np.random.randint(4, 7)
        shells = [f"SHELL_{cycle_id}_{step}" for step in range(hops)]
        
        current_vol: float = np.random.uniform(50000.0, 250000.0)
        
        # Mathematically enforced friction to land in the 85-95% threshold
        target_pvr: float = np.random.uniform(0.85, 0.95) 
        friction_per_hop: float = 1.0 - (target_pvr ** (1.0 / hops))
        
        for step in range(hops):
            source = shells[step]
            target = shells[0] if step == hops - 1 else shells[step + 1]
            
            # Apply mathematical friction (smurfing/fees) on every hop after origin
            if step > 0: 
                current_vol *= (1.0 - friction_per_hop)
            
            # Power-law burst timing: Rapid multi-hop velocity (MVR)
            timestamp = base_time + timedelta(minutes=(cycle_id * 60) + (step * 5))
            
            illicit_txns.append({
                'Transaction_ID': f"TXN_ILLICIT_{cycle_id}_{step}",
                'Source_Account': source,
                'Target_Account': target,
                'Transfer_Weight': np.round(current_vol, 2),
                'Timestamp': timestamp,
                'Is_Synthetic_Fraud': True
            })

    # Unify and sort chronologically to emulate a real database append log
    unified_df = pd.concat([benign_df, pd.DataFrame(illicit_txns)], ignore_index=True)
    unified_df.sort_values(by='Timestamp', inplace=True, ignore_index=True)
    
    logger.info(f"Phase 2: Injection complete. Total ledger edges: {len(unified_df):,}")
    return unified_df
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_benign_haystack(num_accounts=5000, num_txns=50000):
    """
    Engineering the macroeconomic haystack using Log-Normal distributions.
    """
    np.random.seed(42)
    
    # 1. Initialize Nodes (Accounts)
    accounts = [f"ACC_{i:05d}" for i in range(num_accounts)]
    
    # 2. Generate Base Transactions (Log-Normal Distribution)
    mu, sigma = 4.5, 1.2 
    amounts = np.random.lognormal(mean=mu, sigma=sigma, size=num_txns)
    amounts = np.round(np.clip(amounts, 10, 50000), 2)
    
    # 3. Generate Timestamps 
    start_date = datetime(2025, 1, 1)
    timestamps = [start_date + timedelta(days=random.randint(0, 90), hours=random.randint(0, 23), minutes=random.randint(0, 59)) for _ in range(num_txns)]
    timestamps.sort()
    
    transactions = []
    for i in range(num_txns):
        source, target = random.sample(accounts, 2)
        transactions.append({
            "Transaction_ID": f"TXN_BASE_{i:07d}",
            "Source_Account": source,
            "Target_Account": target,
            "Amount": amounts[i],
            "Timestamp": timestamps[i],
            "Is_Synthetic_Fraud": 0,
            "Typology": "Standard_Retail",
            "Risk_Level": "LOW_RISK",
            "Scenario_Label": "Standard_Retail"
        })
        
    df = pd.DataFrame(transactions)
    
    # 4. Governance Enforcement: Inject Counter-Leakage Structuring
    # Legitimate transactions just under reporting thresholds to force topological reasoning
    num_counter_leakage = int(num_txns * 0.05)
    for i in range(num_counter_leakage):
        idx = random.randint(0, num_txns - 1)
        df.at[idx, 'Amount'] = round(random.uniform(7500, 9999), 2)
        df.at[idx, 'Typology'] = "Counter_Leakage_Structuring"
        df.at[idx, 'Risk_Level'] = "MEDIUM_RISK"
        df.at[idx, 'Scenario_Label'] = "Counter_Leakage_Structuring"
        
    df.to_csv("data/raw/synthetic_ledger_baseline.csv", index=False)
    print("Phase 1 Complete: ABM Baseline generated with Counter-Leakage.")
    return df

if __name__ == "__main__":
    generate_benign_haystack()
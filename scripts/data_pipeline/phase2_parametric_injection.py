import pandas as pd
from datetime import timedelta
import random

def inject_illicit_typologies(df, num_cycles=25):
    """
    Parametric injection of U-Turn laundering loops.
    Forces multi-hop velocity (MVR) and 70-90% PVR to test the graph traversal.
    """
    new_txns = []
    base_time = df['Timestamp'].max() - timedelta(days=30)
    
    for i in range(num_cycles): 
        anchor = f"ACC_UTURN_ANCHOR_{i}"
        mules = [f"ACC_MULE_{i}_{j}" for j in range(3)]
        
        principal = round(random.uniform(50000, 150000), 2)
        current_amount = principal
        current_time = base_time + timedelta(days=random.randint(1, 10))
        
        # Hop 1: Placement to first mule
        new_txns.append({"Transaction_ID": f"TXN_UTURN_{i}_1", "Source_Account": anchor, "Target_Account": mules[0], "Amount": current_amount, "Timestamp": current_time, "Is_Synthetic_Fraud": 1, "Typology": "U_Turn"})
        
        # Intermediate Layering (1-5 day delays, 1-3% frictional loss per hop)
        for j in range(2):
            current_time += timedelta(days=random.randint(1, 5))
            fee_decay = random.uniform(0.97, 0.99) 
            current_amount = round(current_amount * fee_decay, 2)
            new_txns.append({"Transaction_ID": f"TXN_UTURN_{i}_{j+2}", "Source_Account": mules[j], "Target_Account": mules[j+1], "Amount": current_amount, "Timestamp": current_time, "Is_Synthetic_Fraud": 1, "Typology": "U_Turn"})
            
        # Integration: Final return hop to anchor
        current_time += timedelta(days=random.randint(1, 5))
        final_amount = round(current_amount * random.uniform(0.95, 0.98), 2) 
        new_txns.append({"Transaction_ID": f"TXN_UTURN_{i}_FINAL", "Source_Account": mules[-1], "Target_Account": anchor, "Amount": final_amount, "Timestamp": current_time, "Is_Synthetic_Fraud": 1, "Typology": "U_Turn"})

    return pd.concat([df, pd.DataFrame(new_txns)], ignore_index=True)

if __name__ == "__main__":
    df_base = pd.read_csv("data/raw/synthetic_ledger_baseline.csv", parse_dates=['Timestamp'])
    df_injected = inject_illicit_typologies(df_base)
    # (Additional functions for Front Business Activity inserted here)
    
    df_injected = df_injected.sort_values(by='Timestamp').reset_index(drop=True)
    df_injected.to_csv("data/raw/synthetic_ledger_final.csv", index=False)
    print("Phase 2 Complete: Parametric U-Turn typologies injected.")
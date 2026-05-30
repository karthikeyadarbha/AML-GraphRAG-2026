import pandas as pd
from datetime import timedelta
import random


def inject_u_turn_typologies(df, num_cycles=25):
    new_txns = []
    base_time = df['Timestamp'].max() - timedelta(days=30)

    for i in range(num_cycles):
        anchor = f"ACC_UTURN_ANCHOR_{i}"
        mules = [f"ACC_MULE_{i}_{j}" for j in range(3)]

        principal = round(random.uniform(50000, 150000), 2)
        current_amount = principal
        current_time = base_time + timedelta(days=random.randint(1, 10))

        # Hop 1: Placement to first mule
        new_txns.append({
            "Transaction_ID": f"TXN_UTURN_{i}_1",
            "Source_Account": anchor,
            "Target_Account": mules[0],
            "Amount": current_amount,
            "Timestamp": current_time,
            "Is_Synthetic_Fraud": 1,
            "Typology": "U_Turn",
            "Risk_Level": "CRITICAL_RISK",
            "Scenario_Label": "U_Turn_Laundering"
        })

        # Intermediate Layering (1-5 day delays, 1-3% frictional loss per hop)
        for j in range(2):
            current_time += timedelta(days=random.randint(1, 5))
            fee_decay = random.uniform(0.97, 0.99)
            current_amount = round(current_amount * fee_decay, 2)
            new_txns.append({
                "Transaction_ID": f"TXN_UTURN_{i}_{j+2}",
                "Source_Account": mules[j],
                "Target_Account": mules[j+1],
                "Amount": current_amount,
                "Timestamp": current_time,
                "Is_Synthetic_Fraud": 1,
                "Typology": "U_Turn",
                "Risk_Level": "CRITICAL_RISK",
                "Scenario_Label": "U_Turn_Laundering"
            })

        # Integration: Final return hop to anchor
        current_time += timedelta(days=random.randint(1, 5))
        final_amount = round(current_amount * random.uniform(0.95, 0.98), 2)
        new_txns.append({
            "Transaction_ID": f"TXN_UTURN_{i}_FINAL",
            "Source_Account": mules[-1],
            "Target_Account": anchor,
            "Amount": final_amount,
            "Timestamp": current_time,
            "Is_Synthetic_Fraud": 1,
            "Typology": "U_Turn",
            "Risk_Level": "CRITICAL_RISK",
            "Scenario_Label": "U_Turn_Laundering"
        })

    return pd.DataFrame(new_txns)


def inject_smurfing_clusters(df, num_clusters=20):
    new_txns = []
    base_time = df['Timestamp'].max() - timedelta(days=25)

    for i in range(num_clusters):
        anchor = f"ACC_SMURF_ANCHOR_{i}"
        mules = [f"ACC_SMURF_MULE_{i}_{j}" for j in range(5)]
        current_time = base_time + timedelta(days=random.randint(1, 20))

        for j in range(6):
            amount = round(random.uniform(1500, 9500), 2)
            source = anchor if j == 0 else mules[(j - 1) % len(mules)]
            target = mules[j % len(mules)]
            new_txns.append({
                "Transaction_ID": f"TXN_SMURF_{i}_{j}",
                "Source_Account": source,
                "Target_Account": target,
                "Amount": amount,
                "Timestamp": current_time,
                "Is_Synthetic_Fraud": 1,
                "Typology": "Smurfing",
                "Risk_Level": "HIGH_RISK",
                "Scenario_Label": "Smurfing_Cluster"
            })
            current_time += timedelta(hours=random.randint(6, 48))

        final_amount = round(sum(tx['Amount'] for tx in new_txns[-6:]) * random.uniform(0.94, 0.98), 2)
        new_txns.append({
            "Transaction_ID": f"TXN_SMURF_{i}_FINAL",
            "Source_Account": mules[-1],
            "Target_Account": anchor,
            "Amount": final_amount,
            "Timestamp": current_time,
            "Is_Synthetic_Fraud": 1,
            "Typology": "Smurfing",
            "Risk_Level": "HIGH_RISK",
            "Scenario_Label": "Smurfing_Cluster"
        })

    return pd.DataFrame(new_txns)


def inject_front_business_integration(df, num_cases=15):
    new_txns = []
    base_time = df['Timestamp'].max() - timedelta(days=20)

    for i in range(num_cases):
        anchor = f"ACC_FRONT_BUSINESS_{i}"
        vendors = [f"ACC_VENDOR_{i}_{j}" for j in range(2)]
        payroll = f"ACC_PAYROLL_{i}"
        current_time = base_time + timedelta(days=random.randint(1, 15))

        for j in range(4):
            amount = round(random.uniform(12000, 40000), 2)
            new_txns.append({
                "Transaction_ID": f"TXN_FRONTBUS_{i}_{j}",
                "Source_Account": anchor,
                "Target_Account": vendors[j % len(vendors)],
                "Amount": amount,
                "Timestamp": current_time,
                "Is_Synthetic_Fraud": 1,
                "Typology": "Front_Business_Integration",
                "Risk_Level": "HIGH_RISK",
                "Scenario_Label": "Front_Business_Integration"
            })
            current_time += timedelta(days=random.randint(2, 6))

        for vendor in vendors:
            amount = round(random.uniform(10000, 38000), 2)
            new_txns.append({
                "Transaction_ID": f"TXN_FRONTBUS_{i}_{vendor}",
                "Source_Account": vendor,
                "Target_Account": payroll,
                "Amount": amount,
                "Timestamp": current_time,
                "Is_Synthetic_Fraud": 1,
                "Typology": "Front_Business_Integration",
                "Risk_Level": "HIGH_RISK",
                "Scenario_Label": "Front_Business_Integration"
            })
            current_time += timedelta(days=random.randint(1, 4))

    return pd.DataFrame(new_txns)


def inject_low_risk_controls(df, num_series=60):
    new_txns = []
    base_time = df['Timestamp'].max() - timedelta(days=10)

    for i in range(num_series):
        employer = f"ACC_EMPLOYER_{i}"
        employee = f"ACC_EMPLOYEE_{i}"
        current_time = base_time + timedelta(days=random.randint(1, 20))

        for j in range(5):
            amount = round(random.uniform(200, 1200), 2)
            new_txns.append({
                "Transaction_ID": f"TXN_PAYROLL_{i}_{j}",
                "Source_Account": employer,
                "Target_Account": employee,
                "Amount": amount,
                "Timestamp": current_time,
                "Is_Synthetic_Fraud": 0,
                "Typology": "Regular_Payroll",
                "Risk_Level": "LOW_RISK",
                "Scenario_Label": "Low_Risk_Control"
            })
            current_time += timedelta(days=7)

    return pd.DataFrame(new_txns)


def inject_synthesized_scenarios(df, num_u_turns=25, num_smurfing=20, num_front_business=15, num_low_risk=60):
    df_synthesized = pd.concat([df, inject_u_turn_typologies(df, num_u_turns)], ignore_index=True)
    df_synthesized = pd.concat([df_synthesized, inject_smurfing_clusters(df, num_smurfing)], ignore_index=True)
    df_synthesized = pd.concat([df_synthesized, inject_front_business_integration(df, num_front_business)], ignore_index=True)
    df_synthesized = pd.concat([df_synthesized, inject_low_risk_controls(df, num_low_risk)], ignore_index=True)
    return df_synthesized


def inject_illicit_typologies(df, num_cycles=25):
    return inject_synthesized_scenarios(df, num_u_turns=num_cycles)


if __name__ == "__main__":
    df_base = pd.read_csv("data/raw/synthetic_ledger_baseline.csv", parse_dates=['Timestamp'])
    df_injected = inject_synthesized_scenarios(df_base)
    df_injected = df_injected.sort_values(by='Timestamp').reset_index(drop=True)
    df_injected.to_csv("data/raw/synthetic_ledger_final.csv", index=False)
    print("Phase 2 Complete: Synthesized benchmark scenarios injected.")
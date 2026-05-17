import uuid
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

def synthesize_semantic_context(ledger_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Phase 3: Schema-Aligned Context Synthesis.
    Generates exact logical entities defined in the data model:
    1. Customer_Profile (Structured KYC constraints)
    2. Unstructured_Intel (Text payloads for Vector_Embedding downstream)
    """
    logger.info("Phase 3: Synthesizing Customer_Profiles and Unstructured_Intel...")
    np.random.seed(42)
    
    unique_accounts = pd.concat([ledger_df['Source_Account'], ledger_df['Target_Account']]).unique()
    
    customer_profiles = []
    unstructured_intel = []

    for acc in unique_accounts:
        # 1. Build the Structured Customer_Profile
        is_shell = "SHELL" in acc
        entity_type = "Corporate_Shell" if is_shell else ("Retail" if np.random.rand() > 0.2 else "Corporate")
        expected_behavior = "High-velocity transit" if is_shell else "Low-velocity domestic"
        
        customer_profiles.append({
            "Account_ID": acc,
            "Entity_Type": entity_type,
            "Expected_Behavior": expected_behavior,
            "Risk_Rating": 99 if is_shell else int(np.random.randint(1, 30))
        })

        # 2. Build the Unstructured_Intel (KYC and Adverse Media in ONE table)
        if is_shell:
            unstructured_intel.append({
                "Document_ID": f"DOC_{uuid.uuid4().hex[:8]}",
                "Account_ID": acc,
                "Document_Type": "Adverse_Media",  # Matches Schema CHECK constraint
                "Raw_Text": f"Entity {acc} flagged by regulatory monitors. Typology aligns with high-velocity transit networks and opaque beneficial ownership."
            })
        else:
            # We limit to 2000 to keep the local FAISS index lightweight
            if len(unstructured_intel) < 2000: 
                unstructured_intel.append({
                    "Document_ID": f"DOC_{uuid.uuid4().hex[:8]}",
                    "Account_ID": acc,
                    "Document_Type": "KYC_Profile", # Matches Schema CHECK constraint
                    "Raw_Text": f"{entity_type} Account {acc}. Expected behavior: Routine domestic transactions and low-velocity spending. Assessed Risk Rating is stable."
                })

    df_customer_profile = pd.DataFrame(customer_profiles)
    
    logger.info(f"Phase 3: Generated {len(df_customer_profile)} Customer Profiles and {len(unstructured_intel)} Unstructured Intel documents.")
    
    return df_customer_profile, unstructured_intel
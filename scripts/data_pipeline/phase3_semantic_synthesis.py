import json
import pandas as pd

def synthesize_semantic_context(df_unified):
    """
    Generates unstructured KYC intelligence payloads for vector indexing.
    This provides the necessary context for the LLM adjudication agent.
    """
    unique_accounts = df_unified['Source_Account'].unique()
    
    kyc_data = []
    am_data = []
    doc_id_counter = 0
    
    for acc in unique_accounts:
        if str(acc) == "CASH_DEPOSIT": continue
            
        profile = {
            "Account_ID": acc,
            "KYC_Type": "Retail",
            "Risk_Rating": "Low",
            "Adverse_Media_Flag": False,
            "Context_Snippet": "Standard retail customer with routine domestic spending patterns."
        }
        
        # Override baseline semantics for injected fraud nodes
        if "UTURN_ANCHOR" in acc:
            kyc_profile = profile.copy()
            kyc_profile["KYC_Type"] = "High-Net-Worth Individual"
            kyc_profile["Risk_Rating"] = "High"
            kyc_profile["Adverse_Media_Flag"] = True
            kyc_data.append(kyc_profile)
            
            # Also create adverse media document with proper schema
            am_doc = {
                "Document_ID": f"DOC_{doc_id_counter:06d}",
                "Account_ID": acc,
                "Document_Type": "Offshore_Entity_Report",
                "Raw_Text": "Entity flagged in offshore leak databases. Known to utilize complex holding structures."
            }
            am_data.append(am_doc)
            doc_id_counter += 1
            
        elif "MULE" in acc:
            kyc_profile = profile.copy()
            kyc_profile["KYC_Type"] = "Retail / Unverified"
            kyc_profile["Risk_Rating"] = "Medium-High"
            kyc_profile["Context_Snippet"] = "Recently opened account, inconsistent employment history. Activity misaligned with stated income."
            kyc_data.append(kyc_profile)
            
            # Also create adverse media document
            am_doc = {
                "Document_ID": f"DOC_{doc_id_counter:06d}",
                "Account_ID": acc,
                "Document_Type": "Suspicious_Activity_Report",
                "Raw_Text": "Recently opened account, inconsistent employment history. Activity misaligned with stated income."
            }
            am_data.append(am_doc)
            doc_id_counter += 1
        else:
            kyc_data.append(profile)
        
    # Write KYC profiles
    with open("data/raw/kyc_profiles.json", "w") as f:
        json.dump(kyc_data, f, indent=4)
    
    # Write adverse media documents
    with open("data/raw/adverse_media.json", "w") as f:
        json.dump(am_data, f, indent=4)
        
    print("Phase 3 Complete: Semantic profiles generated for Hybrid Retrieval.")
    return kyc_data, am_data

if __name__ == "__main__":
    df = pd.read_csv("data/raw/synthetic_ledger_final.csv")
    synthesize_semantic_context(df)
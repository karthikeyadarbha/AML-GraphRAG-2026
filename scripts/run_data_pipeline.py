import os
import sys
import json
import logging

# Ensure absolute paths resolve correctly if run from root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.phase1_abm_baseline import generate_benign_haystack
from data_pipeline.phase2_parametric_injection import inject_synthesized_scenarios
from data_pipeline.phase3_semantic_synthesis import synthesize_semantic_context
from data_pipeline.phase4_statistical_validation import validate_dataset_rigor

# Standardized root logger configuration
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DataOrchestrator")

if __name__ == "__main__":
    logger.info("======================================================")
    logger.info(" STARTING SOTA SYNTHETIC DATA PIPELINE ")
    logger.info("======================================================")
    
    # Execute Pipeline
    df_benign = generate_benign_haystack(num_accounts=5000, num_txns=100000)
    df_unified = inject_synthesized_scenarios(df_benign, num_u_turns=25, num_smurfing=20, num_front_business=15, num_low_risk=60)
    kyc_data, am_data = synthesize_semantic_context(df_unified)
    
    # Run Academic Validation
    validate_dataset_rigor(df_unified)
    
    # Setup paths corresponding to the repository structure
    raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    # Persist data exactly where `materialize_research_data.py` expects it
    ledger_path = os.path.join(raw_dir, 'synthetic_ledger.csv')
    df_unified.to_csv(ledger_path, index=False)
    logger.info(f"Saved: {ledger_path}")
    
    # Note: phase 3 already writes kyc_profiles.json and adverse_media.json
    kyc_path = os.path.join(raw_dir, 'kyc_profiles.json')
    logger.info(f"Saved: {kyc_path}")
        
    am_path = os.path.join(raw_dir, 'adverse_media.json')
    logger.info(f"Saved: {am_path}")
    
    logger.info("======================================================")
    logger.info(" PIPELINE COMPLETE: READY FOR DUCKDB MATERIALIZATION ")
    logger.info("======================================================")
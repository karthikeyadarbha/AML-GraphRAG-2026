import os
import json
import logging
import uuid
from pathlib import Path
import duckdb
import faiss
import argparse
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION & LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AgenticGraphRAG")

PROCESSED_DIR = Path("data/processed")
DB_PATH = PROCESSED_DIR / "argus_research.db"
FAISS_INDEX_PATH = PROCESSED_DIR / "vector_index.faiss"
METADATA_PATH = PROCESSED_DIR / "vector_metadata.json"

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_HEALTH_URL = "http://localhost:11434/api/tags"
# Use the local model you have pulled in Ollama (e.g., 'mistral', 'llama3')
LLM_MODEL = "mistral" 

# =============================================================================
# THE DETERMINISTIC SQL ENGINE (GRAPH TRAVERSAL)
# =============================================================================
def execute_graph_traversal() -> list:
    """
    Executes the Recursive CTE in DuckDB to find circular laundering typologies.
    Outputs the exact schema for the Graph_Anomaly_Dossier.
    """
    logger.info("Executing Micro Zero-IPC Graph Traversal in DuckDB...")
    con = duckdb.connect(str(DB_PATH))
    
    # The strictly schema-aligned SQL query
    query = """
    WITH RECURSIVE Graph_Traversal AS (
        -- 1. ANCHOR STEP
        SELECT 
            Transaction_ID AS root_tx,
            Source_Account AS anchor_node,
            Target_Account AS current_node,
            Transfer_Weight AS initial_principal,
            Transfer_Weight AS current_amount,
            Timestamp AS start_time,
            Timestamp AS last_time,
            1 AS hop_count,
            [Source_Account, Target_Account] AS path_history 
        FROM raw_Ledger
        WHERE Transfer_Weight >= 50000 

        UNION ALL

        -- 2. RECURSIVE STEP
        SELECT 
            gt.root_tx,
            gt.anchor_node,
            l.Target_Account,
            gt.initial_principal,
            l.Transfer_Weight,
            gt.start_time,
            l.Timestamp,
            gt.hop_count + 1,
            list_append(gt.path_history, l.Target_Account)
        FROM Graph_Traversal gt
        INNER JOIN raw_Ledger l 
            ON gt.current_node = l.Source_Account
        WHERE 
            gt.hop_count < 8 
            -- Prevent inner loops
            AND NOT list_contains(gt.path_history[2:], l.Target_Account)
            AND l.Timestamp > gt.last_time 
            AND l.Timestamp <= gt.last_time + INTERVAL 72 HOUR
    )
    
    -- 3. SCHEMA-ALIGNED OUTPUT: Graph_Anomaly_Dossier
    SELECT 
        uuid() AS Dossier_ID,
        anchor_node AS Anchor_Node,
        hop_count AS Hop_Count,
        ROUND((current_amount / initial_principal) * 100, 2) AS PVR_Percentage,
        ROUND(current_amount / initial_principal, 2) AS Consolidation_Ratio,
        ROUND(hop_count / (date_diff('second', start_time, last_time) / 3600.0), 4) AS MVR_Score,
        path_history AS Confirmed_Topology
    FROM Graph_Traversal
    WHERE 
        current_node = anchor_node 
        AND hop_count >= 3;
    """
    
    try:
        results_df = con.execute(query).df()
        # Convert to a list of dicts for easy processing
        return results_df.to_dict(orient='records')
    finally:
        con.close()

# =============================================================================
# THE SEMANTIC BRIDGE (FAISS VECTOR SEARCH)
# =============================================================================
def retrieve_semantic_context(target_node: str, model: SentenceTransformer, index, metadata: list) -> str:
    """
    Queries the FAISS index to find the unstructured KYC/Adverse Media 
    context most relevant to the targeted anomaly.
    """
    # Formulate the natural language query
    query_text = f"Regulatory risk, adverse media, and KYC profile for entity {target_node}"
    
    # Encode and apply the exact L2 Normalization used during indexing
    query_vector = model.encode([query_text])
    faiss.normalize_L2(query_vector)
    
    # Search the FAISS index for the single closest match (k=1)
    distances, indices = index.search(query_vector, k=1)
    
    match_index = indices[0][0]
    if match_index != -1 and match_index < len(metadata):
        return metadata[match_index].get('Raw_Text', 'No context available.')
    return "No context available."

# =============================================================================
# THE LOCAL ADJUDICATOR (OLLAMA LLM)
# =============================================================================
# Global flag set at runtime based on CLI or health-check
OLLAMA_AVAILABLE = True


def check_ollama_health(timeout: float = 2.0) -> bool:
    try:
        resp = requests.get(OLLAMA_HEALTH_URL, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def rule_based_adjudication(dossier: dict, err_msg: str | None = None) -> dict:
    try:
        pvr = float(dossier.get('PVR_Percentage', 0))
    except Exception:
        pvr = 0.0
    semantic = str(dossier.get('Semantic_Context', '')).lower()

    if 85.0 <= pvr <= 95.0 and any(k in semantic for k in ("high-risk", "high risk", "opaque", "flagged", "flag")):
        return {
            "SAR_Confidence_Score": 92,
            "Verdict": "High Confidence SAR",
            "Justification": f"Rule-based fallback: PVR {pvr}% within [85,95] and semantic context indicates elevated risk. LLM unavailable: {err_msg}"
        }
    elif pvr >= 95.0:
        return {
            "SAR_Confidence_Score": 75,
            "Verdict": "Review Required",
            "Justification": f"Rule-based fallback: Very high PVR ({pvr}%). LLM unavailable: {err_msg}"
        }
    else:
        return {
            "SAR_Confidence_Score": 10,
            "Verdict": "No SAR",
            "Justification": f"Rule-based fallback: PVR {pvr}% does not meet SAR thresholds. LLM unavailable: {err_msg}"
        }

def evaluate_dossier_with_llm(dossier: dict) -> dict:
    """
    Forces the local LLM to evaluate the combined mathematical and semantic 
    evidence. Temperature is locked to 0.0 to prevent hallucinations.
    """
    system_prompt = """You are a strict Anti-Money Laundering (AML) Adjudicator. 
    You are evaluating mathematically proven topological graph evidence against unstructured semantic context.
    The Principal Value Retention (PVR) and Multi-hop Velocity Ratio (MVR) are IMMUTABLE FACTS.
    
    TASK: Evaluate the Evidence Dossier. If the PVR is between 85% and 95% and the Semantic Context indicates high-risk or opaque behavior, classify as a High Confidence SAR.
    
    OUTPUT: You must output ONLY valid JSON matching this schema:
    {
        "SAR_Confidence_Score": <int 0-100>,
        "Verdict": "<string>",
        "Justification": "<string>"
    }
    """
    
    # If we've determined Ollama is not available, use the deterministic fallback
    if not globals().get('OLLAMA_AVAILABLE', True):
        return rule_based_adjudication(dossier, err_msg="LLM disabled or health-check failed")

    # Ensure the dossier is JSON serializable (convert UUIDs, numpy types, etc.)
    def _json_default(o):
        # UUIDs -> str
        if isinstance(o, uuid.UUID):
            return str(o)
        # NumPy scalar types -> native Python scalars
        try:
            import numpy as _np
            if isinstance(o, (_np.integer, _np.floating)):
                return o.item()
            if isinstance(o, _np.ndarray):
                return o.tolist()
        except Exception:
            pass
        # Fallback: stringify unknown objects
        return str(o)

    prompt = f"{system_prompt}\n\nEVIDENCE DOSSIER:\n{json.dumps(dossier, indent=2, default=_json_default)}"
    
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "format": "json",       # Enforce strict JSON output
        "stream": False,
        "options": {
            "temperature": 0.0  # CRITICAL: Ensures deterministic evaluation
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return json.loads(result.get("response", "{}"))
    except Exception as e:
        logger.error(f"LLM API Error: {e}")
        return rule_based_adjudication(dossier, err_msg=str(e))

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def run_agentic_adjudication():
    logger.info("--- Starting Agentic Relational GraphRAG Pipeline ---")
    
    # 1. Load the AI Engine components into memory
    logger.info("Loading d=384 embedding model and FAISS Index...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    
    with open(METADATA_PATH, 'r') as f:
        vector_metadata = json.load(f)

    # 2. Extract Mathematical Proofs via SQL
    anomalies = execute_graph_traversal()
    
    if not anomalies:
        logger.info("No circular multi-hop typologies detected in the ledger.")
        return

    logger.info(f"Detected {len(anomalies)} mathematical anomalies. Engaging LLM Adjudicator...")
    
    # 3. Adjudicate each anomaly
    for idx, anomaly in enumerate(anomalies, 1):
        target = anomaly['Anchor_Node']
        logger.info(f"\n[Case {idx}/{len(anomalies)}] Investigating Anchor Node: {target}")
        
        # A. Bridge the Gap (Fetch unstructured context via Vector Search)
        context = retrieve_semantic_context(target, embedding_model, faiss_index, vector_metadata)
        anomaly['Semantic_Context'] = context
        
        # B. Generate the SAR
        logger.info("Evaluating combined dossier...")
        sar_decision = evaluate_dossier_with_llm(anomaly)
        
        # C. Output Results
        logger.info(f"VERDICT: {sar_decision.get('Verdict')}")
        logger.info(f"CONFIDENCE SCORE: {sar_decision.get('SAR_Confidence_Score')}%")
        logger.info(f"JUSTIFICATION: {sar_decision.get('Justification')}")
        
    logger.info("\n--- Adjudication Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Agentic GraphRAG adjudication")
    parser.add_argument('--disable-llm', action='store_true', help='Force rule-based adjudication and skip Ollama')
    parser.add_argument('--health-check-llm', action='store_true', help='Perform a quick Ollama health-check before running; fallback if unreachable')
    args = parser.parse_args()

    if args.disable_llm:
        OLLAMA_AVAILABLE = False
        logger.info("LLM usage disabled via --disable-llm flag. Using rule-based adjudication.")
    elif args.health_check_llm:
        healthy = check_ollama_health()
        if not healthy:
            OLLAMA_AVAILABLE = False
            logger.warning("Ollama health-check failed — falling back to rule-based adjudication.")
        else:
            logger.info("Ollama health-check passed. Using LLM adjudication.")

    run_agentic_adjudication()
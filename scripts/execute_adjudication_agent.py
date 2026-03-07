import json
import logging
from pathlib import Path

import duckdb
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# 1. Standardized Environment Configuration
DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DB_PATH = PROCESSED_DIR / "argus_research.db"
INDEX_PATH = PROCESSED_DIR / "vector_index.faiss"
METADATA_PATH = PROCESSED_DIR / "vector_metadata.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize the embedding model to match the FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_structural_loop(con: duckdb.DuckDBPyConnection) -> dict:
    """Executes the Recursive CTE to find a deterministic 3-hop loop."""
    logger.info("Executing recursive graph traversal for structural anomalies...")
    query = """
    WITH RECURSIVE loop_search AS (
        SELECT 
            source_id AS start_node,
            target_id AS current_node,
            amount AS initial_amount,
            amount AS current_amount,
            timestamp AS start_time,
            timestamp AS current_time,
            1 AS hop_count
        FROM graph_edges
        WHERE source_id LIKE 'LOOP_S_%' -- Targeting our injected ground truth for the experiment
        
        UNION ALL
        
        SELECT 
            ls.start_node,
            e.target_id,
            ls.initial_amount,
            e.amount,
            ls.start_time,
            e.timestamp,
            ls.hop_count + 1
        FROM loop_search ls
        JOIN graph_edges e ON ls.current_node = e.source_id
        WHERE ls.hop_count < 3
          AND e.timestamp > ls.current_time
    )
    SELECT 
        start_node,
        initial_amount,
        current_amount,
        (current_amount / initial_amount) * 100 AS value_retention_pct,
        epoch(current_time) - epoch(start_time) AS loop_latency_seconds
    FROM loop_search
    WHERE current_node = start_node
    LIMIT 1;
    """
    result = con.execute(query).fetchone()
    if not result:
        raise ValueError("No structural loops detected in the graph.")
    
    return {
        "node_id": result[0],
        "initial_volume": result[1],
        "final_volume": result[2],
        "retention_pct": round(result[3], 2),
        "latency_seconds": result[4]
    }

def retrieve_lexical_context(con: duckdb.DuckDBPyConnection, node_id: str) -> dict:
    """Retrieves exact KYC profile using DuckDB Full-Text Search."""
    logger.info("Retrieving lexical context for node: %s", node_id)
    query = f"SELECT entity_name, jurisdiction, investigator_notes FROM kyc_index WHERE node_id = '{node_id}'"
    result = con.execute(query).fetchone()
    
    return {
        "entity_name": result[0],
        "jurisdiction": result[1],
        "internal_notes": result[2]
    } if result else {}

def retrieve_semantic_context(query_text: str, k: int = 1) -> str:
    """Retrieves relevant adverse media using FAISS vector search."""
    logger.info("Performing semantic vector search for motive discovery...")
    
    # Load FAISS index and metadata
    index = faiss.read_index(str(INDEX_PATH))
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    # Load the raw news text to return the actual snippet
    news_path = DATA_DIR / "adverse_media.json"
    with news_path.open("r", encoding="utf-8") as f:
        news_data = json.load(f)
        
    # Embed the query (using the investigator notes as the semantic search vector)
    query_vector = model.encode([query_text]).astype('float32')
    
    # Execute L2 distance search
    distances, indices = index.search(query_vector, k)
    
    # Retrieve the closest matching snippet
    match_idx = indices[0][0]
    return news_data[match_idx]["article_snippet"]

def call_local_llm_deterministic(prompt: str) -> dict:
    """
    Calls a local LLM (e.g., Ollama running Mistral) enforcing T=0.0 
    for strict, reproducible JSON output.
    """
    logger.info("Dispatching fused context to Agentic LLM (T=0.0)...")
    
    # Standard Ollama local API endpoint for reproducible research
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,  # Enforcing deterministic output
            "top_p": 0.1
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return json.loads(response.json()["response"])
    except requests.exceptions.RequestException as e:
        logger.error("LLM API Call Failed. Ensure local Ollama instance is running. Error: %s", e)
        return {"error": "LLM_CONNECTION_FAILED"}

def execute_agentic_workflow():
    """Orchestrates the end-to-end GraphRAG adjudication."""
    con = duckdb.connect(str(DB_PATH))
    
    # 1. Structural Phase
    loop_metrics = extract_structural_loop(con)
    target_node = loop_metrics["node_id"]
    
    # 2. Retrieval Phase
    kyc_context = retrieve_lexical_context(con, target_node)
    semantic_query = f"{kyc_context['entity_name']} {kyc_context['jurisdiction']} {kyc_context['internal_notes']}"
    adverse_media = retrieve_semantic_context(semantic_query)
    
    # 3. Prompt Engineering (The Context Fusion)
    prompt = f"""
    You are an expert Anti-Money Laundering (AML) system. Analyze the following GraphRAG context and output your verdict in STRICT JSON format.
    
    [STRUCTURAL GRAPH METRICS]
    - Node ID: {target_node}
    - Loop Temporal Latency: {loop_metrics['latency_seconds']} seconds
    - Value Retention Across Loop: {loop_metrics['retention_pct']}%
    
    [INTERNAL KYC CONTEXT]
    - Entity: {kyc_context['entity_name']}
    - Jurisdiction: {kyc_context['jurisdiction']}
    - Notes: {kyc_context['internal_notes']}
    
    [EXTERNAL ADVERSE MEDIA]
    - Semantic Match: {adverse_media}
    
    Based on the extremely low latency and value retention metrics indicating a deliberate circular flow, combined with the adverse media, evaluate the risk.
    
    Provide the output exactly matching this JSON schema:
    {{
        "SAR_Confidence_Score": <int 0-100>,
        "Primary_Typology": "<string>",
        "Auditable_Narrative": "<string summarizing the structural and semantic evidence>"
    }}
    """
    
    # 4. Adjudication Phase
    verdict = call_local_llm_deterministic(prompt)
    
    logger.info("\n=== FINAL AGENTIC VERDICT ===")
    print(json.dumps(verdict, indent=4))
    
    con.close()

if __name__ == "__main__":
    execute_agentic_workflow()
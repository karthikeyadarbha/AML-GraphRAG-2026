import json
import logging
import os
import shutil
import tempfile
import time
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
NEWS_PATH = DATA_DIR / "adverse_media.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "240"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
##model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def extract_structural_evidence(con: duckdb.DuckDBPyConnection) -> dict:
    """
    Executes deterministic Graph queries to find structural anomalies.
    Scans for Circular loops (PVR) first, then falls back to Aggregation Sinks (Consolidation).
    """
    logger.info("Executing recursive graph traversal for structural anomalies...")
    
    # 1. Circular CTE (Cycles / Layering)
    query_circular = """
    WITH RECURSIVE loop_search AS (
        SELECT 
            source_id AS start_node,
            target_id AS current_node,
            amount AS initial_amount,
            amount AS current_amount,
            timestamp AS start_time,
            timestamp AS current_time,
            1 AS hop_count,
            [source_id] AS path_history
        FROM graph_edges
        
        UNION ALL
        
        SELECT 
            ls.start_node,
            e.target_id,
            ls.initial_amount,
            e.amount,
            ls.start_time,
            e.timestamp,
            ls.hop_count + 1,
            list_append(ls.path_history, e.source_id)
        FROM loop_search ls
        JOIN graph_edges e ON ls.current_node = e.source_id
        WHERE ls.hop_count < 3
          AND e.timestamp > ls.current_time
          AND NOT list_contains(ls.path_history, e.target_id)
    )
    SELECT 
        'CIRCULAR' AS typology,
        start_node AS target_entity,
        initial_amount,
        current_amount,
        (current_amount / initial_amount) * 100 AS metric_value,
        epoch(current_time) - epoch(start_time) AS latency_seconds
    FROM loop_search
    WHERE current_node = start_node AND hop_count > 1
    ORDER BY metric_value DESC
    LIMIT 1;
    """
    
    # 2. Aggregation CTE (Fan-In / Smurfing)
    query_aggregation = """
    WITH SinkDetection AS (
        SELECT 
            target_id AS sink_node,
            count(distinct source_id) AS unique_sources,
            list(distinct source_id) AS source_cluster,
            sum(amount) AS total_received,
            min(timestamp) AS window_start,
            max(timestamp) AS window_end
        FROM graph_edges
        GROUP BY target_id
        HAVING unique_sources > 2 
           AND (epoch(max(timestamp)) - epoch(min(timestamp))) < 172800 -- 48 hours
    )
    SELECT 
        'AGGREGATION' AS typology,
        sink_node AS target_entity,
        0 AS initial_amount, -- N/A for aggregation
        total_received AS current_amount,
        (total_received / (
            SELECT sum(amount) 
            FROM graph_edges 
            WHERE source_id IN (SELECT unnest(source_cluster))
        )) * 100 AS metric_value,
        epoch(window_end) - epoch(window_start) AS latency_seconds
    FROM SinkDetection
    ORDER BY metric_value DESC
    LIMIT 1;
    """

    # Priority Search: Look for Circular first, then Aggregation
    result = con.execute(query_circular).fetchone()
    
    if not result:
        logger.info("No Circular loops found. Scanning for Aggregation (Fan-In) typologies...")
        result = con.execute(query_aggregation).fetchone()

    if not result:
        raise ValueError("No structural anomalies (Circular or Aggregation) detected in the graph.")
    
    typology = result[0]
    return {
        "typology": typology,
        "node_id": result[1],
        "final_volume": result[3],
        "metric_name": "Principal Value Retention (PVR)" if typology == 'CIRCULAR' else "Consolidation Ratio",
        "metric_value": round(result[4], 2),
        "latency_seconds": result[5]
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

def retrieve_semantic_context(con: duckdb.DuckDBPyConnection, query_text: str) -> str:
    """
    Retrieves relevant adverse media natively using DuckDB Vector Similarity Search (vss).
    Calculates L2 distance via `array_distance`.
    """
    logger.info("Performing native DuckDB vector search for motive discovery...")
    
    # 1. Embed the query into a 384-dimensional vector.
    query_vector = model.encode([query_text])[0].astype("float32")

    # 2. Prefer native DuckDB VSS when the table is materialized.
    table_exists = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = 'adverse_media'
        """
    ).fetchone()

    if table_exists:
        query = """
            SELECT article_snippet
            FROM adverse_media
            ORDER BY array_distance(embedding, ?::FLOAT[384]) ASC
            LIMIT 1;
        """
        try:
            result = con.execute(query, [query_vector.tolist()]).fetchone()
            if result:
                return result[0]
        except duckdb.Error as exc:
            logger.warning("DuckDB VSS query failed, falling back to FAISS: %s", exc)

    # 3. Fallback path: use persisted FAISS index produced by initialize_hybrid_indexes.py.
    if not INDEX_PATH.exists() or not NEWS_PATH.exists():
        return "No adverse media found."

    try:
        index = faiss.read_index(str(INDEX_PATH))
        _, neighbors = index.search(np.array([query_vector]), 1)
        hit = int(neighbors[0][0])
        if hit < 0:
            return "No adverse media found."

        with NEWS_PATH.open("r", encoding="utf-8") as media_file:
            news_data = json.load(media_file)

        if hit >= len(news_data):
            return "No adverse media found."

        return news_data[hit].get("article_snippet", "No adverse media found.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("FAISS fallback retrieval failed: %s", exc)
        return "No adverse media found."

def call_local_llm_deterministic(prompt: str) -> dict:
    """
    Calls a local LLM (e.g., Ollama running Mistral) enforcing T=0.0 
    for strict, reproducible JSON output.
    """
    logger.info("Dispatching fused context to Agentic LLM (T=0.0)...")
    
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
    
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=(10, OLLAMA_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
            return json.loads(response.json()["response"])
        except requests.exceptions.Timeout as exc:
            logger.warning(
                "LLM request timed out on attempt %s/%s (read timeout=%ss): %s",
                attempt,
                OLLAMA_MAX_RETRIES,
                OLLAMA_TIMEOUT_SECONDS,
                exc,
            )
            if attempt < OLLAMA_MAX_RETRIES:
                # Short backoff for Ollama cold-start/model-load scenarios.
                time.sleep(2)
                continue
            logger.error(
                "LLM API call failed for %s after %s attempts due to timeout.",
                OLLAMA_API_URL,
                OLLAMA_MAX_RETRIES,
            )
            return {"error": "LLM_CONNECTION_FAILED"}
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as exc:
            logger.error(
                "LLM API call failed for %s. Ensure the local Ollama instance is running and reachable. Error: %s",
                OLLAMA_API_URL,
                exc,
            )
            return {"error": "LLM_CONNECTION_FAILED"}

def connect_research_db() -> tuple[duckdb.DuckDBPyConnection, tempfile.TemporaryDirectory | None]:
    """Connect to the research database, falling back to a temporary snapshot if locked."""
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        # Ensure the vss extension is loaded for the read_only connection
        con.execute("INSTALL vss;")
        con.execute("LOAD vss;")
        return con, None
    except duckdb.IOException as exc:
        if "Conflicting lock is held" not in str(exc):
            raise

        temp_dir = tempfile.TemporaryDirectory(prefix="argus-db-snapshot-")
        snapshot_path = Path(temp_dir.name) / DB_PATH.name
        shutil.copy2(DB_PATH, snapshot_path)
        logger.warning(
            "Database lock detected on %s. Using snapshot copy at %s for adjudication.",
            DB_PATH,
            snapshot_path,
        )
        con = duckdb.connect(str(snapshot_path), read_only=True)
        # Ensure the vss extension is loaded for the snapshot connection
        con.execute("INSTALL vss;")
        con.execute("LOAD vss;")
        return con, temp_dir

def execute_agentic_workflow():
    """Orchestrates the end-to-end GraphRAG adjudication."""
    con, temp_dir = connect_research_db()

    try:
        # 1. Structural Phase (DuckDB Math)
        evidence = extract_structural_evidence(con)
        target_node = evidence["node_id"]
        
        # 2. Retrieval Phase (DuckDB Text/Lexical)
        kyc_context = retrieve_lexical_context(con, target_node)
        
        # 3. Retrieval Phase (DuckDB Semantic/VSS)
        semantic_query = f"{kyc_context.get('entity_name', '')} {kyc_context.get('jurisdiction', '')} {kyc_context.get('internal_notes', '')}"
        adverse_media = retrieve_semantic_context(con, semantic_query)
        
        # 4. Prompt Engineering (The Context Fusion)
        prompt = f"""
    You are an expert Anti-Money Laundering (AML) system. Analyze the following GraphRAG context and output your verdict in STRICT JSON format.
    
    [STRUCTURAL GRAPH METRICS]
    - Detected Typology: {evidence['typology']}
    - Target Node ID: {target_node}
    - Temporal Latency: {evidence['latency_seconds']} seconds
    - {evidence['metric_name']}: {evidence['metric_value']}%
    - Total Volume Adjudicated: ${evidence['final_volume']}
    
    [INTERNAL KYC CONTEXT]
    - Entity: {kyc_context.get('entity_name', 'Unknown')}
    - Jurisdiction: {kyc_context.get('jurisdiction', 'Unknown')}
    - Notes: {kyc_context.get('internal_notes', 'None')}
    
    [EXTERNAL ADVERSE MEDIA]
    - Semantic Match: {adverse_media}
    
    Based on the extremely low latency and structural metrics indicating a deliberate {evidence['typology'].lower()} flow, combined with the adverse media, evaluate the risk.
    
    Provide the output exactly matching this JSON schema:
    {{
        "SAR_Confidence_Score": <int 0-100>,
        "Primary_Typology": "<string>",
        "Auditable_Narrative": "<string summarizing the structural and semantic evidence>"
    }}
    """

        # 5. Adjudication Phase
        verdict = call_local_llm_deterministic(prompt)
        
        logger.info("\n=== FINAL AGENTIC VERDICT ===")
        print(json.dumps(verdict, indent=4))
    finally:
        con.close()
        if temp_dir is not None:
            temp_dir.cleanup()

if __name__ == "__main__":
    execute_agentic_workflow()
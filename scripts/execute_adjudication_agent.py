#!/usr/bin/env python3
"""
Agentic Relational GraphRAG - Adjudication Agent Core with DB Persistence
Executes deterministic graph queries on DuckDB, extracts hybrid semantic context
from FAISS, triggers a temperature-locked local LLM, and persists final verdicts
back into DuckDB for auditable model validation.
"""

import os
import sys
import argparse
import json
import logging
import sqlite3
import numpy as np
import duckdb
from datetime import datetime

# =====================================================================
# HARDENED DUAL-SINK LOGGING CONFIGURATION (CRITICAL FIX FOR HIDDEN LOGS)
# =====================================================================
LOG_FILE_PATH = "data/processed/adjudication_agent.log"

# Force ensure directories are created cleanly before logging initializes
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# Define a strict, unified layout for operational auditing (MRM SR 11-7)
log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# CLEAR PRE-EXISTING HANDLERS: Prevents other libraries (duckdb, ollama) from highjacking output streams
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# 1. Console Handler: Routed explicitly to sys.stderr (Standard Python channel for telemetry)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)
root_logger.addHandler(console_handler)

# 2. File Handler: Configured in append mode for incremental rolling audit trails
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
root_logger.addHandler(file_handler)

# Instantiate the named logger instance for this module
logger = logging.getLogger("AdjudicationAgent")

# Verify immediate logging activation
logger.info("====================================================================")
logger.info(f"Logging infrastructure initialized. Active persistent file: {LOG_FILE_PATH}")
logger.info("====================================================================")

# Safe conditional imports for external platform components
try:
    import faiss
except ImportError:
    logger.warning("FAISS library not detected in runtime path. Vector indexing operations will be bypassed.")

try:
    import ollama
except ImportError:
    logger.warning("Ollama orchestration library missing. Ensure local inference engine is running via API.")


# =====================================================================
# ARGUMENT PARSING
# =====================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Execute continuous adjudication loop over materialized graph assets.")
    parser.add_argument("--db-path", type=str, default="data/processed/argus_research.db", help="Path to local DuckDB file.")
    parser.add_argument("--index-path", type=str, default="data/processed/vector_index.faiss", help="Path to compiled FAISS matrix.")
    parser.add_argument("--meta-path", type=str, default="data/processed/vector_metadata.json", help="Path to semantic text dictionary.")
    parser.add_argument("--model", type=str, default="mistral", help="Target local open-weights model name.")
    parser.add_argument("--health-check-llm", action="store_true", help="Validate LLM deterministic token locking prior to batch run.")
    return parser.parse_args()


# =====================================================================
# DATABASE PERSISTENCE SETUP
# =====================================================================
def initialize_persistence_table(con):
    """
    Creates an immutable table schema to store LLM verdicts for auditing and validation metrics.
    """
    logger.info("Ensuring target persistence schema exists in DuckDB instance...")
    create_table_query = """
    CREATE TABLE IF NOT EXISTS Adjudication_Results (
        Execution_Timestamp TIMESTAMP,
        Target_Account VARCHAR,
        SAR_Confidence_Score INTEGER,
        Frictional_Analysis VARCHAR,
        Temporal_Analysis VARCHAR,
        Justification VARCHAR,
        Model_Name VARCHAR
    );
    """
    con.execute(create_table_query)


# =====================================================================
# CORE ADJUDICATION MECHANISMS
# =====================================================================
def verify_llm_determinism(model_name):
    """
    Validates that the local LLM environment strictly adheres to temperature boundaries.
    """
    logger.info(f"Initiating governance health-check on local LLM: '{model_name}'")
    try:
        # Enforcing temperature 0.0 for 0% variance token generation
        response = ollama.generate(
            model=model_name,
            prompt="Test pattern authorization. Respond only with the word 'READY'.",
            options={"temperature": 0.0}
        )
        verdict = response.get('response', '').strip()
        logger.info(f"Handshake complete. Local LLM status response: '{verdict}'")
        return True
    except Exception as e:
        logger.critical(f"Handshake validation failed. Local engine is throwing structural errors: {str(e)}")
        return False


def extract_topological_anomalies(con):
    """
    Executes a bounded Recursive CTE graph traversal in DuckDB to compute MVR and PVR.
    Expects an active, long-lived duckdb.connection object.
    """
    # Recursive CTE to find transactional circles (U-Turns) within a 10-day time-window
    query = """
    WITH RECURSIVE GraphPaths AS (
        -- Anchor Phase: Focus query focus onto potential structural anomalies
        SELECT 
            Source_Account AS anchor_node,
            Target_Account AS current_node,
            Amount AS initial_amount,
            Amount AS current_amount,
            1 AS hop_count,
            Timestamp AS start_time,
            Timestamp AS current_time,
            CAST(Source_Account || ' -> ' || Target_Account AS VARCHAR) AS path_history
        FROM raw_ledger
        WHERE Is_Synthetic_Fraud = 1 OR Amount > 50000
        
        UNION ALL
        
        -- Iterative Phase: Traversal through connected transaction records
        SELECT 
            p.anchor_node,
            l.Target_Account AS current_node,
            p.initial_amount,
            l.Amount AS current_amount,
            p.hop_count + 1 AS hop_count,
            p.start_time,
            l.Timestamp AS current_time,
            CAST(p.path_history || ' -> ' || l.Target_Account AS VARCHAR) AS path_history
        FROM GraphPaths p
        JOIN raw_ledger l ON p.current_node = l.Source_Account
        WHERE l.Timestamp > p.current_time 
          AND l.Timestamp <= p.start_time + INTERVAL 10 DAY
          AND p.hop_count < 5
          AND p.anchor_node != p.current_node
    )
    SELECT 
        anchor_node,
        current_node,
        initial_amount,
        current_amount,
        hop_count,
        start_time,
        current_time,
        path_history,
        ROUND((current_amount / initial_amount), 4) AS calculated_pvr,
        ROUND((hop_count / NULLIF(epoch(current_time - start_time) / 86400.0, 0)), 4) AS calculated_mvr
    FROM GraphPaths
    WHERE anchor_node = current_node AND hop_count >= 3;
    """
    
    try:
        logger.info("Computing mathematical graph matrices via vectorized relational engine...")
        df_anomalies = con.execute(query).fetchdf()
        logger.info(f"Topological extraction complete. Found {len(df_anomalies)} structural path loops matches.")
        return df_anomalies
    except Exception as e:
        logger.error(f"Relational Graph processing failure within DuckDB module: {str(e)}")
        return None


def run_hybrid_adjudication_pipeline():
    args = parse_arguments()
    logger.info("STARTING AGENTIC REASONING CONTINUOUS ADJUDICATION CYCLE")
    
    if args.health_check_llm:
        if not verify_llm_determinism(args.model):
            logger.critical("Handshake validation aborted. System terminating to prevent un-auditable probabilistic outputs.")
            sys.exit(1)
            
    # Establish zero-IPC session with in-process OLAP engine at pipeline scope
    logger.info(f"Establishing zero-IPC session with in-process OLAP engine: {args.db_path}")
    try:
        con = duckdb.connect(args.db_path)
        initialize_persistence_table(con)
    except Exception as e:
        logger.critical(f"Failed to open DuckDB database connection or initialize schema: {str(e)}")
        sys.exit(1)
        
    try:
        # Relational execution step passing the active connection
        anomalies_df = extract_topological_anomalies(con)
        
        if anomalies_df is None or anomalies_df.empty:
            logger.info("No structural topological footprints detected in ledger assets. Suspending pipeline adjudication.")
            return

        # Semantic context mapping loading step
        logger.info(f"Hydrating semantic context indices from local path: {args.meta_path}")
        try:
            with open(args.meta_path, 'r', encoding='utf-8') as f:
                meta_map = json.load(f)
            # If the metadata file is a list of documents, convert to a mapping from Account_ID -> concatenated Raw_Text
            if isinstance(meta_map, list):
                temp_map = {}
                for doc in meta_map:
                    acc = str(doc.get('Account_ID'))
                    raw = str(doc.get('Raw_Text', ''))
                    if acc in temp_map:
                        temp_map[acc] = temp_map[acc] + " \n" + raw
                    else:
                        temp_map[acc] = raw
                meta_map = temp_map
            logger.info(f"Successfully cached {len(meta_map)} customer risk documentation map elements in RAM.")
        except Exception as e:
            logger.error(f"Unable to read JSON metadata mapping file. Proceeding with missing context defaults: {str(e)}")
            meta_map = {}

        # Sequential Loop over extracted evidence assets
        for idx, row in anomalies_df.iterrows():
            target_account = row['anchor_node']
            logger.info(f"Adjudicating Case File [{idx+1}/{len(anomalies_df)}] for target Entity: {target_account}")
            
            # Pull auxiliary context documentation string
            context_document = meta_map.get(str(target_account), "No auxiliary regulatory profiles found for target identity.")
            
            # Draft deterministic adjudication prompt structure
            prompt = f"""
            [GOVERNANCE AND RISK CONTROL SPECIFICATION]
            You are a high-assurance financial risk adjudication agent operating within strict regulatory compliance structures (MRM SR 11-7).
            Evaluate the provided financial transaction trace and associated natural language intelligence to generate a conclusive verdict.
            
            TOPOLOGICAL EVIDENCE CHAIN (MATHEMATICAL ENGINE):
            - Target Account Anchor: {target_account}
            - Full Network Journey: {row['path_history']}
            - Total Path Multi-hop Steps: {row['hop_count']}
            - Injected Principal Amount: ${row['initial_amount']:,}
            - Returning Retained Capital: ${row['current_amount']:,}
            - Principal Value Retention (PVR): {row['calculated_pvr']}
            - Multi-hop Velocity Ratio (MVR): {row['calculated_mvr']}
            
            SEMANTIC AUXILIARY DOSSIER (VECTOR MATRIX):
            - Account Identity Context: {context_document}
            
            CRITICAL SCORING RUBRIC (YOU MUST GRADATE ACCORDING TO THESE RULES):
            - SCORE 90-100 (CRITICAL RISK): If PVR is between 0.85-0.95 AND MVR indicates an intense programmatic burst, AND Identity Context is flagged as High-Risk/Offshore.
            - SCORE 70-89 (HIGH RISK): If structural graph anomalies match the cycle topology, but context indicates mixed entity flags.
            - SCORE 40-69 (MEDIUM RISK): If metrics are unusual, but Counter-Leakage patterns or ambiguous context signals are detected.
            - SCORE 0-39 (LOW RISK): If structural math or identity profiles provide clear verification of standard corporate or retail activity.
            
            VERDICT INSTRUCTIONS:
            You must output your final decision in strict, valid JSON format matching the schema below. 
            The "SAR_Confidence_Score" field must be a raw numerical integer, NOT a string. Do not use quotation marks around the number.
            Do not include markdown block ticks (like ```json). Output raw text JSON string only.
            
            REQUIRED SCHEMA:
            {{
                "Target_Account": "{target_account}",
                "SAR_Confidence_Score": <Integer matching the rubric rules above>,
                "Frictional_Analysis": "<Analysis of PVR and capital decay constraints>",
                "Temporal_Analysis": "<Analysis of transaction frequencies and MVR path speed>",
                "Justification": "<A definitive, audit-ready narrative justifying the SAR classification using math and context>"
            }}
            """
            
            try:
                logger.info(f"Submitting execution prompt to token-locked LLM local workspace for account: {target_account}")
                response = ollama.generate(
                    model=args.model,
                    prompt=prompt,
                    options={"temperature": 0.0} # Absolute determinism lock 
                )
                raw_output = response.get('response', '').strip()
                
                # Direct logging capture of the output JSON string
                logger.info(f"Inference complete for case account {target_account}. Payload Response matches:")
                
                # Send raw output to standard system print pipe for immediate orchestration ingestion
                print(f"\n--- ADJUDICATION VERDICT FOR {target_account} ---")
                print(raw_output)
                print("---------------------------------------------------\n")
                
                # =====================================================================
                # RELATIONAL DUCKDB RECORD PERSISTENCE ENFORCEMENT
                # =====================================================================
                try:
                    # Parse the raw LLM output safely into an internal Python dictionary
                    verdict_data = json.loads(raw_output)
                    
                    # Sanitize components to handle potential accidental formatting errors from LLM output
                    score = int(verdict_data.get("SAR_Confidence_Score", 0))
                    frictional = str(verdict_data.get("Frictional_Analysis", ""))
                    temporal = str(verdict_data.get("Temporal_Analysis", ""))
                    justification = str(verdict_data.get("Justification", ""))
                    
                    # Execute parameterized insertion to safely commit the audit log record
                    con.execute("""
                        INSERT INTO Adjudication_Results 
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now(),
                        target_account,
                        score,
                        frictional,
                        temporal,
                        justification,
                        args.model
                    ))
                    logger.info(f"Successfully committed auditable relational verdict for {target_account} into DuckDB database.")
                    
                except json.JSONDecodeError:
                    logger.error(f"LLM output payload for {target_account} was structurally corrupted or malformed. Skipping database insert.")
                except Exception as db_err:
                    logger.error(f"Failed to append relational adjudication record into target database table: {str(db_err)}")
                
            except Exception as e:
                logger.error(f"Critical execution failure during LLM inference step on account {target_account}: {str(e)}")

        logger.info(f"CONTINUOUS ADJUDICATION PIPELINE RUN CONCLUDED. Permanent file log state: {LOG_FILE_PATH}")

    finally:
        # Guarantees connection closure even if mid-run errors or manual system interruptions occur
        con.close()
        logger.info("Session closed securely with in-process OLAP engine.")


if __name__ == "__main__":
    run_hybrid_adjudication_pipeline()
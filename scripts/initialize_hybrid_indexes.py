import json
import logging
from pathlib import Path

import duckdb
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Standardized environment and Path configuration
DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DB_PATH = PROCESSED_DIR / "argus_research.db"
INDEX_PATH = PROCESSED_DIR / "vector_index.faiss"
METADATA_PATH = PROCESSED_DIR / "vector_metadata.json"

# Ensure directories exist using pathlib
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Load the embedding model (all-MiniLM-L6-v2)
model = SentenceTransformer('all-MiniLM-L6-v2')

def initialize_lexical_index():
    """Builds a Full-Text Search index in DuckDB for exact entity lookups."""
    logger.info("Building Lexical Index (DuckDB FTS) for KYC profiles...")
    
    # pathlib objects must be converted to strings for DuckDB SQL commands
    con = duckdb.connect(str(DB_PATH))
    
    kyc_path = DATA_DIR / "kyc_profiles.json"
    
    # Load JSON and build index
    con.execute(f"CREATE OR REPLACE TABLE kyc_index AS SELECT * FROM read_json_auto('{kyc_path}')")
    con.execute("INSTALL fts; LOAD fts;")

    # FTS requires one identifier column plus at least one text column to index.
    fts_columns = ["entity_name", "jurisdiction", "investigator_notes"]
    fts_column_args = ", ".join(f"'{column}'" for column in fts_columns)
    con.execute(
        f"PRAGMA create_fts_index('kyc_index', 'node_id', {fts_column_args}, overwrite=1)"
    )
    
    count = con.execute("SELECT COUNT(*) FROM kyc_index").fetchone()[0]
    logger.info("Lexical Index complete: %s entities indexed.", count)
    con.close()

def initialize_vector_index():
    """Embeds adverse media and persists a FAISS index for semantic recall."""
    logger.info("Building Semantic Vector Index (FAISS) for Adverse Media...")
    
    news_path = DATA_DIR / "adverse_media.json"
    with news_path.open("r", encoding="utf-8") as f:
        news_data = json.load(f)
    
    snippets = [item['article_snippet'] for item in news_data]
    metadata = [item['related_node'] for item in news_data]
    
    # Generate dense embeddings
    embeddings = model.encode(snippets).astype('float32')
    
    # Initialize FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Persist index and metadata mapping using pathlib
    faiss.write_index(index, str(INDEX_PATH))
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f)
        
    logger.info("Vector Index complete: %s snippets embedded.", len(snippets))

if __name__ == "__main__":
    initialize_lexical_index()
    initialize_vector_index()
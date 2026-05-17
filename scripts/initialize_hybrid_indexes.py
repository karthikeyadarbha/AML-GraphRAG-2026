import json
import logging
from pathlib import Path
import numpy as np
import duckdb
import faiss
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standardized Paths
PROCESSED_DIR = Path("data/processed")
DB_PATH = PROCESSED_DIR / "argus_research.db"
FAISS_INDEX_PATH = PROCESSED_DIR / "vector_index.faiss"
METADATA_PATH = PROCESSED_DIR / "vector_metadata.json"

def build_hybrid_semantic_index():
    """
    Reads the unstructured context from DuckDB, encodes it into d=384 vectors, 
    applies L2 normalization, and builds the high-speed local FAISS index.
    """
    logger.info("Starting SOTA Hybrid Index Initialization...")

    # 1. Extract Unstructured Text from DuckDB
    logger.info(f"Connecting to DuckDB at {DB_PATH}...")
    con = duckdb.connect(str(DB_PATH))
    
    try:
        # Querying the unstructured text source table created during materialization
        query = """
            SELECT Document_ID, Account_ID, Document_Type, Raw_Text 
            FROM adverse_media
        """
        df_intel = con.execute(query).df()
        
        if df_intel.empty:
            logger.error("adverse_media table is empty. Did materialize_research_data.py run successfully?")
            return
            
        logger.info(f"Successfully extracted {len(df_intel):,} text records for vectorization.")
    finally:
        con.close()

    # 2. Initialize the d=384 Embedding Model
    logger.info("Loading SOTA all-MiniLM-L6-v2 embedding model...")
    # This specific model outputs exactly 384 dimensions, perfect for local FAISS constraints
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract the raw text column as a list for batch processing
    texts = df_intel['Raw_Text'].tolist()

    # 3. Batch Encode to Vectors
    logger.info("Encoding text to dense vectors. (Utilizing C/SIMD optimizations)...")
    # Batch encoding is significantly faster than a standard python loop
    raw_embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)

    # 4. The Mathematical "L2 Normalization Trick"
    logger.info("Applying L2 Normalization to snap vectors to a unit hypersphere...")
    # By normalizing the vectors, maximizing Cosine Similarity becomes mathematically 
    # equivalent to minimizing Euclidean Distance (L2), which FAISS processes significantly faster.
    faiss.normalize_L2(raw_embeddings)

    # 5. Build the FAISS Index
    logger.info("Building FAISS IndexFlatL2 for high-speed Euclidean search...")
    embedding_dimension = raw_embeddings.shape[1] # Will be 384
    
    # Initialize the Euclidean (L2) distance index
    faiss_index = faiss.IndexFlatL2(embedding_dimension)
    faiss_index.add(raw_embeddings)
    
    # Save the physical FAISS index to disk
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
    logger.info(f"FAISS index written to: {FAISS_INDEX_PATH}")

    # 6. Save the Metadata Mapping
    # FAISS only stores numbers and internal IDs (0, 1, 2...). 
    # We must save a mapping JSON so the Adjudicator agent knows which account matched the vector.
    logger.info("Generating mapping metadata payload...")
    
    # Convert dataframe to a list of dicts, ensuring types are JSON serializable
    metadata_records = df_intel.to_dict(orient='records')
    
    # The list index inherently matches the FAISS vector ID
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata_records, f, indent=4)
        
    logger.info(f"Metadata mapping written to: {METADATA_PATH}")
    logger.info("--- Hybrid Indexing Complete. Ready for Local LLM Adjudication ---")

if __name__ == "__main__":
    build_hybrid_semantic_index()
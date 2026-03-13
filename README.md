# AML-GraphRAG-2026
Applied AI engineering use case - Experimentation for AML dataset with GraphRAG implementation with hybrid search indexing.

# Agentic GraphRAG: Deterministic Reasoning for Anti-Money Laundering

This repository contains the empirical methodology, synthetic data generators, and execution pipelines for the research paper: **"Engineering Determinism in Generative AI: Agentic GraphRAG for Multi-Hop Financial Networks."**

Unlike standard Retrieval-Augmented Generation (RAG) which relies purely on semantic proximity, this architecture utilizes an in-memory analytical engine (DuckDB) to perform deterministic topological graph traversals. The exact structural state is then fused with hybrid-indexed semantic intelligence (DuckDB FTS + FAISS) and passed to a locally hosted Large Language Model under strict greedy decoding constraints (`T=0.0`).

## 1. Prerequisites & Environment Setup

To ensure exact reproducibility, please use the pinned dependencies provided in the manifest.

### Python Environment
```bash
# 1. Clone the repository
git clone [https://github.com/anonymous-researcher/agentic-graphrag.git](https://github.com/anonymous-researcher/agentic-graphrag.git)
cd agentic-graphrag

# 2. Install pinned dependencies
pip install -r requirements.txt


2. Execution Pipeline
The research methodology is divided into four strictly decoupled phases. Execute the following scripts sequentially from the project root.

Phase 1: Materialize Synthetic Ground Truth
Generates the baseline network (10,000 nodes) and deterministically seeds 50 illicit 3-hop circular typologies, linked to specific synthetic KYC notes.

Bash
python scripts/1_materialize_research_data.py
Outputs: data/raw/synthetic_ledger.csv, data/raw/kyc_profiles.json, data/raw/adverse_media.json

Phase 2: Integrity & Idempotency Verification
Acts as the unit-test layer. Verifies schema integrity, ensures zero data duplication, and confirms 1:1 cross-domain linkage between the structural graph and semantic context.

Bash
python scripts/2_verify_data_integrity.py
Expected Output: ALL INTEGRITY TESTS PASSED.

Phase 3: Hybrid Index Initialization
Bifurcates the semantic corpus. Initializes a DuckDB Full-Text Search (FTS) index for O(1) lexical lookups and a FAISS flat L2 index (all-MiniLM-L6-v2, 384-dimensions) for semantic motive discovery.

Bash
python scripts/3_initialize_hybrid_indexes.py

Outputs: data/processed/argus_research.db, data/processed/vector_index.faiss, data/processed/vector_metadata.json

Phase 4: Agentic Adjudication
Executes the GraphRAG reasoning phase. Materializes the recursive paths, retrieves the hybrid context, and dispatches the payload to the LLM.

Bash
python scripts/4_execute_adjudication_agent.py
Expected Output: A strict JSON verdict logging the SAR Confidence Score and Primary Typology.

3. Auditing the T=0.0 Deterministic Constraint
A core claim of this research is that Generative AI can be mathematically constrained to produce auditable JSON outputs for compliance environments.

Reviewers can verify the application of the temperature: 0.0 hyperparameter by observing the network serialization logs in Phase 4. The script intercepts and logs the explicit payload["options"] dictionary passed to the local inference engine before generation, proving the collapse of the probability distribution.

4. Repository Structure
Plaintext
├── data/
│   ├── raw/                 # Generated physical evidence (CSV/JSON)
│   └── processed/           # DuckDB databases and FAISS vector indices
├── scripts/
│   ├── 1_materialize_research_data.py
│   ├── 2_verify_data_integrity.py
│   ├── 3_initialize_hybrid_indexes.py
│   └── 4_execute_adjudication_agent.py
├── requirements.txt         # Pinned execution environment
└── README.md












## Local LLM Prerequisite (Ollama)

The adjudication workflow (`scripts/execute_adjudication_agent.py`) calls a local Ollama API at `http://127.0.0.1:11434` using the `mistral` model by default. You can override this with the `OLLAMA_API_URL` environment variable if your Ollama instance is exposed elsewhere.

Runtime tuning (recommended for slow cold starts):

- `OLLAMA_TIMEOUT_SECONDS` (default: `240`): Read timeout for a single generation call.
- `OLLAMA_MAX_RETRIES` (default: `2`): Number of retries when generation times out.

Example:

```bash
export OLLAMA_API_URL="http://127.0.0.1:11434/api/generate"
export OLLAMA_TIMEOUT_SECONDS=300
export OLLAMA_MAX_RETRIES=3
```

Use this helper before running adjudication (especially in a new shell/session):

```bash
bash scripts/start_ollama.sh
```

You can optionally provide a model name:

```bash
bash scripts/start_ollama.sh mistral
```

Then run:

```bash
/workspaces/AML-GraphRAG-2026/aml-grag/bin/python scripts/execute_adjudication_agent.py
```

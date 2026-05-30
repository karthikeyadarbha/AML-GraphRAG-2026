# Copilot instructions for AML-GraphRAG-2026

This file provides targeted, repository-specific instructions to help Copilot sessions be productive in this project. Keep replies focused on the repository flow, commands, and conventions below.

---

## Quick environment setup

- Install runtime dependencies: `py -3.10 -m pip install -r requirements.txt` (Python 3.10+). On non-Windows: `python3 -m pip install -r requirements.txt`.
- FAISS and sentence-transformers are used for local indexing and embeddings. On Windows prefer conda for faiss (e.g., `conda install -c conda-forge faiss-cpu`) or use a platform wheel; SentenceTransformers requires a PyTorch/TensorFlow backend.

## Recommended run order (one-liner)

- Typical end-to-end sequence:
  - `py -3.10 -m pip install -r requirements.txt && py -3.10 scripts/run_data_pipeline.py && py -3.10 scripts/materialize_research_data.py && py -3.10 scripts/initialize_hybrid_indexes.py && py -3.10 scripts/execute_adjudication_agent.py --health-check-llm`

## Build / test / run commands (explicit)

- Install deps: `py -3.10 -m pip install -r requirements.txt` (Windows) or `python3 -m pip install -r requirements.txt` (POSIX).
- Generate synthetic data (full pipeline):
  - `python scripts/run_data_pipeline.py`
    - Produces: data/raw/synthetic_ledger.csv, data/raw/kyc_profiles.json, data/raw/adverse_media.json
- Materialize DuckDB tables from raw artifacts:
  - `python scripts/materialize_research_data.py`
    - Produces: data/processed/aml_graphrag_adjudication.db
- Build the hybrid vector index (FAISS + metadata):
  - `python scripts/initialize_hybrid_indexes.py`
    - Produces: data/processed/vector_index.faiss and data/processed/vector_metadata.json
- Start the adjudication agent (end-to-end adjudication loop):
  - `python scripts/execute_adjudication_agent.py [--db-path <path>] [--index-path <path>] [--meta-path <path>] [--model mistral] [--health-check-llm] [--include-all-benign] [--suspicious-amount-threshold <float>] [--control-amount-threshold <float>] [--time-window-days <int>] [--max-hop-count <int>]`
  - Key flags to note:
    - `--health-check-llm`: perform a deterministic temperature 0.0 handshake before running.
    - `--include-all-benign`: include all non-fraud transactions in control group.
    - `--suspicious-amount-threshold`, `--control-amount-threshold`: numeric cutoffs used by graph traversal.
    - `--time-window-days`, `--max-hop-count`: control recursive CTE bounds.
  - Example deterministic run with LLM health check:
    - `python scripts/execute_adjudication_agent.py --health-check-llm`
  - The README references `--disable-llm`; the agent uses runtime checks and internal fallbacks — prefer the `--health-check-llm` flow for deterministic adjudication and fallbacks.
- Start Ollama (helper script provided):
  - `./scripts/start_ollama.sh [model_name]` (use WSL or a POSIX shell on Windows, or start Ollama CLI manually)
  - Verify LLM endpoint: `curl -sSf http://127.0.0.1:11434/api/tags`

Testing

- Unit tests are run with pytest from the repo root:
  - Run full suite: `py -3.10 -m pytest -q` or `pytest -q`
  - Run a single test function: `py -3.10 -m pytest tests/test_statistical_validation.py::test_validate_dataset_rigor_passes_with_all_benchmark_levels -q`
  - Run a single test file: `py -3.10 -m pytest tests/test_adjudicator_fallback.py -q`
- Quick dataset validation (no pytest):
  - POSIX/Windows (python):
    - `python -c "import pandas as pd; from scripts.data_pipeline.phase4_statistical_validation import validate_dataset_rigor; df=pd.read_csv('data/raw/synthetic_ledger.csv'); validate_dataset_rigor(df); print('VALIDATION PASSED')"`
  - PowerShell friendly:
    - `py -3.10 -c "import pandas as pd; from scripts.data_pipeline.phase4_statistical_validation import validate_dataset_rigor; df=pd.read_csv('data/raw/synthetic_ledger.csv'); validate_dataset_rigor(df); print('VALIDATION PASSED')"`

Linting

- No repository-provided linter. If linting is required, use team-standard tools (e.g., flake8/ruff) but do not assume they are configured here.

---

## High-level architecture (concise)

- Data generation pipeline (scripts/data_pipeline/phase1..4): creates a synthetic transaction ledger, injects adversarial scenarios (U-turns, smurfing), synthesizes KYC and adverse-media text, and validates dataset rigor.
- Materialization: `materialize_research_data.py` reads data/raw/* and persists three DuckDB tables: raw_ledger, kyc_profiles, adverse_media. This DB is the authoritative in-process OLAP store for graph queries.
- Hybrid semantic index: `initialize_hybrid_indexes.py` extracts adverse_media text from DuckDB, encodes with SentenceTransformer (all-MiniLM-L6-v2, d=384), L2-normalizes vectors, builds a FAISS IndexFlatL2, and saves a JSON metadata mapping linking FAISS IDs to documents.
- Deterministic adjudication agent: `execute_adjudication_agent.py`:
  - Runs recursive CTE graph traversals in DuckDB to find multi-hop loops and computes structural metrics (PVR, MVR).
  - Loads the FAISS metadata map for semantic context lookups.
  - Constructs a tightly constrained prompt matrix and dispatches to a local LLM (Ollama) with temperature=0.0 for determinism.
  - Parses and persists the final JSON verdicts into the Adjudication_Results DuckDB table for auditable outputs.
- Artifacts & locations:
  - Raw inputs: data/raw/*.csv, *.json
  - Persisted OLAP DB: data/processed/aml_graphrag_adjudication.db
  - FAISS index + metadata: data/processed/vector_index.faiss, vector_metadata.json
  - Agent log: data/processed/adjudication_agent.log

---

## Key conventions and repository-specific patterns

- Strict deterministic LLM usage: prompts are sent with options{"temperature": 0.0}. The adjudicator enforces JSON-only replies (no markdown fences) and expects SAR_Confidence_Score as an integer.
- Adjudication persistence schema (`Adjudication_Results`) must have Case_Type non-null; the agent contains migration/backfill logic and will raise a ValueError if NULLs remain (tests verify this). Ensure migrations run prior to summary steps.
- Vector embedding dimension: the pipeline assumes 384-d embeddings (all-MiniLM-L6-v2). FAISS index is built with normalized vectors (faiss.normalize_L2) so cosine ≈ L2 search.
- Topological proofs: multi-hop loops are detected via recursive CTEs in DuckDB; PVR and MVR are derived numerically in SQL and drive the scoring rubric.
- Logging: agent uses a dual-sink logger (stderr + data/processed/adjudication_agent.log). Tests and CI may rely on predictable log initialization.
- Shell scripts are provided for Ollama orchestration (POSIX); on Windows prefer WSL or start the Ollama CLI directly.

---

## Existing docs and AI assistant configs checked

- README.md and scripts/* were incorporated into these instructions.
- No CLAUDE.md, .cursorrules, AGENTS.md, .windsurfrules, CONVENTIONS.md, AIDER_CONVENTIONS.md, or .clinerules files were found in the repo root.

---

Runner provisioning & secrets

- Use self-hosted runners for reliable Ollama model hosting and deterministic CI. Recommended runner specs: 8+ vCPUs, 32+ GB RAM, SSD storage. Expose port 11434 internally for Ollama API.
- Install Ollama on the runner and pre-pull the model (example):
  - `ollama pull mistral`
  - Start Ollama as a systemd service (example): create `/etc/systemd/system/ollama.service` with `ExecStart=/usr/bin/ollama serve` and run `systemctl enable --now ollama`.
- FAISS & Python dependencies:
  - On Ubuntu: conda is recommended for faiss-cpu; e.g., `conda create -n aml python=3.10 && conda activate aml && conda install -c conda-forge faiss-cpu`
  - Preinstall PyTorch backend required by SentenceTransformers.

Secrets and GitHub Actions integration

- Store non-public values in GitHub Secrets: useful keys include:
  - `SELF_HOSTED_RUNNER_LABEL`: label used by self-hosted runners (e.g., "ollama-host")
  - `OLLAMA_MODEL`: default model name to pull (e.g., "mistral")
  - `RUNNER_DOCKER_IMAGE` (optional): pre-built image with Ollama and FAISS for ephemeral runners
- To use a self-hosted runner, update `.github/workflows/ci-ollama.yml` `runs-on:` from `ubuntu-latest` to `['self-hosted', '${{ secrets.SELF_HOSTED_RUNNER_LABEL }}']`.
- For private LLM images or credentials, use repository or organization secrets and avoid embedding tokens in workflow files.

Provisioning notes (quick checklist)

1. Provision VM with recommended specs and open port 11434.
2. Install system packages (curl, jq, build-essential, git, docker if needed).
3. Install Ollama and pull the desired model(s).
4. Install conda, create env, and install Python deps + faiss-cpu + SentenceTransformers.
5. Register the machine as a GitHub self-hosted runner and tag it with the label in secrets.
6. Ensure persistent storage for data/processed and logs; configure logrotate if needed.

CI config tip

- If Ollama must be private or requires GPUs, provision a dedicated runner and set the workflow to use its label. For reproducible CI, prefer a prebuilt runner image (`RUNNER_DOCKER_IMAGE`) that includes Ollama and FAISS.

Summary

Updated copilot instructions with runner provisioning and secrets guidance, plus CI tips to use self-hosted runners for Ollama and FAISS. If desired, I can add an example systemd unit, a runner provisioning script, or a workflow variant that targets self-hosted runners. Which would you like next?
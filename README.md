# AML GraphRAG 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DuckDB](https://img.shields.io/badge/DuckDB-In--Process-orange.svg)](https://duckdb.org/)

This repository contains the Agentic GraphRAG implementation for deterministic AML adjudication using hybrid DuckDB graph reasoning, semantic synthesis, and local LLM adjudication.

The current workflow synthesizes a benchmark dataset, materializes it into DuckDB, and executes an adjudication agent that persists audit-grade verdicts in `data/processed/aml_graphrag_adjudication.db`.

## 🔧 Current implementation

- `scripts/run_data_pipeline.py`: generates the synthetic dataset and validates benchmark coverage.
- `scripts/materialize_research_data.py`: loads raw files into DuckDB tables.
- `scripts/execute_adjudication_agent.py`: executes adjudication and persists results in DuckDB.
- `tests/test_statistical_validation.py`: verifies `Risk_Level` coverage and dataset rigor.
- `tests/test_case_type_integrity.py`: verifies `Case_Type` integrity.
- Persistence target: `data/processed/aml_graphrag_adjudication.db`.
- Logs: `data/processed/adjudication_agent.log`.

## 📁 Pipeline overview

1. `scripts/run_data_pipeline.py`
   - Builds the synthetic ledger baseline.
   - Injects benchmark risk scenarios.
   - Synthesizes semantic KYC/adverse-media context.
   - Runs statistical validation with hard failure on missing benchmark `Risk_Level` coverage.
   - Writes raw outputs under `data/raw/`.

2. `scripts/materialize_research_data.py`
   - Reads generated `data/raw/*` assets.
   - Creates DuckDB tables in `data/processed/aml_graphrag_adjudication.db`.
   - Materializes tables such as `raw_ledger`, `kyc_profiles`, and `adverse_media`.

3. `scripts/execute_adjudication_agent.py`
   - Loads `data/processed/aml_graphrag_adjudication.db`.
   - Ensures `Adjudication_Results` exists.
   - Adds or validates the `Case_Type` column.
   - Persists each adjudication verdict permanently.
   - Writes agent logs to `data/processed/adjudication_agent.log`.

## 🚀 Run the updated pipeline

### 1. Generate synthetic data

```bash
./aml-grag/bin/python scripts/run_data_pipeline.py
```

### 2. Materialize DuckDB tables

```bash
./aml-grag/bin/python scripts/materialize_research_data.py
```

### 3. Execute adjudication

```bash
./aml-grag/bin/python scripts/execute_adjudication_agent.py
```

If you want a safe LLM health-check before full adjudication:

```bash
./aml-grag/bin/python scripts/execute_adjudication_agent.py --health-check-llm
```

## 🧪 Tests

Run the regression tests for the current pipeline:

```bash
./aml-grag/bin/python -m pytest tests/test_statistical_validation.py -q
./aml-grag/bin/python -m pytest tests/test_case_type_integrity.py -q
```

## 📌 Persistence details

Active database:

```text
data/processed/aml_graphrag_adjudication.db
```

Persistent log file:

```text
data/processed/adjudication_agent.log
```

The adjudication agent writes a permanent `Adjudication_Results` table and preserves `Case_Type` values.

## ⚠️ Important note

Current code references only `aml_graphrag_adjudication.db`. Any legacy mentions of `argus_research.db` or older DB names are historical log artifacts, not active persistence targets.

## ℹ️ Troubleshooting

If DuckDB reports a locked database, verify that no other process is currently using `data/processed/aml_graphrag_adjudication.db`, then rerun the materialization or adjudication steps.

## 📌 Recommended workflow

1. `./aml-grag/bin/python scripts/run_data_pipeline.py`
2. `./aml-grag/bin/python scripts/materialize_research_data.py`
3. `./aml-grag/bin/python scripts/execute_adjudication_agent.py`
4. `./aml-grag/bin/python -m pytest tests/test_statistical_validation.py -q`

## 🔒 Data privacy

This repository is synthetic research-only data. No real PII or live financial records are included.

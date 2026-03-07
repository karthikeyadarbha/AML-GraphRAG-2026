# AML-GraphRAG-2026
Applied AI engineering use case - Experimentation for AML dataset with GraphRAG implementation with hybrid search indexing.

## Local LLM Prerequisite (Ollama)

The adjudication workflow (`scripts/execute_adjudication_agent.py`) calls a local Ollama API at `http://localhost:11434` using the `mistral` model.

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

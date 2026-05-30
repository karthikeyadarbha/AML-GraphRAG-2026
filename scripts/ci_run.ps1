<#
PowerShell CI helper for running the data pipeline, materialization, optional Ollama-handshake, and tests on Windows runners.
Usage: PowerShell -File scripts\ci_run.ps1
#>

$ErrorActionPreference = 'Stop'

Write-Host "== CI Run Script =="

# 1) Install Python deps (assumes py launcher present)
Write-Host "Installing Python dependencies..."
py -3.10 -m pip install --upgrade pip
try {
    py -3.10 -m pip install -r requirements.txt
} catch {
    Write-Warning "Dependency install had errors. If FAISS failed on Windows, consider using conda or run on a Linux/macOS runner. Proceeding..."
}

# 2) Run data pipeline
Write-Host "Running data pipeline..."
py -3.10 scripts\run_data_pipeline.py

# 3) Materialize DuckDB
Write-Host "Materializing DuckDB tables..."
py -3.10 scripts\materialize_research_data.py

# 4) Try to contact local Ollama (if user started it); otherwise fall back to rule-based run
$endpoint = 'http://127.0.0.1:11434/api/tags'
$ollamaAvailable = $false
try {
    $resp = Invoke-RestMethod -Uri $endpoint -Method Get -TimeoutSec 5 -ErrorAction Stop
    $ollamaAvailable = $true
} catch {
    Write-Host "Ollama endpoint not reachable. Will run adjudicator with --disable-llm."
}

# 5) Initialize FAISS index if faiss import is available
Write-Host "Attempting to initialize hybrid index (FAISS)..."
try {
    py -3.10 - <<'PY'
import importlib
try:
    importlib.import_module('faiss')
    print('FAISS available: building index...')
    import scripts.initialize_hybrid_indexes as mod
    mod.build_hybrid_semantic_index()
except Exception as e:
    print('FAISS not available or index build failed:', e)
PY
} catch {
    Write-Warning "Index build step encountered an error; continuing."
}

# 6) Run adjudication agent
if ($ollamaAvailable) {
    Write-Host "Running adjudication agent with LLM health check..."
    py -3.10 scripts\execute_adjudication_agent.py --health-check-llm
} else {
    Write-Host "Running adjudication agent in deterministic rule-based fallback mode..."
    py -3.10 scripts\execute_adjudication_agent.py --disable-llm
}

# 7) Run tests
Write-Host "Running pytest..."
try {
    py -3.10 -m pytest -q
} catch {
    Write-Error "Tests failed or pytest not installed. Please inspect output."
    exit 1
}

Write-Host "CI run complete."
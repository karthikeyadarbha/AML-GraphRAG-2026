#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-mistral}"
OLLAMA_API_URL="http://127.0.0.1:11434/api/tags"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Error: 'ollama' CLI is not installed."
  echo "Install it first: curl -fsSL https://ollama.com/install.sh | sh"
  exit 1
fi

if curl -fsS "$OLLAMA_API_URL" >/dev/null 2>&1; then
  echo "Ollama server is already running on 127.0.0.1:11434."
else
  echo "Starting Ollama server in the background..."
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  sleep 2

  if ! curl -fsS "$OLLAMA_API_URL" >/dev/null 2>&1; then
    echo "Error: Ollama server did not become ready."
    echo "Check logs: tail -n 100 /tmp/ollama.log"
    exit 1
  fi
fi

if ollama list | awk 'NR>1 {print $1}' | grep -Eq "^${MODEL_NAME}(:|$)"; then
  echo "Model '$MODEL_NAME' is already available."
else
  echo "Pulling model '$MODEL_NAME'..."
  ollama pull "$MODEL_NAME"
fi

echo "Ollama is ready. You can now run scripts/execute_adjudication_agent.py"

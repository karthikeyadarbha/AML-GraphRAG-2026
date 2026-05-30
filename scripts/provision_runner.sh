#!/usr/bin/env bash
# Provision a self-hosted runner with Ollama, FAISS, and Python dependencies for AML-GraphRAG-2026.
# Intended for Ubuntu 22.04+ (tested best effort). Run as root or with sudo.

set -euo pipefail
IFS=$'\n\t'

# Configuration (override by exporting env vars before running)
OLLAMA_MODEL="${OLLAMA_MODEL:-mistral}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-aml}
"
GITHUB_RUNNER_TOKEN="${GITHUB_RUNNER_TOKEN:-}"
GITHUB_RUNNER_REPO="${GITHUB_RUNNER_REPO:-}"   # e.g. owner/repo
GITHUB_RUNNER_LABEL="${GITHUB_RUNNER_LABEL:-ollama-host}"
RUNNER_NAME="${RUNNER_NAME:-$(hostname)-runner}"

if [ "$(id -u)" -ne 0 ]; then
  echo "This script should be run as root (sudo). Exiting."
  exit 1
fi

echo "== Runner provisioning started =="

apt-get update -y
apt-get install -y curl wget git jq build-essential ca-certificates libglib2.0-0 libnss3 libgomp1 \
  libssl-dev libffi-dev pkg-config unzip

# Optional: Docker for containerized runners
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not installed. Installing Docker (optional for some workflows)..."
  curl -fsSL https://get.docker.com | sh || echo "Docker install failed; continue if not needed."
  systemctl enable --now docker || true
fi

# Install Miniconda if not present
if ! command -v conda >/dev/null 2>&1; then
  echo "Installing Miniconda (headless)..."
  MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
  wget -q "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" -O /tmp/${MINICONDA_INSTALLER}
  bash /tmp/${MINICONDA_INSTALLER} -b -p /opt/miniconda
  rm /tmp/${MINICONA_INSTALLER:-/tmp/${MINICONDA_INSTALLER}} || true
  export PATH="/opt/miniconda/bin:$PATH"
  conda init bash || true
fi

# Create conda env and install faiss + python deps
export PATH="/opt/miniconda/bin:$PATH"
if ! conda info --envs | grep -q "^${CONDA_ENV_NAME}"; then
  echo "Creating conda environment: ${CONDA_ENV_NAME}"
  conda create -y -n "${CONDA_ENV_NAME}" python=3.10
fi

# Activate env for remainder of script
# shellcheck source=/dev/null
source "/opt/miniconda/bin/activate" "${CONDA_ENV_NAME}"

echo "Installing FAISS and PyTorch via conda-forge (faiss-cpu)..."
conda install -y -c conda-forge faiss-cpu pytorch cpuonly -c pytorch || echo "Conda install may have issues; please install manually if needed."

# Install Python pip deps from requirements.txt (project root assumed)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
  echo "Installing pip requirements from ${PROJECT_ROOT}/requirements.txt"
  python -m pip install --upgrade pip
  python -m pip install -r "${PROJECT_ROOT}/requirements.txt" || echo "pip install failed for some packages; you may need to install OS-level build deps or use wheels"
else
  echo "requirements.txt not found at ${PROJECT_ROOT}; skipping pip install"
fi

# Install Ollama (best-effort)
if ! command -v ollama >/dev/null 2>&1; then
  echo "Attempting Ollama install..."
  if curl -fsSL https://ollama.com/install.sh | sh; then
    echo "Ollama installed"
  else
    echo "Ollama install failed or not supported on this host. Install manually and re-run this script to pull models."
  fi
else
  echo "Ollama already installed"
fi

# Pull model
if command -v ollama >/dev/null 2>&1; then
  echo "Pulling Ollama model: ${OLLAMA_MODEL}"
  ollama pull "${OLLAMA_MODEL}" || echo "Model pull failed or requires authentication; please pull manually"

  # Create systemd unit for Ollama if not present
  if [ ! -f /etc/systemd/system/ollama.service ]; then
    echo "Creating systemd unit for Ollama..."
    cat >/etc/systemd/system/ollama.service <<EOF
[Unit]
Description=Ollama Local LLM Service
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/env ollama serve
Restart=on-failure
User=root

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now ollama || echo "Failed to enable/start ollama systemd service; start manually with 'ollama serve'"
  else
    echo "ollama.service already exists"
  fi
else
  echo "Ollama CLI not available; skipping model pull and service setup"
fi

# Optional: register GitHub self-hosted runner if token & repo provided
if [ -n "${GITHUB_RUNNER_TOKEN}" ] && [ -n "${GITHUB_RUNNER_REPO}" ]; then
  echo "Configuring GitHub self-hosted runner for ${GITHUB_RUNNER_REPO}"
  RUNNER_DIR=/opt/actions-runner
  mkdir -p "${RUNNER_DIR}"
  cd "${RUNNER_DIR}"
  # Download latest runner
  RUNNER_VERSION=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | jq -r .tag_name)
  TARBALL="actions-runner-linux-x64-${RUNNER_VERSION#v}.tar.gz"
  curl -sL "https://github.com/actions/runner/releases/download/${RUNNER_VERSION}/${TARBALL}" -o "${TARBALL}"
  tar xzf "${TARBALL}"
  rm -f "${TARBALL}"

  ./config.sh --unattended --url "https://github.com/${GITHUB_RUNNER_REPO}" --token "${GITHUB_RUNNER_TOKEN}" --name "${RUNNER_NAME}" --labels "${GITHUB_RUNNER_LABEL}" || echo "Runner config failed; check token and repo"

  # Install runner as service
  ./svc.sh install || true
  ./svc.sh start || true
  echo "Runner configured (attempted). Verify at https://github.com/${GITHUB_RUNNER_REPO}/settings/actions/runners"
else
  echo "GITHUB_RUNNER_TOKEN or GITHUB_RUNNER_REPO not set; skipping GitHub runner registration. To register a runner automatically, set these env vars and re-run."
fi

# Final notes
cat <<EOF
Provisioning complete.
Verify:
 - Ollama: curl -sSf http://127.0.0.1:11434/api/tags
 - DuckDB file storage: ensure ${PROJECT_ROOT}/data/processed exists and is writable
 - GitHub runner: visit https://github.com/${GITHUB_RUNNER_REPO}/settings/actions/runners (if registration was attempted)

If anything failed, re-run portions of this script manually and check logs. For GPU-enabled Ollama or models, provision GPU drivers and ensure Ollama supports the configuration.
EOF

exit 0

#!/usr/bin/env bash

set -euo pipefail

# primary override inside the data volume, if someone mounts their own config
CONFIG_FILE="/root/.ollama/llm-models.json"
# default config baked into the image
DEFAULT_CONFIG="/etc/ollama/llm-models.json"

echo "Starting Ollama server..."
ollama serve &
SERVER_PID=$!

echo "Waiting for Ollama server to become ready..."
MAX_RETRIES=60
SLEEP_SECONDS=5
READY=0

for i in $(seq 1 "${MAX_RETRIES}"); do
  if ollama list >/dev/null 2>&1; then
    READY=1
    break
  fi
  echo "Ollama not ready yet, attempt ${i}/${MAX_RETRIES}..."
  sleep "${SLEEP_SECONDS}"
done

if [ "${READY}" -ne 1 ]; then
  echo "Ollama did not become ready in time, exiting"
  kill "${SERVER_PID}" || true
  exit 1
fi

echo "Ollama is up, loading model configuration..."
echo ""

MODELS=()

if [ -f "${CONFIG_FILE}" ]; then
  echo "Found override config file: ${CONFIG_FILE}"
  mapfile -t MODELS < <(jq -r '.models[]? // empty' "${CONFIG_FILE}")
elif [ -f "${DEFAULT_CONFIG}" ]; then
  echo "Using default config file: ${DEFAULT_CONFIG}"
  mapfile -t MODELS < <(jq -r '.models[]? // empty' "${DEFAULT_CONFIG}")
else
  echo "No config file found, no models will be pulled"
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "No models configured. Skipping downloads."
  echo "Keeping Ollama server running..."
  wait "${SERVER_PID}"
  exit 0
fi

echo "Models selected for installation:"
for m in "${MODELS[@]}"; do
  echo "  • $m"
done
echo ""

for model in "${MODELS[@]}"; do
  echo "----------------------------------------"
  echo "Starting download of model: ${model}"
  echo "----------------------------------------"
  START_TS=$(date +%s)

  if ollama pull "${model}"; then
    END_TS=$(date +%s)
    DURATION=$((END_TS - START_TS))
    echo "✔ Completed download of ${model} in ${DURATION} seconds"
  else
    echo "✖ Failed to pull ${model}, continuing..."
  fi

  echo ""
done

echo "All model pulls done, keeping server running..."
wait "${SERVER_PID}"

#!/usr/bin/env bash
# [RUNS: LOCAL — needs Ollama installed and model.gguf sitting next to Modelfile]
#
# Registers your fine-tuned GGUF with Ollama, then opens a chat with it.
# This is the payoff: a model YOU trained, running on YOUR machine, offline.

set -euo pipefail
cd "$(dirname "$0")"

MODEL_NAME="${1:-forge-demo}"   # or pass a name: ./import.sh my-model

if [[ ! -f "model.gguf" ]]; then
  echo "model.gguf not found. Download it from Colab and place it here first." >&2
  exit 1
fi

echo "Creating Ollama model '$MODEL_NAME' from Modelfile..."
ollama create "$MODEL_NAME" -f Modelfile

echo
echo "Done. Starting a chat — type /bye to exit."
echo "----------------------------------------------------"
ollama run "$MODEL_NAME"

#!/usr/bin/env bash
set -euo pipefail

# Script per entrare nel .venv, eseguire mypy e avviare Streamlit

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -d ".venv" ]; then
  echo ".venv non trovato. Esegui scripts/setup_venv.sh prima."
  exit 1
fi

echo "Attivo virtualenv..."
. .venv/bin/activate

echo "Eseguo mypy ."
# Se mypy non è installato lo script fallirà — è intenzionale: vogliamo che mypy passi prima di avviare l'app
mypy .

echo "Mypy passato — avvio Streamlit"
streamlit run app.py

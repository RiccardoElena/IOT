#!/usr/bin/env bash
set -euo pipefail

# Script per creare un virtualenv `.venv` e installare le dipendenze da requirements.txt

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 non trovato. Assicurati di avere Python 3 installato."
  exit 1
fi

if [ -d ".venv" ]; then
  echo ".venv già esistente — salto creazione."
else
  echo "Creo virtualenv in .venv..."
  "$PY" -m venv .venv
  echo "Virtualenv creato."
fi

echo "Attivo virtualenv per installare pacchetti..."
. .venv/bin/activate
pip install --upgrade pip

if [ -f requirements.txt ]; then
  echo "Installazione delle dipendenze da requirements.txt..."
  pip install -r requirements.txt
else
  echo "Attenzione: requirements.txt non trovato in $ROOT"
fi

echo "Installazione completata. Per attivare manualmente: source .venv/bin/activate"

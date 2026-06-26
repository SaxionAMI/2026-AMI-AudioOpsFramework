#!/usr/bin/env bash
# Minimal setup for the Audio Ops Framework.
set -euo pipefail

echo "Creating runtime directories..."
mkdir -p storage audio_input dagster_home models mlruns \
         audio_input/example_classifier

if [ ! -f .env ]; then
  cp env.example .env
  echo "Created .env from env.example"
fi

echo
echo "Setup complete. Next:"
echo "  1. Drop training audio under audio_input/example_classifier/<label>/*.wav"
echo "  2. Local:  dagster dev          (then open http://localhost:3000)"
echo "     Docker: docker compose up -d (dagster :3000, inference :8000)"

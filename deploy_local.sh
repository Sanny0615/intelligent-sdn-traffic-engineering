#!/usr/bin/env bash
set -e

echo "===================================================================="
echo "Intelligent SDN Traffic Engineering - Production Deployment"
echo "===================================================================="
echo ""

echo "[1/3] Running automated pytest test suite..."
if [ -d ".venv" ]; then
    .venv/bin/python -m pytest
else
    python3 -m pytest
fi

echo ""
echo "[2/3] Building and starting Docker containers..."
docker compose up --build -d

echo ""
echo "[3/3] Verifying service deployment status..."
docker compose ps

echo ""
echo "===================================================================="
echo "Deployment Successful!"
echo " - FastAPI REST API:      http://localhost:8000/docs"
echo " - Streamlit Dashboard:   http://localhost:8501"
echo "===================================================================="

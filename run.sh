#!/usr/bin/env bash
# =============================================================
# run.sh – Startup script for the Semantic Similarity App
#
# Usage:
#   chmod +x run.sh   (only needed once on Linux/macOS)
#   ./run.sh
#
# On Windows, run this instead:
#   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# =============================================================

echo "======================================"
echo " Semantic Similarity Measurement App  "
echo "======================================"
echo ""
echo "[1/2] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/2] Starting FastAPI server at http://localhost:8000"
echo "      Press CTRL+C to stop the server."
echo ""

# Start the Uvicorn ASGI server
# --host 0.0.0.0  : Listen on all network interfaces
# --port 8000     : Use port 8000
# --reload        : Auto-reload when code changes (development only)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

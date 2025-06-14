#!/bin/sh

# start-mlflow.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- NEW: Wait for the database to be fully ready before proceeding ---
echo "Waiting for PostgreSQL"
python /app/wait-for-postgres.py
echo "PostgreSQL is ready."


echo "Starting MLflow server..."
exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --artifacts-destination "$MLFLOW_DEFAULT_ARTIFACT_ROOT"
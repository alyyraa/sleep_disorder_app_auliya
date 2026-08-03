#!/bin/sh
set -eu

RUNTIME_DIR="${RUNTIME_DIR:-/data}"
DATABASE_PATH="${DATABASE_PATH:-$RUNTIME_DIR/sleep_disorder.db}"
MODELS_PATH="$RUNTIME_DIR/models"
SEED_DIR="/opt/sleep-stress-seed"
SEED_STATE_FILE="/opt/sleep-stress-seed.sha256"
RUNTIME_STATE_FILE="$RUNTIME_DIR/.seed-state.sha256"

mkdir -p "$RUNTIME_DIR"

if [ ! -f "$DATABASE_PATH" ] && [ ! -e "$MODELS_PATH" ]; then
  cp "$SEED_DIR/sleep_disorder.db" "$DATABASE_PATH"
  cp -a "$SEED_DIR/models" "$MODELS_PATH"
  cp "$SEED_STATE_FILE" "$RUNTIME_STATE_FILE"
elif [ ! -f "$DATABASE_PATH" ] || [ ! -d "$MODELS_PATH" ]; then
  echo "ERROR: The Docker runtime volume is incomplete." >&2
  echo "SQLite and model artifacts must be restored or recreated together." >&2
  exit 1
elif [ ! -f "$RUNTIME_STATE_FILE" ]; then
  echo "ERROR: The Docker runtime volume predates deterministic deployment." >&2
  echo "Back it up, then recreate the app_runtime volume from the current image." >&2
  exit 1
elif ! cmp -s "$SEED_STATE_FILE" "$RUNTIME_STATE_FILE"; then
  echo "ERROR: The Docker runtime volume was initialized by another deployment state." >&2
  echo "Back it up, then recreate app_runtime to deploy the current Git state." >&2
  exit 1
fi

# Fail before Gunicorn starts if SQLite metadata and active artifacts have
# drifted apart. This avoids obscure version-directory collisions later.
python - "$DATABASE_PATH" <<'PY'
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

database_path = Path(sys.argv[1])
connection = sqlite3.connect(database_path)
try:
    metadata = connection.execute(
        "SELECT model_version, active_version_id FROM model_metadata WHERE id = 1"
    ).fetchone()
    if metadata is None:
        raise RuntimeError("model_metadata singleton is missing")

    version_name, active_version_id = metadata
    if active_version_id is None:
        raise RuntimeError("model_metadata.active_version_id is missing")

    version = connection.execute(
        "SELECT version, artifact_hashes FROM model_versions WHERE id = ?",
        (active_version_id,),
    ).fetchone()
    if version is None or version[0] != version_name:
        raise RuntimeError("active model metadata does not match model_versions")

    expected_hashes = json.loads(version[1])
    models_path = Path("/app/models")
    for filename, expected_hash in expected_hashes.items():
        artifact = models_path / filename
        if not artifact.is_file():
            raise RuntimeError(f"active artifact is missing: {filename}")
        actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"active artifact hash mismatch: {filename}")
finally:
    connection.close()
PY

exec "$@"

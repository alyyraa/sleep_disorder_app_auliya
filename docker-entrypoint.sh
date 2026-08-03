#!/bin/sh
set -eu

DATABASE_PATH="${DATABASE_PATH:-/app/sleep_disorder.db}"
DATABASE_DIR="$(dirname "$DATABASE_PATH")"

if [ ! -f "$DATABASE_PATH" ]; then
  mkdir -p "$DATABASE_DIR"
  if [ -f /app/sleep_disorder.db ]; then
    cp /app/sleep_disorder.db "$DATABASE_PATH"
  fi
fi

exec "$@"

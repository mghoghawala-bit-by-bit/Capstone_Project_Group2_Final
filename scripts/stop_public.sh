#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/.runtime"
APP_PID_FILE="$RUNTIME_DIR/streamlit.pid"
TUNNEL_PID_FILE="$RUNTIME_DIR/tunnel.pid"

if [[ -f "$TUNNEL_PID_FILE" ]]; then
  PID="$(cat "$TUNNEL_PID_FILE")"
  if kill -0 "$PID" >/dev/null 2>&1; then
    kill "$PID" || true
  fi
  rm -f "$TUNNEL_PID_FILE"
fi

if [[ -f "$APP_PID_FILE" ]]; then
  PID="$(cat "$APP_PID_FILE")"
  if kill -0 "$PID" >/dev/null 2>&1; then
    kill "$PID" || true
  fi
  rm -f "$APP_PID_FILE"
fi

echo "Stopped tunnel and app started by scripts/."

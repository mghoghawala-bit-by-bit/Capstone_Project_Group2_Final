#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"
APP_FILE="$ROOT_DIR/streamlit_app.py"
PORT="${PORT:-8507}"
RUNTIME_DIR="$ROOT_DIR/.runtime"
APP_LOG="$RUNTIME_DIR/streamlit.log"
TUNNEL_LOG="$RUNTIME_DIR/tunnel.log"
APP_PID_FILE="$RUNTIME_DIR/streamlit.pid"
TUNNEL_PID_FILE="$RUNTIME_DIR/tunnel.pid"

mkdir -p "$RUNTIME_DIR"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing Python interpreter at $VENV_PY"
  exit 1
fi

if [[ ! -f "$APP_FILE" ]]; then
  echo "Missing app file at $APP_FILE"
  exit 1
fi

if ! lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
  nohup "$VENV_PY" -m streamlit run "$APP_FILE" \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    >"$APP_LOG" 2>&1 &
  echo "$!" > "$APP_PID_FILE"
fi

timeout 30s bash -c "until curl -fsS http://127.0.0.1:$PORT >/dev/null; do :; done"

if [[ -f "$TUNNEL_PID_FILE" ]] && kill -0 "$(cat "$TUNNEL_PID_FILE")" >/dev/null 2>&1; then
  kill "$(cat "$TUNNEL_PID_FILE")" || true
fi

: > "$TUNNEL_LOG"
nohup ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -R 80:localhost:"$PORT" nokey@localhost.run >"$TUNNEL_LOG" 2>&1 &
echo "$!" > "$TUNNEL_PID_FILE"

PUBLIC_URL="$(timeout 45s sh -c "tail -n +1 -F '$TUNNEL_LOG' | grep -m1 'tunneled with tls termination' | sed -E 's/.*(https:\/\/[^ ]+).*/\1/'" || true)"

if [[ -z "$PUBLIC_URL" ]]; then
  echo "Tunnel started but URL was not detected."
  echo "Check log: $TUNNEL_LOG"
  exit 1
fi

echo "Local app: http://localhost:$PORT"
echo "Public URL: $PUBLIC_URL"
echo "Tunnel log: $TUNNEL_LOG"

#!/usr/bin/env bash
# Usage: ./run.sh [--tunnel]    (set ANTHROPIC_API_KEY for LLM mode; LAB_REPLAY=1 to serve cached LLM answers offline)
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
export PORT
rm -f data/tunnel_url.txt
if [ "$1" = "--tunnel" ]; then
  if ! command -v cloudflared >/dev/null; then echo "cloudflared not found (brew install cloudflared)"; exit 1; fi
  ( cloudflared tunnel --protocol http2 --url "http://localhost:$PORT" --no-autoupdate 2>&1 | \
      while read -r line; do
        u=$(echo "$line" | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | head -1)
        if [ -n "$u" ]; then echo "$u" > data/tunnel_url.txt; echo "  Tunnel: $u"; fi
      done ) &
  trap 'kill 0' EXIT
fi
exec .venv/bin/python -m uvicorn lab.server:app --host 0.0.0.0 --port "$PORT"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

for command in uv docker curl; do
    command -v "$command" >/dev/null || { echo "Install $command before running this demo." >&2; exit 1; }
done
docker compose version >/dev/null
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo 'Created .env. Fill OPENAI_API_KEY and Q_MODEL_MINI, then run ./demo.sh again.'
    exit 1
fi

uv sync --locked
# Check configuration without sourcing .env as shell code or printing credentials.
uv run --no-sync --env-file .env python - <<'PY'
import os
import socket
from urllib.parse import unquote, urlparse

for key in ("POSTGRES_PASSWORD", "DATABASE_URL", "OPENAI_API_KEY", "Q_MODEL_MINI"):
    if not os.environ.get(key, "").strip():
        raise SystemExit(f"Set {key} in .env before running the demo.")
db = urlparse(os.environ["DATABASE_URL"])
if not (db.scheme in ("postgres", "postgresql") and db.hostname in ("localhost", "127.0.0.1")
        and (db.port or 5432) == 5432 and db.path == "/q" and db.username == "q"
        and unquote(db.password or "") == os.environ["POSTGRES_PASSWORD"]):
    raise SystemExit("DATABASE_URL must point to local PostgreSQL on port 5432, database/user q, "
                     "with the password from POSTGRES_PASSWORD.")
for port in (8000, 8501):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            raise SystemExit(f"Port {port} is in use. Stop the existing app and retry.")
PY

docker compose --env-file .env up -d --wait --wait-timeout 60 postgres

api_pid=''
ui_pid=''
cleanup() {
    trap - EXIT INT TERM
    for pid in "$ui_pid" "$api_pid"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo 'Apps stopped. PostgreSQL data is preserved; stop the container with: docker compose stop postgres'
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# API startup creates the schema and upserts fixtures/demo_families.json.
uv run --no-sync --env-file .env uvicorn api.main:app --host 127.0.0.1 --port 8000 &
api_pid=$!

wait_ready() {
    local url=$1 pid=$2 name=$3
    for ((attempt=0; attempt<60; attempt++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$name exited during startup. Check the logs above." >&2
            return 1
        fi
        if curl --fail --silent --max-time 1 "$url" >/dev/null; then
            return 0
        fi
        sleep 1
    done
    echo "$name did not become ready in time. Check the logs above." >&2
    return 1
}
wait_ready http://127.0.0.1:8000/health "$api_pid" API

Q_API_URL=http://127.0.0.1:8000 uv run --no-sync --env-file .env streamlit run streamlit_app.py \
    --server.address 127.0.0.1 --server.port 8501 --server.headless true &
ui_pid=$!
wait_ready http://127.0.0.1:8501/_stcore/health "$ui_pid" Streamlit

echo 'Demo ready: http://127.0.0.1:8501 (10 demo accounts). API docs: http://127.0.0.1:8000/docs'
echo 'Press Ctrl+C to stop the API and Streamlit.'
while kill -0 "$api_pid" 2>/dev/null && kill -0 "$ui_pid" 2>/dev/null; do
    sleep 1
done
echo 'An app process exited; stopping the other app.' >&2
exit 1

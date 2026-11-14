#!/usr/bin/env bash
#
# Launch the OpenRouter-backed V3 demo miner.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[start_demo_miner] WARNING: .venv not found; using current Python." >&2
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "[start_demo_miner] WARNING: .env not found; using process environment." >&2
fi
# With several hotkeys on one host, MINER_ENV_FILE names a file read after .env with what differs per hotkey
# (WALLET_HOTKEY, AXON_PORT, MINER_RECORD_DIR, MINER_WORK_DIR); docs/DEMO_MINER.md, "Several hotkeys".
if [[ -n "${MINER_ENV_FILE:-}" ]]; then
  if [[ ! -f "${MINER_ENV_FILE}" ]]; then
    echo "[start_demo_miner] MINER_ENV_FILE=${MINER_ENV_FILE} does not exist" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${MINER_ENV_FILE}"
  set +a
fi

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY in .env or the process environment}"
: "${NETUID:?set NETUID in .env or the process environment}"
: "${WALLET_NAME:?set WALLET_NAME in .env or the process environment}"
: "${WALLET_HOTKEY:?set WALLET_HOTKEY in .env or the process environment}"
: "${SUBTENSOR_NETWORK:=test}"
export OPENROUTER_API_KEY NETUID WALLET_NAME WALLET_HOTKEY SUBTENSOR_NETWORK

echo "[start_demo_miner] netuid=${NETUID} network=${SUBTENSOR_NETWORK} wallet=${WALLET_NAME}/${WALLET_HOTKEY}"
exec python -u scripts/run_demo_miner.py

#!/usr/bin/env bash
#
# Set up a local SN5 (Hone / RLVR) miner development environment from this
# repository (https://github.com/hone-subnet-org/hone-subnet).
#
# What this does:
#   1. Pins a workspace-local CPython 3.12 under .tools/ (so non-root grading works)
#   2. Creates .venv and installs extras: chain + miner + dev
#   3. Copies .env.example -> .env if missing
#   4. Pulls the release-pinned V3 polyglot sandbox image (for try_task grading)
#   5. Smoke-checks imports and prints next steps
#
# Usage (from repo root):
#   ./setup_miner_dev.sh
#   ./setup_miner_dev.sh --skip-docker   # deps only, no image pull
#
# Local grading MUST run as a non-root user in the docker group — same rule as
# validators (rlvr/v3/release.py). Use ./scripts/run_try_task.sh …
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

SKIP_DOCKER=0
for arg in "$@"; do
  case "${arg}" in
    --skip-docker) SKIP_DOCKER=1 ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: ${arg}" >&2
      exit 2
      ;;
  esac
done

pick_python() {
  # Prefer a workspace-local interpreter so ubuntu/hone can exec the venv.
  if [[ -x "${REPO_ROOT}/.tools/cpython-3.12/bin/python3.12" ]]; then
    echo "${REPO_ROOT}/.tools/cpython-3.12/bin/python3.12"
    return 0
  fi
  local cand src
  for cand in \
    "${HOME}/.local/share/uv/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12" \
    "${HOME}/.local/bin/python3.12" \
    python3.12 python3.11 python3.10
  do
    if [[ "${cand}" == /* ]]; then
      [[ -x "${cand}" ]] || continue
      echo "${cand}"
      return 0
    fi
    if command -v "${cand}" >/dev/null 2>&1; then
      ver="$("${cand}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      case "${ver}" in
        3.10|3.11|3.12) command -v "${cand}"; return 0 ;;
      esac
    fi
  done
  echo "need Python 3.10–3.12 on PATH (or a uv cpython under ~/.local)" >&2
  exit 1
}

ensure_workspace_python() {
  local dest="${REPO_ROOT}/.tools/cpython-3.12"
  if [[ -x "${dest}/bin/python3.12" ]]; then
    return 0
  fi
  local src=""
  for src in \
    "${HOME}/.local/share/uv/python/cpython-3.12.15-linux-x86_64-gnu" \
    "${HOME}/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu"
  do
    if [[ -x "${src}/bin/python3.12" ]]; then
      mkdir -p "${REPO_ROOT}/.tools"
      echo "[setup_miner_dev] seeding workspace CPython from ${src}"
      rm -rf "${dest}"
      cp -a "${src}" "${dest}"
      chmod -R a+rX "${dest}"
      return 0
    fi
  done
  # Fall back: whatever pick_python found; grading-as-ubuntu may need a later seed.
  return 0
}

ensure_workspace_python
PY="$(pick_python)"
echo "[setup_miner_dev] using ${PY} ($("${PY}" --version))"

if [[ ! -d .venv ]]; then
  echo "[setup_miner_dev] creating .venv"
  "${PY}" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
echo "[setup_miner_dev] installing .[chain,miner,dev]"
pip install -e '.[chain,miner,dev]'

if [[ ! -f .env ]]; then
  cp .env.example .env
  # Sensible local defaults when present on this host
  if [[ -d "${HOME}/.bittensor/wallets/bear" ]]; then
    sed -i 's/^WALLET_NAME=.*/WALLET_NAME=bear/' .env
    sed -i 's/^WALLET_HOTKEY=.*/WALLET_HOTKEY=alon/' .env
  fi
  if ! grep -q '^MINER_WORK_DIR=' .env; then
    printf '\n# Local extract/grade scratch\nMINER_WORK_DIR=workspaces\n' >> .env
  fi
  chmod 0600 .env
  echo "[setup_miner_dev] wrote .env from .env.example — edit wallet + OPENROUTER_API_KEY"
else
  echo "[setup_miner_dev] keeping existing .env"
fi

mkdir -p workspaces data
chmod 755 scripts/run_try_task.sh 2>/dev/null || true

if [[ "${SKIP_DOCKER}" -eq 0 ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "[setup_miner_dev] WARNING: docker not found; skip image pull (local grade needs Docker)" >&2
  else
    IMAGE="$(python -c 'from rlvr.policy import RELEASE_POLICY; print(RELEASE_POLICY.v3_image)')"
    echo "[setup_miner_dev] pulling sandbox image:"
    echo "  ${IMAGE}"
    docker pull "${IMAGE}"
  fi
else
  echo "[setup_miner_dev] --skip-docker: not pulling sandbox image"
fi

python - <<'PY'
from rlvr.policy import RELEASE_POLICY
from rlvr.v3 import miner as _m  # noqa: F401
print("[setup_miner_dev] import ok")
print(f"[setup_miner_dev] miners_per_task={RELEASE_POLICY.v3_miners_per_task} commit_quorum={RELEASE_POLICY.v3_commit_quorum}")
PY

cat <<'EOF'

[setup_miner_dev] done.

Next:
  1. Edit .env  — WALLET_NAME / WALLET_HOTKEY / OPENROUTER_API_KEY / AXON_PORT
  2. Local loop (must be non-root; wrapper drops to ubuntu if you are root):
       ./scripts/run_try_task.sh extract /tmp/sn5-workspaces/repo_task
       ./scripts/run_try_task.sh grade tests/fixtures/v3-yamlcpp-e6225d04/reference.diff
  3. Live demo miner (funded registered hotkey + OpenRouter key + public port):
       ./start_demo_miner.sh
  4. Read docs/MINER_DEV.md, docs/DEMO_MINER.md, docs/V3_TASKS.md

Upstream: https://github.com/hone-subnet-org/hone-subnet
EOF

#!/usr/bin/env bash
#
# Run scripts/try_task.py as an unprivileged user in the docker group.
# RELEASE_POLICY grading refuses uid/gid 0 (see rlvr/v3/release.py).
#
# Usage (from repo root):
#   ./scripts/run_try_task.sh extract /tmp/sn5-workspaces/repo_task
#   ./scripts/run_try_task.sh grade tests/fixtures/v3-yamlcpp-e6225d04/reference.diff
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GRADER_USER="${RLVR_GRADER_USER:-ubuntu}"
if [[ "$(id -u)" -eq 0 ]]; then
  if ! id "${GRADER_USER}" >/dev/null 2>&1; then
    echo "[run_try_task] ERROR: create unprivileged user ${GRADER_USER} (docker group) first" >&2
    exit 1
  fi
  exec sudo -u "${GRADER_USER}" -g docker \
    env HOME="$(getent passwd "${GRADER_USER}" | cut -d: -f6)" \
        PATH="${REPO_ROOT}/.venv/bin:/usr/bin:/bin" \
        PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/try_task.py" "$@"
fi

if [[ "$(id -g)" -eq 0 ]]; then
  echo "[run_try_task] ERROR: refuse gid 0; use a normal user in the docker group" >&2
  exit 1
fi

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi
exec python "${REPO_ROOT}/scripts/try_task.py" "$@"

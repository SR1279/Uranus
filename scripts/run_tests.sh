#!/usr/bin/env bash
#
# Run the test suite under the conditions it needs:
#   - as an unprivileged user in the docker group, because V3 grading refuses root;
#   - with temporary files on disk, because one test measures the disk space a
#     directory takes, and an in-memory /tmp allocates none.
#
# Plain `pytest` started as root from the repository root comes here by itself
# (see pytest_configure in tests/conftest.py), so both report the same result.
#
# Each run gets its own temporary directory, so runs started at the same time
# cannot delete each other's files. Passed tests' files are removed as the run
# goes and failed tests' files are kept. A run that leaves nothing removes its
# directory itself (tests/conftest.py); any other is removed by a later run once
# it is a day old. A passing full run keeps a few megabytes for that day.
#
# The script ends by replacing itself with the test run (exec). With setpriv the
# process a caller started becomes pytest itself, so stopping or killing that
# process stops the tests. Without setpriv, sudo relays the signals that can be
# caught; only a SIGKILL of sudo alone leaves the run going.
#
# Usage: ./scripts/run_tests.sh [pytest arguments]
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "$(git rev-parse --is-shallow-repository 2>/dev/null || true)" == "true" ]]; then
  echo "[run_tests] WARNING: shallow clone; the previous-release tests need: git fetch --unshallow origin" >&2
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
TEST_USER="${RLVR_GRADER_USER:-ubuntu}"

if [[ "$(id -u)" -eq 0 ]]; then
  if ! id "${TEST_USER}" >/dev/null 2>&1; then
    echo "[run_tests] ERROR: create unprivileged user ${TEST_USER} (docker group) first" >&2
    exit 1
  fi
  if [[ "$(id -u "${TEST_USER}")" -eq 0 ]]; then
    echo "[run_tests] ERROR: RLVR_GRADER_USER=${TEST_USER} is root; name an unprivileged user" >&2
    exit 1
  fi
  home="$(getent passwd "${TEST_USER}" | cut -d: -f6)"
  # setpriv changes user inside this very process, so the process a caller started is
  # pytest itself and any signal, SIGKILL included, hits it directly. sudo is the
  # fallback: it relays the signals that can be caught, and a SIGKILL of sudo alone
  # leaves the test run going.
  if command -v setpriv >/dev/null 2>&1; then
    switch=(setpriv --reuid="${TEST_USER}" --regid=docker --init-groups)
  else
    switch=(sudo -n -u "${TEST_USER}" -g docker)
  fi
  as_user=("${switch[@]}" env -i
    HOME="${home}"
    USER="${TEST_USER}"
    LOGNAME="${TEST_USER}"
    PATH="${REPO_ROOT}/.venv/bin:/usr/bin:/bin"
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    PYTHONDONTWRITEBYTECODE=1
    PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-}"
    TERM="${TERM:-dumb}"
    LANG="${LANG:-C.UTF-8}"
    GIT_CONFIG_COUNT=1
    GIT_CONFIG_KEY_0=safe.directory
    GIT_CONFIG_VALUE_0="${REPO_ROOT}")
  # Display settings change nothing a test can see, so they are passed on when set.
  for name in NO_COLOR FORCE_COLOR PY_COLORS COLUMNS LINES CI LC_ALL; do
    if [[ -n "${!name:-}" ]]; then
      as_user+=("${name}=${!name}")
    fi
  done
else
  if [[ "$(id -g)" -eq 0 ]]; then
    echo "[run_tests] ERROR: refuse gid 0; use a normal user in the docker group" >&2
    exit 1
  fi
  TEST_USER="$(id -un)"
  home="${HOME:-$(getent passwd "${TEST_USER}" | cut -d: -f6)}"
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  # `env` alone runs the command unchanged; an empty array would stop bash before 4.4 under `set -u`.
  as_user=(env)
fi

# Runs that ended more than a day ago no longer need their files.
"${as_user[@]}" find /var/tmp -maxdepth 1 -type d -name "hone-pytest-${TEST_USER}-run.*" -mmin +1440 \
  -exec rm -rf -- {} + 2>/dev/null || true

# mktemp -d creates a fresh directory with mode 700, owned by the user that runs pytest.
run_dir="$("${as_user[@]}" mktemp -d "/var/tmp/hone-pytest-${TEST_USER}-run.XXXXXX")"

options=(--basetemp="${run_dir}/tmp" -o tmp_path_retention_policy=failed)
# The cache (which --lf reads) lives in the test user's home, who cannot write to a root-owned checkout.
# A caller that switches the cache plugin off gets no cache option, which would only draw a warning.
use_cache=1
for argument in "$@"; do
  if [[ "${argument}" == *no:cacheprovider* ]]; then
    use_cache=0
  fi
done
if [[ "${use_cache}" -eq 1 ]]; then
  if [[ -n "${home}" ]] && "${as_user[@]}" test -d "${home}" -a -w "${home}"; then
    options+=(-o cache_dir="${home}/.cache/hone-pytest")
  else
    echo "[run_tests] ${TEST_USER} has no writable home, so this run keeps no cache (--lf has nothing to read)" >&2
    options+=(-p no:cacheprovider)
  fi
fi

exec "${as_user[@]}" "${PYTHON}" -m pytest "${options[@]}" "$@"

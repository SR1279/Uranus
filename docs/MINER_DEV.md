# SN5 miner development environment

Local setup for mining on **Hone / RLVR** (Bittensor **netuid 5**), following
the rules in this repository:

- Upstream: <https://github.com/hone-subnet-org/hone-subnet>
- Specs: `docs/V3_TASKS.md`, `docs/DEMO_MINER.md`, `docs/PROTOCOL.md`
- Pin: `rlvr.policy.RELEASE_POLICY` (sandbox image, miners/task, commit quorum)

## One-shot setup

From the repo root (`/root/sn88/sn5` on this machine):

```bash
./setup_miner_dev.sh
```

That creates a workspace-local CPython under `.tools/`, a `.venv` with
`.[chain,miner,dev]`, writes `.env` from `.env.example` if missing, and pulls
the release-pinned polyglot sandbox image used by validators and
`scripts/try_task.py`.

Requires **Python 3.10–3.12** and **Docker** (for local grading / live verify).

## What a miner must do (V3)

1. Serve an axon that answers signed V3 offers from validators.
2. For each task:
   - **repo** → return a **Git unified diff** against the provided workspace.
   - **terminal** → return a **Bash script** plus a **trajectory_v1** record.
3. Sign responses with the miner hotkey. Validators only grade after
   **commit-before-reveal** quorum; late or unsigned work does not score.
4. Stay within size / sandbox limits in `RELEASE_POLICY` (`v3_patch_bytes`,
   `v3_script_bytes`, memory/cpus, etc.).

Do **not** invent task formats from older “ARC / manifold” playbooks. Live SN5
is this RLVR protocol only.

## Env vars (`.env`)

| Variable | Role |
| --- | --- |
| `NETUID` | `5` (finney) |
| `SUBTENSOR_NETWORK` | `finney` (or local/test) |
| `WALLET_NAME` / `WALLET_HOTKEY` | Cold/hot under `~/.bittensor/wallets` |
| `AXON_PORT` / `AXON_EXTERNAL_IP` | Public axon endpoint validators dial |
| `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` / `OPENROUTER_MODEL` | Demo miner LLM (Chat Completions) |
| `MINER_WORK_DIR` | Scratch for extracted workspaces |

Wallet keys are **not** stored in this repo. Register the hotkey on netuid 5
before expecting on-chain weight.

## Local grade loop (no chain)

Grading uses the same Docker image and policy path as validators.
`rlvr.v3.release.round_policy` **refuses uid/gid 0**, so run extract/grade as a
normal user in the `docker` group (on this host: `ubuntu`). The wrapper does
that when invoked as root:

```bash
./scripts/run_try_task.sh extract /tmp/sn5-workspaces/repo_task
# produce a unified diff against that workspace, then:
./scripts/run_try_task.sh grade path/to/your.diff

# smoke-test the fixture reference (all six checks pass):
./scripts/run_try_task.sh grade tests/fixtures/v3-yamlcpp-e6225d04/reference.diff

# terminal fixture:
./scripts/run_try_task.sh --fixture tests/fixtures/v3-terminal-urlooker-651460ec \
  extract /tmp/sn5-workspaces/term_task
./scripts/run_try_task.sh --fixture tests/fixtures/v3-terminal-urlooker-651460ec \
  grade tests/fixtures/v3-terminal-urlooker-651460ec/reference.sh
```

A passing local grade is necessary but not sufficient for chain incentive
(validators also enforce timing, signatures, and quorum).

## Running the tests

```bash
./scripts/run_tests.sh
```

Grading deliberately refuses root, so the script runs the suite as an unprivileged
user in the `docker` group (`ubuntu`, or `RLVR_GRADER_USER`), and it keeps
temporary files on disk. The previous-release tests also need full Git history
(`git fetch --unshallow origin`).

A full run takes about 4.5 minutes, because it grades real tasks in Docker. Give it
a 10-minute timeout or run it in the background. The expected result is no failures
and 2 platform skips.

Plain `pytest` or `python -m pytest`, started as root from the repository root, hands
over to the same script and prints one line saying so. It stays root, and says why,
when the hand-over would change the run in another way:

- it was started from another directory, through a wrapper such as a profiler, with
  interpreter options, or on another Python than `.venv`;
- one of the output options `--junitxml`, `--debug`, `--basetemp` or `log_file` is given;
- an environment variable the script does not pass on is set: `TMPDIR`, or any name
  starting with `PYTHON`, `PYTEST_`, `COVERAGE_`, `LD_`, `DOCKER_` or `RLVR_`, other
  than `PYTHONPATH`, `PYTEST_ADDOPTS` and `RLVR_GRADER_USER`;
- a path the run reads is not readable by the test user.

The script changes user with `setpriv`, inside the process you started, so stopping
or killing that process stops the tests. A normal stop (Ctrl-C, `kill`, a timeout)
also removes any grading container the run had started. A hard kill (`kill -9`)
cannot clean up, so it can leave one exited container named `fixture-*` or
`try-task-*`; remove it with `docker rm`.

`RLVR_TESTS_AS_ROOT=1` keeps any run as root. As root, the tests that need an
unprivileged user and on-disk temporary files fail; that is the root refusal working,
not a bug.

Each run has its own directory, `/var/tmp/hone-pytest-<user>-run.*`. Failed tests'
files stay there. A run that leaves nothing removes its directory, and a later run
removes directories older than a day; a passing full run keeps a few megabytes
until then. The cache that `--lf` reads is in the test user's home,
`~/.cache/hone-pytest`, so a run that stays root has a different `--lf` history.

## Live demo miner

```bash
# fill OPENROUTER_API_KEY + wallet + axon in .env first
./start_demo_miner.sh
```

The demo miner (`rlvr.v3.miner`) pulls tasks over the axon, calls OpenRouter, and
returns signed V3 payloads. Replace the OpenRouter step with your own solver while
keeping the protocol surface intact.

## Layout on this host

```
sn5/
  .tools/cpython-3.12/   # workspace-local interpreter (non-root can exec)
  .venv/                 # created by setup_miner_dev.sh
  .env                   # local secrets / axon (gitignored)
  workspaces/            # optional scratch
  docs/MINER_DEV.md      # this file
  setup_miner_dev.sh
  start_demo_miner.sh
  scripts/try_task.py
  scripts/run_try_task.sh
  rlvr/                  # protocol + demo miner
  tests/fixtures/        # reference problems + diffs
```

## Related docs in-repo

- `README.md` — product overview
- `docs/DEMO_MINER.md` — OpenRouter demo miner walkthrough
- `docs/V3_TASKS.md` — task shapes, grading, sandbox
- `docs/PROTOCOL.md` — commit/reveal and axon messages

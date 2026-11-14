"""Shared test setup."""

from __future__ import annotations

import os
import pwd
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from rlvr.config import Settings
from rlvr.neurons.demo_miner import DemoMinerSettings

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "run_tests.sh"
VENV = REPO_ROOT / ".venv"

_ROOT_NOTE = pytest.StashKey[str]()

# The runner starts a fresh interpreter with a short, fixed environment. A run that sets
# a variable from one of these families would silently lose it, so it stays as started.
_FAMILIES_THE_RUNNER_DROPS = ("PYTHON", "PYTEST_", "COVERAGE_", "COV_CORE_", "LD_", "DOCKER_", "RLVR_")
_NAMES_THE_RUNNER_DROPS = {"TMPDIR", "TEMP", "TMP"}
# These it passes on, sets itself, or they are pytest's own bookkeeping.
_NAMES_THAT_SURVIVE = {
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTEST_ADDOPTS",
    "PYTEST_VERSION",
    "PYTEST_CURRENT_TEST",
    "RLVR_GRADER_USER",
    "RLVR_TESTS_AS_ROOT",
}


def _grader_user() -> str:
    return os.environ.get("RLVR_GRADER_USER") or "ubuntu"


def _as_user(user: str) -> list[str]:
    """The same switch the runner uses: in place with setpriv, through sudo without it."""
    if shutil.which("setpriv"):
        return ["setpriv", f"--reuid={user}", "--regid=docker", "--init-groups"]
    return ["sudo", "-n", "-u", user, "-g", "docker"]


def _not_a_plain_pytest_start(config: pytest.Config, arguments: list[str]) -> str | None:
    """A reason unless the process is exactly `pytest ARGS` or `python -m pytest ARGS` on the project's Python."""
    if config.invocation_params.plugins is not None:
        return "pytest was started from inside another program"
    original = list(getattr(sys, "orig_argv", []))
    count = len(arguments)
    if len(original) < 1 + count or (count and original[-count:] != arguments):
        return "pytest was not started from the command line"
    before = original[1 : len(original) - count]
    plain_module = before == ["-m", "pytest"]
    plain_script = len(before) == 1 and Path(before[0]).resolve() in {
        (VENV / "bin" / "pytest").resolve(),
        (VENV / "bin" / "py.test").resolve(),
    }
    if not (plain_module or plain_script):
        return "pytest was started through another program or with interpreter options"
    # sys.prefix is the virtual environment's own directory; the binary behind it is shared
    # by every environment built on the same interpreter.
    if Path(sys.prefix).resolve() != VENV.resolve():
        return f"it runs in {sys.prefix}, not the project's {VENV}"
    return None


def _output_option(config: pytest.Config) -> str | None:
    """An option that writes a file where the caller said, which the test user may not be allowed to do."""
    option = config.option
    if getattr(option, "debug", None):
        return "--debug"
    if getattr(option, "xmlpath", None):
        return "--junitxml"
    if getattr(option, "basetemp", None):
        return "--basetemp"
    if config.getoption("log_file", None):
        return "--log-file"
    try:
        if config.getini("log_file"):
            return "log_file"
    except (ValueError, KeyError):  # the logging plugin is switched off
        pass
    return None


def _paths_the_run_reads(config: pytest.Config, arguments: list[str]) -> list[str]:
    paths = [str(REPO_ROOT), str(config.rootpath)]
    if config.inipath is not None:
        paths.append(str(config.inipath))
    confcutdir = config.getoption("confcutdir", None)
    if confcutdir:
        paths.append(str(confcutdir))
    # config.args is what pytest parsed, with @files and PYTEST_ADDOPTS already expanded.
    for argument in [*config.args, *arguments]:
        argument = str(argument)
        if argument.startswith("--") and "=" in argument:
            argument = argument.split("=", 1)[1]
        elif argument.startswith("@"):
            argument = argument[1:]
        elif argument.startswith("-"):
            continue
        paths.append(argument.split("::")[0])
    paths += os.environ.get("PYTHONPATH", "").split(os.pathsep)
    return sorted({path for path in paths if path and os.path.exists(path)})


def _why_root_stays_root(config: pytest.Config) -> str | None:
    arguments = [str(argument) for argument in config.invocation_params.args]
    reason = _not_a_plain_pytest_start(config, arguments)
    if reason is not None:
        return reason
    if Path.cwd().resolve() != REPO_ROOT:
        return f"it was started outside {REPO_ROOT}"
    if not os.access(RUNNER, os.X_OK):
        return f"{RUNNER} is missing or not executable"
    dropped = sorted(
        name
        for name, value in os.environ.items()
        if value
        and name not in _NAMES_THAT_SURVIVE
        and (name in _NAMES_THE_RUNNER_DROPS or name.startswith(_FAMILIES_THE_RUNNER_DROPS))
    )
    if dropped:
        return f"{', '.join(dropped[:4])} is set, which the unprivileged run would not see"
    output = _output_option(config)
    if output is not None:
        return f"{output} writes a file the unprivileged test user may not be able to write"
    user = _grader_user()
    try:
        if pwd.getpwnam(user).pw_uid == 0:
            return f"RLVR_GRADER_USER={user} is root"
    except KeyError:
        return f"there is no user {user}"
    # Everything the run will read must be readable by the test user.
    try:
        probe = subprocess.run(
            [*_as_user(user), "sh", "-c", 'for p; do [ -r "$p" ] || exit 3; done', "sh", *_paths_the_run_reads(config, arguments)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=10,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"the switch to {user} could not be tried ({exc})"
    if probe.returncode == 3:
        return f"{user} cannot read one of the paths this run uses"
    if probe.returncode != 0:
        return f"commands cannot be run as {user} in the docker group"
    return None


def _stop_on_sigterm() -> None:
    """Make SIGTERM end the run the way Ctrl-C does, so clean-up code runs.

    Left alone, SIGTERM kills pytest on the spot, and a container a grading test started
    is never removed. As an interrupt it unwinds through the tests' own clean-up first.
    """

    def interrupt(_number, _frame):
        raise KeyboardInterrupt("SIGTERM")

    if threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        signal.signal(signal.SIGTERM, interrupt)


def pytest_configure(config: pytest.Config) -> None:
    """Plain `pytest` started as root runs the suite through scripts/run_tests.sh instead.

    V3 grading refuses root on purpose (rlvr/v3/release.py, rlvr/v3/supervisor.py,
    scripts/require_service_user.sh), and one test measures disk allocation, which an
    in-memory /tmp does not have. As root, those tests fail for reasons that are not
    bugs. The runner runs the suite as the unprivileged test user in the docker group,
    with temporary files on disk, so a plain run and the runner report the same result.
    A run that the hand-over would change in some other way (interpreter options, a
    wrapper such as a profiler, output files, another directory) stays root and says so.
    Set RLVR_TESTS_AS_ROOT=1 to keep any run as root.
    """
    _stop_on_sigterm()
    if os.getuid() != 0 or os.environ.get("RLVR_TESTS_AS_ROOT") == "1":
        return
    try:
        reason = _why_root_stays_root(config)
    except Exception as exc:  # noqa: BLE001 - a fault in this check must never stop a run
        reason = f"the hand-over check itself failed ({type(exc).__name__}: {exc})"
    if reason is not None:
        # Said twice, at the start and in the closing summary, and not as a warning: a
        # caller who turns warnings into errors must still get the run they asked for.
        config.stash[_ROOT_NOTE] = (
            f"[tests] running as root because {reason}; the tests that need an unprivileged "
            "user and on-disk temporary files will fail, and --lf reads root's own cache. "
            "Use ./scripts/run_tests.sh."
        )
        print(config.stash[_ROOT_NOTE], file=sys.stderr)
        return
    capture = config.pluginmanager.getplugin("capturemanager")
    if capture is not None:
        capture.stop_global_capturing()
    print(
        f"[tests] started as root; continuing as {_grader_user()} through scripts/run_tests.sh "
        "(RLVR_TESTS_AS_ROOT=1 keeps a run as root)",
        file=sys.stderr,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(str(RUNNER), [str(RUNNER), *config.invocation_params.args])


def pytest_terminal_summary(terminalreporter, config: pytest.Config) -> None:
    note = config.stash.get(_ROOT_NOTE, None)
    if note:
        terminalreporter.write_line(note, yellow=True)


def pytest_unconfigure(config: pytest.Config) -> None:
    """A run started by scripts/run_tests.sh removes its own directory when nothing is left in it."""
    basetemp = getattr(config.option, "basetemp", None)
    if not basetemp:
        return
    run_dir = Path(basetemp).parent
    if not (run_dir.parent == Path("/var/tmp") and run_dir.name.startswith("hone-pytest-") and "-run." in run_dir.name):
        return
    for directory in (Path(basetemp), run_dir):
        try:
            for entry in directory.iterdir():
                if entry.is_symlink() and not entry.exists():
                    entry.unlink()  # pytest's "current" links to directories it already removed
            directory.rmdir()  # only an empty directory goes
        except OSError:
            return


@pytest.fixture(autouse=True)
def _ignore_the_operators_env_file(monkeypatch):
    """Tests run on the defaults, as in a clean checkout, never on a real .env beside them.

    An operator's .env holds live settings and keys, and may not be readable by the
    unprivileged user that runs the tests.
    """
    for settings in (Settings, DemoMinerSettings):
        monkeypatch.setitem(settings.model_config, "env_file", None)

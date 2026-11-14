"""A failed check is rerun on the miner's own answer, to show which hidden tests failed and why."""

from __future__ import annotations

import base64
import json
import zlib

import pytest

from rlvr.neurons.diagnosis import (
    DiagnosisError,
    compare,
    louder,
    parse_failed_check,
    rerun_failed_check,
    split_raw_output,
)
from rlvr.neurons.journal import CHECK_FILES_DIR, DIAGNOSIS_FILE, PATCH_FILE, TASK_FILE, WORKSPACE_FILE
from rlvr.neurons.workbench import Sandbox
from rlvr.v3.feedback import render_failed_check
from rlvr.v3.manifest import Expectation, InvocationCheck
from rlvr.v3.supervisor import ContainerRequest
from tests.test_v3_archive import make_tar, write_archive

HIDDEN_TEST = b'''import unittest

from calc import add, mul


class CalcTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_mul(self):
        self.assertEqual(mul(2, 3), 6)
'''
# Shaped like the checks the task generator writes: hidden tests packed into the
# script, one line per test, and the raw runner output switched off.
CHECK_SCRIPT = f'''import base64, os, re, subprocess, zlib
FILES = {{'hidden_acceptance/test_calc.py': '{base64.b64encode(zlib.compress(HIDDEN_TEST)).decode()}'}}
for p, b in FILES.items():
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(p, "wb") as f:
        f.write(zlib.decompress(base64.b64decode(b)))
try:
    r = subprocess.run(["sh", "-c", 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/work python3 -m unittest discover -v -s hidden_acceptance -p test_calc.py'], capture_output=True, text=True, timeout=60)
    out, code = r.stdout + "\\n" + r.stderr, r.returncode
except subprocess.TimeoutExpired:
    out, code = "", "timeout"
rows = set()
for line in out.splitlines():
    m = re.match(r"^(test\\S* \\([\\w.]+\\)) \\.\\.\\. (ok|FAIL|ERROR|skipped|expected failure|unexpected success)", line)
    if m:
        rows.add("unittest " + m.group(1) + " " + m.group(2)); continue
if True:
    print("\\n".join(sorted(rows)) if rows else "no per-test lines")
print("result:", "ok" if code == 0 else "failed")
if False:
    print("RAW OUTPUT TAIL:\\n" + out[-6000:])
'''
REQUIRED = (
    "unittest test_add (test_calc.CalcTests.test_add) ok\n"
    "unittest test_mul (test_calc.CalcTests.test_mul) ok\n"
    "result: ok\n"
)
WORKSPACE = b"def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a + b\n"
# Fixes add but not mul, so one hidden test still fails.
PATCH = (
    b"diff --git a/calc.py b/calc.py\n"
    b"--- a/calc.py\n"
    b"+++ b/calc.py\n"
    b"@@ -1,4 +1,4 @@\n"
    b" def add(a, b):\n"
    b"-    return a - b\n"
    b"+    return a + b\n"
    b" \n"
    b" \n"
)


def display(argv=("/usr/bin/python3", "-c", CHECK_SCRIPT), required=REQUIRED) -> str:
    check = InvocationCheck(
        check_id="00-calc",
        argv=argv,
        cwd=".",
        stdin=None,
        timeout_s=90,
        max_stdout_bytes=65536,
        max_stderr_bytes=65536,
        expect=Expectation(0, "gold/00-calc.txt", None),
    )
    request = ContainerRequest(
        name="validator-check", argv=argv, cwd="/work", mounts=(), stdin=b"", timeout_s=90,
        max_stdout_bytes=65536, max_stderr_bytes=65536, trusted=False,
    )
    shown = render_failed_check(check=check, request=request, expected_stdout=required.encode(), expected_stderr=None)
    assert shown is not None
    return shown


def test_a_display_reads_back_to_the_exact_check_the_validator_ran():
    check = parse_failed_check(display())
    assert check is not None
    assert check.argv == ("/usr/bin/python3", "-c", CHECK_SCRIPT)
    assert (check.cwd, check.stdin, check.exit_code, check.stdout, check.stderr) == ("/work", b"", 0, REQUIRED, None)
    by_path = parse_failed_check(display(argv=("/usr/bin/python3", "tests/check.py")))
    assert by_path is not None and by_path.argv == ("/usr/bin/python3", "tests/check.py")


@pytest.mark.parametrize("text", ["", "Required stdout: \"x\"", display().replace("Stdin: ", "Input: ")])
def test_a_display_in_another_shape_is_not_guessed_at(text):
    assert parse_failed_check(text) is None


def test_the_rerun_switches_on_the_test_runners_raw_output():
    argv, switched = louder(("/usr/bin/python3", "-c", CHECK_SCRIPT))
    assert switched and 'if True:\n    print("RAW OUTPUT TAIL:\\n" + out[-20000:])' in argv[2]
    assert argv[2].replace("if True:\n    print(\"RAW", "if False:\n    print(\"RAW").replace("-20000", "-6000") == CHECK_SCRIPT
    assert louder(("/usr/bin/python3", "check.py")) == (("/usr/bin/python3", "check.py"), False)
    assert split_raw_output("a ok\nresult: failed\nRAW OUTPUT TAIL:\ntrace\n") == ("a ok\nresult: failed\n", "trace\n")
    assert split_raw_output("no marker\n") == ("no marker\n", "")


RAW = """test_add (test_calc.CalcTests.test_add) ... ok
test_mul (test_calc.CalcTests.test_mul) ... FAIL
test_div (test_calc.CalcTests.test_div) ...
  test_div (test_calc.CalcTests.test_div) (value=0) ... FAIL

======================================================================
FAIL: test_div (test_calc.CalcTests.test_div) (value=0)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/work/hidden_acceptance/test_calc.py", line 14, in test_div
    self.assertRaises(ZeroDivisionError)
AssertionError: ZeroDivisionError not raised

======================================================================
FAIL: test_mul (test_calc.CalcTests.test_mul)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/work/hidden_acceptance/test_calc.py", line 11, in test_mul
    self.assertEqual(mul(2, 3), 6)
AssertionError: 5 != 6

----------------------------------------------------------------------
Ran 3 tests in 0.001s

FAILED (failures=2)
"""


def test_each_hidden_test_that_did_not_pass_is_named_with_its_error():
    required = REQUIRED.replace("result: ok\n", "unittest test_div (test_calc.CalcTests.test_div) ok\nresult: ok\n")
    check = parse_failed_check(display(required=required))
    printed = "unittest test_add (test_calc.CalcTests.test_add) ok\nunittest test_mul (test_calc.CalcTests.test_mul) FAIL\nresult: failed\n"
    found = compare(check, printed, 0, RAW)
    assert not found.matched and found.tests == 3
    assert found.failed == [
        {"test": "test_mul", "result": "FAIL", "message": "AssertionError: 5 != 6"},
        {"test": "test_div", "result": "no result", "message": "AssertionError: ZeroDivisionError not raised"},
    ]
    assert found.summary == (
        "2 of 3 hidden tests did not pass: 1 failed or errored, "
        "1 printed no result (a subtest failed, it crashed or it never ran)"
    )


def test_errors_are_found_when_the_project_indents_the_runners_output():
    # Kivy writes everything on stderr indented by a space, unittest's report included.
    indented = "\n".join(" " + line if line else line for line in RAW.splitlines())
    check = parse_failed_check(display(required=REQUIRED.replace("result: ok\n", "unittest test_div (test_calc.CalcTests.test_div) ok\nresult: ok\n")))
    printed = "unittest test_add (test_calc.CalcTests.test_add) ok\nunittest test_mul (test_calc.CalcTests.test_mul) FAIL\nresult: failed\n"
    found = compare(check, printed, 0, indented)
    assert [(item["test"], item.get("message")) for item in found.failed] == [
        ("test_mul", "AssertionError: 5 != 6"),
        ("test_div", "AssertionError: ZeroDivisionError not raised"),
    ]


def test_a_check_that_states_only_an_overall_result_is_explained_by_the_runners_output():
    check = parse_failed_check(display(required="no per-test lines\nresult: ok\n"))
    found = compare(check, "no per-test lines\nresult: failed\n", 0, RAW)
    assert found.tests == 3
    assert found.failed == [
        {"test": "test_mul", "result": "FAIL", "message": "AssertionError: 5 != 6"},
        {"test": "test_div", "result": "FAIL", "message": "AssertionError: ZeroDivisionError not raised"},
    ]
    assert found.summary == "2 of 3 hidden tests did not pass: 2 failed or errored"


def test_output_without_test_lines_is_compared_line_by_line():
    check = parse_failed_check(display(required="trigger/tenant: scalar 'scoped'\nexit 0\n"))
    found = compare(check, "trigger/tenant: scalar 'base'\nexit 0\n", 0)
    assert found.summary == "1 of 2 required output lines were missing or different"
    assert found.missing_lines == ["trigger/tenant: scalar 'scoped'"] and found.unexpected_lines == ["trigger/tenant: scalar 'base'"]
    assert compare(check, check.stdout, 1).summary == "the check exited 1, not 0"
    assert compare(check, check.stdout, 0).matched


def task_folder(tmp_path, *, task_type="repository_patch_v1", keep_workspace=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive, reference = write_archive(tmp_path, make_tar([("calc.py", "file", WORKSPACE, 0o644)]))
    folder = tmp_path / "task"
    folder.mkdir()
    (folder / TASK_FILE).write_text(json.dumps({"task_type": task_type, "workspace": reference.model_dump(mode="json")}))
    (folder / PATCH_FILE).write_bytes(PATCH)
    if keep_workspace:
        archive.rename(folder / WORKSPACE_FILE)
    return folder


class NoSandbox:
    owns_files = False


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"task_type": "terminal_script_v1"}, "only repository tasks"),
        ({"keep_workspace": False}, "the workspace was not kept"),
    ],
)
def test_a_check_that_cannot_be_rerun_says_why(tmp_path, updates, expected):
    with pytest.raises(DiagnosisError, match=expected):
        rerun_failed_check(task_folder(tmp_path, **updates), display(), NoSandbox(), tmp_path)
    with pytest.raises(DiagnosisError, match="could not be read"):
        rerun_failed_check(task_folder(tmp_path / "other"), "Required stdout: \"x\"", NoSandbox(), tmp_path)


@pytest.fixture
def real_sandbox():
    sandbox, status = Sandbox.detect(concurrency=1)
    if sandbox is None:
        pytest.skip(f"the sandbox is not available here: {status}")
    return sandbox


def test_real_rerun_names_the_hidden_test_the_answer_still_fails(tmp_path, real_sandbox):
    (tmp_path / "work").mkdir()
    folder = task_folder(tmp_path)
    found = rerun_failed_check(folder, display(), real_sandbox, tmp_path / "work")
    assert found["reproduced"] and found["tests"] == 2 and found["failed_total"] == 1
    assert found["failed"][0]["test"] == "test_mul" and found["failed"][0]["result"] == "FAIL"
    assert "AssertionError: 5 != 6" in found["failed"][0]["message"]
    assert found["summary"].startswith("1 of 2 hidden tests did not pass")
    report = (folder / DIAGNOSIS_FILE).read_text()
    assert "test_mul: FAIL" in report and "Traceback" in report
    assert (folder / CHECK_FILES_DIR / "hidden_acceptance" / "test_calc.py").read_bytes() == HIDDEN_TEST
    assert found["check_files"] == ["hidden_acceptance/test_calc.py"]
    assert list((tmp_path / "work").iterdir()) == []  # the rerun's copy of the workspace is gone


def test_real_rerun_of_an_answer_that_now_passes_says_so(tmp_path, real_sandbox):
    folder = task_folder(tmp_path)
    (folder / PATCH_FILE).write_bytes(PATCH + b"@@ -5,2 +5,2 @@\n def mul(a, b):\n-    return a + b\n+    return a * b\n")
    found = rerun_failed_check(folder, display(), real_sandbox, tmp_path)
    assert not found["reproduced"] and "did not happen again" in found["summary"]

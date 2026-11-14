"""The diff check for mistakes with an exact signature that hidden tests caught before (lint.py)."""

from __future__ import annotations

import dataclasses

import pytest

from rlvr.neurons import lint
from rlvr.neurons.demo_miner import DemoMiner
from tests.test_acceptance import RUN, ScriptedSandbox, documented_task, result, run_loop
from tests.test_demo_miner import completion_for
from tests.test_demo_miner_workbench import DONE, FINISHED, Scripted, edit, workspace  # noqa: F401

CORE = (
    "package core\n\n"
    "type CodeError struct {\n\tCode int\n\tMsg  string\n}\n\n"
    "func (ce CodeError) Error() string { return ce.Msg }\n\n"
    "type PtrErr struct{}\n\nfunc (e *PtrErr) Error() string { return \"p\" }\n\n"
    "type ValueOnly struct{}\n\nfunc (e ValueOnly) Error() string { return \"v\" }\n"
)
SEND = "package channel\n\nfunc send() error {\n\treturn &larkcore.CodeError{Code: 1, Msg: \"x\"}\n}\n"


def project(root):
    (root / "core").mkdir()
    (root / "core" / "apiresp.go").write_text(CORE)
    (root / "channel").mkdir()
    (root / "channel" / "send.go").write_text(SEND)
    (root / "vendor" / "x").mkdir(parents=True)
    (root / "vendor" / "x" / "e.go").write_text("package x\n\ntype VErr struct{}\n\nfunc (e VErr) Error() string { return \"\" }\n\nvar _ = &VErr{}\n")
    return root


def test_only_value_receiver_error_types_the_project_builds_with_a_pointer_are_checked(tmp_path):
    types = lint.pointer_error_types(project(tmp_path))
    # PtrErr has a pointer receiver (a value would not compile as an error); ValueOnly is never built with &;
    # vendored code is not the project's.
    assert list(types) == ["CodeError"]
    assert types["CodeError"] == 'channel/send.go: return &larkcore.CodeError{Code: 1, Msg: "x"}'


@pytest.mark.parametrize(
    "lines, found",
    [
        # The two live failures of 2026-11-11 (9b572d481b23 and ace15dab231c).
        (["return resp.CodeError"], 1),
        (["e.Cause = resp.CodeError"], 1),
        (["return nil, resp.CodeError"], 1),
        (["Cause: resp.CodeError,"], 1),
        (['return larkcore.CodeError{Code: 1, Msg: "x"}'], 1),
        (["ce := resp.CodeError", "return ce"], 1),
        (["err := resp.CodeError", "return nil, err"], 1),
        # What the passing answers wrote, and reads of the value that are not errors.
        (["ce := resp.CodeError", "return &ce"], 0),
        (["ce := resp.CodeError", "e.Cause = &ce"], 0),
        (["return &resp.CodeError"], 0),
        (["return nil, &larkcore.CodeError{Code: 1}"], 0),
        (["x := resp.CodeError.Code"], 0),
        (["if resp.CodeError.Code != 0 {"], 0),
        (["// return resp.CodeError"], 0),
        (["ce := resp.CodeError", "ce = other()", "return ce"], 0),
    ],
)
def test_a_value_returned_or_stored_as_an_error_is_found(lines, found):
    diff = "+++ b/service/x/a.go\n" + "".join(f"+\t{line}\n" for line in lines)
    assert len(lint.value_uses(diff, {"CodeError": "core/x.go"})) == found
    # Test files and other languages are not the answer's error paths.
    assert lint.value_uses(diff.replace("a.go", "a_test.go"), {"CodeError": "core/x.go"}) == []
    assert lint.value_uses(diff.replace("a.go", "a.py"), {"CodeError": "core/x.go"}) == []


def test_documentation_that_asks_for_the_value_turns_the_check_off():
    diff = "+++ b/service/x/a.go\n+\treturn resp.CodeError\n"
    types = {"CodeError": "core/x.go: return &CodeError{}"}
    assert len(lint.check(diff, types, {"d.md": "the embedded `resp.CodeError`, **not wrapped**"})) == 1
    assert lint.check(diff, types, {"d.md": "Return the CodeError by value, not a pointer."}) == []


def test_the_prompt_shows_the_line_the_projects_own_form_and_the_fix():
    finding = lint.Finding("service/lingo/catalog/sync.go", "return resp.CodeError", "CodeError")
    text = lint.prompt([finding], {"CodeError": "channel/send.go: return &larkcore.CodeError{Code: 1}"})
    assert "service/lingo/catalog/sync.go: `return resp.CodeError`" in text
    assert "channel/send.go: return &larkcore.CodeError{Code: 1}" in text
    assert "ce := resp.CodeError; return &ce" in text and "run your tests again" in text
    assert "model" not in text.lower()  # harness prompts never describe another model call


@pytest.fixture
def go_project(workspace):
    project(workspace)
    (workspace / "service").mkdir()
    (workspace / "service" / "a.go").write_text("package service\n\nfunc fetch() error {\n\treturn nil\n}\n")
    return workspace


async def no_suite(self, request, docs, deadline):
    return None, None


def go_task():
    request = documented_task()
    return request.model_copy(update={"identity": request.identity.model_copy(update={"primary_language": "go"})})


async def test_a_value_goes_back_once_with_the_evidence_and_the_fixed_version_passes(go_project, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    sandbox = ScriptedSandbox(result(0), result(0))
    provider = Scripted(
        edit("service/a.go", "\treturn nil\n", "\treturn resp.CodeError\n"), RUN, FINISHED,
        edit("service/a.go", "\treturn resp.CodeError\n", "\tce := resp.CodeError\n\treturn &ce\n"), RUN, FINISHED,
        DONE,
    )
    _, submission, notes = await run_loop(go_project, provider, sandbox, request=go_task())
    sent_back = provider.seen[3][-1]["content"]
    assert sent_back.startswith("A check of your change found an error of the project's type `CodeError`")
    assert "service/a.go: `return resp.CodeError`" in sent_back
    assert b"return &ce" in submission
    assert provider.seen[6][-1]["content"].startswith("Before this is submitted")
    assert notes.lint == "passed" and notes.lint_findings == 1


async def test_nothing_is_checked_for_other_languages_or_with_the_setting_off(go_project, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    script = (edit("service/a.go", "\treturn nil\n", "\treturn resp.CodeError\n"), RUN, FINISHED, DONE)
    _, _, notes = await run_loop(go_project, Scripted(*script), ScriptedSandbox(result(0)))  # a Python task
    assert notes.lint == ""
    _, _, notes = await run_loop(
        go_project, Scripted(*script), ScriptedSandbox(result(0)), request=go_task(), miner_lint=False,
    )
    assert notes.lint == ""


@pytest.mark.parametrize(
    "lines, found",
    [
        (["func f(kind string, ce larkcore.CodeError) *E {", "e.Cause = &ce"], 0),
        (["func f(kind string, ce larkcore.CodeError) *E {", "e.Cause = ce"], 1),
        (["func f(ce larkcore.CodeError) error {", "return ce"], 1),
        (["func f(ce *larkcore.CodeError) error {", "return ce"], 0),
        (["var cev larkcore.CodeError", "if errors.As(err, &cev) {", "return cev"], 1),
        (["var cep *larkcore.CodeError", "return cep"], 0),
        # A value parameter in one function does not taint a pointer of the same name in the next.
        (["func f(ce larkcore.CodeError) int {", "return ce.Code", "}", "func g(ce *larkcore.CodeError) error {", "return ce"], 0),
    ],
)
def test_parameters_and_declarations_of_the_value_type_are_followed(lines, found):
    diff = "+++ b/service/x/a.go\n" + "".join(f"+{line}\n" for line in lines)
    assert len(lint.value_uses(diff, {"CodeError": "core/x.go"})) == found


ERRORS = (
    "class CassetteError(Exception):\n"
    "    def __init__(self, code, position=None):\n"
    "        super().__init__(code)\n        self.code, self.position = code, position\n\n\n"
    "class Store:\n    def put(self):\n        pass\n"
)


@pytest.mark.parametrize(
    "added, docs, found",
    [
        # Some sites pass the object's own position, others none: the bare sites are reported, across files.
        ((("pkg/store.py", ['raise CassetteError("closed", position=self._position)', 'raise CassetteError("read_only")']),
          ("pkg/cassette.py", ['raise CassetteError("limit")'])), {}, ['pkg/store.py: `raise CassetteError("read_only")`', 'pkg/cassette.py: `raise CassetteError("limit")`']),
        # A keyword the documentation gives as None for some code is per code, so nothing is reported.
        ((("pkg/store.py", ['raise CassetteError("closed", position=self._position)', 'raise CassetteError("read_only")']),),
         {"d.md": "`'read_only'` with position=None"}, []),
        # A site that passes another keyword says something on purpose; test files are never read.
        ((("pkg/store.py", ['raise CassetteError("closed", position=self._position)', 'raise CassetteError("x", position=None)']),
          ("tests/test_store.py", ['raise CassetteError("y")'])), {}, []),
        # No site passes the object's own state: nothing to compare against.
        ((("pkg/store.py", ['raise CassetteError("closed")', 'raise CassetteError("read_only")']),), {}, []),
    ],
)
def test_error_attributes_set_at_some_raise_sites_only_are_found(added, docs, found):
    diff = "".join(f"+++ b/{path}\n" + "".join(f"+        {line}\n" for line in lines) for path, lines in added)
    gaps = lint.attribute_gaps(diff, docs)
    assert [site for gap in gaps for site in gap.without] == found


def test_the_attribute_prompt_asks_for_the_change_only_where_the_documentation_defines_it():
    gaps = lint.attribute_gaps('+++ b/a.py\n+ raise E("a", line=self._line)\n+ raise E("b")\n', {})
    text = lint.attribute_prompt(gaps)
    assert text.startswith("A check of your change found error attributes set at some raise sites only.")
    assert "`E` pass `line=`" in text and 'a.py: `raise E("b")`' in text
    assert "If the documentation defines the attribute for the class as a whole" in text
    assert "otherwise leave these as they are" in text
    for word in ("model", "catalogue", "list everything"):
        assert word not in text


async def test_a_python_attribute_gap_goes_back_once_and_the_fixed_version_passes(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "errors.py").write_text(ERRORS)
    gap = "        pass\n"
    raises = ('        if self.closed:\n            raise CassetteError("closed", position=self._position)\n'
              '        raise CassetteError("read_only")\n')
    fixed = raises.replace('raise CassetteError("read_only")', 'raise CassetteError("read_only", position=self._position)')
    provider = Scripted(
        edit("pkg/errors.py", gap, raises), RUN, FINISHED,
        edit("pkg/errors.py", raises, fixed), RUN, FINISHED,
        DONE,
    )
    _, submission, notes = await run_loop(workspace, provider, ScriptedSandbox(result(0), result(0)))
    sent_back = provider.seen[3][-1]["content"]
    assert sent_back.startswith("A check of your change found error attributes set at some raise sites only.")
    assert 'pkg/errors.py: `raise CassetteError("read_only")`' in sent_back
    assert b'raise CassetteError("read_only", position=self._position)' in submission
    assert notes.lint == "passed" and notes.lint_findings == 1


@pytest.mark.parametrize(
    "diff, found",
    [
        ('+++ b/pex/cli/commands/lockgraph.py\n+        parser.add_argument("--resolve", type=int)\n'
         '+        parser._negative_number_matcher = re.compile(r"^-[^=\\s]*=")\n', 1),
        ('+++ b/pkg/cli.py\n+class P(argparse.ArgumentParser):\n+    def f(self, s):\n+        return super()._parse_optional(s)\n', 1),
        # Not argparse's state: the file's own helpers, a comment, a file without argparse, a test file.
        ('+++ b/pkg/cli.py\n+import argparse\n+    self._option_string_actions\n+def _parse_optional(self, s):\n'
         '+    # parser._negative_number_matcher\n', 0),
        ('+++ b/pkg/w.py\n+    widget._get_values()\n+    validator._check_value(x)\n', 0),
        ('+++ b/tests/test_cli.py\n+import argparse\n+        parser._negative_number_matcher = 1\n', 0),
    ],
)
def test_argparse_private_state_is_found_only_in_files_that_use_argparse(diff, found):
    assert len(lint.argparse_private_uses(diff)) == found


def test_the_argparse_prompt_names_the_site_and_keeps_argparses_own_reading():
    text = lint.argparse_prompt(["pex/cli/x.py: `parser._negative_number_matcher = m`"])
    assert text.startswith("A check of your change found code that reads or changes argparse's private parser state:")
    assert "- pex/cli/x.py: `parser._negative_number_matcher = m`" in text
    assert "keep argparse's own reading of every option" in text and "run each integer option with -1" in text
    for word in ("model", "catalogue"):
        assert word not in text


async def test_an_argparse_private_change_goes_back_once(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "cli.py").write_text("import argparse\n\n\ndef build():\n    parser = argparse.ArgumentParser()\n    return parser\n")
    old = "    return parser\n"
    hacked = "    parser.add_argument('--resolve', type=int)\n    parser._negative_number_matcher = None\n    return parser\n"
    clean = "    parser.add_argument('--resolve', type=int)\n    return parser\n"
    provider = Scripted(
        edit("pkg/cli.py", old, hacked), RUN, FINISHED,
        edit("pkg/cli.py", hacked, clean), RUN, FINISHED,
        DONE,
    )
    _, submission, notes = await run_loop(workspace, provider, ScriptedSandbox(result(0), result(0)))
    assert provider.seen[3][-1]["content"].startswith("A check of your change found code that reads or changes argparse")
    assert b"_negative_number_matcher" not in submission
    assert notes.lint == "passed" and notes.lint_findings == 1


# The diff questions (lint.questions): the documentation sentences below are quoted from the recorded tasks.
TAPES_DOC = {
    "docs/scenario-tapes.md": (  # 966efbb1635f
        "* The query is split on `&`; empty components are dropped, so a bare `?`\n"
        "  disappears. A component with a `;` is refused. Keys and values are\n"
        "  percent-decoded exactly once (`+` becomes a space), must be valid UTF-8 and\n"
        "  are re-encoded with uppercase escapes. The pairs are sorted by key, then by\n"
    )
}
FIXTURES_DOC = {
    "docs/e2e-fixture-workspaces.md": (  # 8724e1fd451e
        "* `editQuery(app, legacyPath)` returns `'?' + 'file=' +\n"
        "  encodeURIComponent(fixturePath(app, legacyPath))` — the **full absolute\n"
        "  path, encoded exactly once**. Replacement text is taken literally; backslash escapes are kept.\n"
    )
}
CASCADE_DOC = {
    "web/howto/budgeted_classification.md": (  # 0af90493237d
        "  `normalize_distribution` (a foreign label, an invalid `samples()` result, a\n"
        "  negative or non-finite weight, or a distribution without positive weight).\n"
    )
}


def added(path, *lines):
    return f"--- a/{path}\n+++ b/{path}\n@@ -1,1 +1,{len(lines)} @@\n" + "".join(f"+{line}\n" for line in lines)


@pytest.mark.parametrize(
    "path, line, docs, kind",
    [
        # The four recorded lines, each the defect of a failed answer.
        ("scenario_key.go", "\t\tsb.WriteString(url.QueryEscape(p.k))", TAPES_DOC, "form_encoder"),
        ("client-e2e/e2e/app_server/request_contract_journal.py", "class _RenderedReport(str):", {}, "hybrid_type"),
        ("lib/replay-journal.js", "        const { first = '0' } = options(opt);", {}, "js_destructure_default"),
        ("nltk/classify/cascade_model.py", "    total = math.fsum(weights)", CASCADE_DOC, "fsum"),
        # The other re-encoding forms.
        ("pkg/sign.py", "    query = quote_plus(value)", TAPES_DOC, "form_encoder"),
        ("pkg/sign.py", "    query = urlencode(pairs)", TAPES_DOC, "form_encoder"),
        ("src/url.mjs", "  return new URLSearchParams(pairs).toString();", TAPES_DOC, "form_encoder"),
        ("src/url.ts", "  const q = String(new URLSearchParams(pairs));", TAPES_DOC, "form_encoder"),
        ("src/url.js", "  const query = `${new URLSearchParams(pairs)}`;", TAPES_DOC, "form_encoder"),
        ("pkg/types.py", "class Token(bytes):", {}, "hybrid_type"),
        ("lib/opts.cjs", "  let { limit = 10, name } = settings;", {}, "js_destructure_default"),
    ],
)
def test_the_recorded_constructs_are_asked_about(path, line, docs, kind):
    found = lint.questions(added(path, line), docs)
    assert found == [lint.Question(kind, path, line.strip())]


@pytest.mark.parametrize(
    "path, line, docs",
    [
        # 966efbb1635f's right diff wraps the encoder, and c6c48b966bb1 (both diffs) only reads a query.
        ("scenario_key.go", '\treturn strings.ReplaceAll(url.QueryEscape(s), "+", "%20")', TAPES_DOC),
        ("src/atlas.js", "\tconst useAtlas = new URLSearchParams( window.location.search ).get( 'atlas' ) === '1';", TAPES_DOC),
        # A comment that names the encoder is not a call.
        ("scenario_key.go", "// %HH escapes. A space is written as %20: url.QueryEscape escapes a literal", TAPES_DOC),
        # 8724e1fd451e's documentation names encodeURIComponent; others name the function or a form encoding.
        ("src/edit.js", "  return '?' + new URLSearchParams({file}).toString();", FIXTURES_DOC),
        ("scenario_key.go", "\t\tsb.WriteString(url.QueryEscape(p.k))", {"d.md": "Values are escaped with url.QueryEscape."}),
        ("pkg/form.py", "    body = urlencode(pairs)", {"d.md": "The body is application/x-www-form-urlencoded with escapes."}),
        ("pkg/form.py", "    body = quote_plus(text)", {"d.md": "Values are percent-encoded, with + for a space."}),
        ("pkg/form.py", "    body = quote_plus(text)", {"d.md": "Values are percent-encoded and a space is written as `+`."}),
        # Documentation that never speaks of escapes or percent-encoding leaves the encoder alone.
        ("scenario_key.go", "\t\tsb.WriteString(url.QueryEscape(p.k))", {"d.md": "Keys are sorted."}),
        ("scenario_key.go", "\t\tsb.WriteString(url.QueryEscape(p.k))", {}),
        # math.fsum only where the documentation speaks of finite values or overflow.
        ("pkg/cascade.py", "    total = math.fsum(weights)", {"d.md": "Weights are summed."}),
        # Several bases (str, Enum) is an ordinary enum, and the patterns keep to their languages.
        ("pkg/kinds.py", "class Kind(str, Enum):", {}),
        ("lib/kinds.js", "class Kind(str):", {}),
        ("pkg/opts.py", "    const { first = '0' } = options(opt)", {}),
        ("lib/opts.js", "  const { first } = options(opt);", {}),
        ("lib/opts.js", "  if (a == b) { first = 1 }", {}),
        # Test files are never the answer's code.
        ("tests/test_report.py", "class Fake(str):", {}),
        ("pkg/report_test.py", "class Fake(str):", {}),
        ("conftest.py", "class Fake(str):", {}),
        ("lib/journal.test.js", "  const { first = '0' } = options(opt);", {}),
        ("lib/__tests__/journal.js", "  const { first = '0' } = options(opt);", {}),
        ("scenario_key_test.go", "\t\tsb.WriteString(url.QueryEscape(p.k))", TAPES_DOC),
    ],
)
def test_reads_wrapped_encoders_documented_forms_and_test_files_stay_quiet(path, line, docs):
    assert lint.questions(added(path, line), docs) == []


def test_the_decoding_sentence_of_966efbb1635f_does_not_count_as_a_form_encoding():
    # "(`+` becomes a space)" is how the input is decoded; the output still takes uppercase escapes.
    assert not lint._FORM_ENCODING.search(TAPES_DOC["docs/scenario-tapes.md"])
    assert lint._ESCAPE_TALK.search(TAPES_DOC["docs/scenario-tapes.md"])


@pytest.mark.parametrize(
    "sentence",
    [
        "Query values are re-encoded with uppercase escapes, never + for a space.",
        "Do not use url.QueryEscape here: values take uppercase %HH escapes.",
        "Values take uppercase escapes rather than\n  the + for a space of a form encoder.",
    ],
)
def test_a_sentence_that_rejects_the_form_encoding_does_not_silence_the_question(sentence):
    line = "\t\tsb.WriteString(url.QueryEscape(p.k))"
    assert [q.kind for q in lint.questions(added("scenario_key.go", line), {"d.md": sentence})] == ["form_encoder"]


@pytest.mark.parametrize(
    "path, line, docs, asked",
    [
        # A written escape is percent-encoding talk, though the words "escape" and "percent" never appear.
        ("scenario_key.go", "\t\tsb.WriteString(url.QueryEscape(p.k))",
         {"d.md": "A space is written as `%20` and a literal `+` as `%2B`."}, True),
        ("scenario_key.go", "\t\tsb.WriteString(url.QueryEscape(p.k))", {"d.md": "A 20% share is %d of the total."}, False),
        # A line that already writes a space as %20, in any language, is not asked about.
        ("src/url.js", "  return new URLSearchParams(pairs).toString().replace(/\\+/g, '%20');", TAPES_DOC, False),
        ("pkg/sign.py", "    query = urlencode(pairs, quote_via=quote)", TAPES_DOC, False),
        ("pkg/sign.py", "    query = urlencode(pairs, quote_via=urllib.parse.quote)", TAPES_DOC, False),
        ("pkg/sign.py", "    query = urlencode(pairs, quote_via=quote_plus)", TAPES_DOC, True),
        # A contraction or "without" rejects a mention too; a comma ends the rejecting clause.
        ("scenario_key.go", "\t\tsb.WriteString(url.QueryEscape(p.k))",
         {"d.md": "Don't use url.QueryEscape: values take uppercase escapes."}, True),
        ("scenario_key.go", "\t\tsb.WriteString(url.QueryEscape(p.k))",
         {"d.md": "Values are escaped without the + for a space of a form."}, True),
        ("pkg/form.py", "    body = quote_plus(text)", {"d.md": "Values are percent-encoded with no padding, + for a space."}, False),
        ("scenario_key.go", "\t\tsb.WriteString(url.QueryEscape(p.k))",
         {"d.md": "Keys are not sorted, and values are escaped with url.QueryEscape."}, False),
        # Prose the answer edits is not its code.
        ("README.md", "    total = math.fsum(weights)", CASCADE_DOC, False),
        ("docs/usage.rst", "    query = quote_plus(value)", TAPES_DOC, False),
    ],
)
def test_handled_spaces_written_escapes_rejecting_clauses_and_prose(path, line, docs, asked):
    assert bool(lint.questions(added(path, line), docs)) is asked


def test_a_long_line_is_not_read_and_costs_no_time():
    import time

    # The destructuring pattern backtracks steeply on a long line with many defaults and no closing brace.
    long = "  let {" + "x = 1 " * 330  # about 2,000 characters: 0.8 s each before the length limit
    started = time.monotonic()
    assert lint.questions(added("dist/bundle.js", long, long), {}) == []
    assert time.monotonic() - started < 0.5
    edge = "  const { first = '0' } = options(opt);"
    edge += " " * (lint.MAX_QUESTION_LINE - len(edge))
    assert [q.kind for q in lint.questions(added("lib/a.js", edge), {})] == ["js_destructure_default"]
    assert lint.questions(added("lib/a.js", edge + " "), {}) == []


def test_the_prompt_names_the_destructured_key_and_every_builtin_base():
    found = lint.questions(added("lib/a.js", "  const { first: start = '0' } = options(opt);"), {})
    text = lint.question_prompt(found)
    assert "`{first: undefined}`" in text and "`first` left out" in text and "start: undefined" not in text
    found = lint.questions(added("pkg/a.py", "class A(str):", "class B(dict):", "class C(str):"), {})
    assert "subclasses the builtin `str` or `dict` here" in lint.question_prompt(found)


def test_each_line_is_asked_about_once_and_removed_or_deleted_lines_are_not_read():
    diff = (
        added("scenario_key.go", "\tsb.WriteString(url.QueryEscape(p.k))", "\tsb.WriteString(url.QueryEscape(p.v))",
              "\tsb.WriteString(url.QueryEscape(p.k))")
        + "--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-class Old(str):\n"
        + "--- a/keep.py\n+++ b/keep.py\n@@ -1 +1 @@\n-class Gone(str):\n+class Kept:\n"
    )
    assert [q.line for q in lint.questions(diff, TAPES_DOC)] == [
        "sb.WriteString(url.QueryEscape(p.k))", "sb.WriteString(url.QueryEscape(p.v))",
    ]


def test_the_question_prompt_names_each_line_and_the_input_that_decides_it():
    found = (
        lint.questions(added("scenario_key.go", "\tsb.WriteString(url.QueryEscape(p.k))"), TAPES_DOC)
        + lint.questions(added("pkg/journal.py", "class _RenderedReport(str):"), {})
        + lint.questions(added("lib/replay-journal.js", "const { first = '0' } = options(opt);"), {})
        + lint.questions(added("pkg/cascade.py", "    total = math.fsum(weights)"), CASCADE_DOC)
    )
    assert [q.kind for q in found] == ["form_encoder", "hybrid_type", "js_destructure_default", "fsum"]
    text = lint.question_prompt(found)
    assert text.startswith("Questions from a check of your change, each about one line and the one input")
    assert "- scenario_key.go: `sb.WriteString(url.QueryEscape(p.k))`" in text
    assert "a space and a literal +" in text
    assert "- pkg/journal.py: `class _RenderedReport(str):`" in text and "the builtin `str`" in text
    assert "type() and ==" in text
    assert "`{first: undefined}`" in text and "`first` left out" in text
    assert "[1e308, 1e308]" in text
    assert "change code only where a sentence or example of the documentation requires" in text
    assert "keep the code as it is" in text and text.endswith("reply with a one-line summary.")
    # It quotes the code, never anything said about it, and describes no other call (adjudicate.py:10-15).
    for word in ("model", "catalogue", "list everything", "you said", "your summary", "assum"):
        assert word not in text.lower()
    # Only the kinds found are asked about, at most three lines each.
    many = [lint.Question("fsum", "a.py", f"x{i} = math.fsum(v)") for i in range(5)]
    text = lint.question_prompt(many)
    assert text.count("math.fsum sums here") == 1 and "x2 = " in text and "x3 = " not in text
    assert "form encoder" not in text


HYBRID = "class Report(str):\n    pass\n\n\ndef b():\n    return Report('x')\n"


async def test_a_question_goes_back_without_a_sandbox_and_the_solve_finishes(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = Scripted(edit("src/b.py", "def b():\n    return 1\n", HYBRID), FINISHED, FINISHED, DONE)
    _, submission, notes = await run_loop(workspace, provider, None)
    assert provider.seen[2][-1]["content"].startswith("Questions from a check of your change")
    assert provider.seen[3][-1]["content"].startswith("Before this is submitted")
    assert b"class Report(str):" in submission and notes.diff_questions == "hybrid_type"


async def test_a_question_goes_back_once_per_kind_and_the_code_may_stay(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = Scripted(
        edit("src/b.py", "def b():\n    return 1\n", HYBRID), RUN, FINISHED,
        # The answer keeps the class and changes something else: a new version, but the kind was asked.
        edit("src/b.py", "Report('x')", "Report('y')"), RUN, FINISHED,
        DONE,
    )
    _, submission, notes = await run_loop(workspace, provider, ScriptedSandbox(result(0), result(0)))
    asked = provider.seen[3][-1]["content"]
    assert asked.startswith("Questions from a check of your change")
    assert "src/b.py: `class Report(str):`" in asked
    assert sum(m[-1]["content"].startswith("Questions from a check") for m in provider.seen) == 1
    assert provider.seen[6][-1]["content"].startswith("Before this is submitted")
    assert b"class Report(str):" in submission and b"Report('y')" in submission
    assert notes.diff_questions == "hybrid_type"
    assert notes.fields()["diff_questions"] == "hybrid_type" and notes.lint_findings == 0


async def test_a_go_finding_and_a_question_go_back_in_one_message(go_project, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    (go_project / "docs").mkdir()
    (go_project / "docs" / "widgets.md").write_text(TAPES_DOC["docs/scenario-tapes.md"])
    bad = "\tq := url.QueryEscape(k)\n\treturn resp.CodeError\n"
    fixed = "\tq := url.QueryEscape(k)\n\tce := resp.CodeError\n\treturn &ce\n"
    provider = Scripted(
        edit("service/a.go", "\treturn nil\n", bad), RUN, FINISHED,
        edit("service/a.go", bad, fixed), RUN, FINISHED,
        DONE,
    )
    _, submission, notes = await run_loop(go_project, provider, ScriptedSandbox(result(0), result(0)), request=go_task())
    sent_back = provider.seen[3][-1]["content"]
    assert sent_back.startswith("A check of your change found an error of the project's type `CodeError`")
    finding, questions = sent_back.split("\n\nQuestions from a check of your change", 1)
    # The closing rule is about the questions only, never the finding before them.
    assert "These questions are not findings: change code only where" in questions
    assert "service/a.go: `return resp.CodeError`" in finding
    assert "service/a.go: `q := url.QueryEscape(k)`" in questions
    # The fixed version still holds the encoder, which is not asked about again, and goes to the review.
    assert provider.seen[6][-1]["content"].startswith("Before this is submitted")
    assert b"return &ce" in submission
    assert notes.lint == "passed" and notes.lint_findings == 1 and notes.diff_questions == "form_encoder"


async def test_the_questions_share_the_budget_with_the_attribute_check(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "errors.py").write_text(ERRORS)
    raises = ('        if self.closed:\n            raise CassetteError("closed", position=self._position)\n'
              '        raise CassetteError("read_only")\n')
    provider = Scripted(
        edit("pkg/errors.py", "        pass\n", raises), edit("src/b.py", "def b():\n    return 1\n", HYBRID), RUN, FINISHED,
        FINISHED,  # no change: the same version goes on to the attribute check
        edit("src/b.py", "Report('x')", "Report('y')"), RUN, FINISHED,  # the budget of two is spent
        DONE,
    )
    _, _, notes = await run_loop(workspace, provider, ScriptedSandbox(result(0), result(0)))
    assert provider.seen[4][-1]["content"].startswith("Questions from a check of your change")
    assert provider.seen[5][-1]["content"].startswith("A check of your change found error attributes")
    assert provider.seen[8][-1]["content"].startswith("Before this is submitted")
    assert notes.diff_questions == "hybrid_type" and notes.lint == "1 found" and notes.lint_findings == 1


# Second merge (first-deploy-2): the live argparse check of 00:03 (Nov 14) and step 3's questions both run on a
# Python answer, each in its own message, under the one budget of lint.MAX_PROMPTS.
async def test_the_questions_and_the_argparse_check_both_go_back_under_the_shared_budget(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "cli.py").write_text("import argparse\n\n\ndef build():\n    parser = argparse.ArgumentParser()\n    return parser\n")
    hacked = "    parser.add_argument('--resolve', type=int)\n    parser._negative_number_matcher = None\n    return parser\n"
    clean = "    parser.add_argument('--resolve', type=int)\n    return parser\n"
    provider = Scripted(
        edit("pkg/cli.py", "    return parser\n", hacked), edit("src/b.py", "def b():\n    return 1\n", HYBRID), RUN, FINISHED,
        FINISHED,  # no change: the same version goes on to the argparse check
        edit("pkg/cli.py", hacked, clean), RUN, FINISHED,  # the budget of two is spent
        DONE,
    )
    _, submission, notes = await run_loop(workspace, provider, ScriptedSandbox(result(0), result(0)))
    question, argparse_check = provider.seen[4][-1]["content"], provider.seen[5][-1]["content"]
    assert question.startswith("Questions from a check of your change") and "src/b.py: `class Report(str):`" in question
    assert "argparse" not in question
    assert argparse_check.startswith("A check of your change found code that reads or changes argparse")
    assert "pkg/cli.py: `parser._negative_number_matcher = None`" in argparse_check
    assert provider.seen[8][-1]["content"].startswith("Before this is submitted")
    assert b"_negative_number_matcher" not in submission and b"class Report(str):" in submission
    assert notes.diff_questions == "hybrid_type" and notes.lint == "argparse: 1 found" and notes.lint_findings == 1


async def test_no_question_with_the_setting_off(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    script = (edit("src/b.py", "def b():\n    return 1\n", HYBRID), RUN, FINISHED, DONE)
    provider = Scripted(*script)
    _, _, notes = await run_loop(workspace, provider, ScriptedSandbox(result(0)), miner_lint=False)
    assert notes.diff_questions == "" and provider.seen[3][-1]["content"].startswith("Before this is submitted")


async def test_a_refused_question_is_withdrawn_and_not_asked_again(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    refused = dataclasses.replace(
        completion_for(""), reasoning="", finish_reason="content_filter",
        refusal="This request was blocked as it seems to violate Anthropic's Terms of Service",
    )
    provider = Scripted(edit("src/b.py", "def b():\n    return 1\n", HYBRID), RUN, FINISHED, refused, DONE)
    _, submission, notes = await run_loop(workspace, provider, ScriptedSandbox(result(0)))
    question, review = provider.seen[3], provider.seen[4]
    assert question[-1]["content"].startswith("Questions from a check of your change")
    # The question and the finished reply it followed are gone; the review follows that reply instead.
    assert review[-1]["content"].startswith("Before this is submitted") and len(review) == len(question)
    assert review[-2]["content"] == FINISHED.output
    assert notes.refused == "diff check" and notes.diff_questions == "hybrid_type"
    assert b"class Report(str):" in submission

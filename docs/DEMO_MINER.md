# OpenRouter demo miner

The demo miner implements the V3 signed miner protocol. It downloads and verifies
the leased workspace, gives the model read-only file access, uploads the resulting
patch or script and its canonical trajectory, then signs the exact response body.
It never receives the verifier.

The model can list directories and read files in pages using `workspace` tool
requests. Each model turn and file read is recorded in the trajectory. Reads are
limited to 32 KiB per call and directory listings to 200 entries per page. The
miner does not execute repository code outside the sandbox described below.
Temporary workspace files are removed after generation; with a journal, a
repository task's downloaded archive is kept until its verdict, so a failed check
can be rerun (see "Measuring before changing the model").

`MINER_MAX_WORKSPACE_TOOL_CALLS` defaults to 80, enough for the rule-by-rule test below to fit inside a task's time. A call over the budget is not run; the model is asked for its submission instead.
Workspace preparation and all model turns share the solve deadline, reserving
`OPENROUTER_UPLOAD_RESERVE_S` for uploads. More file reads can increase model cost and
solve latency. The miner needs local disk space for the compressed archive, its
expanded tar stream, and extracted files for each concurrent solve.


## Try a real task locally

Two tasks the problem server issued once and retired live under
`tests/fixtures/`, one of each kind. They let you check your whole setup
before mining:

- `v3-yamlcpp-e6225d04`, a repository task: yaml-cpp with a private feature,
  two planted defects, six hidden checks and a reference fix. The answer is a
  unified diff.
- `v3-terminal-urlooker-651460ec`, a terminal task: a pinned Go checkout plus
  incident material, six hidden checks and a reference script. The answer is
  a bash script that runs in the workspace and leaves the required results.

```
python scripts/try_task.py extract /path/to/work      # workspace + instruction
# produce a unified diff against that workspace with your own miner
python scripts/try_task.py grade /path/to/fix.diff    # graded in the pinned sandbox

# the terminal task
python scripts/try_task.py --fixture tests/fixtures/v3-terminal-urlooker-651460ec extract /path/to/work
python scripts/try_task.py --fixture tests/fixtures/v3-terminal-urlooker-651460ec grade /path/to/solve.sh
```

Grading needs Docker and the pinned sandbox image, pulled with
`docker pull "$(python -c 'from rlvr.policy import RELEASE_POLICY; print(RELEASE_POLICY.v3_image)')"`.
The output is the validator's verdict: status, reason code, every check's
outcome, and for a failed check the same display a failure notice carries.
`v3-yamlcpp-e6225d04/reference.diff` passes all six checks; applying only its
first file fails at the fourth. `v3-terminal-urlooker-651460ec/reference.sh`
passes all six; an empty script fails at the first, and the reference script
followed by a change to the recovery inputs fails at the sixth.

## Setup

Requirements:

- Python 3.10–3.12
- a funded hotkey registered on the subnet
- an OpenRouter API key with credits (a normal key; a management key cannot make model calls)
- a public TCP port reachable by validators

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[chain,miner]'
cp .env.example .env
```

Set:

```dotenv
OPENROUTER_API_KEY=<your-key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=anthropic/claude-sonnet-5.5

NETUID=<subnet-id>
SUBTENSOR_NETWORK=test
WALLET_NAME=miner
WALLET_HOTKEY=default
AXON_PORT=8091
```

Keep the API key only in `.env` or the process environment; `.env` is ignored by
Git. Credits are shared by every key on an OpenRouter account, so give this miner
its own key with a credit limit to cap what it can spend.

Start the miner:

```bash
./start_demo_miner.sh
```

The default authorization accepts registered validator hotkeys with a validator
permit. `MINER_MIN_STAKE`, `MINER_MAX_CONCURRENT_REQUESTS`, and
`MINER_REQUIRE_VALIDATOR_PERMIT` control who can spend the model account's
quota.

## How a solve works

The model works through tools rather than writing its answer blind. It reads and
searches the workspace, then makes changes with `edit_file` and `write_file` on a
private copy. Git writes the unified diff from those edits, so a repository task
never depends on the model getting diff syntax right. A terminal task keeps its
Bash script at `submission.sh` in the same way.

The prompts set a fixed order of work. Before writing code, the model writes out a
rules checklist: every rule the documentation states, each error code and exception
type, each "must", "rejects" and "only", the order in which checks run, what must stay
unchanged on failure, the error table line by line, and where case-folding and naming
rules apply, including comparisons against the filesystem. Before finishing, it writes
a table-driven throwaway test from `run_command` with one case per documented rule,
the coverage the hidden tests have (a bool where an int is required, a bytearray where
bytes is required, a key present with the value `undefined` or `None` where only an
absent key may default, each limit at and past its value, each error code, a failure
mid-stream, a name differing only in case on the target, every documented example),
prints how many cases ran, and fixes every check that does not produce the documented
error; while the test passes and time remains, it adds the rules the test still skips.
The same test builds the largest inputs the
documentation allows and times every public operation at that size, with updates
interleaved with reads and, where the documentation promises concurrent use, a short
stress run (`go test -race` in Go): the hidden tests may use those sizes and have about
two minutes in total, compilation included, so an answer that rebuilds an index or
copies a whole collection on every call times out. Edge cases are read strictly, because
the hidden tests treat an input at the edge of a rule as a violation of it (empty after
decoding, whitespace-only, a key present but `undefined`, a selector that fails its
grammar on a missing target), and a precedence rule is tested with an input that breaks
both rules at once. The review then goes back through the documentation's error and
validation sections and confirms each rule against the code, lists the readings the
model chose where the documentation could be read two ways and tests the stricter one,
and any edit it makes is followed by another run of the test.

A model that writes an answer also writes its tests from its own reading of the rules, so
a rule it misreads passes its tests and fails the hidden one. With `MINER_ACCEPTANCE_SUITE`
(on by default), a second model call therefore writes an acceptance suite from the task
text and the documentation it names, never seeing the answer, in the style of the hidden
suites seen so far: one strict case per rule, compound tests named after the rules,
precedence tested with inputs that break two rules at once, the stricter reading at every
edge. The call starts when the task arrives and runs beside the answer; when the answer is
first finished, the harness places the suite in the sandbox with a heredoc command and runs
it (`acceptance-0` and `acceptance-1` in the trajectory, which also records the author's
model turn). A failing run goes back to the model to settle each case against the
documentation, with neither side presumed right: the model keeps its code where a passage
supports it and changes it where the documentation requires the suite's expectation and
nothing supports its own. Until 2026-11-11 the message said the suite was usually right; but in
16 of the 26 graded solves whose suite failed, the hidden tests passed, and a deferred-to
reviewer had just broken a correct JSON Patch answer the same way. The suite is skipped when the
task names no documentation, when less than four minutes remain, or when the author's call
is still running ninety seconds after the answer is finished; the journal's work line shows
`independent suite passed`, `failed (failures=2)` or why it did not run. Where the suite follows
a documented example and the answer follows the model's reading of a rule, the example wins:
an example is the reference implementation's literal output, never a typo.

Since 2026-11-13 the suite is not sent back once and forgotten. Each failing case has to be
settled: the model either changes its code until the case passes, or keeps its code and writes
`REFUTED: <case> | "<passage>"`, quoting the documentation word for word. The harness checks
each quotation against the task text and the documentation it names, with Markdown marks,
quote styles, line breaks and case set aside and `...` allowed between parts; a paraphrase or a
fragment of a few words settles nothing and is named back as such. After every reply the
harness runs the suite again on the answer as it stands (`acceptance-2`, `acceptance-3`, ... in
the trajectory) and sends back the cases still failing that no accepted quotation covers, until
every case is settled, the suite has run five times (`acceptance.MAX_RUNS`), or three quarters of
the solve's time has passed (`acceptance.SETTLE_FRACTION`, about 660 of 880 seconds); then the
review follows, whatever is still open. A run that fails without naming a case (a build or
import error, a crash, a time-out) cannot be refuted; the model fixes its code or repairs the
suite file with a command, and the same such failure twice in a row ends the loop, since nothing
the model did reached it. Go suites run with `go test -vet=off`, because vet's printf warning in a
file nobody touched once failed a suite's build, and the model's own test runs keep vet. The reason: on 2026-11-13 the median solve ended at about 250 of 900
seconds, the suite was still red at submission on 11 of the 24 tasks the top operator passed and
we failed, and the top solvers work to about ten minutes. Letting the model refute a case with a
quotation, rather than having another reader overrule it, matters because the suite fails on
about half of correct answers, and two overrulings by other readers had broken correct answers.
The author also tests only rules a sentence or an example states, copies each sentence into a
`RULE:` comment above its case, and begins with an interface test that uses every documented
name with its documented signature (in Go, assignments such as `var _ func(string) error =
pkg.Check`, so a different signature fails the build). A copy of the suite written into the
answer is taken out before the review. The journal records the runs, the final state (`passed`,
`settled`, `open`, `unreadable`) and the refutations accepted and refused, and the work line
reads `suite ran 3 times, settled at the end, 2 case(s) refuted from the documentation`.

Three further mechanisms since 2026-11-13, after the public honeminer design (Claude Code running
Opus 5.5, which re-checks the work on clean copies whenever the model tries to stop, keeps the best
answer so far and gives time warnings), with the miner now on `anthropic/claude-opus-5.5`:

- A final check before the answer goes out. The working copy is reset to the repository with only
  the answer applied, the state the grader starts from (`Workbench.reset_to_answer`; a suite file
  inside the repository is put back), and the suite runs once more. If it does worse than this same
  code did in the working copy, or worse than an earlier version did, the model is told once,
  with both results, that a recent change broke something or that the answer relies on something
  only the working copy has. Runs are compared only when the suite file is the same: every run
  prints the suite file's sha256 first, so a suite the model repaired starts a fresh comparison.
- The best answer so far. Every suite run records the version it ran on and its score (every
  case passing, every failing case refuted, fewer open cases, and last a run that named no
  case). When the final check still finds an earlier version better after the model had its
  chance, or when a solve has to stop early (wrap-up, cost cap, tool budget, a model error), that
  earlier version's patch is what goes out. The journal's `kept_best` names it.
- Time checks. Once half and once three quarters of the time before the wrap-up has passed, a
  short note after the next tool batch says how many seconds are left.

`MINER_ACCEPTANCE_RUNS=1` turns the final check off together with the settling loop.

The model also tends to notice the points where it chose a reading, and lists them in its
summary as choices the documentation left open; three of the four reasoning failures since the
sandbox fix were on such a point. With `MINER_ADJUDICATOR` (on by default), the loop asks the
model before the review which questions it would put to the documentation's author, each with
the answer its code assumes; a second call that sees the task text, the documentation and the
questions, never the code, answers each as the hidden tests will (a stated sentence first, then
an example, then the project's behaviour, else the stricter reading), and every answer that
differs from the model's goes back to be applied, with the instruction that only a quoted
sentence of the documentation may override it. `MINER_ADJUDICATOR_MODEL` names the reader; it
defaults to `anthropic/claude-opus-5.5`, a stronger reader than the answer's model at a small
cost because its input is only the documentation. The journal's work line shows
`open readings: 3 readings, 1 overruled`.

The wording of those two prompts is deliberate. Anthropic's hosts refuse, as reverse
engineering of model outputs, a prompt that asks the model to catalogue its own earlier
statements or that describes its output being read and ruled on by another model call; the
first wording lost the review of ten of fifteen repository solves that way (`finish_reason`
`content_filter`, nothing generated). The loop now treats such a refusal of a prompt it added
as a withdrawn prompt: the stage counts as done and the solve goes on to the next one, and the
journal's work line shows `the host refused the open points prompt, withdrawn`. A refusal of
the model's own work is retried once, then the work so far is submitted (`model_refused`).

Three Python answers of 2026-11-10/11 failed every hidden test at the first import: a package
directory without `__init__.py`, a package `__init__` without the documented re-exports, an entry
module the documentation named that the answer never touched. With `MINER_SURFACE_CHECK` (on by
default), the loop collects the `from X import a, b` lines written in the documentation and runs
them in the sandbox from the repository root whenever a version of the work is finished, with the
workspace's own package roots on `sys.path` (`surface-0`, `surface-1` in the trajectory). A line
whose module lives in the workspace and fails, or whose module imports but lacks a documented
name, goes back to the model with the reason (a namespace package without `__init__.py` is named
as such); a module not found anywhere in the workspace is not held against the answer. The
journal's work line shows `documented imports passed` or `2 failed`.

A dotted name the documentation states for one of those documented names under its top-level package
(`nltk.BigramLedger`, feb50a592a31) is checked too, after importing the documented submodules
beneath it, the way a hidden test reaches it. When the package itself cannot be imported in the
image, the check uses an empty package in its place, as the hidden tests do, so the attribute
counts only if the documented modules set it when they are imported. A sentence that says what is
*not* there is skipped.

Files that only a command created are not part of the answer (`rlvr/neurons/workbench.py`). The
runner reports each new text file a command leaves in the copy (`created_not_in_answer`, with a
note), and the copy keeps it for the next commands. When the model finishes while the copy still
holds a source or configuration file of that kind (tests and scratch files aside), it is sent back
once to write each one it needs with `write_file` (journal `left_out`). Two answers were lost this
way: 4dc6113b5960 wrote its documented Go package with `cat >` (all 101 hidden tests failed to
build; with the file written in, 91 pass) and db84cf7b56bf lost its package's `__init__.py` (with
the files written in, the whole hidden check passes).

The grader's image has only the standard library, so a project whose top-level package `__init__`
imports a dependency cannot be imported plainly there. Its hidden tests put an empty module in
place of the top-level package and import the documented submodules beneath it (marimo,
2026-11-11, whose `__init__` needs msgspec). When a plain import fails in code the answer did not
touch, the check does the same, and it reports only a failure raised in one of the answer's own
files (judged by the innermost frame inside the workspace) or a missing module of the project
itself. Before this, the marimo check sent back four false failures and the model began rewriting
`marimo/__init__.py`. A documented name that is a submodule counts as present even when the
package does not import it, because `from package import name` imports `package.name` on demand.
At 07:52 on 2026-11-11 the check had sent `from marimo._runtime.runner import cell_runner` back
twice as missing, and the model rewrote the runner package's `__init__.py` for nothing.

Two cost points from the logs of 849 warm loop calls (fit to 0.1%): Sonnet's cache reads cost
$0.10, cache writes $2.50 and output $10.00 per million tokens. Output is half of the loop's cost,
cache writes nearly a third. Every token a tool returns is written once at 25 times the read price,
and `read_file` returns 77% of those bytes. So the system prompts ask for independent calls in
one turn, several reads at once, because each turn rereads the whole conversation.

Commands run on a copy of the workspace, and only the edit tools write the answer. On 2026-11-11
a solve applied the adjudicator's corrections with a Python script run as a command. Its tests
passed on the copy, the submitted diff had none of the corrections, and three hidden tests failed
that the corrections would have passed (a replay of the recorded calls confirms it). On a
repository task the runner now compares the copy with the answer after every command. A changed
file that the answer already changes is taken into the answer, read without following links, and
reported as `answer_updated`. A file put back to its original content, as `git checkout` or
`git stash` would, is not taken, and neither is a change from a command that was cut short. Any
other change to a file of the repository is reported as `not_in_answer` and undone before the
next command, so every test runs on what will be submitted. New files and build trees
(`node_modules`, `target`, `__pycache__` and the like) are left alone. The journal's work line
counts both kinds. The file tools also accept a path written the way commands see it
(`/work/src/x.py` or `./src/x.py`), which three solves had sent and had refused as absolute.

`scripts/replay_solve.py data/solves/tasks/<folder> <output dir>` replays every tool call a kept
solve made, in order, through the current workbench on that task's original workspace, then
reruns the validator's hidden check on the resulting answer. The model's choices stay fixed, so
it shows what the same actions would submit now. Replaying the 20 failed solves that kept their
workspaces found the lost command edits described above in three of them.

Two more rules come from the same failures. A documented size, count or offset is measured
on the input as given (the bytes of a message as fed), because a hidden test can only compute
its expected numbers from the inputs it builds, and limits are tested at the exact size of a
known input, accepted at that size and rejected at one less. And a repository answer the model
finishes with an edit that nothing ran after is sent back once with the instruction to run its
tests again before it is reviewed; the journal's work line notes `sent back to test an edit it
had not run`.

Pairing the two models (2026-11-11). Sonnet 5.5's tool loop takes about 75% of the money (71 solves:
$41 of $55), and the Opus reader under 3%, so the long loop stays on the cheaper model and the
stronger one reads where reading decides the grade. With `MINER_AUDIT` (on by default) and
`MINER_AUDIT_MODEL` (Opus 5.5), once the readings are settled and before the review, Opus gets
the task text, the documentation and the finished diff, and lists at most twelve concrete
violations (`RULE | CALL | EXPECTED | ACTUAL`). They go back to Sonnet as possible violations to
check by running each call; a finding the documentation does not support stays out, with the
sentence that contradicts it quoted. The evidence comes from the 17 failed repository tasks kept
with their hidden suites. A blind Opus judge decided whether a finding named the defect behind a
failing hidden test. Opus at medium effort named it in 10 of the 17, with about four findings
each, for about $0.22 a review. Sonnet with the same prompt named it in 5 of 17, with about nine
findings each, nearly all about something else. Opus at high effort did worse, because four of its
replies spent the 16k-token limit on reasoning and listed nothing. The journal's work line shows
`audit of the diff: 4 findings`, `no findings`, `auditor failed` or
`not run: too little time left`. Expect about $0.3 to $0.5 more per solve, still about half of
running the whole loop on Opus.

The first live version broke a correct answer. In a JSON Patch task the code raised `move` as
the operation's own description of its steps says. The auditor cited the general precedence list
and asked for `target`, Sonnet deferred to "a senior engineer", and the hidden test wanted `move`.
The answer from before the audit, replayed, passes every hidden test. Since 07:57 UTC on
2026-11-11 the auditor reports a point only when no passage of the documentation supports what
the code does, and an operation's own description of its steps decides for it over a general list.
Sonnet is told the findings may be wrong and keeps its code where a passage supports it. Measured
offline, this costs some recall but removes most of the noise:

| auditor prompt | failed tasks with a real defect named | findings on 19 passed answers |
| --- | --- | --- |
| first | 10 of 17 | 27, on 14 answers |
| conservative | 7 of 17 | 7, on 6 answers |

On the broken JSON Patch answer the conservative prompt made no finding in two runs. The per-solve
cost cap in `.env` went from $1.50 to $3.00 at the same time, because one audited solve that later
passed was cut short at $1.54.

The miner also reads the account's OpenRouter credit every ten minutes (`/credits`, which every
key on the account shares, the SN67 miners' too). It logs what is left and how many hours that
lasts at the recent pace, warns below $25, and writes `credit.json` so LOG.md says it first. On
2026-11-11 the account was found with $12.60 left at about $70 a day. A call refused for lack of
credit is logged as such, because every task fails until the account is topped up.

Three reading rules come from the same marimo task, and the implementer, the acceptance author
and the reader share them. A size counts exactly the unit the documentation names, so a delimiter
it keeps out of a record is not counted. A table of error codes is read row by row and literally:
a code covers only the fields its own row names, and anything else takes the general code. The
tape's table gave `range` to `base`/`seq` only, and the hidden test wanted `schema` for a
negative `delete`. A limit of 0 follows its admission rule like any other value, so "0 disables
the resource" admits nothing. Replayed on that task's own questions, the reader now rules all
three as the hidden tests do.

Two more come from a waiver-ledger task on 2026-11-11. An operation-wide statement that every
failure takes one code ("every validation failure is `checkpoint_invalid`", "never raises
`invalid_input`") decides the code of every check in that operation, including one taken from
another section by reference: the reference brings the conditions, not the code. And
requirements that seem to conflict both hold: receipts carry balances "as of that transition"
and a restored ledger must return the same receipts, which replaying the recorded requests in
order and capturing each value at its own step satisfies. The model had asked about both, the
reader had confirmed the wrong code and upheld only one of the two requirements, and the
audit later upheld the other, so the answer flipped between two failing designs. With the
rules the reader rules both as the hidden tests do, in two runs of two, and the earlier tasks'
key rulings are unchanged. The implementer's prompt now says that checkpoints are tested as
round trips of every public call, and the audit's findings message says a fix must keep every
other requirement and carry a changed value through checkpoint and restore.

The stricter reading has two limits, learned from two Go tasks on 2026-11-11, and every prompt
that teaches it carries them (`adjudicate.STRICT_LIMIT`). A decoder accepts insignificant
whitespace around JSON tokens, after the whole document included, members in any order and
escaped member names, as the standard parser does, unless a rule written for the decoder itself
rejects them. A helpdesk checkpoint decoder's rules were headed "accepted only in its canonical
shape" and the encoder writes compact output; the model first accepted whitespace and any order,
then refused both after the acceptance suite expected a trailing space to fail "at the edge of a
rule", and the reader confirmed it. The hidden tests feed a reordered document with spaces and a
document padded with spaces to exactly the byte limit, and the answer as it stood before the
suite passes all 138 of them. Second, a member the input leaves out still fails, but it is not a
member of the wrong type: it takes the error of the rule its missing value meets. A Lingo
checkpoint record `{}` must fail as a stored record that is invalid (`ErrCheckpoint` and
`ErrProtocol`), because its empty `repo_id` and nil entity meet that rule; the reader had ruled
absent fields a plain schema error, and the answer, which passed that test before the ruling,
failed it after. A broader first wording ("it never invents a rejection") fixed both but,
replayed on the questions of the 18 answers that had passed, turned about 13 strict rulings
lenient, so the limits name only what the hidden tests showed; replayed again, the passed
answers' strict rulings stay.

What the hidden suites check beyond the words of the rules is now in the implementer's prompt
(`IMPLICIT_RULES` in `demo_miner.py`), the suite author's checklist and, for the readings among
them, the reader's prompt (`adjudicate.READING_RULES`). It comes from all 27 kept hidden suites,
read on 2026-11-11, and each clause failed a live answer or recurs in several suites:
collaborators the documentation names are used only through their documented methods and looked
up at call time (a lodash answer kept `_.iteratee` from load time, an lxml answer called `.items()`
on a documented duck-typed element); no shortcut skips validation; fields the documentation ties
together are compared (a record's suffix against its path); inputs are read when the documentation
places them (a fingerprint taken for a whole window up front); a streaming decoder reports what the
bytes it holds prove, such as a present non-zero padding byte, before the truncation of the rest;
callbacks run exactly as documented with fresh copies; exceptions from caller code pass through as
the same object; a diagnostic names the line where the problem is found (a failing include at its
own line, input that ends too early at its last line); the feature is wired into every integration
file, the installer payload included; traversals are iterative; Go sentinels are wrapped as
documented; and `json.loads` leniency (duplicates, NaN, 1e999, a BOM) is checked by hand where the
documentation rejects it. Two wordings were narrowed after replays against passed answers' questions:
"fields that must agree" made the reader reject a combination its documentation leaves alone, and
"an error about a directive names the directive's line" put an unclosed block at its opening line
where the hidden test wants the last line.

Re-solved with the code of 14:16 (`scripts/regression_solve.py`), four of the five most recent
failures passed their own hidden tests: the directory resolver and the Lingo catalog (the pointer
form), the helpdesk decoder (the limits of the strict reading) and JSON Patch (the conservative
audit). The waiver ledger went from two failing hidden tests to one: its restore must reproduce a
receipt whose balance counted a reservation that was later cancelled, from a checkpoint that keeps
only each request's final state, and no documented order of replay does that.

In Go, an error the documentation names by its struct type ("the embedded `resp.CodeError`")
is returned and stored as a pointer to a copy (`ce := resp.CodeError; return &ce`), the form
the larksuite SDK's own code returns; its hidden tests match `err.(*larkcore.CodeError)`. Two
answers that returned the value failed after the reader, which never sees the code, confirmed
the value from the documentation's wording; on earlier larksuite tasks the model had chosen the
pointer and passed. The suite author also writes Go for the language version of `go.mod`
(`interface{}` rather than `any` below go 1.18): two suites failed to build, which cost a round
of reconciliation each.

A value has exactly one type (`adjudicate.ONE_TYPE`, in the implementer's, reviewer's,
reader's and suite author's prompts): the one its signature or attribute table gives it. A
passage on how the value is produced ("rendered with the supplied options") decides what it
holds, never its type. A Python answer (a9a5461324c6, request contracts, 2026-11-11) stored a
violation's `report` as rendered JSON text where the attribute table said "the deterministic
report dict". When the independent suite read it as a dict, the answer became a `str` subclass
that also answered `report["key"]`, and all three readers approved it under the rule that
requirements which seem to conflict both hold. The hidden test compared it with `==` to
`json.loads(render_contract_report(...))`, which no hybrid equals. That rule now covers
behaviour only, and the reconciliation message forbids meeting both sides with a hybrid.
Replayed with the new prompt, all three readers ruled the plain dict. On four passed answers'
questions, the only rulings that changed matched what the live majority had ruled.

Existing Python modules keep their import behaviour (`rlvr/neurons/imports.py`, setting
`MINER_IMPORT_CHECK`, journal field `imports`). The validators' image lacks build artifacts
such as compiled extensions and generated version files, so some projects cannot be
imported there at all. Two answers made them importable anyway, to test new code: msgspec
without `msgspec._core` (7efd1df0229c, 17 of 18 hidden tests failed, because the suite
expected its installed-API tests to skip on exactly that import error) and mpmath without
`mpmath._version`. Before the review, every existing module the answer edits is imported in
the sandbox in a fresh interpreter, once from a link farm of the repository with the edited
files' original contents restored and the answer's new files left out, and once with the
answer. A different outcome goes back to the model once, with the advice to test new code
with a stub in its throwaway test instead. Modules are named by their package chain, so a
package imports under its own name, and a module that failed before only because it needs a
file the answer adds is not compared. Replayed on the 22 kept failed Python answers it
found those two and nothing else.

The readers are off by default since 2026-11-12 (`MINER_ADJUDICATOR=false`). Replaying each
of the 14 failures from 2026-11-11 19:28 to 2026-11-12 17:00 up to just before the rulings
and grading it with the kept hidden tests showed the majority of three Opus readers
overruling Sonnet's correct reading twice: a catalog that passed every hidden test before
its rulings failed 17 after them ("reject unsorted records"), and a trace inspector went
from 4 failing tests to 48 ("checkpoint samples the clock"). They rescued none of the 14,
and tasks on which they overruled anything passed 16 of 29 against 9 of 11. They cost
about $0.33 of a $1.20 solve. The independent suite stays: in the same replays it took the
trace inspector from 49 failing tests to 4.

With `MINER_SANDBOX=true`, the model can also test its work. `run_command` runs in
a disposable copy of the workspace, and `run_script` reruns a terminal script from
a clean copy exactly as the grader will. Both use the validators' pinned image and
limits, with no network, as an unprivileged user, and only that disposable copy is
mounted writable. Nothing a command does can reach the submission. This needs
Docker and the pinned image; without them the miner logs a warning and carries on
without testing. Pull it with:

```bash
docker pull "$(python -c 'from rlvr.policy import RELEASE_POLICY; print(RELEASE_POLICY.v3_image)')"
```

Before submitting, the miner checks the answer on a clean copy, the way the grader
starts: it reruns a terminal script that changed since its last clean run, and runs
the task's own `.rlvr/build.py` when there is one. A failure goes back to the model
to fix. The model also gets one review turn to look for anything it missed.
The answer is never lost to a late stop: when time runs short (`MINER_WRAP_UP_RESERVE_S`
before the limit), when a solve passes `MINER_MAX_SOLVE_COST_USD`, when the tool budget
runs out, or when the model fails, the work done so far is submitted.

A solve may run for `OPENROUTER_REQUEST_TIMEOUT_S`, but always stops in time for
the validator to commit before the lease ends. Workspaces are unpacked in
`MINER_WORK_DIR`, which should be on disk, because `/tmp` is often held in memory.

To use another model, change `OPENROUTER_MODEL`. Claude models get prompt
caching automatically, because they cache only when a request asks for it. Leave
`OPENROUTER_TEMPERATURE` unset unless the model accepts it; current Claude models
reject non-default values.

Reasoning text is recorded when the model returns it. Claude does not return
token log probabilities, so trajectories from the default model contain none,
which the trajectory rules allow. For a model that does return them, set
`OPENROUTER_LOGPROBS=true` to record each selected token and its five
alternatives. A turn whose log probabilities do not fit the trajectory format
records none and the solve continues; the miner never fabricates trajectory
evidence.

## Replacing the miner without losing tasks

An ordinary restart takes the miner off the network for a few seconds, and pm2 kills a
solve that is still running. A task a validator sends in that gap scores zero. Test
changes locally first and deploy them together, with:

```bash
python scripts/swap_miner.py            # start a fresh copy, then retire the running one
python scripts/swap_miner.py --status   # which copy is serving
```

The miner runs as one of two pm2 apps, `sn5-miner-a` and `sn5-miner-b`, which take
turns. Every copy opens its port with `SO_REUSEPORT`, so the fresh copy starts listening
beside the running one. Once the fresh copy answers, pm2 stops the old one: it closes
its port at once, so every new connection goes to the fresh copy, finishes the solves it
is running and then exits. The script turns on `net.ipv4.tcp_migrate_req`, so
connections still queued at the old copy when it closes are handed over rather than
reset. A fresh copy that never answers is removed and the running copy keeps serving.
Afterwards the script saves pm2's process list, so a reboot starts the copy that is
serving. A verdict for a task the other copy answered is still accepted, because the
miner looks the task up in the journal.

The first swap, from a miner still running as the single app `sn5-miner`, cannot share
the port, so it is one ordinary restart. It waits for a quiet moment: no task being
solved, nothing received for three minutes and no verdict due. `scripts/restart_when_idle.py`
makes an ordinary restart at such a moment; `--check` only says whether now is one.

## Several hotkeys

A validator sends each task to a random pool of serving miners at once, so an operator
with several registered hotkeys often receives the same task on more than one of them
within seconds. The validator grades byte-identical submissions once and gives every miner
that sent them the same verdict ("it changes no score", `rlvr/v3/round.py`), and the release
policy pays every pass the same whatever its latency (`payment_speed_floor` 1.0). So with
`MINER_SHARED_DIR` set, the model's work is done once (`rlvr/neurons/sharing.py`):

- the first process to receive a task claims it (a file created with `O_EXCL` in the shared
  directory, keyed by the task id and the digests of its workspace and verifier) and solves
  it; the others wait for its answer;
- each process still writes its own trajectory, with its own hotkey and challenge, uploads
  to its own UID's slots and signs its response with its own hotkey; only the model's work,
  the submission and the record of how it was made, is shared;
- if the solving process fails or dies, a waiter with at least five minutes left claims the
  task and solves it itself;
- everything is kept for 30 minutes after its claim, so a task seen again later is solved
  afresh rather than answered from an old round.

A process that reused an answer records `ended_by: shared`, no model calls and the hotkey
that did the work (`shared_from`; the work line says "answer reused from the solve by hotkey
..."). Its verdict is the same as the solver's, and a failed check is rerun from the solver's
journal, which kept the workspace.

One process runs per hotkey, each on its own port and with its own journal, because two
processes answering the same round would otherwise write task folders of the same name.
`.env` holds what all of them share, including `MINER_SHARED_DIR=data/shared`; a small file
per extra hotkey, named by `MINER_ENV_FILE`, holds what differs, and both
`start_demo_miner.sh` and the swap script read it after `.env`:

```bash
# .env.alon2, for a second registered hotkey
WALLET_HOTKEY=alon2
AXON_PORT=8094
MINER_RECORD_DIR=data/solves-alon2
MINER_WORK_DIR=workspaces-alon2
```

```bash
MINER_ENV_FILE=.env.alon2 .venv/bin/python scripts/swap_miner.py --name sn5-alon2
MINER_ENV_FILE=.env.alon2 .venv/bin/python scripts/swap_miner.py --name sn5-alon2 --status
.venv/bin/python scripts/solve_report.py --dir data/solves-alon2
```

Each process serves its own axon (the host's address and its own port) at startup, so open
each port in the firewall, and register each hotkey on netuid 5 first: an unregistered hotkey
stops at startup. The first hotkey keeps its plain `.env` and its pm2 names (`sn5-miner-a`
and `-b`); it shares too once `MINER_SHARED_DIR` is set and it has been swapped.

Two caveats. Sharing saves a model run only when two of these hotkeys receive the same
task; each UID is drawn into a task's pool independently. And `rlvr/scoring/payment.py`
notes that duplication-resistant scoring may come in a later version: identical answers and
trajectories across hotkeys are what such scoring, or a sampled trajectory check, would see.

## API cost

Every model call logs the host that served it, its tokens and what OpenRouter
charged, and every solve logs a total, including solves that fail:

```text
[demo-miner] solve spend: task 1a2b3c4d5e6f7a8b calls=7 input=157578 cached=109770 (70%) written=47808 output=13009 hosts=Anthropic:7 cost=$0.2606
```

OpenRouter picks the host and falls back to another when one fails. Each solve
sends one `session_id`, which keeps its turns on one host while that host's
prompt cache is warm. A `cached` share far below 60% suggests the cache is not
working.

A reply with `finish_reason` `length` reached `OPENROUTER_MAX_TOKENS`, which on Claude
includes reasoning; the loop keeps its text and asks the model to continue in smaller steps,
twice at most. `content_filter` means the host refused the request (Anthropic reports
`native_finish_reason` `refusal`); see the adjudicator section for how the loop goes on.

`OPENROUTER_REQUEST_TIMEOUT_S` caps the whole solve.

Where a solve's money went, measured on the 24 live solves from 05:22 to 14:40 on 2026-11-11
($1.17 a solve): Sonnet's loop 43% (output 48% of it, cache writes 34%, cache reads 18%; Sonnet
charges $2.50 per million tokens written to the cache, $0.10 read from it and $10 of output), the
Opus reader 13%, the Opus audit 13%, the Sonnet suite author 13%, and the review, the rulings'
application, the reconciliation and the listing the rest. The loop's output is mostly the code and
the throwaway tests themselves (full rewrites of a file are negligible), and file reads carry about a
quarter of its input cost, so it has little waste to cut without cutting the work. The audit, which
found nothing in 10 of 13 live solves, is off by default (`MINER_AUDIT=false`); its $0.20 pays for the
three readers' vote (the first reader writes the documentation to the prompt cache, the other two
read it for about $0.09 each, almost all of it their output).

Below `MINER_ECONOMY_BELOW_USD` (default $15) of credit, as the watchdog last read it, solves leave out
the suite's author and the second and third readers, about $0.35 a solve, so the credit lasts until a
top-up: a task the miner cannot answer at all scores zero. Paid experiments (`scripts/regression_solve.py`,
reader replays) draw on the same credit: on 2026-11-11 they spent more than the live miner did between
the morning top-up and 14:40.

## Measuring before changing the model

With `MINER_RECORD_DIR` set, the miner keeps a journal there. It records each
solve's cost, time, tool use and how it ended, and every verdict validators send
back, including the failing check's output. Validators send a verdict for every
graded task unless they turn notices off.

Two files in that directory list every task, and the miner rewrites both after every
task, verdict and rerun:

- `LOG.md` starts with a summary: the pass rate, the spend in total, per task and
  over the last 24 hours, and results by kind of task and by validator. Then it has
  one entry per task, newest first: what the task asked, its kind and language, the
  validator, the verdict and its cause, why it failed, how the solve went (time,
  model calls, tool use, failed commands, the review), what it cost in dollars and
  tokens, and which files the answer changed.
- `tasks.csv` has the same facts, one row per task, for a spreadsheet.

Each task also gets a folder under `tasks/` with its instruction (`instruction.md`),
the answer (`answer.diff` or `answer.sh`), the trajectory and, for a failure, the
failing check (`failed_check.txt`). A repository task's workspace is kept there until
its verdict. When a hidden check fails, the miner reruns that check on its own answer
in the validators' image, the way the validator ran it, because a notice shows only
what the check required, never what the answer printed. The rerun shows which hidden
tests failed and with what error: `diagnosis.txt` in the folder has the details, and the
check's own test files are copied to `check_files/`. A terminal task's checks are never
shown, so its failures cannot be rerun. Verdicts for tasks answered before a restart are
still accepted, because the miner rebuilds its list of answered tasks from the journal.

Summarise the journal from the repository root:

```bash
python scripts/solve_report.py           # also rewrites LOG.md and tasks.csv
python scripts/solve_report.py --rerun   # reruns failed checks the miner has not rerun yet
python scripts/solve_report.py --since 2026-11-10T09:51:00Z   # only tasks received since then
```

Use `--since` to measure from a deploy that changed how the miner works. One case
to know: until the fix deployed on 2026-11-10, the live sandbox could not read the
edit tree (the extracted workspace root is private to the miner, and the sandbox
runs as another user), so every command ran on a copy with the edited files
missing and the model never tested its real code. Tasks from before that fix say
nothing about the model. The miner now runs one real edit-and-command round at
startup and logs `sandbox: ready; commands see the edit tools' work`; a
`WARNING: sandbox self-check failed` line means the same problem is back.

The report gives the pass rate with a 95% range, the spend per solve and per pass,
and each failure sorted by cause:

- **reasoning:** a hidden check failed after a normal finish;
- **stopped early:** time, the spending cap, the tool budget or a model error ended
  the solve;
- **harness:** the answer was rejected before any check ran;
- **miner error:** the miner itself failed and returned no answer;
- **grading limits:** the answer hit the grader's time, memory or output limit, which
  is nearly always an answer that is too slow or too hungry at the documented limits
  (the hidden tests get about 120 s, compilation included), not a validator problem;
- **build failed:** the task's setup step failed.

Reasoning and grading-limit failures are the ones a better loop or a stronger model
could fix. A hundred graded tasks
pin a pass rate near 90% to within about six points either way. That is enough to
find the common causes of failure, but not to tell 92% from 95%.

To compare models before paying for one live, run the real miner on the retired
sample tasks. It records to `data/solves-local` and grades each answer with the
validators' code:

```bash
python scripts/try_miner.py --runs 3
OPENROUTER_MODEL=anthropic/claude-opus-5.5 python scripts/try_miner.py --runs 3
```

Two sample tasks make a smoke test, not a measurement.

### Regression runs on the kept failures

Every failed repository verdict leaves its task, its workspace and the validator's whole hidden
check in the journal, so a failure can be solved again with the current code and graded by the
very tests that failed it. That is the test that a fix holds on the failure it was made for, and
that a later change does not break one fixed earlier. No task has ever been sent twice, so the
point is the fix, not the task.

```bash
.venv/bin/python scripts/regression_solve.py data/solves/tasks/*a6da5e0782c4* data/solves/tasks/*ace15dab231c*
.venv/bin/python scripts/regression_solve.py --jobs 2 data/solves/tasks/<folder> ...
```

Each run sends the kept task to a fresh `DemoMiner` with the same identity and lease, sends the
uploads nowhere, turns answer sharing off, records to `data/solves-regression`, and reruns the kept
check on the new answer in the validators' image. `data/solves-regression/runs/<folder>-<time>/`
keeps the answer and `result.json` (the grade, the failing hidden tests, the cost, the stages and a
digest of the solver code that ran). A run costs about $1 to $2 and as long as a live solve;
two at a time share the machine with the live miner, whose containers keep their own CPU limit.
One run of a task is one sample: a pass shows that the fix can hold, not that it always will.

### Checks of the diff and the reader's vote

`rlvr/neurons/lint.py` checks every finished version of a Go answer for a mistake with an exact
signature: an error type the project returns as a pointer (`&CodeError{...}` in its own code,
with value-receiver methods, so the value compiles as an error too) returned or stored as a value.
A finding goes back with the line, the project's own form and the fix; the documentation can turn
it off by asking for the value. It finds the two live failures of 2026-11-11 and nothing in the
passed larksuite answers. `MINER_LINT=false` turns it off; the journal's `lint` column says what
it found.

The same check asks four questions of the added lines in every language (`lint.questions`), each
the defect of one failed answer and absent from every passed answer and right diff on record: a
form encoder used to re-encode (where the documentation speaks of escapes and names neither that
function nor a form encoding), a single-base subclass of a builtin, a JavaScript destructuring
default on an options object, and `math.fsum` (where the documentation speaks of finite values or
overflow). Each kind is asked once per solve, in the Go check's message when both fire, and shares
its budget of two; the journal's `diff_questions` column lists the kinds asked. Retire a pattern
the first time it fires on a passed answer.

A Go suite written with `any` is spelled with `interface{}` before it runs when the module's
`go.mod` is older than go 1.18 (`acceptance.fit_go_suite`), outside strings and comments.

Three readers rule on the model's open questions (`MINER_ADJUDICATOR_VOTES=3`): the first at once,
the other two six seconds later so they read the documentation from the prompt cache, and the
majority decides each question; a tie keeps the model's own reading, and a reader that fails or is
refused drops out. A single reader answered the same question differently from run to run (the
waiver task's restore code: right in two runs of three), and on the questions of answers that had
passed it overruled a few unrelated points in every run, different ones each time.

### Rules from the 2026-11-13 analysis of 30 failures

A workflow diagnosed every failure since the refusal fix (and the three tasks Opus 5.5 still failed in the
trial), proposed harness changes and had three skeptics check each against the saved hidden tests. What
survived, in `adjudicate.py` and used by the solver, the suite author and the settle rules:

- `EXAMPLE_RULE` replaces "the example wins, an example is never a typo": the passage that speaks to the disputed
  case more directly decides, an output shown without its call fixes only its format, and an example is wrong only
  for the case it contradicts (0689bdd7377c: Opus wrote the right total, then followed a wrong example).
- `LIFECYCLE_RULES` (solver only): eager state transitions, counted callbacks kept in a finally, documented side
  effects on every successful call, injected collaborators never called by snapshots or getters, collections
  validated whole before per-item calls (d91e4c5496ae, ee10b6a11bb4, d1b2cac3a687, 015ff5c79492, 0af90493237d).
- `ERROR_SCOPE`: an attribute defined for an error class as a whole is set on every code (5fbaf671fe6a, 13 of 29
  tests), and a state-named code is reported only for a value in that state.
- `EDGE_RULES` (author and settle rules): the solver's edge readings, so the suite does not assert the opposite.
- The command-line rule follows what the documentation puts on stdout and stderr for each outcome (c448520a0ec1).
- `surface.py` no longer blames the answer for a project module with no source anywhere (a compiled extension)
  imported by untouched code (7efd1df0229c).

Refuted and not built: a deviation-sentence prompt, re-running the suite as first written, interpreter parity for
the suite, and a second gap suite.

### Fixes from the review of 2026-11-13 and the first UID 83 failure

An adversarial review of the day's harness changes confirmed 28 defects, fixed together: every suite command
takes its own trajectory call id (a failed run used to leave its number to the next, and a duplicate id loses
the answer); each run prints its failure lines again after its output (`acceptance.FAIL_LIST_MARK`), so the tail
the workbench keeps always names every failing case; a failed unittest fixture, a Go panic, a run in which no
case ran and a missing suite file can neither pass nor be settled; a refutation covers only the cases it names
(a Go parent expands into the subtests failing when it was written) and is read in every reply, numbered or in
inline code; the final check reruns the working copy when the version or the refutations changed, compares it
with earlier versions of the same suite file and the clean copy with the working copy, ignores runs cut short,
offers the refutation route, and sends an earlier version only when the final one no longer builds or loads;
stored versions never contain the suite file. From UID 83's first failure (ed00cb107823, a tuple turned into a
list), ONE_TYPE now keeps a copy of loaded data in its loaded types, and both the solver and the suite author
compare such copies with the repository's own files.

### Cost and accuracy, measured 2026-11-13

An Opus 5.5 repository solve costs about $1.7: output 36% (new files, reasoning at effort high, self-written
test scripts), the suite author's single call 26%, cache writes 25%, cache reads 14%. A cost analysis over 14
Opus trajectories found no large saving that keeps accuracy: every proposal recomputed to $0.002-0.04 a solve.
It found one bug: `search_text` on a single file returned no hits (grep printed no file name without `-H`), which
sent the model to read whole 32 KiB pages; fixed. On the 10 hard tasks UID 39's operator passed and we failed,
the full harness with Opus passed 7 for $17.65 (0689b now passes under EXAMPLE_RULE). A first-finish coverage
question (`MINER_COVERAGE_PROMPT`) passed 2 of 5 like the baseline at 19% more cost, so it is off.

A grammar stated once for a kind of value (9fd0ec887882, 2026-11-13): the header grammar was stated under the
profile's header config and the record's header patch was typed only as a patch object; the hidden test checked the
grammar there too. EDGE_RULES, the solver's checklist and the suite author's patterns now carry such a grammar to
every input that holds a value of that kind, with that input's own error code and pointer.

**Fresh state per case and percent-encoding (2026-11-13).** A Go task failed because the suite author put several records in one shared default lane, so the documented head-of-line rule decided a case written for another rule, and the solver bent correct lane code to make it pass. The suite author is now told to give each case, and each table row that changes state, a fresh instance from one builder helper, to give records the case is not about their own key, lane, group or quota, and to test an interaction between two rules only where a RULE: sentence states it. The same task also used Go's url.QueryEscape, which writes a space as +, where the documentation asked for re-encoding with escapes. EDGE_RULES and the solver's edge block now say that percent-encoding writes every byte outside the unreserved set as an uppercase %HH escape, so a space is %20 and a + is %2B, and that standard form encoders are not that encoding unless the documentation names them.

**Fallback chains (2026-11-13).** A Python task documented a context's route segment as "info_name, or command.name when there is no info_name", plus an error "when neither supplies a nonempty, NUL-free string". The solver read that as trying each source in turn, so an empty info_name fell back to the command's name, while the hidden test expected the configuration error. `adjudicate.FALLBACK_RULE` now says a fallback named for a missing value is taken only when the value is not given (left out, or passed as its declared default; a key present in a mapping is always given), and a given value that breaks its own rule is rejected with its documented error. A value goes to the fallback only where a sentence or example names that state ("absent or empty defaults to", "X if non-empty, else Y") or the project's own code reads that same parameter with a truthiness test. The rule is part of EDGE_RULES, so the suite author and the settle rules get it, and it sits in the solver prompt next to the absent-key rule.

**Terminal tasks on their own model (2026-11-13).** `MINER_TERMINAL_MODEL` names the model for terminal_script_v1 solves; empty keeps `OPENROUTER_MODEL` for every task type. alon runs terminal tasks on anthropic/claude-sonnet-5.5 and repository tasks on Opus 5.5. Sonnet passed 24 of 24 terminal tasks (Nov 10–13, mean $0.22) under the same prompts, while Opus passed 4 of 5 at $0.42. Terminal tasks run no suite author, reader or audit, so the solver loop is their only model call. `DemoMiner.solver_model()` decides the model in one place: the solver call (which names a model only when it differs from the default), the trajectory's model name and the journal's `model` column. `/health` still reports `OPENROUTER_MODEL`.

**Rules mined from the 50 kept hidden suites (2026-11-13).** These were found before any live failure, by reading every kept hidden suite for behaviours the solver wasn't told about. Each rule was then checked by two skeptics for evidence and for any opposite-direction test among the 50. `adjudicate.GRAMMAR_RULE`: a documented grammar or number form is checked whole and exactly, because the suites probe the nearest value outside it. Python's `re.match` with `$`, `\d`, `int()`, `strip()` and case folding are lenient, and so are JavaScript's `Number()` and `trim()` and Go's `strconv.ParseInt` and `strings.TrimSpace`, unless the documentation names the helper or the project's own parser. `adjudicate.SECTION_RULE`: an operation's own section and its steps decide the order of its checks and the code of each defect over a general table, and a no-op sentence never skips the checks listed before it. `adjudicate.SHAPE_RULE`: a result shown only by its part names is a plain dict keyed by those names (a tuple only where an example unpacks it), open-layout items are stored whole, and non-default options survive a checkpoint. IMPLICIT_RULES now says documented classes normalise their own constructor arguments. The run prompt now asks for a probe on what the project's own producers emit. These rules sit in the solver prompt only; the test writer is maintained separately.

**Python error attributes set at some raise sites only (2026-11-13).** `lint.attribute_gaps` reads the finished diff of a Python answer (non-test files only) and groups the added raise sites of each error class across files. Some sites may pass a keyword taken from the raising object's own state (`position=self._position`) while others pass no keyword at all. In that case the bare sites go back once, with the condition spelled out: if the documentation defines the attribute for the class as a whole, set it on every code through one helper; otherwise leave them. A keyword the documentation gives as None is skipped. The check costs no model call and shares the Go diff check's limit of two prompts per solve. It comes from 5fbaf671fe6a, which lost 13 of 29 hidden tests to a position set at only four of ten sites.

**Public functions defined by a formula (2026-11-13).** A Python task defined its public `canonical_channel()` only as `normalize(...)` and stated the 1..64 bound in a constants table. The answer enforced the bound only in a private wrapper the parsers used, but the hidden test called the public function directly. IMPLICIT_RULES now says the tests call each documented public function directly. A function defined by a formula or library call still follows the module-wide error rule, and its own module's bounds for the values it takes or returns, counted on the canonical result when the bound names that form. Other entry points reach the check by calling the function, each keeping its own error code. Predicates still answer False, Go functions without an error result keep their plain result, and a sentence written for that function decides.

**Wrappers, text spellings, object strings and argparse state (2026-11-14).** Four fixes from two live failures, each passed by two skeptics:
- The precedence sentence now covers a wrapper the documentation writes as `x.commit(build(x, ...))`: Python looks up `x.commit` before `build` runs, so bind the inner result first and test the wrapper with each wrong-type argument.
- The 18:21 checklist sentence is narrowed: a spelling the documentation gives for how one text format writes a non-text value (a decimal id in CSV) belongs to that format's parser alone, and a record passed directly takes the value's own type rule.
- `VALUE_STRING_RULE`: "the X string" of a value the project represents as an object is that object's own string conversion (pex's `Version` prints a pinned 1.0 as 1), unless the documentation names another form.
- `lint.argparse_private_uses` sends back added lines that change argparse's private parser state, such as `_negative_number_matcher`, which turned `--resolve -1` into a usage error instead of the documented error.

**Go error kinds on the chain (2026-11-14).** Where the documentation maps a kind of error to a code ("cancellation and deadline errors become canceled"), the Go solver now finds that kind with `errors.Is` or `errors.As` on each error's chain, including errors a collaborator returns, not only `ctx.Err()`. The exception is documentation that decides by where an error came from: there a collaborator's error keeps its documented handling. This came from 4dc6113b5960, whose five subtests would have failed this way had its build not failed first.

**Deduplicate before the work (2026-11-14).** Where an operation removes duplicate items, keeping the first, IMPLICIT_RULES now says the deduplication comes before each item's work. Every item is still validated and keyed in order, but the engine or collaborator its result comes from runs once per distinct key, and a later duplicate reuses the result or is dropped; duplicates are not sent through it and filtered out afterwards. Documented exceptions decide ("sink still runs", "filtering happens before duplicate suppression"). This came from a Pex routing report whose hidden test counted the calls to the selection engine with a recorder.

**Steps are the mechanism (2026-11-14).** `STEPS_RULE` follows EXAMPLE_RULE in the solver prompt. Steps and definitions that say how a value is computed or which entries form a group are the mechanism. A clause that only gives a reason or a consequence ("and therefore the same lanes") adds no grouping or propagation they don't perform. An example only such an added mechanism could produce is wrong. It comes from the second mining round over six newer hidden suites, where two answers had added per-key lanes and source-to-target extras that the defining steps did not have.

**Live A/B of the repository solver (2026-11-14).** `MINER_REPOSITORY_ALT_MODEL` splits repository tasks between two models by a hash of the task id (`repository_arm`). Odd ids run on the alternate model, the rest on `OPENROUTER_MODEL`, so both arms see the same task mix and a retry stays in its arm. Only the solver's calls change: the suite author keeps `OPENROUTER_MODEL` in both arms. The journal's `model` column records the arm. alon runs Sonnet 5.5 against Opus 5.5 for about two days. Empty disables the split.

**The probe phase (2026-11-14, off by default).** `MINER_PROBE_PHASE` (`off`, `shadow`, `alternate`, `live`) sends one timed message after the review. It asks the model to add the discriminating variant of each probe it already ran, from six families: the same bad value through every other carrier, same key with everything else different, comparisons against the documented source and non-default round trips, present-but-invalid optional inputs, recording stand-ins and numeric extremes, and printed runtime types with an independent mini-encoder. The governing rule comes first: change code only where a sentence or example decides against it. Each change must be marked `CHANGED: <file> | "<sentence>"`; quotations are checked word for word, record-only, with one request for missing ones and no revert by the harness. In shadow mode the answer from before the phase goes out and the probed version is kept for grading. A forced stop inside the phase sends the version from before it. `alternate` splits tasks by a hash independent of the repository model A/B. It stays off until a paired shadow gate shows rescues without breaks.

**A dependency named like a project file (2026-11-14).** The documented-imports check counts a missing module as the project's when its source is somewhere in the workspace. For a top-level name that search now applies only to dotted names: piccolo's piccolo/utils/pydantic.py made the missing pydantic dependency look like a project module, so a documented import that failed in untouched code was held against the answer twice, and the import check sent back the workaround in between. A missing project module that the answer's own file imports is still reported.

**The interpreter the hidden tests name (2026-11-14).** Every sandbox command runs with PATH=/usr/local/bin:/usr/bin:/bin (rlvr/v3/supervisor.py), so `python3` has only the standard library, while /opt/rlvr-venv/bin/python in the same image also has pydantic 2, httpx, anyio, cryptography and a few more. Some hidden checks run with the venv interpreter, and only the task's own text can say so. When the instruction or the documentation names /opt/rlvr-venv/bin/python, the documented-imports check and the import check run with it (`surface.interpreter`), falling back to python3 if the sandbox lacks it; otherwise both commands are unchanged.

**An unknown id passed to a call is reported as passed (2026-11-14).** `adjudicate.ERROR_ID_RULE` follows ERROR_SCOPE in the solver prompt only. An error about an element that a call names by a well-formed id or key argument carries that id, also when no element has it, unless a sentence, table row or example gives that code another value; a general clause giving no id to faults of the whole input or the run does not cover such a call. It comes from a live failure (121c13873690, `resolve_argv(plan, 'absent', ...)` with no step id) and three kept suites that document the same convention. A wider first draft was rejected: ten suites expect an empty attribute for elements of the input that no call argument names.

**A byte order mark is one only where a document starts (2026-11-14).** The solver's json.loads sentence now says that a byte order mark is U+FEFF at the start of the input, or at the start of a line only where each line is its own document (JSON Lines), and that U+FEFF anywhere else is ordinary content. A decoder read "a line has a BOM" as the code point anywhere and rejected it inside a JSON string (4c763391ef17); three other suites accept it there, and a CSV suite keeps it at a later row's start as text.

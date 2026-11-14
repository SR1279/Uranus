"""JavaScript files an answer adds or edits must still parse with the project's own bundler chain.

d45cbc99a6dc (2026-11-14, a sandbox library bundled with browserify 16): the answer wrote a private class method
(`#request (event, args) {`) in lib/sandbox/datasets.js. Node runs that, so the model's own tests and the
independent suite passed. The project, though, bundles lib/sandbox when a sandbox is created, and the bundler's
parser (detective, on acorn-node) rejects the syntax: "Parsing file /work/lib/sandbox/datasets.js: Unexpected
token (54:4)". The hidden tests reach the feature through the package's entry point, so all 16 failed in their
set-up, on a task 90% of the other miners passed.

The check runs the parsers the project itself uses, where the sandbox has them, on every JavaScript file the
answer touches:

- browserify's (detective from module-deps), when package.json lists browserify or a tool built on it;
- Babel with the project's own configuration, when @babel/core and a configuration file are present.

A rejection counts only when the same parser accepts the repository's own version of that file, or, for a new
file, an untouched file beside it: a parser that fails on the project's own sources is not in their path. So a
finding means "this project's toolchain accepted this code before your change and rejects it now".
"""

from __future__ import annotations

import base64
import json
import re
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

LANGUAGES = frozenset({"javascript", "typescript"})
MAX_FILES = 16
MAX_PROMPTS = 2
MIN_TIME_S = 90.0
TIMEOUT_S = 60
MAX_PAYLOAD_CHARS = 56 * 1024
MAX_FILE_BYTES = 512 * 1024
DELIMITER = "__HONE_TOOLCHAIN__"
MARKER = "TOOLCHAIN-PARSE "
DONE = "TOOLCHAIN-PARSE-DONE"
_SOURCE_RE = re.compile(r"\.(?:js|cjs|mjs|jsx)$")
_TEST_RE = re.compile(r"(^|/)(tests?|__tests__|spec|specs|e2e|fixtures)/|\.(test|spec)\.[cm]?jsx?$")


@dataclass(frozen=True)
class Rejection:
    path: str
    tool: str
    message: str


def source_files(touched: Iterable[str]) -> list[str]:
    """The JavaScript sources among the files an answer touches, tests and fixtures left out."""
    return sorted(path for path in set(touched) if _SOURCE_RE.search(path) and not _TEST_RE.search(path))[:MAX_FILES]


_SCRIPT = r"""
const fs = require('fs'), path = require('path'), zlib = require('zlib');
const FILES = __FILES__;
const ORIGINALS = JSON.parse(zlib.inflateSync(Buffer.from(__ORIGINALS__, 'base64')).toString('utf8'));
const cwd = process.cwd();
function resolveFrom(name, from) { try { return require.resolve(name, {paths: [from || cwd]}); } catch (e) { return null; } }
function load(name, from) { const found = resolveFrom(name, from); if (!found) { return null; } try { return require(found); } catch (e) { return null; } }
function dirOf(name, from) { const found = resolveFrom(name + '/package.json', from); return found ? path.dirname(found) : null; }
let pkg = {};
try { pkg = JSON.parse(fs.readFileSync('package.json', 'utf8')); } catch (e) { pkg = {}; }
const deps = Object.assign({}, pkg.dependencies, pkg.devDependencies, pkg.optionalDependencies, pkg.peerDependencies);
const parsers = [];
if (['browserify', 'watchify', 'budo', 'module-deps', 'karma-browserify', 'browserify-middleware'].some((name) => deps[name])) {
  const moduleDeps = dirOf('module-deps', dirOf('browserify') || cwd) || dirOf('module-deps');
  const detective = (moduleDeps && load('detective', moduleDeps)) || load('detective');
  if (typeof detective === 'function') {
    parsers.push(['browserify (its parser, detective)', (source) => { detective(source); }]);
  }
}
const babelConfigs = ['babel.config.js', 'babel.config.cjs', 'babel.config.json', '.babelrc', '.babelrc.js', '.babelrc.cjs', '.babelrc.json'];
if (babelConfigs.some((name) => fs.existsSync(name)) || pkg.babel) {
  const babel = load('@babel/core');
  if (babel && typeof babel.transformSync === 'function') {
    parsers.push(['Babel with the project\'s configuration', (source, file) => {
      try {
        babel.transformSync(source, {filename: path.resolve(file), cwd: cwd, code: false, ast: false});
      } catch (error) {
        if (error && (error.code === 'BABEL_PARSE_ERROR' || error.name === 'SyntaxError')) { throw error; }
      }
    }]);
  }
}
function rejects(parse, source, file) {
  try { parse(source, file); return null; } catch (error) { return String((error && error.message) || error).split('\n')[0].slice(0, 200); }
}
function sources(dir) {
  try { return fs.readdirSync(dir).filter((name) => /\.(js|cjs|mjs|jsx)$/.test(name)).map((name) => path.join(dir, name)); } catch (e) { return []; }
}
function neighbours(file) {
  // Untouched JavaScript files beside the new file, or in the nearest directory above it that has any.
  let dir = path.dirname(file);
  for (let depth = 0; depth < 4; depth += 1) {
    const found = sources(dir).filter((name) => !FILES.includes(name.split(path.sep).join('/')));
    if (found.length) { return found.slice(0, 4); }
    if (dir === '.' || dir === '') { break; }
    dir = path.dirname(dir);
  }
  return [];
}
for (const file of FILES) {
  let source;
  try { source = fs.readFileSync(file, 'utf8'); } catch (e) { continue; }
  for (const [tool, parse] of parsers) {
    const message = rejects(parse, source, file);
    if (!message) { continue; }
    let accepted = false;
    if (Object.prototype.hasOwnProperty.call(ORIGINALS, file)) {
      accepted = rejects(parse, Buffer.from(ORIGINALS[file], 'base64').toString('utf8'), file) === null;
    } else {
      const beside = neighbours(file);
      accepted = beside.length > 0 && beside.every((name) => {
        try { return rejects(parse, fs.readFileSync(name, 'utf8'), name) === null; } catch (e) { return false; }
      });
    }
    if (accepted) { console.log('__MARKER__' + JSON.stringify({path: file, tool: tool, message: message})); }
  }
}
console.log('__DONE__ ' + parsers.length);
"""


def check_command(touched: Iterable[str], originals: Mapping[str, bytes], working_directory: str = ".") -> str | None:
    """The sandbox command, or None when the answer touches no JavaScript source or the originals do not fit.

    `touched` lists every file the answer changes; `originals` holds the repository's content of those that
    existed before, so that a parser which already rejected a file is not held against the answer.
    """
    working = (working_directory or ".").strip("/")
    prefix = "" if working in ("", ".") else working + "/"
    # An edited file whose original is too large to carry is left out: without it a rejection cannot be told
    # from one the repository already had. Files outside the working directory are not the project's sources.
    files = [
        path for path in source_files(touched)
        if path.startswith(prefix) and (path not in originals or len(originals[path]) <= MAX_FILE_BYTES)
    ]
    if not files:
        return None
    relative = [path[len(prefix):] for path in files]
    carried = {path[len(prefix):]: base64.b64encode(originals[path]).decode("ascii") for path in files if path in originals}
    payload = base64.b64encode(zlib.compress(json.dumps(carried).encode("utf-8"), 9)).decode("ascii")
    if len(payload) > MAX_PAYLOAD_CHARS:
        return None
    script = (
        _SCRIPT.replace("__FILES__", json.dumps(relative))
        .replace("__ORIGINALS__", json.dumps(payload))
        .replace("__MARKER__", MARKER)
        .replace("__DONE__", DONE)
    )
    directory = "/work" if not prefix else "/work/" + working
    return f"cd {directory} && node - <<'{DELIMITER}'\n{script.strip()}\n{DELIMITER}\n"


def rejections(output: str) -> list[Rejection] | None:
    """What the project's parsers newly reject, or None when the check did not finish."""
    if DONE not in output:
        return None
    found: list[Rejection] = []
    for line in output.splitlines():
        if line.startswith(MARKER):
            try:
                item = json.loads(line[len(MARKER):])
            except ValueError:
                continue
            found.append(Rejection(str(item.get("path", "")), str(item.get("tool", "")), str(item.get("message", ""))))
    return found


def parsers_tried(output: str) -> int:
    """How many of the project's parsers the sandbox had; 0 means there was nothing to check with."""
    match = re.search(re.escape(DONE) + r" (\d+)", output)
    return int(match.group(1)) if match else 0


def prompt(found: list[Rejection]) -> str:
    lines = "\n".join(f"- {r.path}: {r.tool} says: {r.message}" for r in found)
    return (
        "A check parsed each JavaScript file your change adds or edits with this project's own bundler chain, in "
        f"the validators' image, and it rejects:\n{lines}\n\n"
        "The same tool accepts the repository's own files beside it, so the construct at that position is newer "
        "than this project's toolchain reads: a private `#name` member, a class field or a static block are the "
        "usual cases. Node runs such a file directly, which is why your own tests pass. A test that reaches the "
        "feature through the project's own entry point, which bundles or compiles these sources first, fails on "
        "every case. Rewrite the construct at that position in the syntax the neighbouring files use (for private "
        "state an underscore-prefixed member, a Symbol key or a WeakMap; for class fields, assignments in the "
        "constructor), keep the behaviour exactly as it is, run your tests again and reply with a one-line summary."
    )

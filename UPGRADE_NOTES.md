# cdagent v4 - auto-dispatch enforcement + import-aware call graph

Run the checks: `python test_fixes.py && python test_agents.py && python test_auto_dispatch.py`
(82 checks total: 18 + 48 + 16 new)

## What changed and why

### 1. Auto-dispatch now covers read/analysis tools, not just writes (specialists.py)

**Symptom reported:** specialists only activated when the user manually said
"use the security specialist" (or similar) - a read-only task (e.g. a
security audit) never woke a specialist automatically even with a strong
domain match, because the old safety net (`_Guard` / `execute_with_specialists`
in `install()`) only intercepted `write_file`/`edit_file`/`edit_files`.

**Root cause:** `arm_router()` (called once per user prompt) only appends a
text directive to the prompt and arms a guard that watches for *writes*.
If the top-level model's plan for a task never reaches a write - e.g. it
just calls `read_file`/`search_code` and answers in prose - the guard never
fires and `dispatch_specialists` is never called unless the model chooses to
call it, which it may not.

**Fix, staying within the approach you chose (LLM-driven dispatch + a
broader safety net, not a forced pre-router):**
- `_DOMAIN_READ_TOOLS` now covers `read_file`, `search_code`, `find_symbol`,
  `find_callers`, `repo_map`, `list_dir`, `run_command`, `git_diff`,
  `git_blame`, `git_log`, `git_status`.
- `_Guard` now tracks two independent counters: `write_rescues` (capped at 1
  per prompt - unchanged behaviour, since redoing a write is wasteful) and
  `analysis_rescues` (capped at 20 - effectively unbounded for one prompt,
  since re-reading is cheap and a multi-step read-only task needs every
  attempt redirected, not just the first).
- `_args_touch_domain()` decides whether a SPECIFIC call is this specialist's
  business (not just "some specialist is armed"), using three signals in
  order of strength: (a) file extension via `_EXT_DOMAIN`, (b) a trigger
  keyword in the call's own arguments, (c) - for content-inspecting tools
  only (`read_file`/`search_code`/`find_symbol`/`find_callers`/`git_diff`/
  `git_blame`) - a trigger keyword in the originally armed task text. Signal
  (c) is deliberately withheld from purely structural tools (`list_dir`,
  `git_log`, `git_status`, `repo_map`) so an unrelated `list_dir(".")` isn't
  swept in just because the task happens to mention a domain keyword
  elsewhere.

**Known limitation (by design, not a bug):** this is still LLM-driven
dispatch with a safety net, per your explicit choice - not a forced
pre-router. If the top-level model never calls any tool that matches
`_DOMAIN_READ_TOOLS`/`_DOMAIN_WRITE_TOOLS` for a domain task (e.g. it just
answers from its own reasoning with no tool call at all), nothing intercepts
it, because there is no tool call to intercept. The `suggest_routing_note`
DIRECTIVE text still does the work of pushing the model toward calling
`dispatch_specialists` first in that case.

### 2. Import-aware cross-file call graph (repo_intel.py)

**Bug found (reproduced before fixing, not assumed):** `DependencyGraph`
resolved every call by NAME ONLY. Two files each defining a same-named
function (e.g. `validate()`) had their callers merged: `find_callers` would
report a caller of `b.py::validate` as also possibly calling `c.py::validate`,
even though the caller's own `from b import validate` names exactly one of
them.

**Fix:** new `ImportResolver` class builds a precise `(file, local_name) ->
defining file` table for Python from the exact `from X import Y as Z` /
`import X` strings the parser already extracts, handling relative-import
dots (`from .utils import helper`) and aliases (`import Y as Z`).
`DependencyGraph._rebuild()`'s `link()` now tries this precise resolution
FIRST and only falls back to the old name-only/fan-out-capped heuristic when
a call isn't import-resolvable (locally defined symbols, dynamic dispatch,
non-Python languages). `references()` gained a `by_def` field that groups
callers under the specific definition they resolve to; `find_callers`'s
rendered output uses this to show per-definition caller lists instead of one
merged list when a name has multiple definitions.

**Independent bug found and fixed while building this:** the import-string
extraction in `_fallback_python()` was dropping `as` aliases entirely
(`a.name` instead of `a.name + " as " + a.asname`) - this silently broke
alias resolution for BOTH the old heuristic and the new resolver. Fixed at
the source so the raw import strings shown in outlines are also now
accurate.

**Known limitation (stated plainly, not hidden):** Python-only. JS/TS/CSS/
HTML/tree-sitter-backed languages still use the name-only graph exactly as
before - the fallback path was deliberately left untouched rather than
half-implemented for languages where the import syntax is different enough
that a rushed resolver would likely be wrong in ways that are harder to spot
than "no precision" was. This is a real scope boundary, not an oversight:
extending precise resolution to JS/TS (import/require/ESM) would be a
similarly-sized follow-up, not a small addition to this one.

**Visibility:** `/repomap` now shows import-resolver stats (files indexed,
name->file bindings resolved) and call-graph precision (% of edges resolved
via import vs name-only guess), so precision is checkable at runtime, not
just claimed in a docstring.

## Files changed
- `specialists.py`: `_Guard`, `_DOMAIN_READ_TOOLS`, `_args_touch_domain`,
  `execute_with_specialists` (the interception logic in `install()`).
- `repo_intel.py`: new `ImportResolver` class; `DependencyGraph.__init__`/
  `update_file`/`_rebuild`/`references`; `_fallback_python`'s import-string
  extraction (alias bug); `RepoIntel.health`/`stats` (new telemetry rows).
- `coder3.py`: `find_callers` tool method (renders `by_def` grouping).
- `test_auto_dispatch.py` (new): 16 end-to-end checks for both upgrades.

## Verified, not assumed
- Reproduced the name-collision bug BEFORE fixing it (two same-named
  `validate()` functions, confirmed wrong caller attribution).
- Reproduced the alias-dropping bug BEFORE fixing it (`from .utils import
  helper as h` failed to resolve; traced to the parser discarding `asname`).
- Found and fixed a regex-grouping bug in my own first draft of
  `_PY_IMPORT_RE` (the from/import alternation was nested wrong, causing a
  crash - `NoneType.split()` - on any plain `import x` statement) via the
  existing `test_fixes.py` suite catching the regression on a real rerun.
- Found and fixed an over-broad task-text fallback in my own first draft of
  `_args_touch_domain` (it would have intercepted an unrelated `list_dir`
  call just because the task text mentioned a domain keyword elsewhere) -
  caught by my own new test, not shipped silently.
- All 66 pre-existing checks (test_fixes.py + test_agents.py) still pass
  unmodified. 16 new checks added and passing. 82/82 total.
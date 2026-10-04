# cdagent - tree-sitter/CST + LSP-semantic upgrade, and the step-14 specialist bug

Run the checks: `python3 test_fixes.py && python3 test_agents.py && python3 test_step_budget.py && python3 test_treesitter.py`
(94 checks total; stubs rich/pygments/prompt_toolkit/openai/google.generativeai only if missing)

Requires `pip install tree-sitter tree-sitter-python tree-sitter-javascript tree-sitter-typescript
tree-sitter-css tree-sitter-html` for the new JS/TS/TSX tree-sitter-primary parsing and the
semantic-navigation tools to work at full accuracy - see `requirements.txt`. Everything degrades
gracefully without them (falls back to the pre-existing regex/ast parsers; semantic tools report
"repo intelligence unavailable" instead of crashing).

## 1. The "specialists fail after step 14" bug - root cause and fix

Two stacked bugs, both confirmed by reading the code before touching it:

1. **`SPEC_MAX_STEPS` was hardcoded to 14.** That number was tuned back when symbol reads/edits ran
   on regex fallbacks. Real semantic-navigation work (`find_definition` -> `find_references` ->
   read symbol -> edit -> verify, often repeated per call site on a multi-file change) routinely
   needs more than 14 tool turns for anything beyond a one-line tweak - specialists were being cut
   off mid-task on ordinary refactors. Raised to a budget sized for that (`specialists.py`).

2. **Status was inferred from `"FAILED" in text`, in TWO DIFFERENT PLACES that disagreed with each
   other**, and neither recognised `"stopped after N steps (INCOMPLETE)"` as anything other than
   success:
   - `dispatch_specialists`: `ok = sum(1 for r in results.values() if "FAILED" not in r)`
   - `execute_plan`: `bad = ("FAILED" in out) or out.startswith("error:")`

   Consequence: a plan step whose specialist ran out of its step budget was silently marked
   `"done"` and the plan kept going as if the step had actually finished - the step-14 cutoff
   *looked* like success further up the chain. The check is also fragile the other way: a
   specialist whose own summary legitimately contains the word "FAILED" (e.g. a test specialist
   reporting `"3 tests FAILED, 12 passed"`) would have been misclassified as a hard failure.

   **Fix:** every specialist result now carries an explicit `[[status:ok|incomplete|failed]]` tag
   (`run_specialist`), aggregated into one `[[dispatch_status:...]]` trailer on
   `dispatch_specialists`' own return value (`specialists.py`) - no substring guessing, no shared/
   global state (an earlier draft used a module-level "last call" channel; caught it racing against
   itself in testing and replaced it with a value threaded through the actual return string).
   `execute_plan` reads that trailer directly and now records a step as `"incomplete"` (a new,
   distinct `PlanStep.status`, separate from `"fail"`) rather than either silently "done" or a hard
   failure, and stops there so later steps' "ALREADY DONE" context stays honest.

   The identical bug existed in the separate plain-subagent path (`coder3.py`'s `dispatch_agents` /
   `run_subagent`) and got the same fix, scoped to that file's simpler needs (no plan-gating there,
   just a UI log line, so a lighter marker-regex classifier was enough rather than the full tagging
   system).

   Tests: `test_step_budget.py` (13 checks) - reproduces the exact scenario end-to-end, including
   the false-positive case above.

## 2. Tree-sitter made PRIMARY for JS/TS/TSX (not Python/CSS/HTML - see why below)

The tree-sitter *loader* itself already worked correctly (confirmed by testing it directly).
The actual gap: `repo_intel.py` hardcoded `_FALLBACK_LANGS = {"python", "javascript", "typescript",
"tsx", "css", "html"}` and always tried the regex/ast-based fallback FIRST for all five, so
tree-sitter was never actually reached for any of them even though the grammars loaded fine.

Direct comparison on a real-world TS snippet (decorated Angular-style class, object-literal arrow
method, nested named function expression) proved the fallback JS/TS parser (`_fallback_js`, genuinely
regex-based) misses real, common patterns:
```js
const config = {
  onLoad: () => console.log("loaded"),        // <- MISSED by the old fallback
  handlers: { click: function handleClick(e){} }  // <- MISSED by the old fallback
};
```
Tree-sitter finds both for free, plus the class/method/decorator/exports the fallback already
handled. **Python/CSS/HTML were deliberately left as-is**: their "fallback" parsers are themselves
real parsers (`ast` module, `html.parser`, a brace/string/comment-aware CSS scanner) - not regex
approximations - and are already at least as accurate as tree-sitter for their languages (Python's
`ast` in particular gives exact decorator ranges and qualnames a generic tree-sitter walk would only
reconstruct heuristically). Switching those to tree-sitter-primary would have been a downgrade.

What changed in `repo_intel.py`:
- `_parse_file`: js/ts/tsx now try tree-sitter first, falling back to the regex walk only if the
  grammar is unavailable/broken or produces nothing. python/css/html unchanged (fallback-first).
- `_DEF_NODE_TYPES` (js/ts/tsx): added `variable_declarator` and `pair` as definition-bearing node
  types, gated by `is_real_def()` (only counts if the bound *value* is itself a function/arrow/class
  - `const x = 5` is not a definition) - this is what catches `const Foo = () => {}` and
  `{ onClick: () => {} }`, which have no dedicated grammar node of their own. Also added
  `type_alias_declaration`, `enum_declaration`, `generator_function_declaration`.
- **Bug caught and fixed during my own end-to-end verification** (not hypothetical): a definition
  node's own byte range excludes an enclosing `export` keyword, so patching an exported symbol with
  replacement text that itself starts with `export function ...` produced `export export function
  ...` in the file. Fixed with `effective_span()`, which widens the recorded symbol span to the
  enclosing `export_statement`/`lexical_declaration` when present. Verified correct for plain
  exported functions, exported const-arrow functions, and decorated exported classes (where the
  grammar already includes the decorator in the class's own node boundary for free - no extra
  handling needed there).
- `qualname` is now populated consistently for every tree-sitter language (previously only
  implicitly present for some), which also fixes a latent inconsistency for go/java/rust/c/cpp.

Tests: `test_treesitter.py` (15 checks) - object-literal/nested-function-expression detection,
JSX/TSX component detection, Python/CSS/HTML regression (unchanged output), and the export/decorator
duplication-bug fix verified through the real `Workspace.edit_file` tool end-to-end.

## 3. New LSP-style semantic-navigation tools

`repo_intel.py`'s `ImportResolver` was Python-only; extended to also resolve ES-module imports
(`import { X } from "./x"`, default imports, namespace imports, side-effect imports) for JS/TS/TSX,
relative specifiers only (bare npm specifiers like `"react"` have no in-repo file and are correctly
left unresolved rather than guessed at).

Three new `RepoIntel` methods, built on that resolver + each language's real parser, go beyond the
pre-existing `find_symbol`/`find_callers` (which answer "where is X defined/called, by name,
anywhere in the repo" - ambiguous the moment two files define a same-named function) by resolving a
**specific occurrence** through the import graph:

- **`goto_definition(path, line, col=None, symbol=None)`** - resolves the identifier at an exact
  usage site to its exact definition. Import-aware: if the file imports that exact name from a
  specific file, returns ONLY that definition, not every same-named symbol in the repo. Falls back
  to name-only (same ambiguity semantics as the old `references()`) when not import-resolvable.
- **`find_references_at(path, line, col=None, symbol=None)`** - every occurrence of that identifier
  across the (import-connected) repo, labeled `definition` / `call` / `read` so a declaration is
  distinguishable from a usage at a glance - restricted to files actually linked via an import, not
  every file that happens to contain the substring.
- **`rename_symbol(path, line, col, new_name, symbol=None)`** (always a dry run in `repo_intel.py` -
  this module stays a pure read/analysis layer, consistent with the rest of its API) - computes a
  safe multi-file rename: refuses invalid identifiers, refuses if `new_name` would collide with
  something that already exists in a file that would be touched, returns the changes for the caller
  to apply.

Wired into `coder3.py` as three new tools (`goto_definition`, `find_references`, `rename_symbol`) in
the shared `TOOL_SPECS` schema and `Workspace.execute`'s dispatch table. `Workspace.rename_symbol`
drives the actual write: computes the rename via `repo_intel`, then shows one combined diff across
every affected file, ONE approval for the whole rename (matching `edit_files`' existing atomic
multi-file pattern - a rename lands everywhere or not at all, never half-applied), then writes and
re-parses. Available to every specialist (frontend/backend/python/security/test/devops/docs) by
default, same as every other tool; Review Specialist's curated read-only `allowed_tools` gained the
two read-only ones (`goto_definition`, `find_references`) but not the mutating `rename_symbol`.
Also added to `_DOMAIN_READ_TOOLS`/`_DOMAIN_WRITE_TOOLS`/`_CONTENT_INSPECTING_TOOLS` so the
top-level model's delegate-to-specialist "rescue" logic treats them the same as the tools they
extend (`find_symbol`/`find_callers` and `write_file`/`edit_file`/`edit_files` respectively).

Tests: exercised end-to-end (not just unit-level) through `RepoIntel` directly and through the full
`Workspace` tool interface, for both TS (import-resolved class + method) and Python (import-resolved
function), including the dry-run/collision/invalid-identifier paths for rename.

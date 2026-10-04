# Fix notes — AST/tree-sitter revival + specialist routing (this patch)

## 1. Tree-sitter / AST / CST layer was silently dead

**Root cause:** `repo_intel.py` loaded grammars via `tree_sitter_languages`
(the grantjenks package). That package is abandoned and was built against
the pre-0.22 `tree_sitter` Python API. Against any current `tree_sitter`
(0.22+, tested on 0.26.0), every call raised:

    TypeError: __init__() takes exactly 1 argument (2 given)

Because `repo_intel.py` is deliberately defensive (never crash coder3 if
tree-sitter isn't available), this failure was silently swallowed:
`HAVE_TREE_SITTER` became `False` at import time and every caller
(`outline_text`, `extract_symbol`, `cst_patch`, `cst_insert`, the symbol
graph / PageRank ranking) fell back to full-file reads and substring/fuzzy
edits — with no visible error. That's what produced the repeated whole-file
`read_file` calls over overlapping line ranges.

**Fix:** `repo_intel.py`'s loader now uses the actively-maintained
per-language PyPI packages (`tree_sitter_python`, `tree_sitter_javascript`,
`tree_sitter_typescript`, `tree_sitter_go`, `tree_sitter_java`,
`tree_sitter_rust`, `tree_sitter_ruby`, `tree_sitter_c`, `tree_sitter_cpp`,
`tree_sitter_css`, `tree_sitter_html`) with the modern one-arg
`tree_sitter.Language(pkg.language())` / `tree_sitter.Parser(lang)` API.
`get_parser(lang_name)` keeps its old signature (cached Parser per
language name), so no other line in `repo_intel.py` or `coder3.py` needed
to change. `_parse_file` also now degrades per-language instead of killing
AST parsing for every language if one grammar package is missing.

**New/changed dependencies** — replace `tree-sitter-languages` with:

```
tree-sitter>=0.22
tree-sitter-python
tree-sitter-javascript
tree-sitter-typescript
tree-sitter-go
tree-sitter-java
tree-sitter-rust
tree-sitter-ruby
tree-sitter-c
tree-sitter-cpp
tree-sitter-css
tree-sitter-html
```

Install with:

```
pip uninstall -y tree-sitter-languages
pip install "tree-sitter>=0.22" tree-sitter-python tree-sitter-javascript \
    tree-sitter-typescript tree-sitter-go tree-sitter-java tree-sitter-rust \
    tree-sitter-ruby tree-sitter-c tree-sitter-cpp tree-sitter-css tree-sitter-html
```

(You only strictly need the packages for languages you actually use —
Python/JS/TS/CSS/HTML cover most web+Python repos — but installing all of
them costs little and keeps every supported language working.)

**Verified:** outline extraction, exact symbol-boundary reads, real
AST-node-level CST patching (JS/Python/CSS), cross-file import-aware call
graph resolution. All pre-existing tests in `test_fixes.py` and
`test_agents.py` still pass (52 total).

## 2. Specialist auto-routing was too conservative for iterative/debug work

**Root cause:** the specialist router (`route_confidence` in
`specialists.py`) and its safety-net guard already cover reads, edits, and
refactors — not just file creation — but only force-delegate on a "strong"
match, and `is_small_edit()` (keyed on verbs like fix/update/tweak/adjust)
vetoed strong-routing for almost all ordinary debugging phrasing, since
"fix the animation timing in app.js" and "fix the typo in a.py" both start
with "fix". That made specialists look like they only ever fired for
"create X" requests.

**Fix:** added `_has_substantive_domain_match()`, which lets a task
override the `is_small_edit` veto only when it matches a trigger that names
real domain content (e.g. "gsap", "sql injection", "models.py") — not just
a bare filename/extension or a small-edit verb combined with nothing but a
filename. So "fix the typo in a.py" still correctly stays undelegated
(genuinely trivial), while "fix the hero animation timing in app.js" now
correctly force-routes to `frontendspecialist` (previously it didn't).

**Verified:** 13 representative prompts covering both cases, plus an
end-to-end simulation of the reported scenario — arming the router on a
GSAP-animation debugging prompt, then confirming a `read_file(app.js, 580,
710)` call (matching the pattern in the reported session log) is now
correctly intercepted and redirected to `frontendspecialist` instead of
running raw.

## Files changed
- `repo_intel.py` — tree-sitter loader + `_parse_file` per-language fallback
- `specialists.py` — `route_confidence`, `_has_substantive_domain_match`,
  `_is_bare_filename_or_verb_pattern`, `_SMALL_EDIT_VERBS`,
  `_BARE_FILENAME_TRIGGERS` (new)

No public tool names, tool arguments, or test contracts changed.

## 3. Plan-step owner could be assigned to the wrong specialist

**Root cause:** unlike runtime tool routing (`route_confidence`, deterministic
regex), plan-step `owner` is chosen by the planner LLM freely from a bare
name+label list, with `plan_from_dict` only validating the owner is a *real*
key — never that it's the *right* one for that step's content. A cheap/fast
planner model can pick a plausible-sounding but wrong owner (e.g.
"testspecialist" for a step titled "Enhance Frontend Chat Rendering &
Styling" — that's frontendspecialist's job; testspecialist only writes/runs
tests).

**Fix:** `_corrected_step_owner()` in `specialists.py`, called from
`plan_from_dict` for every step. Re-scores the step's title+detail with the
same deterministic scorer runtime routing already uses (`_score_specialist`),
and overrides the planner's `owner` ONLY when a different specialist scores
decisively higher (>= STRONG_SCORE and beats the runner-up by 2+) — a close
call or thin-signal title is left untouched, trusting the planner. Zero extra
LLM calls, keyword-only (deliberately no file-extension bonus: a step editing
`test_server.py` is a `.py` file but unambiguously testspecialist's job, so
extension alone would misfire here).

**Verified:** the reported case (testspecialist -> frontendspecialist for a
frontend-rendering step) corrects when the step has real descriptive detail;
an already-correct owner and a genuinely ambiguous/cross-cutting step are
left unchanged. All 52 tests still pass.

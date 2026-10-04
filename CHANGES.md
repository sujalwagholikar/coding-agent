# cdagent v3 - what changed (all agents: main, specialists, workers, subagents, planner)

Run the checks:  `python test_fixes.py && python test_agents.py`  (66 checks; stubs UI/LLM libs only if missing)

## Root causes found (with proof in the tests)
1. Subagents (`dispatch_agents`) could not edit at all: their write-lock wrapper had the signature
   `(path, old, new)` but `edit_file` is `(path, new, old, symbol)` -> every text edit had old/new SWAPPED and
   failed, every `symbol=` edit crashed. Now ONE shared lock (`install_write_lock`) used by subagents and specialists.
2. History was re-sent in full every turn: a file dump read at step 2 was paid for again on every later step.
   -> `compact()` on both providers keeps the last 2 tool-result turns verbatim, stubs older big ones.
   All agents inherit it (they all use these providers).
3. `find_callers`/PageRank silently returned nothing without `networkx`; symbol tools silently died with a broken
   `tree_sitter_languages`. -> built-in pure-Python call graph + PageRank; built-in parsers for
   Python (ast), JS/TS, CSS/SCSS, HTML (incl. inline <script>/<style>); tree-sitter only for go/java/rust/ruby/c/cpp.
4. Specialist/planner/worker prompts steered to basic tools (tool list without repo_map/find_callers, "ALWAYS use
   edit_file old/new"). -> single shared WORKFLOW in the base prompt + a per-specialist NAVIGATION hint.

## Token diet
- Base prompt + tool schemas: main ~5.5k -> ~2.9k tokens/turn, specialist ~4.1k -> ~2.5k tokens/turn.
- read_file on files > 400 lines returns a symbol outline (full=true for raw); raw read cap 60k -> 24k chars.
- search_code: 200 -> 60 matches, pruned walk (skips node_modules/.git/binaries). repo_map: 400 -> 150 entries, depth 3.
- run_command output tail 12k -> 6k chars.
- write_file refuses to re-send an existing file > 120 lines unless overwrite=true.
- Removed: old regex outline (`_symbol_outline`), networkx dependency, whole-file chunker, verbose tool/prompt prose.

## New capabilities (available to every agent)
- `find_symbol(name)`: where is a function/class/method/CSS rule/HTML #id defined (whole repo).
- `edit_file(insert_after=<symbol>, new=...)`: add code next to an existing symbol, re-indented.
- symbol= reads/edits for HTML (#id, section.hero) and functions/rules inside inline <script>/<style>.
- Python edits that would introduce a syntax error are rejected before approval/write.
- Indentation-tolerant edit matching (deterministic) before the fuzzy fallback; fuzzy is pruned (fast on big files).
- Step budget warnings (2 left / 1 left) for main agent, subagents and specialists; step-limit = INCOMPLETE, not a green tick.
- `/repomap` shows real backend health, symbol-vs-text usage counters and tokens saved by compaction.

## Not covered
- browsertester.py / browseragent.py were not in the zip: they get compaction + the lean prompt automatically
  (same providers/build_provider) but their own prompts are unchanged.
- The tree-sitter path (go/java/rust/ruby/c/cpp) could not be tested here (no network to install it).
- Call graph is name-based, not a type-aware LSP. YAML/Dockerfile/Markdown aren't symbol-indexed (ranges/`old` still work).

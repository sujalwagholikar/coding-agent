"""
repo_intel.py - graph-based code intelligence for coder3.py
=============================================================

Upgrades four subsystems that coder3.py previously handled with regex /
raw-string primitives, WITHOUT changing coder3.py's public tool surface
(tool names/args stay the same - only their implementations get smarter,
and each one degrades gracefully to the old behaviour if tree_sitter or
networkx aren't installed):

  1. Graph-based RepoMap        : parse every source file into a real AST
     (tree_sitter), build a symbol graph (def/class nodes + "who calls
     whom" / "who imports whom" edges - a lightweight LSP-style symbol
     graph), then run PageRank over it so repo_map() can rank and surface
     the files/symbols that actually matter, not just list everything.

  2. AST-level chunking          : read_file_smart() extracts just the
     requested function/class (plus its signature-level neighbours)
     instead of dumping the whole file or a blind line range, using the
     AST to find exact node boundaries.

  3. CST-aware patching          : cst_patch() applies an edit by locating
     the target *node* (function/class/block) in the concrete syntax tree
     rather than doing a raw substring replace, so it survives whitespace/
     formatting drift; fuzzy_replace() is the fallback for free-text diffs
     that don't map to one clean AST node, using a token-level fuzzy
     matcher instead of requiring a byte-exact substring.

  4. Multi-tier caching + dependency graph : ASTCache holds parsed trees
     and file hashes in memory (L1) and on disk under .coder3/ast_cache
     (L2), keyed by content hash, so re-analysing an unchanged file across
     a long session or across subagents costs nothing. DependencyGraph
     tracks import/call edges so "what does this touch" is a graph lookup,
     not a fresh repo-wide regex grep every time.

Supported languages (anything else falls back to the old regex outline):
Python, JavaScript, JSX, TypeScript, TSX, Go, Java, Rust, Ruby, C, C++.

Everything here is defensive: if tree_sitter/tree_sitter_languages/
networkx are missing, or a file fails to parse, callers get the previous
regex-based behaviour back with zero crashes - this is a strict upgrade,
never a new hard dependency for coder3.py to run at all.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Optional deps - everything degrades gracefully without them.
#
# NOTE ON WHY THIS ISN'T `tree_sitter_languages`:
# `tree_sitter_languages` (grantjenks) bundles prebuilt grammars for the OLD
# tree-sitter C ABI / Python binding (tree_sitter <= 0.21, `Language(so_path,
# name)` two-arg constructor). Modern `tree_sitter` (0.22+) changed both the
# Python API (`Parser()` no longer takes a language positionally at
# construction the old way, `Language.__init__` takes ONE capsule arg) and
# the ABI grantjenks's wheels were compiled against, so on any current
# tree_sitter install `get_parser(...)` raises
# `TypeError: __init__() takes exactly 1 argument (2 given)` for EVERY
# language, every time - `HAVE_TREE_SITTER` silently becomes False and every
# caller below (outline_text, extract_symbol, cst_patch, cst_insert, the
# PageRank symbol graph) falls back to the old regex path with no visible
# error. That is the actual root cause of "it keeps re-reading whole files
# instead of using symbol=/outline" - the AST layer was never running.
#
# Fix: use the actively-maintained, per-language `tree_sitter_<lang>` PyPI
# packages instead (tree_sitter_python, tree_sitter_javascript, ...), each of
# which ships a `language()` factory matched to whatever `tree_sitter` build
# is installed, and build `Language`/`Parser` via the current one-arg API:
#     lang   = tree_sitter.Language(tree_sitter_python.language())
#     parser = tree_sitter.Parser(lang)
# `get_parser(lang_name)` below reproduces the old call signature (one cached
# Parser per language name) so every other line in this file - including
# `_parse_file`'s `get_parser(lang)` call - needs no further change.
# --------------------------------------------------------------------------- #
import importlib as _importlib

# language name (as used throughout this file, via _EXT_LANG) -> the PyPI
# module that provides it, and which attribute of that module (if not
# `language`) to call. tsx/typescript both live in tree_sitter_typescript
# under two different factory functions.
_TS_LANG_MODULES: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "java": ("tree_sitter_java", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "css": ("tree_sitter_css", "language"),
    "html": ("tree_sitter_html", "language"),
}

_parser_cache: dict[str, Any] = {}
_language_load_errors: dict[str, str] = {}


def _build_parser(lang_name: str):
    """Build (and cache) a tree_sitter.Parser for `lang_name` using the
    modern per-language package + one-arg Language/Parser API. Raises on
    failure (caller decides what "no parser" means); result is cached so
    repeated calls (one per file parsed) cost nothing after the first."""
    if lang_name in _parser_cache:
        return _parser_cache[lang_name]
    mod_name, factory_name = _TS_LANG_MODULES[lang_name]
    mod = _importlib.import_module(mod_name)
    capsule = getattr(mod, factory_name)()
    language = _ts.Language(capsule)
    parser = _ts.Parser(language)
    _parser_cache[lang_name] = parser
    return parser


def get_parser(lang_name: str):
    """Drop-in replacement for `tree_sitter_languages.get_parser`: returns a
    cached Parser for `lang_name`, raising if that language's grammar
    package isn't installed or fails to build - callers already wrap this in
    try/except and treat failure as "no AST for this file"."""
    try:
        return _build_parser(lang_name)
    except Exception as e:                                      # noqa: BLE001
        _language_load_errors[lang_name] = f"{type(e).__name__}: {e}"[:160]
        raise


try:
    import tree_sitter as _ts
    get_parser("python").parse(b"x = 1")        # import can succeed while parsing is broken
    HAVE_TREE_SITTER = True
    TS_ERROR = ""
except Exception as _ts_exc:                                   # pragma: no cover
    HAVE_TREE_SITTER = False
    TS_ERROR = f"{type(_ts_exc).__name__}: {_ts_exc}"[:160]

HAVE_NETWORKX = True      # kept for import compatibility: the graph/PageRank are built in now, no networkx needed

_EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".go": "go", ".java": "java",
    ".rs": "rust", ".rb": "ruby", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".mjs": "javascript", ".css": "css", ".scss": "css", ".html": "html", ".htm": "html",
}

# Tree-sitter node types that count as a "definition" per language, mapped to
# a uniform kind label, plus which child field holds the name.
#
# js/ts/tsx additionally map "variable_declarator" and "pair": these are how
# the dominant modern JS/TS patterns actually declare a function - there is
# no separate "arrow_function_declaration" node type in the grammar, an
# arrow/function-expression is just a VALUE, and its name lives on the
# enclosing binding (variable_declarator's `name` field for
# `const Foo = () => {}`, pair's `key` field for `{ onClick: () => {} }`
# object/class-field methods). Walked generically (see _find_def_name /
# _walk_ts below) rather than matched as their own top-level def type,
# because only the SUBSET of variable_declarator/pair nodes whose value is
# itself a function/arrow/class counts as a definition - `const x = 5` is
# not one, and the walker checks that before recording a symbol.
_DEF_NODE_TYPES: dict[str, dict[str, str]] = {
    "python": {"function_definition": "func", "class_definition": "class"},
    "javascript": {"function_declaration": "func", "class_declaration": "class",
                   "method_definition": "method", "generator_function_declaration": "func",
                   "variable_declarator": "func", "pair": "method"},
    "typescript": {"function_declaration": "func", "class_declaration": "class",
                   "method_definition": "method", "interface_declaration": "interface",
                   "generator_function_declaration": "func", "type_alias_declaration": "type",
                   "enum_declaration": "enum", "variable_declarator": "func", "pair": "method"},
    "tsx": {"function_declaration": "func", "class_declaration": "class",
            "method_definition": "method", "interface_declaration": "interface",
            "generator_function_declaration": "func", "type_alias_declaration": "type",
            "enum_declaration": "enum", "variable_declarator": "func", "pair": "method"},
    "go": {"function_declaration": "func", "method_declaration": "method",
           "type_declaration": "type"},
    "java": {"method_declaration": "method", "class_declaration": "class",
              "interface_declaration": "interface"},
    "rust": {"function_item": "func", "struct_item": "struct", "impl_item": "impl"},
    "ruby": {"method": "func", "class": "class"},
    "c": {"function_definition": "func", "struct_specifier": "struct"},
    "cpp": {"function_definition": "func", "struct_specifier": "struct",
            "class_specifier": "class"},
}
# For "variable_declarator"/"pair" def types above: only counts as a
# definition if its value is one of these node types (a plain
# `const x = 5` or `{ key: 5 }` is NOT a definition and must be skipped).
_JS_FUNC_VALUE_TYPES = frozenset({
    "arrow_function", "function_expression", "function", "generator_function",
})
# Node types whose text (minus this file's own defs) is scanned for call-like

# identifiers, to build the "who calls whom" edges cheaply without a full
# type-checker/LSP.
_CALL_NODE_TYPES = {"call", "call_expression", "method_invocation"}
_IMPORT_NODE_TYPES = {
    "import_statement", "import_from_statement", "import_declaration",
    "use_declaration",
}

_IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
                ".mypy_cache", ".pytest_cache", ".idea", ".vscode", ".coder3"}
_MAX_PARSE_BYTES = 1_500_000     # don't parse absurd generated/minified files
_SYMBOL_MAX_PER_FILE = 3000      # symbols indexed per file (was 40: later functions/rules were invisible)
_OUTLINE_MAX = 40                # symbols shown per file in the one-line repo_map outline


# =========================================================================== #
#  Tier-1 (in-memory) + Tier-2 (on-disk) cache of parsed files
# =========================================================================== #
@dataclass
class _CacheEntry:
    mtime: float
    size: int
    content_hash: str
    symbols: list[dict]           # [{name, kind, start_line, end_line, calls:[...], byte_range}]
    imports: list[str]
    lang: str
    module_calls: list = None        # names called from top-level code (e.g. `main()` at the bottom)

    def __post_init__(self):
        if self.module_calls is None:
            self.module_calls = []


class ASTCache:
    """Multi-tier cache: L1 = in-memory dict (per-process, instant), L2 = a
    JSON file under <root>/.coder3/ast_cache.json (survives process restarts
    and is shared across subagent processes hitting the same workspace).
    Keyed by (relpath, content hash) so a file that hasn't changed since it
    was last parsed - the overwhelmingly common case across a long session
    or many subagent dispatches - never gets re-parsed or re-walked."""

    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.RLock()
        self._mem: dict[str, _CacheEntry] = {}
        self._disk_path = root / ".coder3" / "ast_cache.json"
        self._load_disk()

    def _load_disk(self) -> None:
        try:
            if self._disk_path.is_file():
                raw = json.loads(self._disk_path.read_text(encoding="utf-8"))
                for rel, d in raw.items():
                    self._mem[rel] = _CacheEntry(**d)
        except Exception:
            pass  # corrupt/missing cache -> just start cold, never fatal

    def _save_disk(self) -> None:
        try:
            self._disk_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {rel: e.__dict__ for rel, e in self._mem.items()}
            self._disk_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass  # caching is a pure speed optimisation - never let it break a call

    @staticmethod
    def _hash(data: bytes) -> str:
        return hashlib.sha1(data).hexdigest()[:16]

    def get_or_parse(self, path: Path, rel: str) -> Optional[_CacheEntry]:
        if path.suffix.lower() not in _EXT_LANG:
            return None
        try:
            stat = path.stat()
        except Exception:
            return None
        with self._lock:
            hit = self._mem.get(rel)
            if hit and hit.mtime == stat.st_mtime and hit.size == stat.st_size:
                return hit                                     # L1 hit, zero I/O beyond stat()
        try:
            data = path.read_bytes()
        except Exception:
            return None
        h = self._hash(data)
        with self._lock:
            hit = self._mem.get(rel)
            if hit and hit.content_hash == h:
                hit.mtime, hit.size = stat.st_mtime, stat.st_size   # touched but unchanged content
                return hit
        entry = _parse_file(path, data)
        if entry is None:
            return None
        entry.mtime, entry.size, entry.content_hash = stat.st_mtime, stat.st_size, h
        with self._lock:
            self._mem[rel] = entry
        self._save_disk_async()
        return entry

    def invalidate(self, rel: str) -> None:
        with self._lock:
            self._mem.pop(rel, None)

    def all_entries(self) -> dict[str, "_CacheEntry"]:
        """Snapshot of every currently-cached (rel_path -> parsed entry) pair
        - used by semantic-navigation (find_references_at/rename_symbol) to
        enumerate candidate files without reaching into the private `_mem`
        dict directly."""
        with self._lock:
            return dict(self._mem)

    _saving = False

    def _save_disk_async(self) -> None:
        # Fire-and-forget, debounced: one writer at a time, never stalls a tool call.
        if self._saving:
            return
        self._saving = True

        def run():
            try:
                time.sleep(1.0)
                self._save_disk()
            finally:
                self._saving = False
        threading.Thread(target=run, daemon=True).start()


# --------------------------------------------------------------------------- #
# Built-in fallback parsers (no dependencies). Used first for Python/JS/TS/CSS
# (stdlib `ast` for Python) and whenever tree-sitter is missing or broken, so
# symbol reads/edits keep working instead of silently degrading to plain text.
# --------------------------------------------------------------------------- #
import ast as _ast
import bisect as _bisect

_FALLBACK_LANGS = {"python", "javascript", "typescript", "tsx", "css", "html"}
import keyword as _keyword
_PY_KW = frozenset(_keyword.kwlist) | frozenset(_keyword.softkwlist)
_JS_KW = frozenset("if for while switch catch function return with else do try typeof new await "
                   "super import delete void throw yield case in of".split())


def _mask_code(text: str, line_comments: bool, backtick: bool) -> str:
    """Blank out comments and string contents (same length, newlines kept) so
    brace matching / regexes aren't fooled by '}' inside a string or comment."""
    out: list[str] = []
    i, n = 0, len(text)
    blank = lambda seg: re.sub(r"[^\n]", " ", seg)                # noqa: E731
    while i < n:
        c = text[i]
        if c == "/" and text[i + 1:i + 2] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(blank(text[i:j])); i = j; continue
        if line_comments and c == "/" and text[i + 1:i + 2] == "/" and text[i - 1:i] != ":":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(blank(text[i:j])); i = j; continue
        if c in "\"'" or (backtick and c == "`"):
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2; continue
                if text[j] == c:
                    j += 1; break
                if text[j] == "\n" and c != "`":
                    break
                j += 1
            j = min(j, n)
            out.append(blank(text[i:j])); i = j; continue
        out.append(c); i += 1
    return "".join(out)


def _brace_pairs(masked: str) -> dict[int, int]:
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for m in re.finditer(r"[{}]", masked):
        if m.group() == "{":
            stack.append(m.start())
        elif stack:
            pairs[stack.pop()] = m.start()
    return pairs


def _skip_parens(masked: str, i: int) -> int:
    """i points at '(' -> index just after its matching ')' (or -1)."""
    depth = 0
    for j in range(i, len(masked)):
        if masked[j] == "(":
            depth += 1
        elif masked[j] == ")":
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


class _SymMaker:
    """Builds symbol dicts in the same schema the tree-sitter path produces."""
    def __init__(self, text: str):
        self.text = text
        self.ascii = text.isascii()
        self.ls = [0] + [m.end() for m in re.finditer(r"\n", text)]

    def line(self, off: int) -> int:
        return _bisect.bisect_right(self.ls, off)

    def byte(self, off: int) -> int:
        return off if self.ascii else len(self.text[:off].encode("utf-8"))

    def make(self, name: str, kind: str, s: int, e: int, calls=(), **extra) -> dict:
        d = {"name": name, "kind": kind, "start_line": self.line(s), "end_line": self.line(max(s, e - 1)),
             "start_byte": self.byte(s), "end_byte": self.byte(e), "calls": sorted(calls)[:20]}
        d.update(extra)
        return d


def _fallback_css(text: str, scss: bool) -> list[dict]:
    masked = _mask_code(text, line_comments=scss, backtick=False)
    pairs = _brace_pairs(masked)
    mk = _SymMaker(text)
    syms: list[dict] = []
    scan = re.compile(r"[{};]")

    def block(i: int, end: int, prefix: str) -> None:
        while i < end:
            m = scan.search(masked, i, end)
            if not m:
                break
            if m.group() != "{":
                i = m.end(); continue
            j = m.start()
            close = pairs.get(j, end - 1)
            pre = masked[i:j]
            ss = i + (len(pre) - len(pre.lstrip()))
            sel = " ".join(re.sub(r"/\*.*?\*/", "", text[ss:j], flags=re.S).split())
            if sel.startswith("@"):
                syms.append(mk.make(sel, "at-rule", ss, close + 1))
                if sel.split()[0].lower() in ("@media", "@supports", "@layer", "@container", "@document"):
                    block(j + 1, close, sel + " > ")
            elif sel:
                syms.append(mk.make(prefix + sel, "rule", ss, close + 1))
            i = close + 1

    block(0, len(masked), "")
    return syms


_JS_DECLS = [
    (re.compile(r"^[ \t]*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*(?=\()", re.M), "func"),
    (re.compile(r"^[ \t]*(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)", re.M), "class"),
    (re.compile(r"^[ \t]*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=\n]+)?=\s*(?:async\s*)?"
                r"(?:function\b|\([^()]*\)\s*(?::[^=;{\n]*)?=>|[A-Za-z_$][\w$]*\s*=>)", re.M), "func"),
    (re.compile(r"^[ \t]*(?:export\s+)?(?:declare\s+)?(?:interface|enum)\s+([A-Za-z_$][\w$]*)", re.M), "interface"),
]
_JS_METHOD = re.compile(r"^[ \t]+(?:(?:public|private|protected|static|readonly|override)\s+)*(?:async\s+)?"
                        r"(?:get\s+|set\s+)?\*?(#?[A-Za-z_$][\w$]*)\s*(?=\()", re.M)
_JS_CALL = re.compile(r"(?<![\w$])([A-Za-z_$][\w$]*)\s*\(|\.\s*([A-Za-z_$][\w$]*)\s*\(")


def _fallback_js(text: str) -> tuple[list[dict], list[str], list[str]]:
    masked = _mask_code(text, line_comments=True, backtick=True)
    pairs = _brace_pairs(masked)
    mk = _SymMaker(text)
    found: list[tuple[int, int, str, str]] = []                     # (start, end, name, kind)

    def body_end(kind: str, m: re.Match) -> Optional[int]:
        """End offset of the declaration matched by `m`, or None if no body is found."""
        pos, head = m.end(), m.group()
        if head.rstrip().endswith("=>") or (kind == "func" and re.search(r"=>\s*$", head)):
            k = pos                                   # arrow function: `{...}` or a one-line expression
            while k < len(masked) and masked[k] in " \t":
                k += 1
            if masked[k:k + 1] == "{" and k in pairs:
                e = pairs[k] + 1
                return e + 1 if masked[e:e + 1] == ";" else e
            eol = masked.find("\n", k)
            return len(masked) if eol < 0 else eol
        k = pos
        if kind == "func":                            # skip the (params) so `{` in destructuring isn't the body
            p = masked.find("(", pos - 1 if masked[pos - 1:pos] == "(" else pos)
            if p >= 0:
                k = max(_skip_parens(masked, p), pos)
        b = masked.find("{", k)
        return pairs[b] + 1 if b >= 0 and b in pairs else None

    for rx, kind in _JS_DECLS:
        for m in rx.finditer(masked):
            e = body_end(kind, m)
            if e is None:
                continue
            s = m.start() + (len(m.group()) - len(m.group().lstrip()))
            found.append((s, e, m.group(1), kind))
    # methods inside classes: only members at the class's own brace level
    for s, e, name, kind in list(found):
        if kind != "class":
            continue
        ob = masked.find("{", s)
        for m in _JS_METHOD.finditer(masked, ob + 1, e):
            if m.group(1) in _JS_KW:
                continue
            if masked.count("{", ob + 1, m.start()) != masked.count("}", ob + 1, m.start()):
                continue                              # nested inside another member, not a class-level method
            p = masked.find("(", m.end() - 1)
            k = _skip_parens(masked, p)
            b = masked.find("{", k) if k > 0 else -1
            if b < 0 or b not in pairs or ";" in masked[k:b]:
                continue                              # overload/abstract declaration without a body
            ms = m.start() + (len(m.group()) - len(m.group().lstrip()))
            found.append((ms, pairs[b] + 1, f"{name}.{m.group(1)}", "method"))

    seen, syms = set(), []
    for s, e, name, kind in sorted(found):
        if (s, name) in seen or len(syms) >= _SYMBOL_MAX_PER_FILE:
            continue
        seen.add((s, name))
        calls = set() if kind == "class" else \
            {a or b for a, b in _JS_CALL.findall(masked[s:e])} - _JS_KW - {name.split(".")[-1]}
        syms.append(mk.make(name.split(".")[-1], kind, s, e, calls, qualname=name))
    imports = [l.strip()[:200] for l in re.findall(r"^\s*import\b[^\n]*", text, re.M)][:30]
    top = list(masked)                                         # blank every function/class -> what's left is top-level code
    for s_, e_, _n, _k in found:
        for i in range(s_, min(e_, len(top))):
            if top[i] != "\n":
                top[i] = " "
    mcalls = {a or b for a, b in _JS_CALL.findall("".join(top))} - _JS_KW
    return syms, imports, sorted(mcalls)[:40]


def _fallback_python(data: bytes, text: str) -> tuple[list[dict], list[str], list[str]]:
    tree = _ast.parse(text)                                          # SyntaxError -> caller falls back
    ls = [0] + [m.end() for m in re.finditer(b"\n", data)]
    lines_b = data.split(b"\n")
    syms: list[dict] = []
    imports: list[str] = []

    def calls_of(node) -> set[str]:
        out = set()
        for n in _ast.walk(node):
            if isinstance(n, _ast.Call):
                f = n.func
                if isinstance(f, _ast.Name):
                    out.add(f.id)
                elif isinstance(f, _ast.Attribute):
                    out.add(f.attr)
        return out

    def visit(node, prefix: str, in_class: bool) -> None:
        for ch in _ast.iter_child_nodes(node):
            if isinstance(ch, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                first = min([d.lineno for d in ch.decorator_list] + [ch.lineno])
                lb = lines_b[first - 1]
                sb = ls[first - 1] + (len(lb) - len(lb.lstrip()))    # decorators included in the range
                eb = ls[ch.end_lineno - 1] + ch.end_col_offset
                q = prefix + ch.name
                is_cls = isinstance(ch, _ast.ClassDef)
                syms.append({"name": ch.name, "qualname": q,
                             "kind": "class" if is_cls else ("method" if in_class else "func"),
                             "start_line": first, "end_line": ch.end_lineno, "def_line": ch.lineno,
                             "start_byte": sb, "end_byte": eb,
                             "calls": [] if is_cls else sorted(calls_of(ch) - {ch.name})[:20]})
                visit(ch, q + ".", is_cls)
            elif isinstance(ch, (_ast.Assign, _ast.AnnAssign)) and (in_class or node is tree):
                tgt = ch.targets[0] if isinstance(ch, _ast.Assign) else ch.target
                if isinstance(tgt, _ast.Name) and (tgt.id.isupper() or ch.end_lineno - ch.lineno >= 3):
                    lb = lines_b[ch.lineno - 1]
                    syms.append({"name": tgt.id, "qualname": prefix + tgt.id, "kind": "const",
                                 "start_line": ch.lineno, "end_line": ch.end_lineno,
                                 "start_byte": ls[ch.lineno - 1] + (len(lb) - len(lb.lstrip())),
                                 "end_byte": ls[ch.end_lineno - 1] + ch.end_col_offset, "calls": []})
            elif isinstance(ch, _ast.Import):
                imports.append("import " + ", ".join(
                    a.name + (f" as {a.asname}" if a.asname else "") for a in ch.names))
            elif isinstance(ch, _ast.ImportFrom):
                imports.append(f"from {'.' * ch.level}{ch.module or ''} import " + ", ".join(
                    a.name + (f" as {a.asname}" if a.asname else "") for a in ch.names))
            elif isinstance(ch, _ast.stmt):
                visit(ch, prefix, in_class)

    visit(tree, "", False)
    syms.sort(key=lambda s: s["start_byte"])
    mcalls: set[str] = set()
    for stmt in tree.body:
        if not isinstance(stmt, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            mcalls |= calls_of(stmt)
    return syms[:_SYMBOL_MAX_PER_FILE], [i[:200] for i in imports[:30]], sorted(mcalls)[:40]


import html.parser as _hp

_HTML_VOID = frozenset("area base br col embed hr img input link meta param source track wbr".split())
_HTML_KEEP = frozenset("head body header nav main section article aside footer form dialog template".split())
_JS_TYPES = ("", "module", "text/javascript", "application/javascript", "text/babel", "text/jsx")


def _fallback_html(text: str) -> list[dict]:
    """Symbols for HTML: elements with an id, landmarks (header/nav/main/section/...),
    and <script>/<style> blocks - plus the JS functions / CSS rules INSIDE inline blocks,
    so single-file pages are navigable by symbol too."""
    mk = _SymMaker(text)
    found: list[dict] = []

    class _P(_hp.HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.stack: list[dict] = []

        def _pos(self) -> int:
            ln, col = self.getpos()
            return mk.ls[ln - 1] + col if ln - 1 < len(mk.ls) else len(text)

        def _done(self, rec: dict, end: int, ce: int) -> None:
            rec["end"], rec["ce"] = end, ce
            a, t = rec["attrs"], rec["tag"]
            if a.get("id") or t in _HTML_KEEP or t in ("script", "style"):
                found.append(rec)

        def _rec(self, tag, attrs) -> dict:
            start = self._pos()
            raw = self.get_starttag_text() or ""
            return {"tag": tag, "attrs": {k: (v or "") for k, v in attrs},
                    "start": start, "cs": start + len(raw)}

        def handle_starttag(self, tag, attrs):
            rec = self._rec(tag, attrs)
            if tag in _HTML_VOID or (self.get_starttag_text() or "").rstrip().endswith("/>"):
                self._done(rec, rec["cs"], rec["cs"])
            else:
                self.stack.append(rec)

        def handle_startendtag(self, tag, attrs):           # <x/> never opens a scope
            rec = self._rec(tag, attrs)
            self._done(rec, rec["cs"], rec["cs"])

        def handle_endtag(self, tag):
            pos = self._pos()
            gt = text.find(">", pos)
            end = gt + 1 if gt >= 0 else len(text)
            for k in range(len(self.stack) - 1, -1, -1):
                if self.stack[k]["tag"] == tag:
                    while len(self.stack) - 1 > k:          # implicitly closed (<p>, <li>, ...)
                        self._done(self.stack.pop(), pos, pos)
                    self._done(self.stack.pop(), end, pos)
                    return

    p = _P()
    try:
        p.feed(text)
        p.close()
    except Exception:                                        # malformed HTML: keep what we found
        pass
    for r in p.stack:
        p._done(r, len(text), len(text))

    syms: list[dict] = []
    for r in sorted(found, key=lambda r: (r["start"], -r["end"])):
        a, tag = r["attrs"], r["tag"]
        if a.get("id"):
            name = f"#{a['id']}"
        else:
            cls = (a.get("class", "").split() or [""])[0]
            name = f"{tag}.{cls}" if cls else tag
        extra = {"aliases": [a["id"], f"{tag}#{a['id']}"]} if a.get("id") else {}
        syms.append(mk.make(name, "element", r["start"], r["end"], qualname=name, tag=tag, **extra))
        if tag in ("script", "style") and r["ce"] - r["cs"] > 1 and not a.get("src"):
            seg = text[r["cs"]:r["ce"]]
            if tag == "script" and a.get("type", "").lower() not in _JS_TYPES:
                continue
            try:
                inner = _fallback_js(seg)[0] if tag == "script" else _fallback_css(seg, False)
            except Exception:
                continue
            dl, db = mk.line(r["cs"]) - 1, mk.byte(r["cs"])
            for s in inner:
                s["start_line"] += dl; s["end_line"] += dl
                s["start_byte"] += db; s["end_byte"] += db
                syms.append(s)
    syms.sort(key=lambda s: (s["start_byte"], -s["end_byte"]))
    return syms[:_SYMBOL_MAX_PER_FILE]


def _parse_fallback(path: Path, data: bytes, lang: str) -> Optional[_CacheEntry]:
    try:
        text = data.decode("utf-8", "replace")
        mcalls: list[str] = []
        if lang == "python":
            syms, imps, mcalls = _fallback_python(data, text)
        elif lang == "css":
            syms, imps = _fallback_css(text, path.suffix.lower() == ".scss"), []
        elif lang == "html":
            syms, imps = _fallback_html(text), []
        else:
            syms, imps, mcalls = _fallback_js(text)
    except Exception:
        return None
    return _CacheEntry(mtime=0, size=0, content_hash="", symbols=syms[:_SYMBOL_MAX_PER_FILE],
                       imports=imps, lang=lang, module_calls=mcalls)


def _parse_file(path: Path, data: bytes) -> Optional[_CacheEntry]:
    lang = _EXT_LANG.get(path.suffix.lower())
    if not lang or len(data) > _MAX_PARSE_BYTES:
        return None
    # Python/CSS/HTML fallbacks are themselves real parsers (ast, a
    # brace-matching CSS scanner, html.parser) - not regex approximations -
    # so they stay primary for those three: they're already as accurate as a
    # grammar and give richer output (Python: exact qualnames + decorator
    # ranges via `ast`; nothing to gain from re-deriving that with tree-sitter).
    #
    # js/ts/tsx are different: _fallback_js is genuinely regex-based and
    # verifiably misses real-world patterns a grammar catches for free -
    # object-literal methods (`{ onClick: () => {} }`), named function
    # expressions nested in object values, etc. (confirmed by direct
    # comparison: tree-sitter finds `onLoad`/`handlers.click` from
    # `const config = { onLoad: () => ..., handlers: { click: function
    # handleClick(){} } }`; the regex fallback finds neither). So for these
    # three, tree-sitter is PRIMARY and the regex walk is the fallback,
    # tried only if tree-sitter's own grammar is unavailable/broken or
    # produces nothing.
    ts_first = lang in ("javascript", "typescript", "tsx")
    if lang in _FALLBACK_LANGS and not ts_first:
        entry = _parse_fallback(path, data, lang)
        if entry is not None and entry.symbols:
            return entry
    if HAVE_TREE_SITTER and lang in _DEF_NODE_TYPES:
        entry = _parse_treesitter(path, data, lang)
        if entry is not None and (entry.symbols or not ts_first):
            return entry
    if ts_first:
        # tree-sitter unavailable/broken/empty: regex walk as last resort.
        return _parse_fallback(path, data, lang)
    return None


def _parse_treesitter(path: Path, data: bytes, lang: str) -> Optional[_CacheEntry]:
    try:
        parser = get_parser(lang)
        tree = parser.parse(data)
    except Exception:
        # This one language's grammar package may be missing/broken even
        # though tree_sitter core + python are fine (HAVE_TREE_SITTER=True
        # only guarantees the python grammar loaded at import time).
        return None
    def_types = _DEF_NODE_TYPES.get(lang, {})
    func_value_types = _JS_FUNC_VALUE_TYPES if lang in ("javascript", "typescript", "tsx") else frozenset()
    symbols: list[dict] = []
    imports: list[str] = []

    def node_text(n) -> str:
        return data[n.start_byte:n.end_byte].decode("utf-8", "replace")

    def field(n, name: str):
        for i, c in enumerate(n.children):
            if n.field_name_for_child(i) == name:
                return c
        return None

    def find_name(n) -> Optional[str]:
        """Name for a definition node. `variable_declarator`/`pair` (the
        js/ts "const X = () => {}" / "{ key: () => {} }" patterns) take
        their name from a sibling FIELD, not a descendant of the value -
        the arrow_function/function_expression node itself has no name.
        Everything else falls back to the first identifier-ish child
        (function_declaration.name, class_declaration.name, etc)."""
        if n.type == "variable_declarator":
            nm = field(n, "name")
            return node_text(nm) if nm is not None and nm.type == "identifier" else None
        if n.type == "pair":
            nm = field(n, "key")
            return node_text(nm) if nm is not None and nm.type in ("property_identifier", "string") else None
        for child in n.children:
            if child.type in ("identifier", "type_identifier", "property_identifier"):
                return node_text(child)
        return None

    def is_real_def(n) -> bool:
        """variable_declarator/pair only count as a definition when their
        VALUE is itself a function/arrow/class - `const x = 5` and
        `{ retries: 3 }` are ordinary bindings, not definitions, and must
        not be recorded (would flood the outline with noise)."""
        if n.type not in ("variable_declarator", "pair"):
            return True
        val = field(n, "value")
        return val is not None and val.type in func_value_types

    def collect_calls(n, own_name: str, out: set[str]):
        if n.type in _CALL_NODE_TYPES:
            for c in n.children:
                if c.type in ("identifier", "attribute", "member_expression", "field_expression"):
                    txt = node_text(c).split("(")[0].strip()
                    txt = txt.split(".")[-1] if "." in txt else txt
                    if txt and txt != own_name:
                        out.add(txt)
                    break
        for c in n.children:
            collect_calls(c, own_name, out)

    def qual_kind(n) -> str:
        base = def_types[n.type]
        if n.type == "variable_declarator" and field(n, "value") is not None \
                and field(n, "value").type in ("generator_function", "function_expression") \
                and node_text(field(n, "value")).lstrip().startswith("function*"):
            return "func"
        return base

    def effective_span(n, ancestors: list) -> tuple[int, int]:
        """Widen a definition's byte range to include an enclosing
        `export`/`export default` keyword, and (for `const X = ...`) the
        `const`/`let`/`var` keyword and trailing `;` - matching what
        _fallback_js already did (confirmed by direct comparison) and, more
        importantly, what a model naturally writes back when it patches an
        exported symbol. Without this, patching 'plainHelper' with
        replacement text that itself starts with 'export function ...'
        produced 'export export function ...' - a real, reproducible
        double-keyword bug caught in end-to-end testing, not hypothetical.
        `ancestors` is [closest_parent, ..., root] so a variable_declarator
        can see past its immediate `lexical_declaration` parent to check
        THAT node's own parent for an export_statement wrapper."""
        start, end = n.start_byte, n.end_byte
        chain = [n] + ancestors
        if n.type == "variable_declarator" and len(ancestors) >= 1 and ancestors[0].type == "lexical_declaration":
            start, end = ancestors[0].start_byte, ancestors[0].end_byte     # include const/let/var + trailing ';'
            chain = ancestors                             # check export_statement against lexical_declaration's parent
        outer = chain[1] if len(chain) > 1 else None
        if outer is not None and outer.type == "export_statement":
            start = outer.start_byte
            end = max(end, outer.end_byte)
        return start, end

    def walk(n, depth: int, class_stack: list[str], ancestors: list):
        if n.type in _IMPORT_NODE_TYPES:
            txt = node_text(n).strip().replace("\n", " ")
            if len(txt) < 200:
                imports.append(txt)
        recorded = False
        rec_name = None
        if n.type in def_types and len(symbols) < _SYMBOL_MAX_PER_FILE and is_real_def(n):
            rec_name = find_name(n)
            if rec_name:                                # unnamed (e.g. anonymous default export) -> skip, still descend
                kind = qual_kind(n)
                if n.type in ("method_definition", "pair") and class_stack:
                    kind = "method" if kind not in ("class", "interface", "enum", "type") else kind
                qualname = ".".join(class_stack + [rec_name]) if class_stack and kind == "method" else rec_name
                calls: set[str] = set()
                collect_calls(n, rec_name, calls)
                sb, eb = effective_span(n, ancestors)
                symbols.append({
                    "name": rec_name, "qualname": qualname, "kind": kind,
                    "start_line": data[:sb].count(b"\n") + 1, "end_line": data[:eb].count(b"\n") + 1,
                    "start_byte": sb, "end_byte": eb,
                    "calls": sorted(calls)[:20],
                })
                recorded = True
        nested_class_stack = class_stack + [rec_name or "?"] \
            if recorded and n.type in ("class_declaration", "class") else class_stack
        child_ancestors = [n] + ancestors
        for c in n.children:
            walk(c, depth + 1, nested_class_stack, child_ancestors)

    try:
        walk(tree.root_node, 0, [], [])
    except RecursionError:
        pass
    if not symbols and not imports:
        return None
    return _CacheEntry(mtime=0, size=0, content_hash="", symbols=symbols,
                        imports=imports[:30], lang=lang)


# =========================================================================== #
#  Dependency / symbol graph + PageRank ranking
# =========================================================================== #
_NON_CALLABLE = frozenset({"const", "element", "rule", "at-rule", "heading"})
_MAX_NAME_FANOUT = 8          # a name defined in >8 places (get, run, ...) is too ambiguous to draw edges for


# --------------------------------------------------------------------------- #
# Import-aware cross-file resolution.
#
# The name-only graph below (`DependencyGraph._rebuild`/`link`) treats every
# call to `validate(...)` as possibly targeting EVERY `validate` defined
# anywhere in the repo (deduped only by an arbitrary fan-out cap). That is
# wrong whenever two files define a same-named symbol and a third file
# imports a SPECIFIC one of them - the caller gets attributed to both
# definitions instead of the one it actually imported. `ImportResolver`
# fixes this for Python (the language `_fallback_python` already gives us
# real `from X import Y` strings for) by turning each file's import lines
# into a precise "local name -> defining file" table, so `references()`/
# `find_callers()` can prefer an exact import-resolved edge over the
# ambiguous by-name fallback. Not a full type checker (it won't follow
# re-exports through `__init__.py` wildcard imports, or resolve `self.x()`
# through a class hierarchy) but it removes the single most common source
# of wrong "who calls this" answers in a multi-file repo: same-named
# functions/classes in different modules.
# --------------------------------------------------------------------------- #
_PY_IMPORT_RE = re.compile(
    r"^(?:(?P<from>from\s+(?P<level>\.*)(?P<mod>[\w.]*)\s+import\s+(?P<names>.+))"
    r"|(?P<plain>import\s+(?P<mods>[\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+(?:\s+as\s+\w+)?)*)))$"
)
# ES module imports: `import {A, B as C} from "./x"`, `import D from "./x"`,
# `import * as ns from "./x"`, and the side-effect form `import "./x"`.
# Relative specifiers only ("./" or "../") - bare specifiers ("react",
# "lodash") are npm packages with no in-repo file to resolve to, and are
# left alone (goto_definition on those correctly reports "not resolvable"
# rather than guessing).
_JS_IMPORT_RE = re.compile(
    r'^import\s+(?:'
    r'(?P<named>\{[^}]*\})'
    r'|(?P<default>[A-Za-z_$][\w$]*)\s*,?\s*(?P<named2>\{[^}]*\})?'
    r'|(?:\*\s+as\s+(?P<ns>[A-Za-z_$][\w$]*))'
    r')?\s*(?:from\s+)?[\'"](?P<spec>\.\.?/[^\'"]+)[\'"]'
)
_JS_IMPORT_NAME_RE = re.compile(r'([A-Za-z_$][\w$]*)\s*(?:as\s+([A-Za-z_$][\w$]*))?')
_JS_RESOLVE_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", "/index.ts", "/index.tsx", "/index.js")


def _split_import_names(names: str) -> list[tuple[str, str]]:
    """'foo, bar as baz' -> [('foo','foo'), ('bar','baz')] (local_name, real_name)."""
    out = []
    for part in names.split(","):
        part = part.strip().strip("()")
        if not part:
            continue
        if " as " in part:
            real, local = part.split(" as ", 1)
        else:
            real = local = part
        out.append((local.strip(), real.strip()))
    return out


def _resolve_js_specifier(rel_dir: str, spec: str, by_stem: dict[str, str]) -> Optional[str]:
    """'./user-service' relative to rel_dir -> the actual indexed file path,
    trying each real extension/index-file form in turn (the import
    specifier itself never includes the extension)."""
    base = str((Path(rel_dir) / spec).as_posix())
    base = re.sub(r"^(\./)+", "", base)
    for ext in _JS_RESOLVE_EXTS:
        cand = base + ext if not ext.startswith("/") else base + ext
        stem = cand.rsplit(".", 1)[0] if "." in Path(cand).name else cand
        if stem in by_stem:
            return by_stem[stem]
    # exact stem match as a last resort (handles ext already implied oddly)
    return by_stem.get(base)


class ImportResolver:
    """Per-repo table: (file, local_name) -> defining file, built from the
    exact `from X import Y` / `import X` strings every fallback/tree-sitter
    parser already emits into `_CacheEntry.imports`. Python-precise; for
    other languages this degrades to a no-op (callers fall back to the
    name-only graph, exactly as before)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # module dotted path (best-effort, relative-resolved) -> relpath
        self._module_to_file: dict[str, str] = {}
        # (file, local_name) -> relpath of the file that DEFINES it
        self._resolved: dict[tuple[str, str], str] = {}
        self._dirty = True
        self._entries: dict[str, _CacheEntry] = {}

    def update_file(self, rel: str, entry: Optional[_CacheEntry]) -> None:
        with self._lock:
            if entry is None:
                self._entries.pop(rel, None)
            else:
                self._entries[rel] = entry
            self._dirty = True

    @staticmethod
    def _module_candidates(rel_dir: str, level: int, mod: str) -> list[str]:
        """Best-effort dotted-module -> file-path candidates, handling
        relative-import dots (level>0) by walking up `rel_dir`."""
        base_parts = [p for p in Path(rel_dir).parts if p not in (".", "")]
        if level:
            # level=1 -> same package (this dir); level=2 -> parent package; etc.
            up = level - 1
            base_parts = base_parts[: len(base_parts) - up] if up else base_parts
        mod_parts = mod.split(".") if mod else []
        parts = base_parts + mod_parts
        stem = "/".join(parts) if parts else ""
        cands = []
        if stem:
            cands.append(stem + ".py")
            cands.append(stem + "/__init__.py")
        return cands

    def _rebuild(self) -> None:
        # Index every known file by its "module stem" (path without extension)
        # so absolute-style `import pkg.mod` can also resolve even without a
        # real installed-package view of the repo. Shared across Python and
        # JS/TS - a JS `./utils` specifier and a Python `from . import utils`
        # both ultimately want "this directory's utils file, minus extension".
        by_stem: dict[str, str] = {}
        for rel in self._entries:
            p = Path(rel)
            by_stem[str(p.with_suffix(""))] = rel
            if p.name == "__init__.py":
                by_stem[str(p.parent)] = rel

        resolved: dict[tuple[str, str], str] = {}
        for rel, e in self._entries.items():
            if e.lang == "python":
                rel_dir = str(Path(rel).parent)
                for line in e.imports:
                    m = _PY_IMPORT_RE.match(line.strip())
                    if not m:
                        continue
                    if m.group("from"):
                        level = len(m.group("level") or "")
                        mod = m.group("mod") or ""
                        names = _split_import_names(m.group("names"))
                        target_file = None
                        for cand in self._module_candidates(rel_dir, level, mod):
                            stem = cand[:-3] if cand.endswith(".py") else cand
                            if stem in by_stem:
                                target_file = by_stem[stem]
                                break
                        if target_file is None:
                            # absolute import with no relative dots: also try
                            # matching by trailing path segments (repo root may
                            # not be the actual package root we assumed)
                            mod_parts = mod.split(".") if mod else []
                            for stem, f in by_stem.items():
                                if stem.split("/") == mod_parts or stem.endswith("/" + mod.replace(".", "/")):
                                    target_file = f
                                    break
                        if target_file and target_file != rel:
                            for local, _real in names:
                                resolved[(rel, local)] = target_file
                    elif m.group("mods"):
                        for one in m.group("mods").split(","):
                            one = one.strip()
                            modname = one.split(" as ")[0].strip()
                            local = one.split(" as ")[-1].strip() if " as " in one else modname.split(".")[0]
                            stem = modname.replace(".", "/")
                            if stem in by_stem and by_stem[stem] != rel:
                                resolved[(rel, local)] = by_stem[stem]
            elif e.lang in ("javascript", "typescript", "tsx"):
                rel_dir = str(Path(rel).parent)
                for line in e.imports:
                    m = _JS_IMPORT_RE.match(line.strip())
                    if not m or not m.group("spec"):
                        continue
                    target_file = _resolve_js_specifier(rel_dir, m.group("spec"), by_stem)
                    if not target_file or target_file == rel:
                        continue                          # npm package (unresolvable) or self-import
                    if m.group("ns"):
                        resolved[(rel, m.group("ns"))] = target_file
                    if m.group("default"):
                        resolved[(rel, m.group("default"))] = target_file
                    named = m.group("named") or m.group("named2")
                    if named:
                        for nm in _JS_IMPORT_NAME_RE.finditer(named):
                            local = nm.group(2) or nm.group(1)
                            resolved[(rel, local)] = target_file
        with self._lock:
            self._resolved = resolved
            self._dirty = False

    def resolve(self, from_file: str, local_name: str) -> Optional[str]:
        """Which file DEFINES `local_name` as used inside `from_file`, per
        that file's own imports? None if not import-resolvable (locally
        defined, dynamic, or a language this resolver doesn't cover)."""
        with self._lock:
            if self._dirty:
                self._rebuild()
            return self._resolved.get((from_file, local_name))

    def stats(self) -> dict:
        with self._lock:
            if self._dirty:
                self._rebuild()
            return {"files_indexed": len(self._entries), "resolved_imports": len(self._resolved)}


class DependencyGraph:
    """Symbol-level call/import graph, pure Python (no networkx): nodes are `file::qualname`
    plus one `file::<module>` node per file (imports + top-level calls). Edges are resolved
    BY NAME (same-file definitions win), so this is a fast heuristic - not a type-aware LSP.
    Rebuilt lazily when any file's parse changes."""

    def __init__(self, cache: ASTCache):
        self.cache = cache
        self._lock = threading.RLock()
        self._entries: dict[str, _CacheEntry] = {}
        self._file_symbols: dict[str, list[dict]] = {}
        self._nodes: dict[str, dict] = {}
        self._fwd: dict[str, set[str]] = {}
        self._rev: dict[str, set[str]] = {}
        self._dirty = True
        self.imports = ImportResolver()          # precise cross-file resolution (Python + JS/TS/TSX)
        self.precise_edges = 0                   # telemetry: edges resolved via imports, not name-guess
        self.ambiguous_edges = 0                 # telemetry: edges resolved by name-only fallback

    def update_file(self, rel: str, entry: Optional[_CacheEntry]) -> None:
        with self._lock:
            if entry is None:
                self._entries.pop(rel, None)
                self._file_symbols.pop(rel, None)
            else:
                self._entries[rel] = entry
                self._file_symbols[rel] = entry.symbols
            self._dirty = True
        self.imports.update_file(rel, entry)

    @staticmethod
    def _qual(s: dict) -> str:
        return s.get("qualname") or s["name"]

    def _rebuild(self) -> None:
        nodes: dict[str, dict] = {}
        by_name: dict[str, list[str]] = {}
        for rel, e in self._entries.items():
            nodes[f"{rel}::<module>"] = {"file": rel, "name": "<module>", "line": 1}
            for s in e.symbols:
                if s["kind"] in _NON_CALLABLE:
                    continue
                nid = f"{rel}::{self._qual(s)}"
                nodes[nid] = {"file": rel, "name": s["name"], "line": s["start_line"], "kind": s["kind"]}
                by_name.setdefault(s["name"], []).append(nid)
        fwd: dict[str, set[str]] = {k: set() for k in nodes}
        rev: dict[str, set[str]] = {k: set() for k in nodes}
        self.precise_edges = 0
        self.ambiguous_edges = 0

        def link(src: str, callee: str, src_file: str) -> None:
            # 1) precise path: does src_file's OWN imports resolve this exact
            #    name to one specific file? If so, only link to that file's
            #    definition(s) of the name - never to a same-named symbol in
            #    an unrelated file the caller never imported.
            target_file = self.imports.resolve(src_file, callee)
            if target_file is not None:
                targets = [t for t in by_name.get(callee, []) if t.split("::", 1)[0] == target_file]
                if targets:
                    for t in targets:
                        if t != src:
                            fwd[src].add(t); rev[t].add(src)
                            self.precise_edges += 1
                    return
                # resolver named a file but that file has no matching def
                # (e.g. re-export) - fall through to the name-only guess
                # rather than dropping the edge entirely.
            # 2) fallback: name-only, same-file preferred, fan-out capped -
            #    this is the pre-existing heuristic, kept for locally defined
            #    symbols, dynamic dispatch, and non-Python languages.
            targets = by_name.get(callee, [])
            same = [t for t in targets if t.split("::", 1)[0] == src_file]
            targets = same or targets
            if len(targets) > _MAX_NAME_FANOUT:
                return
            for t in targets:
                if t != src:
                    fwd[src].add(t); rev[t].add(src)
                    self.ambiguous_edges += 1

        stems: dict[str, list[str]] = {}
        for rel in self._entries:
            stems.setdefault(Path(rel).stem, []).append(rel)
        for rel, e in self._entries.items():
            for s in e.symbols:
                if s["kind"] in _NON_CALLABLE:
                    continue
                for callee in s.get("calls", []) or []:
                    link(f"{rel}::{self._qual(s)}", callee, rel)
            for callee in e.module_calls or []:
                link(f"{rel}::<module>", callee, rel)
            # imports: module -> module (file level) and module -> the imported symbol
            for line in e.imports:
                toks = set(re.findall(r"[A-Za-z_]\w*", line))
                for tok in toks:
                    for other in stems.get(tok, []):
                        if other != rel:
                            fwd[f"{rel}::<module>"].add(f"{other}::<module>")
                            rev[f"{other}::<module>"].add(f"{rel}::<module>")
                            for nm in toks:
                                for nid in by_name.get(nm, []):
                                    if nid.startswith(other + "::"):
                                        fwd[f"{rel}::<module>"].add(nid); rev[nid].add(f"{rel}::<module>")
        self._nodes, self._fwd, self._rev, self._dirty = nodes, fwd, rev, False

    def pagerank(self) -> dict[str, float]:
        """node id -> importance (power iteration, damping 0.85). Never raises."""
        with self._lock:
            if self._dirty:
                self._rebuild()
            n = len(self._nodes)
            if n == 0:
                return {}
            d = 0.85
            pr = {k: 1.0 / n for k in self._nodes}
            for _ in range(40):
                dangling = sum(pr[k] for k, outs in self._fwd.items() if not outs)
                base = (1 - d) / n + d * dangling / n
                new = {k: base for k in self._nodes}
                for k, outs in self._fwd.items():
                    if outs:
                        share = d * pr[k] / len(outs)
                        for t in outs:
                            new[t] += share
                delta = sum(abs(new[k] - pr[k]) for k in new)
                pr = new
                if delta < 1e-7:
                    break
            return pr

    def file_scores(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        for nid, sc in self.pagerank().items():
            rel = nid.split("::", 1)[0]
            scores[rel] = scores.get(rel, 0.0) + sc
        return scores

    def references(self, name: str, limit: int = 25) -> dict:
        """{defs: [...], callers: [...], importers: [...], by_def: {...}} for a
        function/class name. `callers` stays a flat, deduped list for
        backward compatibility with existing callers of this API; `by_def`
        is new and groups callers under the SPECIFIC definition they were
        resolved to (via ImportResolver where possible) - this is what tells
        two same-named functions in different files apart instead of
        merging their callers into one list, which is the wrong answer as
        soon as a repo has more than one `validate`/`run`/`handle` etc."""
        with self._lock:
            if self._dirty:
                self._rebuild()
            want = name.strip().rstrip("()")
            defs = [nid for nid, m in self._nodes.items()
                    if m["name"] != "<module>" and (m["name"] == want or nid.split("::", 1)[1] == want)]
            callers: list[str] = []
            seen: set[str] = set()
            by_def: dict[str, list[str]] = {}
            for d in defs:
                def_label = f"{self._nodes[d]['file']}:{self._nodes[d]['line']}  {d.split('::', 1)[1]}"
                by_def[def_label] = []
                for c in sorted(self._rev.get(d, ())):
                    m = self._nodes[c]
                    label = f"{m['file']}:{m['line']}  " + ("(module level)" if m["name"] == "<module>"
                                                             else c.split("::", 1)[1])
                    by_def[def_label].append(label)
                    if c not in seen:
                        seen.add(c)
                        callers.append(label)
            importers = sorted({rel for rel, e in self._entries.items()
                                if any(re.search(rf"\b{re.escape(want)}\b", l) for l in e.imports)})
            return {"defs": [f"{self._nodes[d]['file']}:{self._nodes[d]['line']}  {d.split('::', 1)[1]}"
                             for d in defs][:limit],
                    "callers": callers[:limit], "importers": importers[:limit],
                    "by_def": {k: v[:limit] for k, v in by_def.items()},
                    "ambiguous": len(defs) > 1}

    def callers_of(self, symbol_name: str, limit: int = 20) -> list[str]:
        return self.references(symbol_name, limit)["callers"]


# =========================================================================== #
#  Public API used by coder3.py's Workspace
# =========================================================================== #
class RepoIntel:
    """One instance per Workspace. Owns the AST cache + dependency graph and
    exposes the four upgraded operations. Every method is safe to call even
    when tree_sitter/networkx are absent - it just returns None/[] and the
    caller in coder3.py falls back to the previous behaviour."""

    def __init__(self, root: Path):
        self.root = root
        self.cache = ASTCache(root)
        self.graph = DependencyGraph(self.cache)
        self._scanned_once = False

    # -- 1. graph-based repo map ------------------------------------------- #
    def ensure_scanned(self, files: list[Path]) -> None:
        """Parse (or fetch from cache) every given file and feed the
        dependency graph. Cheap on repeat calls thanks to ASTCache's
        mtime/hash short-circuit."""
        for f in files:
            rel = str(f.relative_to(self.root)) if f.is_absolute() else str(f)
            entry = self.cache.get_or_parse(f, rel)
            self.graph.update_file(rel, entry)
        self._scanned_once = True

    def ranked_files(self, files: list[Path], top_n: int = 25) -> list[tuple[str, float]]:
        """Files sorted by PageRank importance (files nothing depends on and
        that call nothing rank low; central modules everything imports/calls
        rank high). Empty list if graph tooling isn't available - caller
        should then fall back to plain alphabetical listing."""
        self.ensure_scanned(files if files else self._workspace_files())
        scores = self.graph.file_scores()
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_n]

    def symbol_outline(self, path: Path, max_chars: int = 150) -> str:
        """One short line for repo_map: top-level names (classes show their method count),
        capped in length - a 60-file map must stay cheap to read."""
        rel = str(path.relative_to(self.root)) if path.is_absolute() else str(path)
        entry = self.cache.get_or_parse(path, rel)
        if entry is None or not entry.symbols:
            return ""
        counts: dict[str, int] = {}
        for s_ in entry.symbols:
            if s_["kind"] == "method" and "." in (s_.get("qualname") or ""):
                head = s_["qualname"].rsplit(".", 1)[0]
                counts[head] = counts.get(head, 0) + 1
        parts: list[str] = []
        for s_ in entry.symbols:
            q = s_.get("qualname") or s_["name"]
            if s_["kind"] == "method" or (s_["kind"] in ("func",) and q != s_["name"]):
                continue
            n = counts.get(q)
            parts.append(f"{s_['name']}[{n}]" if n else s_["name"])
        out, total = [], 0
        for i, p_ in enumerate(parts):
            if total + len(p_) > max_chars:
                out.append(f"...+{len(parts) - i}")
                break
            out.append(p_)
            total += len(p_) + 2
        return ", ".join(out)

    # -- 2. AST-level chunking / selective node extraction ------------------ #
    def resolve_symbol(self, path: Path, name: str) -> tuple[Optional[dict], str]:
        """(symbol, "") on success, else (None, human-readable reason incl. closest
        matches) - so a failed symbol= call tells the model what to do next instead
        of silently turning into a text edit / whole-file read."""
        rel = str(path.relative_to(self.root)) if path.is_absolute() else str(path)
        entry = self.cache.get_or_parse(path, rel)
        if entry is None or not entry.symbols:
            return None, (f"no symbols could be indexed for {rel} (unsupported file type, or the file "
                          f"failed to parse) - use start/end line ranges or `old` text matching instead")
        want = " ".join(name.strip().rstrip("{").split())
        for pfx in ("def ", "class ", "function ", "async "):
            if want.startswith(pfx):
                want = want[len(pfx):].strip()
        want = want.rstrip("()").replace("::", ".")
        key = lambda s: s.get("qualname") or s["name"]              # noqa: E731
        exact = [s for s in entry.symbols if key(s) == want]
        if len(exact) == 1:
            return exact[0], ""
        loose = exact or [s for s in entry.symbols if s["name"] == want or key(s).endswith("." + want)
                          or key(s).endswith(" > " + want)
                          or (s["kind"] == "rule" and want in [p.strip() for p in s["name"].split(",")])
                          or want in s.get("aliases", ())]
        if len(loose) == 1:
            return loose[0], ""
        if len(loose) > 1:
            listing = ", ".join(f"{key(s)} (lines {s['start_line']}-{s['end_line']})" for s in loose[:8])
            return None, (f"'{name}' is ambiguous in {rel}: {listing}. Use a qualified name "
                          f"(e.g. Class.method) or a start/end line range.")
        names = [key(s) for s in entry.symbols]
        close = difflib.get_close_matches(want, names, n=4, cutoff=0.5)
        shown = ", ".join(names[:25]) + (" ..." if len(names) > 25 else "")
        return None, (f"symbol '{name}' not found in {rel}."
                      + (f" Closest: {', '.join(close)}." if close else "")
                      + f" Indexed symbols: {shown}")

    def extract_symbol(self, path: Path, symbol_name: str) -> Optional[dict]:
        """Return {name, kind, start_line, end_line, text} for one symbol, located by
        parse boundaries. None if not found/ambiguous/unsupported."""
        sym, _err = self.resolve_symbol(path, symbol_name)
        if sym is None:
            return None
        try:
            data = path.read_bytes()
            text = data[sym["start_byte"]:sym["end_byte"]].decode("utf-8", "replace")
        except Exception:
            return None
        return {**sym, "text": text}

    def find_symbols(self, name: str, limit: int = 25) -> list[dict]:
        """Where is `name` DEFINED? Searches every indexed file (functions, classes, methods,
        CSS rules, HTML ids, constants). Exact matches first, then prefix/substring."""
        if time.time() - getattr(self, "_last_scan", 0) > 5:
            self.ensure_scanned(self._workspace_files())
            self._last_scan = time.time()
        want = " ".join(name.strip().split()).rstrip("()")
        low = want.lower()
        ranked: list[tuple[int, dict]] = []
        for rel, entry in list(self.cache._mem.items()):
            for s in entry.symbols:
                q = s.get("qualname") or s["name"]
                if q == want or s["name"] == want or want in s.get("aliases", ()):
                    r = 0
                elif q.endswith("." + want) or q.endswith(" > " + want):
                    r = 1
                elif low and low in q.lower():
                    r = 2 if q.lower().startswith(low) else 3
                else:
                    continue
                ranked.append((r, {"file": rel, "kind": s["kind"], "name": q,
                                   "start_line": s["start_line"], "end_line": s["end_line"]}))
        ranked.sort(key=lambda t: (t[0], t[1]["file"], t[1]["start_line"]))
        return [d for _, d in ranked[:limit]]

    def cst_insert(self, path: Path, symbol_name: str, new_code: str) -> Optional[str]:
        """Return the file text with `new_code` inserted right AFTER symbol `symbol_name`
        (re-indented to match it, blank-line separated). None if the symbol can't be resolved."""
        import textwrap
        sym, _err = self.resolve_symbol(path, symbol_name)
        if sym is None:
            return None
        raw = path.read_bytes()
        line_end = raw.find(b"\n", sym["end_byte"])
        line_end = len(raw) if line_end < 0 else line_end + 1
        ls = raw.rfind(b"\n", 0, sym["start_byte"]) + 1
        first = raw[ls:sym["start_byte"]].decode("utf-8", "replace")
        indent = first if not first.strip() else ""
        block = textwrap.indent(textwrap.dedent(new_code.strip("\n")), indent).rstrip("\n") + "\n"
        before = raw[:line_end].decode("utf-8", "replace")
        after = raw[line_end:].decode("utf-8", "replace")
        if before and not before.endswith("\n"):
            before += "\n"
        if not before.endswith("\n\n"):
            before += "\n"                                   # blank line between symbol and new code
        if after.strip() and not after.startswith("\n"):
            block += "\n"
        return before + block + after

    def outline_text(self, path: Path, max_lines: int = 250) -> Optional[str]:
        """Multi-line outline (qualified name, kind, line range) for read_file on big files."""
        rel = str(path.relative_to(self.root)) if path.is_absolute() else str(path)
        entry = self.cache.get_or_parse(path, rel)
        if entry is None or len(entry.symbols) < 3:
            return None
        rows = []
        for s in entry.symbols[:max_lines]:
            q = s.get("qualname") or s["name"]
            rows.append(f"  {s['start_line']:>5}-{s['end_line']:<5} {s['kind']:<9} {q}")
        more = len(entry.symbols) - max_lines
        return "\n".join(rows) + (f"\n  ... {more} more symbols not listed (read_file(symbol=<name>) still works for them; "
                                            f"use search_code to find a name)" if more > 0 else "")

    # -- 3. CST-aware patching + fuzzy fallback ------------------------------ #
    def cst_patch(self, path: Path, symbol_name: str, new_body: str) -> Optional[str]:
        """Replace an entire function/class node's source text by AST
        location instead of substring match - immune to whitespace/comment
        drift elsewhere in the file, and never accidentally matches a
        second occurrence the way a raw string replace can. Returns the new
        full file text, or None if the symbol can't be located (caller then
        falls back to fuzzy_replace or a manual edit_file)."""
        sym = self.extract_symbol(path, symbol_name)
        if sym is None:
            return None
        try:
            data = path.read_bytes().decode("utf-8", "replace")
        except Exception:
            return None
        lines = data.splitlines(keepends=True)
        # start_byte/end_byte were computed against the original bytes; recompute
        # via a byte-accurate re-read to stay correct on non-ASCII content.
        raw = path.read_bytes()
        prefix = raw[:sym["start_byte"]].decode("utf-8", "replace")
        suffix = raw[sym["end_byte"]:].decode("utf-8", "replace")
        k = sym.get("def_line", sym["start_line"]) - sym["start_line"]      # decorator lines above `def`
        if k > 0 and not new_body.lstrip().startswith("@"):
            olines = sym["text"].splitlines(keepends=True)
            ind = olines[k][: len(olines[k]) - len(olines[k].lstrip())]
            new_body = "".join(olines[:k]) + ind + new_body.lstrip()
        return prefix + new_body + suffix

    @staticmethod
    def fuzzy_replace(text: str, old_snippet: str, new_snippet: str,
                       threshold: float = 0.85) -> tuple[Optional[str], str]:
        """`old_snippet` didn't match byte-for-byte. Tier A: same lines ignoring indentation /
        trailing spaces (deterministic; must be unique; new text is re-indented to fit).
        Tier B: best difflib window >= threshold (pruned with quick_ratio so big files stay fast)."""
        old_lines = old_snippet.strip("\n").splitlines()
        text_lines = text.splitlines(keepends=True)
        tl = [l.rstrip("\r\n") for l in text_lines]
        n = len(old_lines)
        if n == 0:
            return None, "error: empty snippet"
        want = [l.strip() for l in old_lines]
        hits = [i for i in range(len(tl) - n + 1) if [x.strip() for x in tl[i:i + n]] == want]
        if len(hits) > 1:
            return None, (f"error: snippet matches {len(hits)} places ignoring whitespace "
                          f"(lines {', '.join(str(h + 1) for h in hits[:5])}) - include more context")
        best_ratio, best_i, how = 0.0, -1, ""
        if len(hits) == 1:
            best_i, best_ratio, how = hits[0], 1.0, "matched ignoring indentation"
            f_ind = tl[best_i][: len(tl[best_i]) - len(tl[best_i].lstrip())]
            o_ind = old_lines[0][: len(old_lines[0]) - len(old_lines[0].lstrip())]
            new_snippet = "\n".join((f_ind + l[len(o_ind):]) if l.startswith(o_ind) and l.strip() else l
                                    for l in new_snippet.splitlines())
        else:
            sm = difflib.SequenceMatcher(None, autojunk=False)
            sm.set_seq2(old_snippet)
            for i in range(0, max(1, len(tl) - n + 1)):
                sm.set_seq1("\n".join(tl[i:i + n]))
                if sm.real_quick_ratio() <= best_ratio or sm.quick_ratio() <= best_ratio:
                    continue
                r = sm.ratio()
                if r > best_ratio:
                    best_ratio, best_i = r, i
            how = f"fuzzy-matched (similarity {best_ratio:.0%})"
        if best_i < 0 or best_ratio < threshold:
            return None, (f"error: no fuzzy match found (best similarity {best_ratio:.0%}, need "
                           f"{threshold:.0%}). Re-read the exact lines (read_file symbol=/start/end) "
                           f"and retry, or use symbol= to replace the whole function/rule.")
        block = new_snippet if new_snippet.endswith("\n") or new_snippet == "" else new_snippet + "\n"
        result = "".join(text_lines[:best_i]) + block + "".join(text_lines[best_i + n:])
        return result, f"{how} at line {best_i + 1}"

    # -- 4. cache/dependency introspection ---------------------------------- #
    def invalidate(self, rel: str) -> None:
        self.cache.invalidate(rel)

    def _workspace_files(self) -> list[Path]:
        import os
        out: list[Path] = []
        for dp, dns, fns in os.walk(self.root):
            dns[:] = [d for d in dns if d not in _IGNORE_DIRS and not d.startswith(".")]
            for fn in fns:
                if Path(fn).suffix.lower() in _EXT_LANG:
                    out.append(Path(dp) / fn)
            if len(out) > 4000:
                break
        return out

    def _fresh_scan(self) -> None:
        # the graph only knows files somebody already parsed -> scan lazily, at most every 5s
        if time.time() - getattr(self, "_last_scan", 0) > 5:
            self.ensure_scanned(self._workspace_files())
            self._last_scan = time.time()

    def callers_of(self, symbol_name: str) -> list[str]:
        self._fresh_scan()
        return self.graph.callers_of(symbol_name)

    def references(self, symbol_name: str) -> dict:
        self._fresh_scan()
        return self.graph.references(symbol_name)

    # -- 5. LSP-style semantic navigation ----------------------------------- #
    # goto_definition / find_references_at / rename_symbol give
    # occurrence-level, import-aware navigation - "what does the name AT
    # THIS EXACT SPOT resolve to", not just "everywhere a symbol with this
    # name is defined anywhere in the repo" (which `references()` above
    # already covered, and which is ambiguous the moment two files define a
    # same-named function). Built on ImportResolver (now Python + JS/TS/TSX)
    # for the cross-file part and each language's own real parser
    # (tree-sitter for js/ts/tsx, `ast` for python) for locating the exact
    # occurrence, rather than a second independent implementation.
    _WORD_RE = re.compile(r"[A-Za-z_$][\w$]*")

    def _word_at(self, path: Path, line: int, col: Optional[int]) -> Optional[tuple[str, int, int]]:
        """(word, line, col) of the identifier at 1-based `line`/0-based
        `col`. If `col` is omitted, the FIRST identifier on that line that
        isn't a language keyword - good enough for "click the line the
        model already quoted from read_file output" without asking it to
        count columns by hand."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        lines = text.splitlines()
        if not (1 <= line <= len(lines)):
            return None
        raw = lines[line - 1]
        if col is not None:
            for m in self._WORD_RE.finditer(raw):
                if m.start() <= col < m.end():
                    return m.group(), line, m.start()
            return None
        for m in self._WORD_RE.finditer(raw):
            if m.group() not in _JS_KW and m.group() not in _PY_KW:
                return m.group(), line, m.start()
        return None

    def goto_definition(self, path: Path, line: int, col: Optional[int] = None,
                        symbol: Optional[str] = None) -> dict:
        """Resolve the identifier at `path:line[:col]` (or an explicit
        `symbol` name, skipping the word-lookup) to its definition site.
        Import-aware: if the current file imports that exact name from a
        specific file, returns ONLY that definition - not every same-named
        symbol repo-wide. Falls back to the name-only defs list (same
        ambiguity handling as `references()`) when the name isn't a
        locally-imported one (e.g. it's defined in the same file, or the
        language/import form isn't resolver-covered).
        Returns {ok, word, definitions: [{file, line, end_line, kind, qualname}],
                 resolved_via: "import"|"name"|"none", error}."""
        self._fresh_scan()
        rel = str(path.relative_to(self.root)) if path.is_absolute() else str(path)
        if symbol:
            word = symbol.strip().rstrip("()").split(".")[-1]
        else:
            hit = self._word_at(path if path.is_absolute() else self.root / rel, line, col)
            if hit is None:
                return {"ok": False, "error": f"no identifier found at {rel}:{line}"
                        + (f":{col}" if col is not None else ""), "definitions": []}
            word, line, col = hit
        target_file = self.graph.imports.resolve(rel, word)
        entry = self.cache.get_or_parse(self.root / target_file, target_file) if target_file else None
        if entry is not None:
            hits = [s for s in entry.symbols
                    if s["name"] == word or (s.get("qualname") or "").split(".")[-1] == word]
            if hits:
                return {"ok": True, "word": word, "resolved_via": "import",
                        "definitions": [{"file": target_file, "line": s["start_line"],
                                         "end_line": s["end_line"], "kind": s["kind"],
                                         "qualname": s.get("qualname") or s["name"]} for s in hits]}
        # not import-resolvable (locally defined, dynamic import, or a
        # language the resolver doesn't cover yet) - fall back to name-only,
        # same ambiguity semantics as references()/find_symbol.
        refs = self.graph.references(word)
        if refs["defs"]:
            defs = []
            for d in refs["defs"]:
                fm = re.match(r"^(.+):(\d+)\s+(.+)$", d)
                if fm:
                    defs.append({"file": fm.group(1), "line": int(fm.group(2)),
                                "end_line": int(fm.group(2)), "kind": "?", "qualname": fm.group(3)})
            return {"ok": True, "word": word,
                    "resolved_via": "name" if len(defs) == 1 else "name (ambiguous)",
                    "definitions": defs}
        return {"ok": False, "word": word, "resolved_via": "none", "definitions": [],
                "error": f"no definition found for '{word}' (checked import-resolved target "
                        f"and repo-wide by name)"}

    def find_references_at(self, path: Path, line: int, col: Optional[int] = None,
                           symbol: Optional[str] = None, limit: int = 50) -> dict:
        """Every occurrence of the identifier at `path:line[:col]` across the
        repo, split into `definition` (the declaration site itself, if
        indexed as a symbol), `calls` (used as a function/constructor call),
        and `reads` (referenced but not called - passed as an argument,
        assigned from, read in an expression - the kind distinction a plain
        text search can't make). Text-scan based (regex word-boundary, like
        the existing `importers` field in references()) rather than a full
        type-checker, so it can't tell apart two unrelated same-named locals
        in different functions - it reports where the NAME is import-linked
        to this exact definition when resolvable, and everywhere else that
        name appears in files that import it, which is the same precision
        ceiling the rest of this module already operates at."""
        self._fresh_scan()
        rel = str(path.relative_to(self.root)) if path.is_absolute() else str(path)
        if symbol:
            word = symbol.strip().rstrip("()").split(".")[-1]
        else:
            hit = self._word_at(path if path.is_absolute() else self.root / rel, line, col)
            if hit is None:
                return {"ok": False, "error": f"no identifier found at {rel}:{line}", "occurrences": []}
            word = hit[0]
        pat = re.compile(rf"\b{re.escape(word)}\b")
        call_pat = re.compile(rf"\b{re.escape(word)}\s*\(")
        # Which files are actually relevant: the defining file(s) + every
        # file whose imports resolve `word` back to one of them (import-aware,
        # not "every file that happens to contain this substring").
        gd = self.goto_definition(path, line, col, symbol=symbol)
        def_files = {d["file"] for d in gd.get("definitions", [])}
        candidate_files = set(def_files)
        for f, e in self.cache.all_entries().items():
            if any(re.search(rf"\b{re.escape(word)}\b", l) for l in e.imports):
                candidate_files.add(f)
        if not candidate_files:                          # not import-tracked anywhere - scan whole repo by name
            candidate_files = set(self.cache.all_entries().keys())
        occurrences = []
        for f in sorted(candidate_files):
            fp = self.root / f
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            def_lines = {d["line"] for d in gd.get("definitions", []) if d["file"] == f}
            for i, raw in enumerate(text.splitlines(), start=1):
                if not pat.search(raw):
                    continue
                is_def = i in def_lines
                kind = "definition" if is_def else ("call" if call_pat.search(raw) else "read")
                occurrences.append({"file": f, "line": i, "kind": kind, "text": raw.strip()[:160]})
                if len(occurrences) >= limit:
                    break
            if len(occurrences) >= limit:
                break
        return {"ok": True, "word": word, "definition_files": sorted(def_files), "occurrences": occurrences}

    def rename_symbol(self, path: Path, line: int, col: Optional[int], new_name: str,
                      symbol: Optional[str] = None) -> dict:
        """Compute (but never write) a rename of every occurrence of the
        identifier at `path:line[:col]` across every file this resolver can
        connect to that definition - word-boundary text substitution
        restricted to the SAME candidate-file set find_references_at
        computes (import-linked files + the defining file), so an unrelated
        same-named local in an unconnected file is left untouched. Refuses
        if `new_name` isn't a valid identifier, or if it already appears to
        collide with an existing distinct symbol in any file that would be
        touched (best-effort safety check, not a type-checker).

        Always a dry run: this class is a read/analysis layer and never
        writes to disk anywhere else either - returns
        {ok, word, new_name, changes: [{file, count, new_text}]} for the
        CALLER (coder3.Workspace.rename_symbol) to show as a diff, get
        approval for, snapshot and write through the same pipeline every
        other multi-file edit goes through, instead of this method
        bypassing that entirely."""
        if not self._WORD_RE.fullmatch(new_name) or new_name in _JS_KW or new_name in _PY_KW:
            return {"ok": False, "error": f"'{new_name}' is not a safe identifier to rename to"}
        refs = self.find_references_at(path, line, col, symbol=symbol, limit=5000)
        if not refs.get("ok"):
            return refs
        word = refs["word"]
        if word == new_name:
            return {"ok": False, "error": "new name is the same as the current name"}
        touched = sorted({o["file"] for o in refs["occurrences"]})
        for f in touched:
            e = self.cache.all_entries().get(f)
            if e and any((s.get("qualname") or s["name"]).split(".")[-1] == new_name for s in e.symbols):
                return {"ok": False, "error": (f"refusing to rename: '{new_name}' already defines something "
                                               f"in {f} - this would collide, not just rename")}
        pat = re.compile(rf"\b{re.escape(word)}\b")
        changes = []
        for f in touched:
            fp = self.root / f
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            new_text, n = pat.subn(new_name, text)
            if n == 0:
                continue
            changes.append({"file": f, "count": n, "old_text": text, "new_text": new_text})
        return {"ok": True, "word": word, "new_name": new_name, "changes": changes}

    def health(self) -> list[str]:
        """Real smoke-parse per backend - not just 'was the import ok'."""
        samples = {".py": "def a():\n    return 1\n", ".js": "function a(){ return 1 }\nconst b = () => 2;\n",
                   ".css": ".a { color: red }\n"}
        rows = []
        for ext, src in samples.items():
            e = _parse_file(Path("x" + ext), src.encode())
            n = len(e.symbols) if e else 0
            rows.append(f"  {ext:<4} {'OK  ' if n else 'FAIL'} ({n} symbols parsed from a test snippet)")
        rows.append("  tree-sitter (only used for go/java/rust/ruby/c/cpp): "
                    + ("working" if HAVE_TREE_SITTER else f"NOT working - {TS_ERROR or 'not installed'}"))
        e = _parse_fallback(Path("x.html"), b'<header id="top"><style>.a{color:red}</style></header>', "html")
        rows.append(f"  .html {'OK  ' if e and len(e.symbols) >= 3 else 'FAIL'} ({len(e.symbols) if e else 0} symbols parsed from a test snippet)")
        ist = self.graph.imports.stats()
        total_edges = self.graph.precise_edges + self.graph.ambiguous_edges
        pct = (100 * self.graph.precise_edges / total_edges) if total_edges else 0.0
        rows.append(f"  import resolver (Python, precise cross-file refs): {ist['files_indexed']} files indexed, "
                    f"{ist['resolved_imports']} name->file bindings resolved")
        rows.append(f"  call-graph precision: {self.graph.precise_edges} edges resolved via import (exact), "
                    f"{self.graph.ambiguous_edges} via name-only guess ({pct:.0f}% precise)")
        return rows

    def stats(self) -> str:
        n_files = len(self.graph._entries)
        n_syms = sum(len(v) for v in self.graph._file_symbols.values())
        backend = ["built-in py/js/ts/css parsers"]
        if HAVE_TREE_SITTER:
            backend.append("tree-sitter (go/java/rust/ruby/c/cpp)")
        backend.append("built-in call graph + PageRank + import-aware resolver")
        backend_s = " + ".join(backend)
        return f"repo intel: {n_files} files parsed, {n_syms} symbols indexed [{backend_s}]"
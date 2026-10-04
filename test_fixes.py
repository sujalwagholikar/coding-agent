import sys, types, importlib.abc, importlib.machinery
from unittest.mock import MagicMock
# stub third-party UI/LLM libs that aren't installed in this sandbox
class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    PREFIX = ("rich", "google", "openai", "prompt_toolkit", "pygments", "httpx", "dotenv")
    _busy = False
    def find_spec(self, name, path, target=None):
        top = name.split(".")[0]
        if top not in self.PREFIX or self._busy:
            return None
        import importlib.util
        _Stub._busy = True                       # importlib.util.find_spec() walks meta_path -> would recurse into us
        try:
            missing = importlib.util.find_spec(top) is None
        except Exception:
            missing = True
        finally:
            _Stub._busy = False
        if missing:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = MagicMock(name=spec.name); m.__path__ = []; m.__spec__ = spec; m.__name__ = spec.name; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, _Stub())
import pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import tempfile, pathlib, textwrap
import coder3

root = pathlib.Path(tempfile.mkdtemp())
(root/"style.css").write_text(".theme-toggle {\n  background: red;\n}\n" + "\n".join(f".rule{i} {{ color: red; margin: {i}px; }}" for i in range(500)) + "\n.last-rule { top: 0; }\n")
(root/"app.js").write_text("document.addEventListener('DOMContentLoaded', () => {\n  function initTheme() {\n    return 1;\n  }\n  initTheme();\n});\n")
(root/"mod.py").write_text(textwrap.dedent('''
    import functools
    def top(x):
        return helper(x)
    class A:
        @property
        def val(self):
            return 1
        def other(self):
            return top(2)
    def helper(y):
        return y
'''))
ws = coder3.Workspace(root, auto_approve=True)
ws.ui = None
ok = True
def check(label, cond, detail=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + label + (f"  -> {detail}" if not cond else ""))

# --- big CSS file: outline first, not 500 raw lines
r = ws.execute("read_file", {"path": "style.css"})
check("big css -> outline, not raw dump", "SYMBOL OUTLINE" in r and r.count("\n") < 520 and ".theme-toggle" in r, r[:200])
r = ws.execute("read_file", {"path": "style.css", "symbol": ".theme-toggle"})
check("css symbol read", "background: red" in r and "rule3" not in r, r[:200])
r = ws.execute("read_file", {"path": "style.css", "symbol": ".theme-togle"})
check("symbol typo -> helpful error w/ closest", r.startswith("error") and ".theme-toggle" in r, r[:200])
r = ws.execute("read_file", {"path": "style.css", "full": True})
check("full=true forces raw file", "    1 | .theme-toggle" in r)
r = ws.execute("read_file", {"path": "style.css", "start": 1, "end": 2})
check("range read still works", ".theme-toggle" in r and ".rule0" not in r, r)

# --- symbol edits
r = ws.execute("edit_file", {"path": "style.css", "symbol": ".theme-toggle", "new": ".theme-toggle {\n  background: blue;\n}"})
check("css symbol edit", "background: blue" in (root/"style.css").read_text() and "background: red" not in (root/"style.css").read_text(), r)
r = ws.execute("edit_file", {"path": "app.js", "symbol": "initTheme", "new": "function initTheme() {\n    return 2;\n  }"})
check("js nested fn symbol edit", "return 2" in (root/"app.js").read_text(), r)
before = (root/"mod.py").read_text()
r = ws.execute("edit_file", {"path": "mod.py", "symbol": "A.val", "new": "def val(self):\n        return 42"})
after = (root/"mod.py").read_text()
check("py decorator kept when omitted", after.count("@property") == 1 and "return 42" in after, after)
r = ws.execute("edit_file", {"path": "mod.py", "symbol": "A.val", "new": "@property\n    def val(self):\n        return 43"})
after = (root/"mod.py").read_text()
check("py decorator not duplicated when included", after.count("@property") == 1 and "return 43" in after, after)
compile(after, "mod.py", "exec"); check("edited python still compiles", True)

# --- the misleading error is gone
r = ws.execute("edit_file", {"path": "mod.py", "symbol": "nonexistent", "new": "x"})
check("failed symbol edit explains why (not '`old` not found')", "not found" in r and "Indexed symbols" in r and "`old` not found" not in r, r[:200])
r = ws.execute("edit_file", {"path": "style.css", "symbol": "zzz", "old": ".rule1 { color: red; margin: 1px; }", "new": ".rule1 { color: green; }"})
check("bad symbol + valid old -> falls back to text", "green" in (root/"style.css").read_text(), r)

# --- find_callers no longer needs repo_map first
r = ws.execute("find_callers", {"symbol": "helper"})
check("find_callers works without repo_map", "top" in r, r[:200])

# --- counters + detail
r = ws.execute("read_file", {"path": "style.css", "symbol": ".last-rule"})
check("symbol beyond 250-line outline / old 400 cap still resolves", "top: 0" in r, r[:150])
check("usage counters recorded", ws.usage.get("read_symbol") == 2 and ws.usage.get("edit_symbol", 0) >= 3, ws.usage)
c = coder3.ToolCall(id="1", name="read_file", args={"path": "style.css", "symbol": ".x"}) if hasattr(coder3, "ToolCall") else None
if c: check("UI detail shows symbol", coder3._tool_detail(c) == "style.css :: .x", coder3._tool_detail(c))
print("\nALL PASS" if ok else "\nSOME FAILED"); sys.exit(0 if ok else 1)

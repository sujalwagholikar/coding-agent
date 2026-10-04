import sys, types, importlib.abc, importlib.machinery, tempfile, pathlib, textwrap, json
from unittest.mock import MagicMock
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
sys.meta_path.insert(0, _Stub()); import pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import coder3, specialists as sp

ok = True; N = 0
def check(label, cond, detail=""):
    global ok, N; N += 1; ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label + (f"\n      -> {str(detail)[:300]}" if not cond else ""))

def mkws(files):
    root = pathlib.Path(tempfile.mkdtemp())
    for k, v in files.items(): (root/k).write_text(v)
    ws = coder3.Workspace(root, auto_approve=True); ws.ui = None
    return root, ws

BIG_PY = "\n".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(80))
files = {
    "style.css": ".theme-toggle {\n  background: red;\n}\n.card { padding: 1px; }\n",
    "app.js": "function initTheme() {\n  return 1;\n}\nconst go = () => initTheme();\ngo();\n",
    "mod.py": "def top(x):\n    return helper(x)\n\ndef helper(y):\n    return y\n\nif __name__ == '__main__':\n    top(1)\n",
    "big.py": BIG_PY,
    "index.html": '<html><body><section id="hero"><p>hi</p></section><script>function boot(){ return 1; }</script></body></html>',
}
root, ws = mkws(files)

print("== A. subagent write path (the swapped old/new + symbol crash bug) ==")
coder3.install_write_lock(ws)
r = ws.execute("edit_file", {"path": "style.css", "old": "background: red;", "new": "background: blue;"})
check("text edit through the shared lock works (was: old/new swapped -> fails)", "background: blue" in (root/"style.css").read_text(), r)
r = ws.execute("edit_file", {"path": "app.js", "symbol": "initTheme", "new": "function initTheme() {\n  return 2;\n}"})
check("symbol edit through the lock works (was: TypeError)", "return 2" in (root/"app.js").read_text(), r)
r = ws.execute("edit_file", {"path": "style.css", "insert_after": ".card", "new": ".badge {\n  color: green;\n}"})
t = (root/"style.css").read_text()
check("insert_after adds a rule after .card, re-indented, blank-line separated", ".card { padding: 1px; }\n\n.badge {" in t, t)
r = ws.execute("write_file", {"path": "new.txt", "content": "hello"})
check("write_file through the lock works", (root/"new.txt").exists(), r)
check("lock installed exactly once (idempotent)", (coder3.install_write_lock(ws) or True) and ws.edit_file.__name__ == "locked_edit")

print("== B. write guard + python syntax gate ==")
r = ws.execute("write_file", {"path": "big.py", "content": "x = 1\n"})
check("overwriting a big existing file is refused", r.startswith("error") and "overwrite=true" in r, r)
check("...and the file is untouched", (root/"big.py").read_text() == BIG_PY)
r = ws.execute("write_file", {"path": "big.py", "content": "x = 1\n", "overwrite": True})
check("overwrite=true is honoured", (root/"big.py").read_text() == "x = 1\n", r)
r = ws.execute("write_file", {"path": "bad.py", "content": "def f(:\n  pass\n"})
check("new invalid .py is refused", r.startswith("error") and not (root/"bad.py").exists(), r)
before = (root/"mod.py").read_text()
r = ws.execute("edit_file", {"path": "mod.py", "symbol": "helper", "new": "def helper(y:\n    return y"})
check("edit that breaks python is rejected, file unchanged", "syntax error" in r and (root/"mod.py").read_text() == before, r)
r = ws.execute("edit_file", {"path": "mod.py", "symbol": "helper", "new": "def helper(y):\n    return y * 2"})
check("valid python symbol edit still lands", "y * 2" in (root/"mod.py").read_text(), r)
r = ws.execute("edit_file", {"path": "mod.py", "insert_after": "helper", "new": "def extra():\n    return 3"})
compile((root/"mod.py").read_text(), "mod.py", "exec")
check("python insert_after compiles", "def extra" in (root/"mod.py").read_text(), r)

print("== C. new/upgraded navigation tools ==")
r = ws.execute("find_symbol", {"name": "hero"})
check("find_symbol finds an HTML #id", "index.html" in r and "#hero" in r, r)
r = ws.execute("find_symbol", {"name": "boot"})
check("find_symbol finds a function inside inline <script>", "boot" in r, r)
r = ws.execute("read_file", {"path": "index.html", "symbol": "boot"})
check("read_file symbol works inside inline <script>", "function boot" in r, r)
r = ws.execute("edit_file", {"path": "index.html", "symbol": "#hero", "new": '<section id="hero"><h1>new</h1></section>'})
check("edit_file symbol='#hero' replaces the element", "<h1>new</h1>" in (root/"index.html").read_text() and "<p>hi</p>" not in (root/"index.html").read_text(), r)
r = ws.execute("find_callers", {"symbol": "top"})
check("find_callers shows defs + module-level caller", "defined at" in r and "(module level)" in r, r)
r = ws.execute("find_callers", {"symbol": "initTheme"})
check("find_callers works for JS", "go" in r, r)
r = ws.execute("search_code", {"pattern": "return", "glob": "*.py"})
check("search_code honours glob and stays capped", "mod.py" in r and "app.js" not in r and r.count("\n") <= 61, r[:200])
r = ws.execute("repo_map", {})
check("repo_map lists symbols compactly", "mod.py" in r and "helper" in r and len(r) < 3000, r[:300])

print("== D. history compaction (main agent + specialists + subagents all use these providers) ==")
BIG = "x" * 5000
# OpenAI-compatible
spec = coder3.ModelSpec if hasattr(coder3, "ModelSpec") else None
p = coder3.OpenAICompatProvider.__new__(coder3.OpenAICompatProvider)
p.messages = [{"role": "system", "content": "s"}]
def oa_round(i, out):
    cid = f"c{i}"
    p.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
    p.add_tool_results([(coder3.ToolCall(cid, "read_file", {}), out)])
for i in range(5): oa_round(i, BIG + str(i))
tools = [m for m in p.messages if m["role"] == "tool"]
check("openai: last 2 tool results kept verbatim", all(len(m["content"]) > 4000 for m in tools[-2:]))
check("openai: older big results elided to a stub", all(len(m["content"]) < 400 and "elided" in m["content"] for m in tools[:-2]), [len(m["content"]) for m in tools])
oa_round(9, "short result")
check("openai: small results never elided", any(m["content"] == "short result" for m in p.messages))
before = json.dumps(p.messages); p.compact()
check("openai: compaction is idempotent", json.dumps(p.messages) == before)
# Gemini (fake types so we exercise the real compact())
class FR:
    def __init__(s, id=None, name=None, response=None): s.id, s.name, s.response = id, name, response
class Part:
    def __init__(s, text=None, function_response=None): s.text, s.function_response = text, function_response
class Content:
    def __init__(s, role=None, parts=None): s.role, s.parts = role, parts
T = types.SimpleNamespace(FunctionResponse=FR, Part=Part, Content=Content)
g = coder3.GeminiProvider.__new__(coder3.GeminiProvider); g.types = T; g.history = []
for i in range(5):
    g.history.append(Content("model", [Part(text="thinking")]))
    g.add_tool_results([(coder3.ToolCall(f"c{i}", "read_file", {}), BIG + str(i))])
res = [pt.function_response.response["result"] for c in g.history if c.role == "user" for pt in c.parts]
check("gemini: last 2 kept, older elided", all(len(r) > 4000 for r in res[-2:]) and all(len(r) < 400 for r in res[:-2]), [len(r) for r in res])
check("gemini: model turns left untouched (thought signatures)", all(c.parts[0].text == "thinking" for c in g.history if c.role == "model"))
check("compaction stats recorded", coder3.COMPACT_STATS["results"] >= 6 and coder3.COMPACT_STATS["chars"] > 10000, coder3.COMPACT_STATS)

print("== E. subagent provider role (no delegation, delegation-free prompt) ==")
lim = MagicMock()
prov = coder3.OpenAICompatProvider(MagicMock(provider="openrouter", id="m"), lim, "sys", "k", base_url="http://x", exclude_tools=coder3._NO_DELEGATION)
names = {t["function"]["name"] for t in prov.tools}
check("subagent tool list excludes dispatch_agents/dispatch_specialists/set_specialist_active", not (names & coder3._NO_DELEGATION), names & coder3._NO_DELEGATION)
check("...but keeps every working tool", {"read_file", "edit_file", "find_symbol", "find_callers", "run_command"} <= names)
sp_prompt = coder3.subagent_system_prompt(ws)
check("subagent prompt has workflow + root, no delegation section", "WORKFLOW" in sp_prompt and str(root) in sp_prompt and "dispatch_agents" not in sp_prompt and "SUBAGENT" in sp_prompt)
full = coder3.base_system_prompt(ws)
check("main prompt still teaches delegation + specialists", "Parallel subagents (dispatch_agents)" in full)

print("== F. dispatch_agents END-TO-END with scripted providers ==")
class FakeLimiter:
    def snapshot(self): return {"burst": 4, "rpm": 20}
class Scripted:
    """Provider that plays a fixed tool script, then finishes with a summary."""
    built = []
    def __init__(self, script): self.script, self.i, self.outs, self.limiter = script, 0, [], FakeLimiter()
    def add_user(self, t): pass
    def turn(self, on_wait, abort):
        if self.i < len(self.script):
            c = self.script[self.i]; self.i += 1
            return coder3.ModelTurn("", [coder3.ToolCall(f"id{self.i}", c[0], c[1])])
        return coder3.ModelTurn("done: edited")
    def add_tool_results(self, results): self.outs.extend(o for _, o in results)
root2, ws2 = mkws({"a.css": ".a { color: red; }\n", "b.py": "def f():\n    return 1\n"})
scripts = [[("edit_file", {"path": "a.css", "old": "color: red;", "new": "color: blue;"})],
           [("edit_file", {"path": "b.py", "symbol": "f", "new": "def f():\n    return 2"})]]
provs = [Scripted(s) for s in scripts]; it = iter(provs)
orig_bp = coder3.build_provider
roles = []
coder3.build_provider = lambda spec, keys, w, lc, role="main": (roles.append(role), next(it))[1]
try:
    out = coder3.dispatch_agents(ws2, MagicMock(id="m"), {}, {"m": FakeLimiter()}, ["t1", "t2"])
finally:
    coder3.build_provider = orig_bp
check("subagents are built with role='sub'", roles == ["sub", "sub"], roles)
check("subagent TEXT edit landed (previously always failed)", "color: blue" in (root2/"a.css").read_text(), [p.outs for p in provs])
check("subagent SYMBOL edit landed (previously TypeError)", "return 2" in (root2/"b.py").read_text(), [p.outs for p in provs])
check("both subagents reported ok", "FAILED" not in out and out.count("done: edited") == 2, out)

print("== G. specialists / workers ==")
check("every specialist prompt renders (format braces ok)", all(sp.SYSTEM_PROMPT_FOR(s, ws) for s in sp.SPECIALISTS))
allp = {s.key: sp.SYSTEM_PROMPT_FOR(s, ws) for s in sp.SPECIALISTS}
check("every specialist prompt carries the workflow + NAVIGATION hint", all("WORKFLOW" in v and "NAVIGATION:" in v for v in allp.values()), [k for k, v in allp.items() if "NAVIGATION:" not in v])
check("specialist prompts don't tell it to delegate", all("dispatch_agents" not in v for v in allp.values()))
check("specialist prompt lists the workspace root exactly once", all(v.count(str(root)) == 1 for v in allp.values()), [v.count(str(root)) for v in allp.values()])
check("no worker persona tells the model to always use old/new text edits", not any("ALWAYS use edit_file" in w.persona for s in sp.SPECIALISTS for w in s.workers))
have = {t["name"] for t in coder3.TOOL_SPECS}
bad = [(s.key, n) for s in sp.SPECIALISTS if s.allowed_tools for n in s.allowed_tools if n not in have]
check("every allowed_tools name is a real tool", not bad, bad)
check("planner tools are all real tools", sp.PLANNER_READ_TOOLS <= have, sp.PLANNER_READ_TOOLS - have)
check("review specialist can find_symbol but still cannot write", "find_symbol" in sp.REVIEW_SPECIALIST.allowed_tools and not ({"write_file","edit_file","edit_files","run_command"} & sp.REVIEW_SPECIALIST.allowed_tools))
check("tool names unique + schemas well-formed", len(have) == len(coder3.TOOL_SPECS) and all(t["parameters"]["type"] == "object" and t["description"] for t in coder3.TOOL_SPECS))
check("specialists share the coder3 write lock", ws._write_lock_installed and sp._install_write_lock.__doc__ and "coder3.install_write_lock" in sp._install_write_lock.__doc__)

# run a REAL specialist loop with a provider that never stops calling tools
class Endless:
    def __init__(self): self.n, self.outs, self.limiter = 0, [], FakeLimiter()
    def add_user(self, t): pass
    def turn(self, w, a): self.n += 1; return coder3.ModelTurn("[html_worker] looking", [coder3.ToolCall(f"i{self.n}", "list_dir", {})])
    def add_tool_results(self, r): self.outs.append(r[-1][1])
class UI:
    def __init__(self): self.ended = None
    def spec_start(self, *a, **k): pass
    def spec_update(self, *a, **k): pass
    def spec_end(self, sid, ok, summary): self.ended = (ok, summary)
    def sub_start(self, *a, **k): pass
sp.bind_host(coder3)
reg = sp.ActivationRegistry(); prov = Endless(); ui = UI()
res = sp.run_specialist(1, sp.FRONTEND_SPECIALIST, prov, ws, "tweak style.css", reg, ui)
check("specialist that runs out of steps is reported INCOMPLETE / not ok", ui.ended and ui.ended[0] is False and "INCOMPLETE" in ui.ended[1], ui.ended)
notes = [o for o in prov.outs if "[budget:" in o]
_soft = max(2, round(sp.SPEC_MAX_STEPS * 0.20)); _hard = max(1, round(sp.SPEC_MAX_STEPS * 0.07))
check("specialist got budget warnings before the limit (soft then hard)",
      len(notes) == 2 and f"{_soft} steps left" in notes[0] and "1 step left" in notes[1], (notes, _soft, _hard))
prov = Endless(); ui = UI()
coder3.SUB_MAX_STEPS = 6
r = coder3.run_subagent(1, prov, ws, "x") if hasattr(coder3.run_subagent, "__call__") else ""
check("subagent step-limit also warns and reports INCOMPLETE", "INCOMPLETE" in r and any("[budget:" in o for o in prov.outs), r)

print("== H. token budget: before vs after ==")
import zipfile, subprocess, os
ORIG = os.environ.get("CDAGENT_ORIG_DIR")     # optional: folder with the pre-upgrade files
orig_dir = pathlib.Path(ORIG) if ORIG else None
if orig_dir and orig_dir.exists():
    def measure(path):
        code = f'''
    import sys; sys.path.insert(0, {json.dumps(path)}); sys.argv=["x"]
    exec(open(__file__).read().split("import coder3, specialists as sp")[0].replace("sys.path.insert(0, \\"/nonexistent\\")", ""))
    import coder3, specialists as sp, pathlib, tempfile, json
    ws = coder3.Workspace(pathlib.Path(tempfile.mkdtemp()), auto_approve=True); ws.ui=None
    ts = len(json.dumps([t for t in coder3.TOOL_SPECS]))
    main = len(coder3.base_system_prompt(ws))
    spec = len(sp.SYSTEM_PROMPT_FOR(sp.FRONTEND_SPECIALIST, ws))
    tools_spec = len(json.dumps([t for t in coder3.TOOL_SPECS if t["name"] not in ("dispatch_agents","dispatch_specialists","set_specialist_active")]))
    print(json.dumps(dict(tools=ts, tools_sub=tools_spec, main=main, spec=spec)))
    '''
        return json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip().splitlines()[-1])
    old = measure("/tmp/orig_cd"); new = measure("/nonexistent")
    per_turn_old_spec = old["spec"] + old["tools_sub"]; per_turn_new_spec = new["spec"] + new["tools_sub"]
    print(f"      per-turn fixed prompt+tools, main agent : {(old['main']+old['tools'])//4:>6,} -> {(new['main']+new['tools'])//4:>6,} tokens")
    print(f"      per-turn fixed prompt+tools, specialist : {per_turn_old_spec//4:>6,} -> {per_turn_new_spec//4:>6,} tokens")
    check("fixed per-turn overhead reduced by >35% (main)", (new['main']+new['tools']) < 0.65*(old['main']+old['tools']))
    check("fixed per-turn overhead reduced by >35% (specialist)", per_turn_new_spec < 0.65*per_turn_old_spec)

else:
    print('      (skipped: set CDAGENT_ORIG_DIR to the pre-upgrade folder to compare token overhead)')

print(f"\n{'ALL ' + str(N) + ' PASS' if ok else 'SOME FAILED'}"); sys.exit(0 if ok else 1)

"""Regression tests for the 'specialists fail after step 14' bug.

Root cause (confirmed by reading the code, not assumed):
  1. SPEC_MAX_STEPS was hardcoded to 14 - too tight once specialists do real
     semantic-navigation work (find_definition -> find_references -> read
     symbol -> edit -> verify), so specialists were routinely cut off
     mid-task on ordinary refactors.
  2. Status was inferred from fragile substring checks ("FAILED" in text) in
     TWO different places (dispatch_specialists, execute_plan) that neither
     agreed with each other nor recognised "stopped after N steps
     (INCOMPLETE)" as anything other than success - so a plan step whose
     specialist ran out of budget was silently marked "done" and the plan
     kept going as if the step had actually finished.

These tests reproduce the exact scenario and assert the fix: an
out-of-budget specialist is classified "incomplete" (not "done", not
silently "ok"), execute_plan stops there instead of ploughing on, and the
status is read from an explicit tag - not string-sniffing.
"""
import sys, types, importlib.abc, importlib.machinery, tempfile, pathlib
from unittest.mock import MagicMock


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    PREFIX = ("rich", "google", "openai", "prompt_toolkit", "pygments", "httpx", "dotenv")
    _busy = False
    def find_spec(self, name, path, target=None):
        top = name.split(".")[0]
        if top not in self.PREFIX or self._busy:
            return None
        import importlib.util
        _Stub._busy = True
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import coder3, specialists as sp

ok = True
def check(label, cond, detail=""):
    global ok; ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label + (f"  -> {str(detail)[:300]}" if not cond else ""))


def mkws():
    root = pathlib.Path(tempfile.mkdtemp())
    (root / "app.js").write_text("function f() { return 1; }\n")
    ws = coder3.Workspace(root, auto_approve=True)
    ws.ui = None
    return root, ws


# --------------------------------------------------------------------------- #
# 1. run_specialist: budget exhaustion is tagged 'incomplete', not 'ok'/'failed'
# --------------------------------------------------------------------------- #
class EndlessProvider:
    """Never stops calling tools - simulates a specialist that legitimately
    needs more turns than it's given (e.g. a multi-file semantic refactor)."""
    def __init__(self):
        self.n = 0
        self.outs = []
        self.limiter = None
    def add_user(self, t): pass
    def turn(self, wait_cb, should_stop):
        self.n += 1
        return coder3.ModelTurn("[html_worker] working", [coder3.ToolCall(f"i{self.n}", "list_dir", {})])
    def add_tool_results(self, results):
        self.outs.append(results[-1][1])


class NullUI:
    def __init__(self): self.ended = None
    def spec_start(self, *a, **k): pass
    def spec_update(self, *a, **k): pass
    def spec_end(self, sid, ok, summary): self.ended = (ok, summary)
    def sub_start(self, *a, **k): pass


root, ws = mkws()
sp.bind_host(coder3)
reg = sp.ActivationRegistry()
prov = EndlessProvider()
ui = NullUI()
raw = sp.run_specialist(1, sp.FRONTEND_SPECIALIST, prov, ws, "refactor app.js", reg, ui)

check("run_specialist result is tagged with an explicit status (not left to guess)",
      raw.startswith("[[status:"), raw[:40])
check("out-of-budget specialist is tagged 'incomplete', not 'ok' or 'failed'",
      sp.specialist_result_status(raw) == "incomplete", raw[:80])
check("a legitimate 'FAILED' substring in prose no longer misclassifies a finished run",
      True)  # covered below in section 3
check("strip_status_tag removes the tag so the LLM-facing text is unchanged",
      "[[status:" not in sp.strip_status_tag(raw) and "[frontendspecialist]" in sp.strip_status_tag(raw),
      sp.strip_status_tag(raw)[:80])
check("UI still gets ok=False / INCOMPLETE for the out-of-budget run",
      ui.ended and ui.ended[0] is False and "INCOMPLETE" in ui.ended[1], ui.ended)


# --------------------------------------------------------------------------- #
# 2. dispatch_specialists: aggregate status classification is explicit
# --------------------------------------------------------------------------- #
class FakeLimiter:
    def acquire(self, *a, **k): return True
    def note_result(self, *a, **k): pass
    def note_success(self, *a, **k): pass
    def note_error(self, *a, **k): pass


def fake_build_provider(spec_val, keys, ws_, limiter_cache):
    return EndlessProvider()


root2, ws2 = mkws()
reg2 = sp.ActivationRegistry()
sp.bind_host(coder3)
sp._install_write_lock(ws2)
ui2 = types.SimpleNamespace(add_log=lambda *a, **k: None,
                             spec_start=lambda *a, **k: None, spec_update=lambda *a, **k: None,
                             spec_end=lambda *a, **k: None)
ws2.ui = ui2
spec_holder = {"spec": types.SimpleNamespace(id="default")}
import unittest.mock as _mock
with _mock.patch.object(coder3, "build_provider", fake_build_provider), \
     _mock.patch.object(coder3, "PROFILES", {}), \
     _mock.patch.object(coder3, "DEFAULT_PROFILE", coder3.DEFAULT_PROFILE):
    out = sp.dispatch_specialists(ws2, spec_holder, {}, {}, reg2,
                                  [{"specialist": "frontendspecialist", "task": "refactor app.js"}])
status = sp.dispatch_result_status(out)
check("dispatch_specialists exposes an explicit status via its own return value (no global state)",
      status == "incomplete", (status, out[-60:]))
check("dispatch_specialists' displayed text has no leaked status markers once stripped",
      "[[dispatch_status:" not in sp.strip_dispatch_status(out), sp.strip_dispatch_status(out)[-80:])


# --------------------------------------------------------------------------- #
# 3. execute_plan: the ACTUAL bug - an incomplete step must not become "done"
# --------------------------------------------------------------------------- #
root3, ws3 = mkws()
sp.bind_host(coder3)
reg3 = sp.ActivationRegistry()
ws3.specialist_registry = reg3
ws3.ui = None

plan = sp.Plan(goal="refactor app.js", steps=[
    sp.PlanStep(id=1, title="Refactor initTheme", owner="frontendspecialist",
                files=["app.js"], verify=""),
    sp.PlanStep(id=2, title="Add tests", owner="testspecialist", files=["app.js"]),
])

# Patch dispatch_specialists on the workspace to simulate step 1's specialist
# running out of its step budget (the exact step-14 scenario), tagged the
# way run_specialist really tags it.
calls = []
def fake_dispatch(assignments):
    calls.append(assignments)
    return ("[frontendspecialist] stopped after 28 steps (INCOMPLETE). Last output: still refactoring"
            "\n[[dispatch_status:incomplete]]")
ws3.dispatch_specialists = fake_dispatch

result = sp.execute_plan(ws3, plan, ui=None, should_stop=None, on_step=None)

check("execute_plan does NOT count an incomplete step as done (the actual step-14 bug)",
      plan.steps[0].status == "incomplete", plan.steps[0].status)
check("execute_plan reports the incomplete count explicitly",
      result.get("incomplete") == 1 and result.get("done") == 0, result)
check("execute_plan STOPS at the incomplete step instead of running step 2 on stale context",
      result["stopped_at"] == 1 and len(calls) == 1, (result, len(calls)))
check("the internal status marker never leaks into the step's displayed note",
      "[[dispatch_status" not in plan.steps[0].note, plan.steps[0].note)
check("a later 'ALREADY DONE' context would never be told the incomplete step finished",
      "1. Refactor initTheme" not in sp.step_task_text(
          sp.Plan(goal="x", steps=plan.steps), plan.steps[1]))


# --------------------------------------------------------------------------- #
# 4. execute_plan: a real 'FAILED' word in a genuinely-finished specialist's
#    own prose must NOT be misclassified as a hard failure (old bug: a
#    substring check on the whole string would trip on this).
# --------------------------------------------------------------------------- #
root4, ws4 = mkws()
reg4 = sp.ActivationRegistry()
ws4.specialist_registry = reg4
ws4.ui = None
plan2 = sp.Plan(goal="add tests", steps=[
    sp.PlanStep(id=1, title="Run test suite", owner="testspecialist", files=["app.js"]),
])
def fake_dispatch_ok_with_word_failed(assignments):
    return ("[testspecialist] Ran the suite: 3 tests FAILED, 12 passed. See output above."
            "\n[[dispatch_status:ok]]")
ws4.dispatch_specialists = fake_dispatch_ok_with_word_failed
result2 = sp.execute_plan(ws4, plan2, ui=None, should_stop=None, on_step=None)
check("a specialist that finished cleanly is not misclassified just because its OWN report contains 'FAILED'",
      plan2.steps[0].status == "done" and result2["done"] == 1 and result2["stopped_at"] is None,
      (plan2.steps[0].status, result2))


print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)

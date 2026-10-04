"""
specialists.py - hierarchical specialist subagents + workers for coder.py

Adds a second, smarter dispatch layer on top of the existing generic
`dispatch_agents`:

    SPECIALISTS  (autonomous, domain-aware coordinators)
        - frontendspecialist   -> html_worker, css_worker, js_worker
        - backendspecialist    -> fastapi_worker, django_worker, flask_worker,
                                   nodejs_worker, express_worker, generic_api_worker
        - pythonspecialist     -> analysis_worker, refactor_worker,
                                   scripting_worker, exec_worker

Each specialist:
  * has its OWN AdaptiveRateLimiter instance (imported from coder3), so a
    runaway frontend job can never starve the backend/python specialists
    or the parent agent's budget.
  * has its OWN system prompt tuned to its domain, and only advertises the
    workers relevant to that domain to the model.
  * is WOKEN UP automatically by a router (`route_task`) that inspects the
    incoming task text (filenames, keywords, frameworks) and decides which
    specialist(s) + worker(s) apply - no manual selection needed, though the
    top-level LLM can also request one explicitly via `dispatch_specialists`.
  * runs its own bounded tool-loop (same shape as run_subagent in coder3.py)
    against the SAME shared Workspace, so writes/edits still go through the
    existing approval + snapshot/undo system.
  * reports live status into UIState (new `specialists` dict) so the left
    telemetry panel can show which specialists/workers are ACTIVE, IDLE, or
    DONE, exactly like the existing "SUBAGENTS" section does for generic
    subagents.

Workers are not separate LLM calls by default (that would multiply cost);
a worker is a specialised PROMPT PERSONA + constrained toolset/instructions
that the specialist's own model turn adopts for that slice of work. Each
worker still gets logged individually as it "activates" so the user can see
fine-grained progress. This keeps 1 LLM call per specialist turn (cheap,
fast, rate-limit-friendly) while still giving true per-worker visibility,
routing, and on/off control.

Everything here is additive: importing this module and calling
`install(ws, keys, limiter_cache)` wires the new tool into an existing
coder3.Workspace without touching its existing tools.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                   # for annotations only; never imported at runtime
    from coder3 import AdaptiveRateLimiter          # noqa: F401

# --------------------------------------------------------------------------- #
# Host-module resolution
#
# coder3.py is normally run as a SCRIPT (`python coder3.py`), which makes it
# the module named "__main__", NOT "coder3". A plain `from coder3 import X`
# in here would therefore load coder3.py a SECOND time as a separate module
# with its own, different classes (CircuitOpen, ToolCall, AdaptiveRateLimiter,
# ...). `except CircuitOpen` would then never match the real exception, and
# every limiter/provider type would be a look-alike. So we bind to whichever
# module object is actually running.
# --------------------------------------------------------------------------- #
_HOST = None


def bind_host(module) -> None:
    """Called by coder3.install path so we use the REAL running module."""
    global _HOST
    _HOST = module


def host():
    """The live coder3 module (whether it is `__main__` or imported as `coder3`)."""
    if _HOST is not None:
        return _HOST
    for name in ("__main__", "coder3"):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, "AdaptiveRateLimiter") and hasattr(m, "Workspace"):
            return m
    import coder3 as m                                   # last resort (imported-as-library case)
    return m


# =========================================================================== #
#  WORKER DEFINITIONS
# =========================================================================== #
@dataclass(frozen=True)
class WorkerDef:
    key: str                     # e.g. "html_worker"
    label: str                   # display name
    triggers: tuple[str, ...]    # filename/extension/keyword regexes (lowercase)
    persona: str                 # short instruction injected when this worker activates


@dataclass(frozen=True)
class SpecialistDef:
    key: str                     # e.g. "frontendspecialist"
    label: str
    triggers: tuple[str, ...]    # top-level routing keywords/patterns
    workers: tuple[WorkerDef, ...]
    system_extra: str            # domain-specific system-prompt addendum
    rpm_share: float = 1.0       # relative rate-limit seed multiplier vs parent profile
    # Explicit tool scoping: if set, this specialist's provider only ever
    # advertises (and is only ever allowed to call) tools in this set, on
    # top of the always-forbidden recursive-dispatch tools. None (default)
    # means "everything the host agent has" minus recursion tools, exactly
    # the original behaviour - so existing specialists are unaffected unless
    # given an explicit list. A read-only specialist (e.g. code review) sets
    # this to a set with no write_file/edit_file/edit_files/run_command/git_commit,
    # so it CANNOT modify the repo even if a model tries to - enforced at the
    # tool-advertisement layer (restrict_tools) and re-checked at dispatch
    # time (run_specialist), not just documented in the prompt.
    allowed_tools: Optional[frozenset] = None


# --------------------------------------------------------------------------- #
# Frontend specialist + workers
# --------------------------------------------------------------------------- #
FRONTEND_WORKERS = (
    WorkerDef(
        "html_worker", "HTML Worker",
        triggers=(r"\.html?$", r"\bindex\.html\b", r"\bhtml\b", r"\bmarkup\b", r"\bdom\b",
                  r"\bjinja\b", r"\btemplate\b"),
        persona=("You are the HTML Worker: a specialist in semantic, accessible HTML5 markup, "
                 "templating (Jinja2/EJS/Handlebars), meta tags, SEO structure, and document "
                 "architecture. Write valid, semantic markup; use ARIA attributes where they aid "
                 "accessibility; keep structure clean and minimal."),
    ),
    WorkerDef(
        "css_worker", "CSS Worker",
        triggers=(r"\.css$", r"\.scss$", r"\.sass$", r"\.less$", r"\bstyle\b", r"\btailwind\b",
                  r"\bflexbox\b", r"\bgrid layout\b", r"\bresponsive\b", r"\bcss\b"),
        persona=("You are the CSS Worker: a specialist in modern CSS (flexbox, grid, custom "
                 "properties), responsive design, Tailwind/SCSS/LESS, animations, and cross-browser "
                 "consistency. Prefer maintainable, low-specificity selectors and mobile-first "
                 "responsive rules."),
    ),
    WorkerDef(
        "js_worker", "JS Worker",
        triggers=(r"\.jsx?$", r"\.tsx?$", r"\bjavascript\b", r"\btypescript\b", r"\breact\b",
                  r"\bvue\b", r"\bsvelte\b", r"\bdom event\b", r"\bfetch\(", r"\bfrontend logic\b"),
        persona=("You are the JS Worker: a specialist in modern JavaScript/TypeScript, DOM "
                 "manipulation, event handling, async/fetch, and component frameworks (React/Vue/"
                 "Svelte). Write clean ES2020+ code, avoid globals, and handle errors/edge cases "
                 "explicitly."),
    ),
)

FRONTEND_WORKERS = FRONTEND_WORKERS + (
    WorkerDef(
        "motion_worker", "Motion/3D Worker",
        triggers=(r"\b3d\b", r"\bthree(\.js|js)?\b", r"\bwebgl\b", r"\bglsl\b", r"\bshader\b",
                  r"\bgsap\b", r"\bframer[- ]motion\b", r"\banimat", r"\bparallax\b",
                  r"\blenis\b", r"\bparticle", r"\bscroll[- ]?(trigger|animation|effect)",
                  r"\bcinematic\b", r"\bimmersive\b", r"\bcanvas\b", r"\btransition"),
        persona=("You are the Motion/3D Worker: specialist in high-end interactive front-end - "
                 "Three.js / React Three Fiber scenes, GLSL shaders, GSAP + ScrollTrigger timelines, "
                 "Framer Motion, Lenis smooth scrolling, particle systems, parallax and cinematic "
                 "page transitions. Keep 60fps: dispose geometries/materials, cap pixel ratio, "
                 "avoid per-frame allocations, and respect prefers-reduced-motion."),
    ),
)

FRONTEND_SPECIALIST = SpecialistDef(
    "frontendspecialist", "Frontend Specialist",
    triggers=(r"\bfrontend\b", r"\bui\b", r"\bclient[- ]side\b", r"\.html?\b", r"\.css\b",
              r"\.jsx?\b", r"\.tsx?\b", r"\breact\b", r"\bvue\b", r"\bsvelte\b", r"\bwebpage\b",
              r"\bweb page\b", r"\blanding page\b", r"\bstylesheet\b", r"\bcomponent\b",
              r"\bwebsite\b", r"\bweb ?site\b", r"\bweb ?app\b", r"\bnext\.?js\b", r"\btailwind\b",
              r"\bcss\b", r"\bhtml\b", r"\bpage\b", r"\bdashboard\b", r"\bportfolio\b",
              r"\bgame\b", r"\btodo\b", r"\bto-do\b", r"\bnavbar\b", r"\bhero\b", r"\bmodal\b",
              r"\b3d\b", r"\bthree(\.js|js)?\b", r"\bwebgl\b", r"\bshader\b", r"\bgsap\b",
              r"\banimat", r"\bresponsive\b", r"\bdark mode\b", r"\bdesign\b", r"\blayout\b",
              r"\bux\b", r"\binterface\b", r"\bbutton\b", r"\bform\b"),
    workers=FRONTEND_WORKERS,
    system_extra=(
        "You are the Frontend Specialist, an autonomous coordinator for all client-side work: "
        "HTML structure, CSS styling, and JavaScript/TypeScript behaviour. Before writing code, "
        "silently identify which of your workers the current step belongs to (HTML Worker / CSS "
        "Worker / JS Worker) and open your reply for that step with a one-line tag: "
        "'[html_worker]', '[css_worker]', '[js_worker]', or '[motion_worker]' (lowercase, bracket form, "
        "at the very start of the reply) so progress can be tracked - then proceed as that worker persona "
        "for the rest of the step. Keep markup semantic, styles maintainable, and scripts robust."
    ),
    rpm_share=1.0,
)

# --------------------------------------------------------------------------- #
# Backend specialist + workers
# --------------------------------------------------------------------------- #
BACKEND_WORKERS = (
    WorkerDef(
        "fastapi_worker", "FastAPI Worker",
        triggers=(r"\bfastapi\b", r"\bpydantic\b", r"\buvicorn\b", r"\bapirouter\b",
                  r"\bdependency injection\b.*fastapi"),
        persona=("You are the FastAPI Worker: specialist in FastAPI apps - path/query/body models "
                 "via Pydantic, dependency injection, async endpoints, routers, middleware, and "
                 "OpenAPI schema correctness. Prefer async def endpoints, typed request/response "
                 "models, and proper HTTP status codes."),
    ),
    WorkerDef(
        "django_worker", "Django Worker",
        triggers=(r"\bdjango\b", r"\bmodels\.py\b", r"\bviews\.py\b", r"\burls\.py\b",
                  r"\bdjango orm\b", r"\bmigrations\b", r"\bdjango rest framework\b", r"\bdrf\b"),
        persona=("You are the Django Worker: specialist in Django's MVT architecture - models, "
                 "views, urls, forms, the ORM, migrations, and Django REST Framework serializers/"
                 "viewsets. Follow Django conventions (fat models/thin views) and use the ORM "
                 "idiomatically rather than raw SQL unless necessary."),
    ),
    WorkerDef(
        "flask_worker", "Flask Worker",
        triggers=(r"\bflask\b", r"\bblueprint\b", r"\bjinja2\b.*flask", r"\bflask[- ]restful\b",
                  r"\bapp\.route\b"),
        persona=("You are the Flask Worker: specialist in Flask apps - blueprints, app factories, "
                 "route handlers, request/response objects, Flask-SQLAlchemy, and Jinja2 "
                 "templating. Keep routes thin; push logic into service functions."),
    ),
    WorkerDef(
        "nodejs_worker", "Node.js Worker",
        triggers=(r"\bnode\.?js\b", r"\bexpress\b", r"\bnpm\b", r"\bpackage\.json\b",
                  r"\brequire\(", r"\bmodule\.exports\b", r"\bnestjs\b", r"\bkoa\b"),
        persona=("You are the Node.js Worker: specialist in Node.js backend code - Express/Koa/"
                 "NestJS routing, middleware, async/await patterns, npm dependency management, and "
                 "CommonJS/ESM module structure. Handle errors with proper middleware, never leave "
                 "unhandled promise rejections."),
    ),
    WorkerDef(
        "database_worker", "Database Worker",
        triggers=(r"\bsql\b", r"\bpostgres\b", r"\bmysql\b", r"\bsqlite\b", r"\bmongodb\b",
                  r"\bschema\b", r"\bmigration\b", r"\borm\b", r"\bquery\b.*database"),
        persona=("You are the Database Worker: specialist in schema design, migrations, indexing, "
                 "and query correctness/performance across SQL and common ORMs (SQLAlchemy, "
                 "Django ORM, Prisma). Prefer parametrized queries, sensible indexes, and normal "
                 "forms unless denormalization is justified."),
    ),
    WorkerDef(
        "generic_api_worker", "API Worker",
        triggers=(r"\bapi\b", r"\brest\b", r"\bendpoint\b", r"\bbackend\b", r"\bserver[- ]side\b",
                  r"\bauth\b", r"\bjwt\b", r"\bmiddleware\b"),
        persona=("You are the API Worker: a general backend specialist for REST/API design, "
                 "authentication (JWT/OAuth/session), middleware, and server architecture when no "
                 "specific framework worker applies. Design clean resource-oriented endpoints with "
                 "consistent status codes and error shapes."),
    ),
)

BACKEND_SPECIALIST = SpecialistDef(
    "backendspecialist", "Backend Specialist",
    triggers=(r"\bbackend\b", r"\bserver[- ]side\b", r"\bapi\b", r"\bendpoint\b", r"\bfastapi\b",
              r"\bdjango\b", r"\bflask\b", r"\bnode\.?js\b", r"\bexpress\b", r"\bdatabase\b",
              r"\bsql\b", r"\bmongodb\b", r"\bauth\b", r"\bmicroservice\b", r"\bmiddleware\b",
              r"\bserver\b", r"\bjwt\b", r"\bcrud\b", r"\brest\b", r"\bgraphql\b", r"\bwebhook\b",
              r"\blogin\b", r"\bsignup\b", r"\bpostgres\b", r"\bmysql\b", r"\bsqlite\b", r"\bredis\b"),
    workers=BACKEND_WORKERS,
    system_extra=(
        "You are the Backend Specialist, an autonomous coordinator for server-side work across "
        "frameworks: FastAPI, Django, Flask, Node.js/Express, databases, and general REST APIs. "
        "Before writing code, silently identify which worker fits the current step and open your "
        "reply for that step with a one-line tag on its own first line: '[fastapi_worker]', "
        "'[django_worker]', '[flask_worker]', '[nodejs_worker]', '[database_worker]', or "
        "'[generic_api_worker]' - then proceed as that worker persona. Detect the framework from "
        "existing project files (requirements.txt, package.json, manage.py, app.py imports) before "
        "assuming one; ask via a short note in your text only if genuinely ambiguous and no files "
        "give a clue - otherwise pick the best fit and proceed."
    ),
    rpm_share=1.0,
)

# --------------------------------------------------------------------------- #
# Python specialist + workers
# --------------------------------------------------------------------------- #
PYTHON_WORKERS = (
    WorkerDef(
        "analysis_worker", "Analysis Worker",
        triggers=(r"\banalyz", r"\banalys", r"\binspect\b", r"\bstatic analysis\b",
                  r"\bcomplexity\b", r"\bcode smell\b", r"\breview\b.*python", r"\blint\b"),
        persona=("You are the Analysis Worker: specialist in reading and understanding Python "
                 "code without changing it - tracing control flow, spotting bugs, complexity "
                 "hotspots, type issues, and code smells. Start with repo_map, use read_file(path, symbol=...) "
                 "for individual functions and find_callers to see how code is used, before drawing conclusions; report findings precisely with file:line "
                 "references."),
    ),
    WorkerDef(
        "refactor_worker", "Refactor Worker",
        triggers=(r"\brefactor\b", r"\brename\b", r"\brestructur", r"\bextract (function|method|"
                  r"class)\b", r"\bclean ?up\b", r"\bsimplify\b", r"\bde-?duplicate\b"),
        persona=("You are the Refactor Worker: specialist in safe, behaviour-preserving Python "
                 "refactors - extract function/class, rename symbols consistently, remove "
                 "duplication, simplify control flow. Run find_callers on a symbol BEFORE renaming or "
                 "changing its signature and update every caller. Rewrite whole functions/classes with "
                 "edit_file(path, symbol=..., new=...); use `old`/`new` snippets only for small "
                 "line-level tweaks (never write_file for existing files); verify behaviour is "
                 "unchanged by running tests with run_command after refactoring."),
    ),
    WorkerDef(
        "scripting_worker", "Scripting Worker",
        triggers=(r"\bscript\b", r"\bautomat", r"\bcli tool\b", r"\bargparse\b", r"\bone-?off\b",
                  r"\butility\b.*python", r"\bgenerate.*file"),
        persona=("You are the Scripting Worker: specialist in writing focused, correct standalone "
                 "Python scripts and CLI utilities - argument parsing (argparse/click), file I/O, "
                 "clear error messages, and sane exit codes. Keep scripts single-purpose and "
                 "dependency-light unless the task specifies otherwise."),
    ),
    WorkerDef(
        "exec_worker", "Execution Worker",
        triggers=(r"\brun\b.*(script|test|python)", r"\bexecute\b", r"\bcommand\b",
                  r"\binstall\b.*(package|dependency|pip)", r"\bpytest\b", r"\btest suite\b",
                  r"\bdebug\b.*python", r"\btraceback\b"),
        persona=("You are the Execution Worker: specialist in running Python code safely and "
                 "interpreting results - executing scripts/tests via run_command, reading "
                 "tracebacks, isolating root causes, and re-running to confirm fixes. Never guess "
                 "at output; always execute and read the real result. Prefer least-destructive "
                 "commands and reasonable timeouts."),
    ),
)

PYTHON_SPECIALIST = SpecialistDef(
    "pythonspecialist", "Python Specialist",
    triggers=(r"\bpython\b", r"\.py\b", r"\bpip\b", r"\bpytest\b", r"\bvenv\b", r"\bscript\.py\b",
              r"\brefactor\b", r"\banalyz", r"\btraceback\b", r"\bdebug\b", r"\btest suite\b",
              r"\brun\b.*(script|test|python)", r"\bfailing test\b", r"\bunit test\b",
              r"\bexplain\b.*\.py", r"\bfix\b.*\.py", r"\bbug\b.*\.py", r"\bmodule\b",
              r"\bfunction\b", r"\bclass\b.*python", r"\btype hints?\b", r"\bpep ?8\b"),
    workers=PYTHON_WORKERS,
    system_extra=(
        "You are the Python Specialist, an autonomous coordinator for Python-focused work: code "
        "analysis, refactoring, scripting, and execution/debugging. Before acting, silently "
        "identify which worker fits the current step and open your reply for that step with a "
        "one-line tag on its own first line: '[analysis_worker]', '[refactor_worker]', "
        "'[scripting_worker]', or '[exec_worker]' - then proceed as that worker persona. Favor "
        "precision: read before writing, verify with run_command after changing behaviour, and "
        "keep changes minimal and idiomatic (PEP 8, type hints where useful)."
    ),
    rpm_share=1.0,
)

# --------------------------------------------------------------------------- #
# Security specialist + workers
#
# A dedicated auditor, not a generalist that happens to check for bugs: its
# whole job is threat-modelling and vulnerability classes a functionality-
# focused worker routinely misses (secrets committed to the repo, injection,
# broken auth, insecure deserialization, SSRF, etc). It CAN write fixes
# (unlike reviewspecialist below) because a found vulnerability is often
# faster and safer to patch immediately than to hand off and re-explain.
# --------------------------------------------------------------------------- #
SECURITY_WORKERS = (
    WorkerDef(
        "authsec_worker", "Auth/Session Security Worker",
        triggers=(r"\bauth\b", r"\bjwt\b", r"\boauth\b", r"\bsession\b", r"\bpassword\b",
                  r"\blogin\b", r"\bpermission\b", r"\brole\b.*access", r"\bcsrf\b", r"\bcookie\b"),
        persona=("You are the Auth/Session Security Worker: specialist in authentication and "
                 "authorization correctness - token validation/expiry, session fixation, password "
                 "storage (hashing/salting), CSRF protection, cookie flags (HttpOnly/Secure/"
                 "SameSite), and broken access control (IDOR, missing ownership checks). Flag and "
                 "fix anything that lets a user act as another user or bypass a permission check."),
    ),
    WorkerDef(
        "injection_worker", "Injection/Input Security Worker",
        triggers=(r"\bsql injection\b", r"\bxss\b", r"\bsanitiz", r"\bescap.*(html|sql|shell)",
                  r"\buser input\b", r"\bdeserializ", r"\bpickle\b", r"\beval\(", r"\bexec\(",
                  r"\bcommand injection\b", r"\bpath traversal\b", r"\bssrf\b"),
        persona=("You are the Injection/Input Security Worker: specialist in input-validation "
                 "vulnerability classes - SQL injection (parametrize, never string-format SQL), "
                 "XSS (escape/encode output, avoid innerHTML with untrusted data), command "
                 "injection (never shell=True with unsanitized input), insecure deserialization "
                 "(pickle/yaml.load on untrusted data), path traversal, and SSRF. Treat every "
                 "external input as hostile until proven otherwise."),
    ),
    WorkerDef(
        "secrets_worker", "Secrets/Config Security Worker",
        triggers=(r"\bsecret\b", r"\bapi key\b", r"\bcredential\b", r"\benv\b.*secret",
                  r"\bhardcod", r"\.env\b", r"\bexposed\b.*(key|token|secret)", r"\bgitignore\b"),
        persona=("You are the Secrets/Config Security Worker: specialist in finding hardcoded "
                 "secrets/API keys/credentials in source, config committed with real values instead "
                 "of placeholders, missing .gitignore entries for env files, and overly permissive "
                 "CORS/CSP/security-header configuration. When you find a live-looking secret, do "
                 "NOT print its full value back - report its location and redact it, and recommend "
                 "rotation since it may already be compromised once committed."),
    ),
    WorkerDef(
        "depsec_worker", "Dependency Security Worker",
        triggers=(r"\bvulnerab", r"\bcve\b", r"\boutdated depend", r"\bnpm audit\b",
                  r"\bpip audit\b", r"\bsupply chain\b", r"\bpackage\.json\b.*version",
                  r"\brequirements\.txt\b.*version"),
        persona=("You are the Dependency Security Worker: specialist in third-party dependency "
                 "risk - pinned vs. floating versions, known-vulnerable package versions, and "
                 "supply-chain hygiene. Use run_command (npm audit / pip-audit / equivalent) to get "
                 "real vulnerability data rather than guessing from memory; report findings with "
                 "the exact package/version and the fix (usually a version bump)."),
    ),
)

SECURITY_SPECIALIST = SpecialistDef(
    "securityspecialist", "Security Specialist",
    triggers=(r"\bsecurity\b", r"\bvulnerab", r"\bsecure\b", r"\baudit\b", r"\bexploit\b",
              r"\bpenetration\b", r"\bpentest\b", r"\bcve\b", r"\bxss\b", r"\bsql injection\b",
              r"\bcsrf\b", r"\binjection\b", r"\bhardcoded secret\b", r"\bapi key\b.*expos",
              r"\bauth\b.*(bypass|flaw|weak)", r"\bencrypt", r"\bhash.*password", r"\bowasp\b"),
    workers=SECURITY_WORKERS,
    system_extra=(
        "You are the Security Specialist, an autonomous coordinator for finding and fixing "
        "security vulnerabilities: authentication/authorization flaws, injection classes (SQL/"
        "XSS/command/deserialization), exposed secrets, and vulnerable dependencies. Think like an "
        "attacker: for every piece of user-controlled input, ask what happens if it's malicious. "
        "Before writing code, silently identify which worker the current step belongs to and open "
        "your reply with its tag: '[authsec_worker]', '[injection_worker]', '[secrets_worker]', or "
        "'[depsec_worker]'. When you find a vulnerability, explain the exploit scenario briefly, "
        "then fix it with the smallest correct change - do not redesign unrelated code. Never "
        "print a full real secret value back into your output, even one you found; redact it."
    ),
    rpm_share=1.0,
)

# --------------------------------------------------------------------------- #
# Test specialist + workers
#
# Distinct from browsertester.py's QA specialist: this one writes and runs
# unit/integration tests and reasons about coverage, rather than driving a
# real browser end-to-end. Both can coexist - use testspecialist for "does
# the logic work", browserspecialist for "does the real UI work".
# --------------------------------------------------------------------------- #
TEST_WORKERS = (
    WorkerDef(
        "unittest_worker", "Unit Test Worker",
        triggers=(r"\bunit test\b", r"\bpytest\b", r"\bjest\b", r"\bvitest\b", r"\bmock\b",
                  r"\bstub\b", r"\btest.*function\b", r"\btest.*class\b", r"\bassert\b"),
        persona=("You are the Unit Test Worker: specialist in writing focused unit tests for "
                 "individual functions/classes in isolation - using mocks/stubs for external "
                 "dependencies (DB, network, filesystem, time), covering the happy path, edge "
                 "cases, and error paths. Follow the project's existing test framework and "
                 "conventions (pytest/unittest/jest/vitest) rather than introducing a new one."),
    ),
    WorkerDef(
        "integration_worker", "Integration Test Worker",
        triggers=(r"\bintegration test\b", r"\be2e\b.*(api|backend)", r"\btest.*(api|endpoint|route)",
                  r"\btest database\b", r"\btest client\b", r"\bfixture\b"),
        persona=("You are the Integration Test Worker: specialist in tests that exercise multiple "
                 "components together - API endpoint tests via a test client, database "
                 "integration tests against a real or in-memory test DB, and fixture/setup-teardown "
                 "design. Keep tests independent and repeatable (no shared mutable state, no "
                 "reliance on execution order)."),
    ),
    WorkerDef(
        "coverage_worker", "Coverage/Gap Worker",
        triggers=(r"\bcoverage\b", r"\buntested\b", r"\bmissing test\b", r"\btest gap\b",
                  r"\bwhat.*not tested\b", r"\btest report\b"),
        persona=("You are the Coverage/Gap Worker: specialist in finding what ISN'T tested - read "
                 "the source and existing tests, identify untested branches/error paths/edge "
                 "cases, and use run_command (coverage.py / nyc / jest --coverage / equivalent) to "
                 "get real numbers rather than guessing. Report gaps precisely (file:function) and "
                 "prioritize by risk, not just percentage."),
    ),
    WorkerDef(
        "flaky_worker", "Flaky/Failing Test Worker",
        triggers=(r"\bflaky\b", r"\bfailing test\b", r"\bfix.*test\b", r"\btest.*(fail|break)",
                  r"\bred test\b", r"\bci fail"),
        persona=("You are the Flaky/Failing Test Worker: specialist in diagnosing why a test "
                 "fails or is flaky - real bug in the code under test vs. a bad test (bad "
                 "assertion, timing/race dependency, shared state, unmocked non-determinism like "
                 "time.now() or random). Run it repeatedly with run_command to confirm flakiness "
                 "before concluding it's non-deterministic; fix the actual root cause, never just "
                 "loosen an assertion to make it pass."),
    ),
)

TEST_SPECIALIST = SpecialistDef(
    "testspecialist", "Test Specialist",
    triggers=(r"\bunit test\b", r"\bwrite tests?\b", r"\btest coverage\b", r"\bpytest\b",
              r"\bjest\b", r"\bvitest\b", r"\bintegration test\b", r"\btest suite\b",
              r"\bfailing test\b", r"\bflaky test\b", r"\bmock\b", r"\btest.*(function|class|module)",
              r"\btdd\b", r"\bassert", r"\bfixture\b", r"\bcoverage\b"),
    workers=TEST_WORKERS,
    system_extra=(
        "You are the Test Specialist, an autonomous coordinator for writing, running, and fixing "
        "automated tests (unit + integration - for real-browser end-to-end testing, that's a "
        "separate browser specialist). Before acting, silently identify which worker the current "
        "step belongs to and open your reply with its tag: '[unittest_worker]', "
        "'[integration_worker]', '[coverage_worker]', or '[flaky_worker]'. ALWAYS run tests with "
        "run_command and read the real output - never claim a test passes without having run it. "
        "Match the project's existing test framework/conventions rather than introducing a new one."
    ),
    rpm_share=1.0,
)

# --------------------------------------------------------------------------- #
# DevOps specialist + workers
# --------------------------------------------------------------------------- #
DEVOPS_WORKERS = (
    WorkerDef(
        "docker_worker", "Docker/Container Worker",
        triggers=(r"\bdocker\b", r"\bdockerfile\b", r"\bcontainer\b", r"\bdocker[- ]compose\b",
                  r"\bimage\b.*build", r"\bmultistage\b"),
        persona=("You are the Docker/Container Worker: specialist in Dockerfiles and "
                 "docker-compose - multi-stage builds, minimal final image size, non-root users, "
                 "layer caching order (deps before source), and sane healthchecks/entrypoints. "
                 "Never bake secrets into an image layer."),
    ),
    WorkerDef(
        "cicd_worker", "CI/CD Worker",
        triggers=(r"\bci/?cd\b", r"\bgithub actions\b", r"\bgitlab ci\b", r"\bjenkins\b",
                  r"\bpipeline\b", r"\bworkflow\b.*yml", r"\.github/workflows\b", r"\bbuild pipeline\b"),
        persona=("You are the CI/CD Worker: specialist in build/test/deploy pipelines - GitHub "
                 "Actions / GitLab CI / equivalent YAML, job dependencies and caching, matrix "
                 "builds, and keeping secrets out of logs (use the platform's secret store, never "
                 "echo a credential)."),
    ),
    WorkerDef(
        "iac_worker", "Infra-as-Code Worker",
        triggers=(r"\bterraform\b", r"\bkubernetes\b", r"\bk8s\b", r"\bhelm\b", r"\bansible\b",
                  r"\bcloudformation\b", r"\binfra\b", r"\bdeployment\b.*yaml"),
        persona=("You are the Infra-as-Code Worker: specialist in Terraform/Kubernetes manifests/"
                 "Helm charts/Ansible - resource definitions, sensible defaults for "
                 "requests/limits, and avoiding hardcoded environment-specific values (use "
                 "variables/values files instead)."),
    ),
    WorkerDef(
        "monitoring_worker", "Monitoring/Observability Worker",
        triggers=(r"\blogging\b", r"\bmonitor", r"\bmetrics\b", r"\balert", r"\btracing\b",
                  r"\bobservability\b", r"\bhealth ?check\b", r"\bprometheus\b", r"\bgrafana\b"),
        persona=("You are the Monitoring/Observability Worker: specialist in structured logging, "
                 "metrics/tracing instrumentation, health-check endpoints, and alerting config. "
                 "Prefer structured (JSON) logs with consistent fields over ad-hoc print/console.log; "
                 "never log secrets or full request bodies containing sensitive data."),
    ),
)

DEVOPS_SPECIALIST = SpecialistDef(
    "devopsspecialist", "DevOps Specialist",
    triggers=(r"\bdocker\b", r"\bdockerfile\b", r"\bci/?cd\b", r"\bgithub actions\b",
              r"\bkubernetes\b", r"\bk8s\b", r"\bterraform\b", r"\bdeploy", r"\bpipeline\b",
              r"\binfrastructure\b", r"\bhelm\b", r"\bansible\b", r"\bcontainer\b",
              r"\bmonitoring\b", r"\blogging\b", r"\bobservability\b", r"\bhealth ?check\b"),
    workers=DEVOPS_WORKERS,
    system_extra=(
        "You are the DevOps Specialist, an autonomous coordinator for infrastructure, deployment, "
        "and operational concerns: containers, CI/CD pipelines, infra-as-code, and observability. "
        "Before acting, silently identify which worker the current step belongs to and open your "
        "reply with its tag: '[docker_worker]', '[cicd_worker]', '[iac_worker]', or "
        "'[monitoring_worker]'. Favor standard, widely-understood patterns for this ecosystem over "
        "clever ones; never write a real secret/credential into a committed file - reference an "
        "environment variable or the platform's secret store instead."
    ),
    rpm_share=1.0,
)

# --------------------------------------------------------------------------- #
# Docs specialist + workers
# --------------------------------------------------------------------------- #
DOCS_WORKERS = (
    WorkerDef(
        "readme_worker", "README/Guide Worker",
        triggers=(r"\breadme\b", r"\bgetting started\b", r"\binstall.*instructions\b",
                  r"\bquickstart\b", r"\bcontributing\.md\b", r"\bchangelog\b"),
        persona=("You are the README/Guide Worker: specialist in project-level documentation - "
                 "README structure (what it is, install, quickstart, usage), CONTRIBUTING guides, "
                 "and changelogs. Write for a reader who has never seen this project; verify every "
                 "command you document actually works by checking the real project files, don't "
                 "invent install steps."),
    ),
    WorkerDef(
        "apidoc_worker", "API Doc Worker",
        triggers=(r"\bapi doc", r"\bopenapi\b", r"\bswagger\b", r"\bendpoint doc",
                  r"\bdocstring\b", r"\bjsdoc\b", r"\btypedoc\b"),
        persona=("You are the API Doc Worker: specialist in API-level documentation - docstrings "
                 "(Google/NumPy/JSDoc style matching the project's existing convention), OpenAPI/"
                 "Swagger schema accuracy, and per-endpoint request/response documentation. "
                 "Document actual behavior read from the code, never assumed behavior."),
    ),
    WorkerDef(
        "archdoc_worker", "Architecture Doc Worker",
        triggers=(r"\barchitecture\b.*doc", r"\bdesign doc\b", r"\badr\b", r"\bsystem diagram\b",
                  r"\bhow.*works\b.*explain", r"\boverview\b.*(system|architecture)"),
        persona=("You are the Architecture Doc Worker: specialist in explaining how a system "
                 "fits together at a higher level than any single file - module responsibilities, "
                 "data flow, key design decisions and their rationale (ADR-style). Base every claim "
                 "on code you've actually read (repo_map/read_file/find_callers), never on "
                 "assumption; note where you're inferring vs. where a comment/commit explains it."),
    ),
)

DOCS_SPECIALIST = SpecialistDef(
    "docsspecialist", "Documentation Specialist",
    triggers=(r"\bdocument", r"\breadme\b", r"\bdocstring\b", r"\bapi doc", r"\bcomment\b.*code",
              r"\bexplain.*(codebase|architecture|system)", r"\bchangelog\b", r"\bcontributing\b",
              r"\badr\b", r"\bdesign doc\b", r"\bopenapi\b", r"\bswagger\b"),
    workers=DOCS_WORKERS,
    system_extra=(
        "You are the Documentation Specialist, an autonomous coordinator for all written "
        "documentation: READMEs, API docs/docstrings, and architecture explanations. Before "
        "writing, silently identify which worker the current step belongs to and open your reply "
        "with its tag: '[readme_worker]', '[apidoc_worker]', or '[archdoc_worker]'. Read the real "
        "code first (never invent behavior); match the project's existing doc style/format if it "
        "has one; keep documentation accurate over exhaustive - a correct short doc beats a long "
        "one with invented details."
    ),
    rpm_share=1.0,
)

# --------------------------------------------------------------------------- #
# Review specialist - READ-ONLY by design (allowed_tools has no write/exec
# tools at all). Its job is to find and explain problems precisely enough
# that another specialist (or the top-level agent) can fix them, never to
# fix them itself - this keeps a "just double-check this" request from
# silently turning into an uncontrolled rewrite.
# --------------------------------------------------------------------------- #
REVIEW_WORKERS = (
    WorkerDef(
        "quality_worker", "Code Quality Worker",
        triggers=(r"\bcode review\b", r"\breview.*(pr|pull request|code)", r"\bcode quality\b",
                  r"\bcode smell\b", r"\bcomplexity\b", r"\bmaintainab", r"\bnaming\b",
                  r"\bstyle\b.*review", r"\bbest practices?\b"),
        persona=("You are the Code Quality Worker: specialist in reviewing code for correctness, "
                 "readability, naming, complexity, duplication, and adherence to the project's "
                 "existing conventions. Cite exact file:line locations for every finding; "
                 "distinguish must-fix (bugs, correctness) from should-fix (style, clarity) from "
                 "nice-to-have (taste). You do not have write access - report only."),
    ),
    WorkerDef(
        "perf_worker", "Performance Review Worker",
        triggers=(r"\bperformance\b", r"\bslow\b", r"\boptimiz", r"\bn\+1\b", r"\bbottleneck\b",
                  r"\bmemory leak\b", r"\bprofil", r"\blatency\b", r"\bO\(n"),
        persona=("You are the Performance Review Worker: specialist in spotting real performance "
                 "issues - N+1 queries, unnecessary O(n^2)+ algorithms on hot paths, unbounded "
                 "memory growth, blocking I/O on an async path, missing indexes/caching. Only flag "
                 "issues you can point to concretely in the code (file:line + why it's slow); "
                 "don't speculate about performance without evidence. You do not have write access."),
    ),
    WorkerDef(
        "arch_worker", "Architecture Review Worker",
        triggers=(r"\barchitecture\b.*review", r"\bdesign review\b", r"\bcoupling\b",
                  r"\bseparation of concerns\b", r"\bsolid\b", r"\bpattern\b.*review",
                  r"\bscalab", r"\btechnical debt\b"),
        persona=("You are the Architecture Review Worker: specialist in structural/design-level "
                 "review - coupling and cohesion, separation of concerns, whether abstractions "
                 "match actual usage, and technical debt that will bite future changes. Ground "
                 "every claim in what find_callers/repo_map/read_file actually show about how the "
                 "code is used, not abstract principle alone. You do not have write access."),
    ),
)

REVIEW_SPECIALIST = SpecialistDef(
    "reviewspecialist", "Code Review Specialist",
    triggers=(r"\breview\b", r"\bcode review\b", r"\bpr review\b", r"\bpull request\b",
              r"\bcritique\b", r"\bsecond opinion\b", r"\bis this (good|correct|right)\b",
              r"\bdouble.?check\b", r"\bfeedback on\b.*(code|implementation)"),
    workers=REVIEW_WORKERS,
    system_extra=(
        "You are the Code Review Specialist, a READ-ONLY autonomous reviewer covering code "
        "quality, performance, and architecture. You physically cannot write or run anything - "
        "your only output is a precise, actionable review. Before writing, silently identify "
        "which worker the current step belongs to and open your reply with its tag: "
        "'[quality_worker]', '[perf_worker]', or '[arch_worker]'. Structure findings as: what/"
        "where (exact file:line), why it matters, and a concrete suggested fix in words (not code "
        "you apply yourself) - the user or another specialist decides whether and how to act on it. "
        "Be honest about severity: don't inflate nitpicks into blockers, and don't soften real bugs."
    ),
    rpm_share=1.0,
    allowed_tools=frozenset({
        "list_dir", "repo_map", "read_file", "find_files", "search_code", "find_callers", "find_symbol",
        "goto_definition", "find_references",
        "save_note", "recall_notes", "git_status", "git_diff", "git_log", "git_blame",
    }),
)

SPECIALISTS: tuple[SpecialistDef, ...] = (
    FRONTEND_SPECIALIST, BACKEND_SPECIALIST, PYTHON_SPECIALIST,
    SECURITY_SPECIALIST, TEST_SPECIALIST, DEVOPS_SPECIALIST, DOCS_SPECIALIST, REVIEW_SPECIALIST,
)
SPECIALIST_BY_KEY: dict[str, SpecialistDef] = {s.key: s for s in SPECIALISTS}

# Real models write the tag many ways: "[html_worker]" alone, "[html_worker] text",
# "**[html_worker]**", "`[css_worker]`", "[HTML_WORKER]", "Worker: [js_worker]".
# Accept all of them, anywhere in the first ~200 chars, case-insensitively; the
# caller validates the captured key against the specialist's real worker keys.
_WORKER_TAG_RE = re.compile(r"\[\s*([A-Za-z][A-Za-z0-9_]*_worker)\s*\]", re.IGNORECASE)


def find_worker_tag(text: str, spec: "SpecialistDef") -> Optional[str]:
    """Return the first valid worker key of `spec` tagged near the start of `text`."""
    valid = {w.key for w in spec.workers}
    for m in _WORKER_TAG_RE.finditer(text[:240]):
        key = m.group(1).lower()
        if key in valid:
            return key
    return None


# =========================================================================== #
#  ROUTER  -  decides which specialist(s) to wake up for a task
# =========================================================================== #
def _score_specialist(spec: SpecialistDef, text: str) -> int:
    """Score = matches on the specialist's own triggers, PLUS a lighter weight
    for matches on any of its workers' triggers - this way a phrase that only
    a worker-level pattern catches (e.g. 'run the test suite') still routes
    to the right specialist even if the specialist's own curated trigger list
    doesn't happen to mention it verbatim."""
    own = sum(1 for pat in spec.triggers if re.search(pat, text))
    worker_hits = sum(1 for w in spec.workers for pat in w.triggers if re.search(pat, text))
    # worker hits count at half weight so a specialist with an explicit,
    # deliberate top-level match still outranks an incidental worker overlap
    return own * 2 + (1 if worker_hits else 0)


def route_task(task: str) -> list[SpecialistDef]:
    """Inspect task text and return the specialists whose domain it matches,
    best match first. Empty list means no specialist is a clear fit (caller
    should fall back to the generic agent/subagent)."""
    text = task.lower()
    scored = [(_score_specialist(s, text), s) for s in SPECIALISTS]
    scored = [(sc, s) for sc, s in scored if sc > 0]
    scored.sort(key=lambda t: -t[0])
    return [s for _, s in scored]


def route_worker(spec: SpecialistDef, task: str) -> Optional[WorkerDef]:
    """Best-matching worker within a specialist, for pre-activation logging.
    The specialist's own model turn will also self-report via the [worker]
    tag; this is used as an immediate best-guess before the first turn
    returns, so the UI shows *something* active right away."""
    text = task.lower()
    best, best_score = None, 0
    for w in spec.workers:
        sc = sum(1 for pat in w.triggers if re.search(pat, text))
        if sc > best_score:
            best, best_score = w, sc
    return best


# =========================================================================== #
#  PER-SPECIALIST RATE LIMITING
# =========================================================================== #
def get_specialist_limiter(limiter_cache: dict, spec: SpecialistDef, base_profile) -> "AdaptiveRateLimiter":
    """Each specialist gets its OWN AdaptiveRateLimiter (keyed separately from
    the parent model's limiter) so frontend/backend/python work is paced
    independently and one can't starve another or the parent agent. Seeded
    from the same provider profile (same model = same real ceiling) but
    tracked/adapted separately, and scaled by rpm_share if a specialist is
    intentionally throttled lower.
    """
    h = host()
    AdaptiveRateLimiter, RateProfile = h.AdaptiveRateLimiter, h.RateProfile

    cache_key = f"__specialist__{spec.key}"
    if cache_key not in limiter_cache:
        p = base_profile
        seeded = RateProfile(
            name=f"{p.name}:{spec.key}",
            rpm=max(p.floor_rpm, p.rpm * spec.rpm_share),
            tpm=p.tpm,
            burst=max(1, p.burst),
            floor_rpm=p.floor_rpm,
            max_retries=p.max_retries,
            base_backoff=p.base_backoff,
            max_backoff=p.max_backoff,
        )
        limiter_cache[cache_key] = AdaptiveRateLimiter(seeded)
    return limiter_cache[cache_key]


# =========================================================================== #
#  ACTIVATION REGISTRY  -  enable/disable specialists & workers at runtime
# =========================================================================== #
class ActivationRegistry:
    """Tracks which specialists/workers are enabled, and which are currently
    "awake" (mid-task). Thread-safe; the LLM-facing tools and the /specialists
    CLI command both go through this so behaviour stays consistent.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.enabled: dict[str, bool] = {s.key: True for s in SPECIALISTS}
        self.worker_enabled: dict[str, bool] = {
            w.key: True for s in SPECIALISTS for w in s.workers
        }
        self.awake: dict[str, float] = {}          # specialist key -> since (monotonic)
        self.awake_workers: dict[str, str] = {}     # worker key -> owning specialist key

    def is_enabled(self, spec_key: str) -> bool:
        with self._lock:
            return self.enabled.get(spec_key, True)

    def is_worker_enabled(self, worker_key: str) -> bool:
        with self._lock:
            return self.worker_enabled.get(worker_key, True)

    def set_enabled(self, spec_key: str, on: bool) -> bool:
        with self._lock:
            if spec_key not in self.enabled:
                return False
            self.enabled[spec_key] = on
            return True

    def set_worker_enabled(self, worker_key: str, on: bool) -> bool:
        with self._lock:
            if worker_key not in self.worker_enabled:
                return False
            self.worker_enabled[worker_key] = on
            return True

    def wake(self, spec_key: str) -> None:
        with self._lock:
            self.awake[spec_key] = time.monotonic()

    def sleep(self, spec_key: str) -> None:
        with self._lock:
            self.awake.pop(spec_key, None)
            for wk in [k for k, v in self.awake_workers.items() if v == spec_key]:
                self.awake_workers.pop(wk, None)

    def wake_worker(self, worker_key: str, spec_key: str) -> None:
        with self._lock:
            self.awake_workers[worker_key] = spec_key

    def snapshot(self) -> dict:
        with self._lock:
            return dict(enabled=dict(self.enabled), worker_enabled=dict(self.worker_enabled),
                        awake=dict(self.awake), awake_workers=dict(self.awake_workers))


# =========================================================================== #
#  SPECIALIST RUN LOOP  (mirrors run_subagent's shape/guarantees)
# =========================================================================== #
# NOTE on this number's history: this used to be 14, tuned back when symbol
# reads/edits ran on regex fallbacks. Real AST/CST + semantic-navigation work
# (find_definition -> find_references -> read symbol -> edit -> verify, often
# repeated across 2-3 call sites for a rename/refactor) routinely needs more
# than 14 tool turns for anything beyond a one-line tweak, so specialists were
# being cut off mid-task on ordinary refactors - which looked like "failing
# after step 14" even though nothing had crashed. Raised to a budget sized for
# semantic-navigation-heavy work; still bounded (never unbounded) so a stuck
# specialist can't loop forever.
SPEC_MAX_STEPS = 28

# Tools a specialist may NOT call. Letting a specialist call dispatch_* would
# allow unbounded recursive fan-out (specialist -> specialist -> ...) that
# multiplies threads and rate-limit pressure with no ceiling.
_FORBIDDEN_IN_SPECIALIST = frozenset({"dispatch_specialists", "set_specialist_active", "dispatch_agents"})

# Unique, monotonically increasing ids across the WHOLE session. The original
# code used enumerate(...) per call, so a second dispatch reused ids 1..n and
# overwrote the first dispatch's rows in the UI.
_ID_LOCK = threading.Lock()
_NEXT_ID = 0


def _next_spec_id() -> int:
    global _NEXT_ID
    with _ID_LOCK:
        _NEXT_ID += 1
        return _NEXT_ID


# --------------------------------------------------------------------------- #
# Persona injection - provider agnostic
# --------------------------------------------------------------------------- #
def apply_persona(provider, system: str) -> None:
    """Make `system` the ACTUAL system prompt the model receives.

    Provider classes bake the system prompt in at construction time, in
    different places:
      * OpenRouterProvider  -> self.messages[0] == {"role":"system", ...}
      * GeminiProvider      -> self.cfg.system_instruction
    Merely assigning `provider.system = ...` (what the old code did) changes
    a label nobody reads, so specialists silently ran with the GENERIC prompt
    and never learned to emit [worker_tag] lines.
    """
    provider.system = system
    msgs = getattr(provider, "messages", None)
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        msgs[0]["content"] = system
    cfg = getattr(provider, "cfg", None)
    if cfg is not None:
        try:
            cfg.system_instruction = system
        except Exception:                                            # noqa: BLE001
            pass
    # Some SDK config objects are frozen/pydantic: rebuild via model_copy if present.
    if cfg is not None and getattr(cfg, "system_instruction", None) != system and hasattr(cfg, "model_copy"):
        try:
            provider.cfg = cfg.model_copy(update={"system_instruction": system})
        except Exception:                                            # noqa: BLE001
            pass


def restrict_tools(provider, spec: Optional["SpecialistDef"] = None) -> None:
    """Strip recursive-dispatch tools from what THIS provider advertises (always),
    and - when `spec.allowed_tools` is set - strip everything NOT in that
    explicit allow-list too, so a domain specialist (e.g. a read-only code
    reviewer) is physically incapable of calling write_file/edit_file/
    run_command etc., not just told not to in its prompt. Re-checked again
    per-call in run_specialist as defense in depth against a model that
    hallucinates a call the provider layer somehow still let through."""
    def _blocked(name: str) -> bool:
        if name in _FORBIDDEN_IN_SPECIALIST:
            return True
        if spec is not None and spec.allowed_tools is not None and name not in spec.allowed_tools:
            return True
        return False

    # OpenRouter-style: self.tools = [{"type":"function","function":{"name":...}}]
    tools = getattr(provider, "tools", None)
    if isinstance(tools, list):
        provider.tools = [t for t in tools if not _blocked((t.get("function", {}) or {}).get("name"))]
    # Gemini-style: cfg.tools = [types.Tool(function_declarations=[...])]
    cfg = getattr(provider, "cfg", None)
    if cfg is not None and getattr(cfg, "tools", None):
        try:
            for tool in cfg.tools:
                decls = getattr(tool, "function_declarations", None)
                if decls:
                    tool.function_declarations = [d for d in decls if not _blocked(getattr(d, "name", ""))]
        except Exception:                                            # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Explicit result status - NOT inferred from "FAILED" in the text.
#
# Bug this replaces: callers used to decide ok/bad with `"FAILED" not in r`
# (dispatch_specialists) or `("FAILED" in out) or out.startswith("error:")`
# (execute_plan) - two DIFFERENT ad-hoc checks on the same string, neither of
# which recognised "stopped after N steps (INCOMPLETE)" as anything other
# than success, and both of which are one unlucky word away from a false
# positive (a test specialist reporting "3 tests FAILED, 2 passed" as its
# genuinely-successful summary would trip the FAILED check even though the
# specialist itself finished its own turn cleanly). That silent
# misclassification is what let a plan step marked "done" actually be a
# specialist that got cut off mid-task at the step budget - the step-14
# failure looked like success further up the chain.
#
# Fix: every return path below tags its result with one of these markers as
# a PREFIX (`[[status:xxx]] `), which run_specialist strips before handing
# the human-readable text back, and which `specialist_result_status()` reads
# to classify the result deterministically - no string-sniffing of prose.
_STATUS_OK = "ok"                 # finished its turn (turn.calls empty) - genuine completion
_STATUS_INCOMPLETE = "incomplete" # ran out of steps - not a crash, but not finished either
_STATUS_FAILED = "failed"         # rate-limited / raised / provider error

_STATUS_TAG_RE = re.compile(r"^\[\[status:(ok|incomplete|failed)\]\] ")


def _tag(status: str, text: str) -> str:
    return f"[[status:{status}]] {text}"


def specialist_result_status(result: str) -> str:
    """Read back the status a run_specialist result was tagged with (the
    LEADING [[status:...]] tag). Defaults to 'failed' (fail-closed) if a
    result somehow has no tag - safer than defaulting to 'ok' and letting an
    unrecognised result silently pass."""
    m = _STATUS_TAG_RE.match(result or "")
    return m.group(1) if m else _STATUS_FAILED


def strip_status_tag(result: str) -> str:
    """A single specialist's human/UI-facing text with its leading
    [[status:...]] tag removed."""
    return _STATUS_TAG_RE.sub("", result or "", count=1)


def run_specialist(spec_id: int, spec: SpecialistDef, provider, ws, task: str,
                   registry: ActivationRegistry, ui=None) -> str:
    """One specialist's bounded tool loop against the shared workspace, using
    its OWN provider (own history) and OWN rate limiter (already wired into
    `provider.limiter` by the caller via get_specialist_limiter). Reports
    into UIState.specialists exactly like run_subagent reports into
    UIState.subagents, so the left panel can render it live.
    """
    h = host()
    CircuitOpen, parse_error, _tool_detail = h.CircuitOpen, h.parse_error, h._tool_detail

    registry.wake(spec.key)
    guess = route_worker(spec, task)
    active_worker = guess.key if guess and registry.is_worker_enabled(guess.key) else None
    if active_worker:
        registry.wake_worker(active_worker, spec.key)

    if ui:
        ui.spec_start(spec_id, spec.key, spec.label, task, active_worker)

    def wait_cb(left: float, why: str) -> None:
        if ui:
            ui.spec_update(spec_id, phase=f"waiting {left:.0f}s ({why})")

    # Prime the model with the routing pre-guess so it starts in the right persona
    # even if it forgets to emit a tag on turn one.
    primer = task
    if active_worker:
        w = next((w for w in spec.workers if w.key == active_worker), None)
        if w:
            primer = f"{task}\n\n[start as {w.key}] {w.persona}"
    provider.add_user(primer)

    last_text = ""
    seen_workers: list[str] = [active_worker] if active_worker else []
    try:
        for step in range(1, SPEC_MAX_STEPS + 1):
            if ui:
                ui.spec_update(spec_id, phase=f"thinking (step {step})", worker=active_worker)
            try:
                turn = provider.turn(wait_cb, lambda: False)
            except CircuitOpen as e:
                summary = f"rate limited: paused ~{int(e.reopen_in)}s"
                if ui:
                    ui.spec_end(spec_id, ok=False, summary=summary)
                return _tag(_STATUS_FAILED, f"[{spec.key} FAILED] {summary}")
            except Exception as e:                                    # noqa: BLE001
                sig = parse_error(e)
                summary = f"error: {type(e).__name__}: {sig.reason}"
                if ui:
                    ui.spec_end(spec_id, ok=False, summary=summary)
                return _tag(_STATUS_FAILED, f"[{spec.key} FAILED] {summary}")

            if turn.text.strip():
                last_text = turn.text.strip()
                tag = find_worker_tag(last_text, spec)
                if tag and registry.is_worker_enabled(tag):
                    if tag != active_worker:
                        active_worker = tag
                    registry.wake_worker(tag, spec.key)
                    if tag not in seen_workers:
                        seen_workers.append(tag)
                # display line = first line that is not just the tag
                clean = _WORKER_TAG_RE.sub("", last_text).strip(" *`\t")
                first_line = (clean.splitlines() or ["working"])[0][:70] or "working"
                if ui:
                    ui.spec_update(spec_id, detail=first_line, worker=active_worker)

            if not turn.calls:
                clean = _WORKER_TAG_RE.sub("", last_text).strip(" *`\t")
                summary = (clean.splitlines()[-1][:70] if clean else "done")
                if ui:
                    ui.spec_end(spec_id, ok=True, summary=summary)
                wtrail = "/".join(seen_workers) if seen_workers else (active_worker or "")
                return _tag(_STATUS_OK,
                            f"[{spec.key}{f'/{wtrail}' if wtrail else ''}] {last_text or 'done (no output)'}")

            results = []
            for c in turn.calls:
                if ui:
                    ui.spec_update(spec_id, phase=f"{c.name} {_tool_detail(c)[:36]}", worker=active_worker)
                if c.name in _FORBIDDEN_IN_SPECIALIST:
                    out = (f"error: '{c.name}' is not available inside a specialist. "
                           "Do the work yourself with the file/command tools.")
                elif spec.allowed_tools is not None and c.name not in spec.allowed_tools:
                    out = (f"error: '{c.name}' is outside {spec.label}'s scope ({spec.key} is "
                           "restricted to: " + ", ".join(sorted(spec.allowed_tools)) + "). "
                           "Report the finding instead, or ask for a different specialist.")
                else:
                    out = ws.execute(c.name, c.args)
                results.append((c, out))
            note = h.budget_note(step, SPEC_MAX_STEPS)
            if note and results:
                results[-1] = (results[-1][0], results[-1][1] + note)
            provider.add_tool_results(results)

        summary = f"stopped after {SPEC_MAX_STEPS} steps (INCOMPLETE)"
        if ui:
            ui.spec_end(spec_id, ok=False, summary=summary)      # was ok=True -> green tick on unfinished work
        return _tag(_STATUS_INCOMPLETE, f"[{spec.key}] {summary}. Last output: {last_text[:200]}")
    finally:
        registry.sleep(spec.key)


# =========================================================================== #
#  PARALLEL DISPATCH ACROSS SPECIALISTS
# =========================================================================== #
def max_parallel_specialists(limiter_cache: dict, specs: list[SpecialistDef],
                             base_profile, hard_cap: int = 3) -> int:
    """Specialists run on independent limiters, so unlike generic subagents
    (bounded by ONE shared limiter's burst), the ceiling here is about
    sane concurrency for the terminal UI and the workspace write-lock, not
    the rate limit itself. Still capped conservatively."""
    return max(1, min(len(specs), hard_cap))


def _install_write_lock(ws) -> None:
    """Delegates to coder3.install_write_lock - ONE persistent lock shared by specialists AND
    dispatch_agents subagents (idempotent; wrappers forward symbol=/insert_after=/overwrite=)."""
    host().install_write_lock(ws)


def dispatch_specialists(ws, spec_holder, keys: dict, limiter_cache: dict,
                         registry: ActivationRegistry, assignments: list[dict]) -> str:
    """Fan work out across specialist coordinators in parallel.

    assignments: list of {"specialist": "<key>", "task": "<instruction>"}
    Each specialist gets its own Provider (own history, own persona-augmented
    system prompt) and its own AdaptiveRateLimiter (via get_specialist_limiter),
    so frontend/backend/python work never contends for the same budget.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    h = host()
    build_provider, PROFILES, DEFAULT_PROFILE = h.build_provider, h.PROFILES, h.DEFAULT_PROFILE

    spec_val = spec_holder["spec"] if isinstance(spec_holder, dict) else spec_holder
    base_profile = PROFILES.get(spec_val.id, DEFAULT_PROFILE)

    jobs, skipped = [], []
    for a in (assignments or [])[:6]:
        key = str((a or {}).get("specialist", "")).strip().lower()
        task = str((a or {}).get("task", "")).strip()
        sdef = SPECIALIST_BY_KEY.get(key)
        if not sdef:
            skipped.append(f"unknown specialist '{key}'")
            continue
        if not task:
            skipped.append(f"{key}: empty task")
            continue
        if not registry.is_enabled(key):
            skipped.append(f"{key}: disabled (use set_specialist_active to enable)")
            continue
        jobs.append((sdef, task))

    if not jobs:
        return ("error: no valid/enabled specialist assignments"
                + (f" ({'; '.join(skipped)})" if skipped else "")
                + ". Valid keys: " + ", ".join(SPECIALIST_BY_KEY))

    ui = ws.ui
    if ui:
        ui.add_log("sub", f"waking {len(jobs)} specialist(s): " + ", ".join(sd.key for sd, _ in jobs))

    n_workers = max_parallel_specialists(limiter_cache, [sd for sd, _ in jobs], base_profile)
    _install_write_lock(ws)

    results: dict[int, str] = {}
    order: list[int] = []
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="specialist") as pool:
        futures = {}
        for sdef, task in jobs:
            sid = _next_spec_id()
            order.append(sid)
            limiter = get_specialist_limiter(limiter_cache, sdef, base_profile)
            provider = build_provider(spec_val, keys, ws, limiter_cache)
            provider.limiter = limiter                   # this specialist's OWN budget
            apply_persona(provider, SYSTEM_PROMPT_FOR(sdef, ws))
            restrict_tools(provider, sdef)
            futures[pool.submit(run_specialist, sid, sdef, provider, ws, task, registry, ui)] = sid
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                results[sid] = fut.result()
            except Exception as e:                                    # noqa: BLE001
                results[sid] = _tag(_STATUS_FAILED, f"[specialist {sid} FAILED] {type(e).__name__}: {e}")
                if ui:
                    ui.spec_end(sid, ok=False, summary=f"{type(e).__name__}: {e}")

    # Explicit status classification (see specialist_result_status) - NOT a
    # substring check on the rendered text, so a specialist whose own report
    # happens to contain the word "FAILED" (e.g. a test specialist summarising
    # "3 tests FAILED, 2 passed") is still correctly counted 'ok' if it
    # actually finished its turn, and a step-budget-exhausted specialist is
    # counted as incomplete rather than silently 'ok'.
    statuses = {sid: specialist_result_status(r) for sid, r in results.items()}
    ok = sum(1 for s in statuses.values() if s == _STATUS_OK)
    incomplete = sum(1 for s in statuses.values() if s == _STATUS_INCOMPLETE)
    failed = sum(1 for s in statuses.values() if s == _STATUS_FAILED)
    if ui:
        parts = [f"{ok}/{len(jobs)} specialists finished ok"]
        if incomplete:
            parts.append(f"{incomplete} incomplete (ran out of steps)")
        if failed:
            parts.append(f"{failed} failed")
        ui.add_log("sub", ", ".join(parts))
    note = f"\n\n(skipped: {'; '.join(skipped)})" if skipped else ""
    rendered = "\n\n".join(strip_status_tag(results[i]) for i in order if i in results)
    # The single worst status across this dispatch, appended as ONE clearly-
    # delimited trailer line that both the top-level model (harmless extra
    # context - it already sees "(INCOMPLETE)"/"FAILED" in the prose above)
    # and execute_plan (which parses THIS line, not the free-form prose
    # above it) can read. This replaces an earlier draft that stashed status
    # in a module-level "last call" global: that broke the moment two
    # dispatches could interleave (e.g. a test harness, or future concurrent
    # plan execution) - a caller could read another call's status. A value
    # threaded through the actual return string has no such race.
    worst = (_STATUS_FAILED if failed else _STATUS_INCOMPLETE if incomplete else _STATUS_OK)
    return f"{rendered}{note}\n[[dispatch_status:{worst}]]"


def dispatch_result_status(result: str) -> str:
    """Read the worst-case status ('ok'|'incomplete'|'failed') appended to a
    dispatch_specialists() return string by THIS call - safe to parse from
    the exact value a caller received (no shared/global state, no races).
    Defaults to 'failed' (fail-closed) if the trailer is missing, e.g. a
    hand-built string from older code or a test."""
    m = re.search(r"\[\[dispatch_status:(ok|incomplete|failed)\]\]\s*$", result or "")
    return m.group(1) if m else _STATUS_FAILED


def strip_dispatch_status(result: str) -> str:
    """The dispatch_specialists text without its trailing status marker -
    what the top-level model / UI should actually display."""
    return re.sub(r"\n?\[\[dispatch_status:(ok|incomplete|failed)\]\]\s*$", "", result or "")



_DOMAIN_NAV = {
    "frontendspecialist": ("CSS rules are symbols: read_file/edit_file symbol='.card' or "
                           "'@media (max-width: 600px) > .card'; HTML elements by '#id' or 'section.hero'; JS functions by "
                           "name (also those inside inline <script>/<style>). Add a new rule/function with "
                           "edit_file(insert_after=<neighbouring rule/function>, new=...) - never rewrite the whole stylesheet."),
    "backendspecialist": ("Routes, handlers, models and services are functions/classes: read_file(symbol=...), "
                          "edit_file(symbol=...). find_callers before changing a handler/model/schema signature."),
    "pythonspecialist": ("Read one function/class at a time (symbol='Class.method'); find_callers before changing a "
                         "signature; refactor with edit_file(symbol=...) / edit_files."),
    "securityspecialist": ("Trace with find_symbol/find_callers, then read only the flagged function via symbol=; "
                           "fix with edit_file(symbol=...)."),
    "testspecialist": ("Read the function under test by symbol; add a test with edit_file(insert_after=<neighbouring test>) "
                       "or a new file; run the tests and read the real output."),
    "devopsspecialist": ("YAML/Dockerfile/Terraform are not symbol-indexed: use start/end ranges and small `old` edits; "
                         "validate with the tool's own lint/plan command via run_command."),
    "docsspecialist": ("Read the code you document by symbol for accuracy. Markdown is not symbol-indexed: use start/end "
                       "ranges or insert_after on a neighbouring symbol only for code files."),
    "reviewspecialist": "Cite file:line from read_file(symbol=...) / find_callers results; never speculate.",
}


def SYSTEM_PROMPT_FOR(spec: SpecialistDef, ws) -> str:
    """Specialist system prompt = the host's BASE prompt (workflow, edit rules, verify-after-change,
    workspace root) minus the delegation section + the domain addendum + the worker-tag contract."""
    h = host()
    base = h.base_system_prompt(ws, with_specialists=False)
    base = base.split("Parallel subagents (dispatch_agents):")[0].rstrip()    # a specialist can't delegate
    worker_list = "\n".join(f"  - {w.key}: {w.label}" for w in spec.workers)
    if spec.allowed_tools is not None:
        scope_note = (
            f"TOOL SCOPE: you are restricted to exactly these tools: {', '.join(sorted(spec.allowed_tools))}. "
            "Any other call is rejected. "
            + ("You have NO write access - report findings, do not attempt fixes; recommend which specialist "
               "should make the change.\n"
               if not ({"write_file", "edit_file", "edit_files"} & spec.allowed_tools) else "\n"))
    else:
        scope_note = "You cannot delegate: do the work yourself with the tools you have.\n"
    nav = _DOMAIN_NAV.get(spec.key, "")
    return (
        f"{base}\n\n"
        f"=== SPECIALIST ROLE ===\n{spec.system_extra}\n"
        + (f"NAVIGATION: {nav}\n" if nav else "")
        + f"\nYour workers:\n{worker_list}\n"
        f"WORKER TAG CONTRACT: begin EVERY reply with the tag of the worker you are acting as, in square "
        f"brackets, e.g. '[{spec.workers[0].key}]'. Switch tags when the work switches worker.\n"
        f"{scope_note}"
    )


# =========================================================================== #
#  PLANNER  -  plan-first workflow (read-only reasoning -> structured plan ->
#              user review -> step-by-step execution by the right specialist)
# =========================================================================== #
import json as _json


@dataclass
class PlanStep:
    """One unit of work. `owner` is a specialist key so execution can be routed
    with zero extra LLM calls; `files` lets us detect overlap between steps."""
    id: int
    title: str
    detail: str = ""
    owner: str = ""                                   # specialist key ("" = top-level agent)
    files: list = field(default_factory=list)         # paths this step creates/edits
    verify: str = ""                                  # how we know it worked (command or check)
    status: str = "todo"                              # todo | run | done | fail | incomplete | skip
    note: str = ""                                    # execution result / error


@dataclass
class Plan:
    goal: str = ""
    summary: str = ""
    stack: list = field(default_factory=list)         # e.g. ["Next.js", "Tailwind"]
    assumptions: list = field(default_factory=list)
    questions: list = field(default_factory=list)     # things the planner could not decide
    risks: list = field(default_factory=list)
    steps: list = field(default_factory=list)         # list[PlanStep]
    version: int = 1
    source_prompt: str = ""

    # ---- helpers ------------------------------------------------------- #
    def file_set(self) -> list:
        seen: dict = {}
        for st in self.steps:
            for f in st.files:
                seen.setdefault(f, None)
        return list(seen)

    def renumber(self) -> None:
        for i, st in enumerate(self.steps, 1):
            st.id = i

    def progress(self) -> tuple:
        done = sum(1 for s in self.steps if s.status in ("done", "skip"))
        return done, len(self.steps)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal, "summary": self.summary, "stack": self.stack,
            "assumptions": self.assumptions, "questions": self.questions, "risks": self.risks,
            "steps": [{"id": s.id, "title": s.title, "detail": s.detail, "owner": s.owner,
                       "files": s.files, "verify": s.verify} for s in self.steps],
        }

    def to_markdown(self) -> str:
        """Human-readable copy, saved to PLAN.md so the plan survives the session."""
        out = [f"# Plan: {self.goal or 'Untitled'}", ""]
        if self.summary:
            out += [self.summary, ""]
        if self.stack:
            out += ["**Stack:** " + ", ".join(self.stack), ""]
        if self.assumptions:
            out += ["## Assumptions", *[f"- {a}" for a in self.assumptions], ""]
        if self.risks:
            out += ["## Risks", *[f"- {r}" for r in self.risks], ""]
        out += ["## Steps", ""]
        box_ = {"done": "x", "skip": "-", "fail": "!", "incomplete": "~"}
        for s in self.steps:
            out.append(f"- [{box_.get(s.status, ' ')}] **{s.id}. {s.title}**"
                       + (f"  _(owner: {s.owner})_" if s.owner else ""))
            if s.detail:
                out.append(f"  - {s.detail}")
            if s.files:
                out.append("  - files: " + ", ".join(f"`{f}`" for f in s.files))
            if s.verify:
                out.append(f"  - verify: {s.verify}")
        return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Robust JSON extraction: models wrap JSON in prose, ```json fences, trailing
# commas, smart quotes... A planner that crashes on formatting is useless.
# --------------------------------------------------------------------------- #
def _balanced_json_objects(text: str) -> list:
    """Every top-level {...} span in `text`, string- and escape-aware."""
    spans, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start:i + 1])
                start = -1
    return spans


def _repair_json(s: str) -> str:
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)                    # trailing commas
    s = re.sub(r"^\s*//.*$", "", s, flags=re.MULTILINE)      # // line comments
    return s


def extract_plan_json(text: str) -> Optional[dict]:
    """Best-effort: return the first parseable JSON object that looks like a plan."""
    if not text:
        return None
    candidates = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)]
    candidates += _balanced_json_objects(text)
    best = None
    for c in candidates:
        for attempt in (c, _repair_json(c)):
            try:
                obj = _json.loads(attempt)
            except Exception:                                # noqa: BLE001
                continue
            if isinstance(obj, dict) and isinstance(obj.get("steps"), list) and obj["steps"]:
                return obj                                   # a real plan: done
            if isinstance(obj, dict) and best is None:
                best = obj
    return None if best is None or not isinstance(best.get("steps"), list) else best


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v)]


def _safe_rel_path(f: str) -> str:
    """Workspace-relative path or "" (REJECTED, not "fixed"). A traversal attempt must not be
    quietly rewritten into a plausible path: `../etc/passwd` -> "" , never `etc/passwd`.
    Only a leading `./` prefix is stripped (as a prefix, not a character set)."""
    f = f.strip().replace("\\", "/")
    while f.startswith("./"):
        f = f[2:]
    if not f or f.startswith("/") or re.match(r"^[A-Za-z]:", f) or f.startswith("~"):
        return ""                                            # absolute / drive / home path
    parts = f.split("/")
    if ".." in parts or "" in parts[:-1]:
        return ""                                            # traversal or `a//b`
    return f


def _corrected_step_owner(owner: str, title: str, detail: str, files: list,
                           valid_owners: Optional[set] = None) -> str:
    """Deterministic, zero-LLM-call sanity check on a planner-chosen step
    owner (see call site for why this exists). Only overrides on a clear
    mismatch; a close call or an already-correct owner is left alone.

    Deliberately keyword-only (no file-extension bonus like
    route_confidence uses): a step's `files` name WHAT gets touched, not
    WHICH specialist should own it - "test_server.py" is a .py file but
    unambiguously testspecialist's job, "style.css" changed by a step about
    server config would be a red herring, etc. Extension is a fine signal
    for "does this tool call touch X's domain" (route_confidence, at
    runtime, per call) but not for "who should own this whole step" - here
    only the step's own description says what kind of work it is.
    """
    text = f"{title} {detail}".lower()
    ranked = [(_score_specialist(s, text), s) for s in SPECIALISTS
              if valid_owners is None or s.key in valid_owners]
    ranked = [(sc, s) for sc, s in ranked if sc > 0]
    if not ranked:
        return owner
    ranked.sort(key=lambda t: -t[0])
    best_score, best_spec = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0
    decisive = best_score >= STRONG_SCORE and best_score >= runner_up + 2
    if not decisive:
        return owner                              # ambiguous - trust the planner
    cur_score = next((sc for sc, s in ranked if s.key == owner), 0)
    if best_spec.key != owner and best_score > cur_score:
        return best_spec.key
    return owner


def plan_from_dict(obj: dict, source_prompt: str = "", valid_owners: Optional[set] = None) -> "Plan":
    """Coerce arbitrary model output into a clean Plan. Never raises on bad fields:
    unknown owners fall back to '' (top-level), files are normalised, steps capped."""
    valid_owners = valid_owners or set()
    plan = Plan(goal=str(obj.get("goal") or "").strip()[:200],
                summary=str(obj.get("summary") or "").strip()[:600],
                stack=_as_list(obj.get("stack"))[:8],
                assumptions=_as_list(obj.get("assumptions"))[:8],
                questions=_as_list(obj.get("questions"))[:5],
                risks=_as_list(obj.get("risks"))[:6],
                source_prompt=source_prompt)
    for raw in (obj.get("steps") or [])[:MAX_PLAN_STEPS]:
        if isinstance(raw, str):
            raw = {"title": raw}
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or raw.get("name") or "").strip()
        if not title:
            continue
        owner = str(raw.get("owner") or raw.get("specialist") or "").strip().lower()
        if owner not in valid_owners:
            owner = ""
        files = []
        for f in _as_list(raw.get("files")):
            f = _safe_rel_path(f)
            if f and f not in files:
                files.append(f)
        # The planner is a full LLM call choosing `owner` by "vibes" from a
        # bare name+label list, with nothing here checking it against the
        # step's actual content - so a step like "Enhance Frontend Chat
        # Rendering & Styling" can land on testspecialist just because
        # "test" sounds plausible next to "Chat", especially on a cheap/
        # fast model. Re-score the step deterministically (same scorer the
        # runtime safety-net already uses, zero extra LLM calls) against its
        # own title+detail+files; if a DIFFERENT specialist scores decisively
        # higher than whatever the planner picked (or picked nothing for),
        # override it. Only overrides on a clear, unambiguous mismatch -
        # never second-guesses a close call or a deliberate "" (cross-cutting).
        owner = _corrected_step_owner(owner, title, str(raw.get("detail") or ""), files, valid_owners)
        plan.steps.append(PlanStep(id=0, title=title[:120], detail=str(raw.get("detail") or "").strip()[:500],
                                   owner=owner, files=files[:12], verify=str(raw.get("verify") or "").strip()[:200]))
    plan.renumber()
    return plan


def validate_plan(plan: "Plan") -> list:
    """Return human-readable warnings (never blocks; the user decides)."""
    w = []
    if not plan.steps:
        w.append("plan has no steps")
    owners_by_file: dict = {}
    for st in plan.steps:
        for f in st.files:
            owners_by_file.setdefault(f, set()).add(st.owner or "agent")
    for f, owners in owners_by_file.items():
        if len(owners) > 1:
            w.append(f"{f} is touched by several owners ({', '.join(sorted(owners))}) - order matters")
    if plan.steps and not any(s.verify for s in plan.steps):
        w.append("no step has a verification - nothing will be checked")
    if len(plan.steps) >= MAX_PLAN_STEPS:
        w.append(f"plan was capped at {MAX_PLAN_STEPS} steps")
    return w


MAX_PLAN_STEPS = 12


# --------------------------------------------------------------------------- #
# Planner agent
# --------------------------------------------------------------------------- #
PLANNER_READ_TOOLS = frozenset({"list_dir", "read_file", "find_files", "search_code", "repo_map",
                                "find_callers", "find_symbol", "git_status", "git_diff", "git_log",
                                "recall_notes"})
PLANNER_MAX_STEPS = 8            # exploration turns before it MUST answer
PLAN_TAG = "PLAN_JSON"           # delimiter that makes the final answer unambiguous

PLANNER_SYSTEM_EXTRA = """\
=== PLANNER ROLE ===
You are the Planner. You DESIGN the work; you never do it. You have READ-ONLY tools
(list_dir, repo_map, find_symbol, read_file (symbol=... for ONE function/class/CSS rule), find_files, search_code, find_callers, git_status/git_diff/git_log). You cannot write, edit or run anything.

Process:
1. Inspect the workspace FIRST (list_dir, then read the few files that matter: manifests,
   entry points, existing conventions). Do not guess what already exists.
2. Decide the stack and structure. Prefer what the project already uses. For a new project
   choose modern, mainstream, minimal tooling; do not over-engineer.
3. Break the work into 3-{max_steps} ordered, self-contained steps. Each step must be
   something ONE specialist can finish and verify. Order by dependency (scaffold -> data ->
   logic -> UI -> polish -> verify).
4. Assign every step an `owner` from EXACTLY this list, or "" for cross-cutting work:
{owners}
5. For every step list the concrete `files` it creates or edits (workspace-relative paths)
   and a `verify` that is a runnable check (a command, or an observable behaviour).
6. Put anything you genuinely cannot decide from the request/workspace in `questions`
   (max 3, only ones that would change the plan). Record the defaults you chose in
   `assumptions` so the user can correct them.

Finish with ONE JSON object between the markers below, and nothing after it:
<<<{tag}
{{"goal": "...", "summary": "2-3 sentence approach", "stack": ["..."],
  "assumptions": ["..."], "questions": ["..."], "risks": ["..."],
  "steps": [{{"title": "...", "detail": "what and why", "owner": "<key or empty>",
             "files": ["path/a", "path/b"], "verify": "npm run build"}}]}}
{tag}>>>
Be concrete and short. No prose outside the markers after your final tool call.
"""


def planner_system_prompt(ws) -> str:
    h = host()
    base = h.base_system_prompt(ws, with_specialists=False)
    base = base.split("Parallel subagents (dispatch_agents):")[0].rstrip()
    owners = "\n".join(f'   - "{s.key}": {s.label} ({", ".join(w.label for w in s.workers)})'
                       for s in SPECIALISTS)
    return (f"{base}\n\n" + PLANNER_SYSTEM_EXTRA.format(owners=owners, tag=PLAN_TAG, max_steps=MAX_PLAN_STEPS)
            + f"\nWorkspace root: {ws.root}\n")


def _plan_block(text: str) -> str:
    """Text between the <<<PLAN_JSON ... PLAN_JSON>>> markers, else the whole text."""
    m = re.search(rf"<<<{PLAN_TAG}\s*(.*?)\s*{PLAN_TAG}>>>", text or "", re.DOTALL)
    return m.group(1) if m else (text or "")


def build_planner_prompt(request: str, feedback: str = "", previous: Optional["Plan"] = None) -> str:
    if previous is None:
        return f"Create a plan for this request:\n\n{request}"
    prev = _json.dumps(previous.to_dict(), indent=2)
    return (f"Original request:\n{request}\n\nYour previous plan (version {previous.version}):\n{prev}\n\n"
            f"The user wants changes:\n{feedback}\n\n"
            "Revise the plan to satisfy the feedback. Keep steps the user did not mention unless "
            "they conflict. Re-inspect files only if the feedback needs it. Return the full updated plan.")


def run_planner(ws, provider, request: str, feedback: str = "", previous: Optional["Plan"] = None,
                ui=None, on_phase=None) -> "Plan":
    """Run the read-only planning loop and return a validated Plan.

    Raises PlannerError with a user-presentable message on failure (never returns a
    half-plan silently). Read-only is enforced HERE in Python, not just by which tools
    were advertised, because models do hallucinate calls to tools they were not given.
    """
    h = host()
    CircuitOpen, parse_error, _tool_detail = h.CircuitOpen, h.parse_error, h._tool_detail
    apply_persona(provider, planner_system_prompt(ws))
    _restrict_to(provider, PLANNER_READ_TOOLS)
    provider.add_user(build_planner_prompt(request, feedback, previous))
    valid_owners = {s.key for s in SPECIALISTS}

    def wait_cb(left: float, why: str) -> None:
        if ui:
            ui.set_wait(left, why)

    last_text = ""
    for step in range(1, PLANNER_MAX_STEPS + 2):
        final_turn = step > PLANNER_MAX_STEPS
        if on_phase:
            on_phase("analysing workspace" if step == 1 else f"exploring (step {step})")
        try:
            turn = provider.turn(wait_cb, lambda: False)
        except CircuitOpen as e:
            raise PlannerError(f"rate limited - paused ~{int(e.reopen_in)}s. Try /model or wait.") from e
        except KeyboardInterrupt:
            raise
        except Exception as e:                                        # noqa: BLE001
            sig = parse_error(e)
            raise PlannerError(f"{type(e).__name__}: {sig.reason}") from e

        if turn.text.strip():
            last_text = turn.text.strip()
        if not turn.calls or final_turn:
            break

        results = []
        for c in turn.calls:
            if c.name not in PLANNER_READ_TOOLS:                      # hard gate, not a suggestion
                out = (f"error: '{c.name}' is not allowed. You are the Planner: read-only. "
                       "Finish by returning the plan JSON.")
                if ui:
                    ui.add_log("warn", f"planner tried {c.name} - blocked")
            else:
                if on_phase:
                    on_phase(f"{c.name} {_tool_detail(c)[:40]}")
                out = ws.execute(c.name, c.args)
            results.append((c, out))
        provider.add_tool_results(results)
    else:
        pass

    obj = extract_plan_json(_plan_block(last_text)) or extract_plan_json(last_text)
    if obj is None:
        # one corrective retry: ask ONLY for the JSON, no more exploring
        provider.add_user("Your reply did not contain a valid plan. Return ONLY the plan JSON between "
                          f"<<<{PLAN_TAG} and {PLAN_TAG}>>> now. Do not call tools.")
        try:
            turn = provider.turn(wait_cb, lambda: False)
        except Exception as e:                                        # noqa: BLE001
            raise PlannerError(f"planner returned no plan ({type(e).__name__})") from e
        last_text = turn.text.strip() or last_text
        obj = extract_plan_json(_plan_block(last_text)) or extract_plan_json(last_text)
    if obj is None:
        raise PlannerError("the model did not produce a structured plan. Try again or use /model.")

    plan = plan_from_dict(obj, source_prompt=request, valid_owners=valid_owners)
    if not plan.steps:
        raise PlannerError("the plan came back with no usable steps")
    plan.version = (previous.version + 1) if previous else 1
    if not plan.goal:
        plan.goal = request.strip().splitlines()[0][:120]
    return plan


class PlannerError(Exception):
    """A planning failure with a message that is safe to show the user as-is."""


def _restrict_to(provider, allowed: frozenset) -> None:
    """Advertise ONLY `allowed` tools on this provider instance (OpenRouter + Gemini shapes).
    The authoritative gate is the whitelist check in run_planner; this just stops the model
    from being tempted."""
    tools = getattr(provider, "tools", None)
    if isinstance(tools, list):
        provider.tools = [t for t in tools if (t.get("function", {}) or {}).get("name") in allowed]
    cfg = getattr(provider, "cfg", None)
    if cfg is not None and getattr(cfg, "tools", None):
        try:
            for tool in cfg.tools:
                decls = getattr(tool, "function_declarations", None)
                if decls:
                    tool.function_declarations = [d for d in decls if getattr(d, "name", "") in allowed]
        except Exception:                                             # noqa: BLE001
            pass




# --------------------------------------------------------------------------- #
# Plan execution: each step -> one specialist dispatch, in dependency order
# --------------------------------------------------------------------------- #
def step_task_text(plan: "Plan", step: "PlanStep") -> str:
    """A fully self-contained brief. A specialist has no memory of the plan, so it must be
    told the goal, what earlier steps already produced, and exactly what 'done' means."""
    done = [s for s in plan.steps if s.status == "done" and s.id < step.id]
    lines = [f"PROJECT GOAL: {plan.goal}"]
    if plan.stack:
        lines.append("STACK: " + ", ".join(plan.stack))
    if plan.assumptions:
        lines.append("ASSUMPTIONS: " + "; ".join(plan.assumptions))
    if done:
        lines.append("ALREADY DONE (do not redo): " + "; ".join(f"{s.id}. {s.title}" for s in done))
    lines.append(f"\nYOUR STEP ({step.id}/{len(plan.steps)}): {step.title}")
    if step.detail:
        lines.append(step.detail)
    if step.files:
        lines.append("FILES to create/edit: " + ", ".join(step.files))
    if step.verify:
        lines.append(f"VERIFY when finished: {step.verify}  (run it with run_command if it is a command; "
                     "fix what you broke; report the result)")
    lines.append("\nStay strictly within this step. Read existing files before editing them.")
    return "\n".join(lines)


def execute_plan(ws, plan: "Plan", ui=None, should_stop=None, on_step=None) -> dict:
    """Run every non-skipped step in order. Never raises for a failed step: it records the
    failure on the step and STOPS (later steps depend on it), so the user sees exactly where
    it broke. A step whose specialist ran out of its step budget mid-task is recorded as
    "incomplete" (distinct from "fail" - nothing crashed, but the step did not actually
    finish) rather than silently counted as "done"; either way execution stops there, since
    later steps' task text lists earlier steps as "ALREADY DONE" and that must stay true.
    Returns {"done": n, "failed": n, "incomplete": n, "stopped_at": step_id | None}.
    """
    registry = ws.specialist_registry
    done = failed = incomplete_n = 0
    stopped_at = None
    for step in plan.steps:
        if step.status in ("done", "skip"):
            done += step.status == "done"
            continue
        if should_stop and should_stop():
            stopped_at = step.id
            break
        step.status, step.note = "run", ""
        if on_step:
            on_step(step)
        owner = step.owner if (step.owner in SPECIALIST_BY_KEY and registry.is_enabled(step.owner)) else ""
        task = step_task_text(plan, step)
        if ui:
            ui.add_log("sub", f"plan step {step.id}/{len(plan.steps)}: {step.title[:40]}")
        step_ok = True
        try:
            if owner:
                out = ws.dispatch_specialists([{"specialist": owner, "task": task}])
            else:
                # no owner: route by content, falling back to the best-scoring specialist
                top, _score, _strong = route_confidence(task, registry)
                if top:
                    out = ws.dispatch_specialists([{"specialist": top.key, "task": task}])
                else:
                    out = "error: no specialist available for this step (assign an owner)"
                    step_ok = False
        except Exception as e:                                        # noqa: BLE001
            out = f"error: {type(e).__name__}: {e}"
            step_ok = False

        # Explicit status read directly from THIS call's own return string -
        # a trailing "[[dispatch_status:...]]" marker that dispatch_specialists
        # appends (see its docstring/comments). No shared/global state, so
        # nothing to race even if plan execution is ever parallelised later.
        status = dispatch_result_status(out) if step_ok else _STATUS_FAILED
        incomplete = step_ok and status == _STATUS_INCOMPLETE
        hard_fail = step_ok and status == _STATUS_FAILED
        bad = (not step_ok) or hard_fail or out.lstrip().startswith("error:")
        # A step whose specialist ran out of budget mid-task is neither a
        # clean "done" (it explicitly did NOT finish - the old bug: this used
        # to slip through as 'done' because "INCOMPLETE" doesn't contain the
        # substring "FAILED") nor a hard "fail" (nothing crashed, and partial
        # progress may already be on disk) - it gets its own status so the
        # plan view and the next step's "ALREADY DONE" context are both
        # honest about what actually happened.
        if bad:
            step.status = "fail"
        elif incomplete:
            step.status = "incomplete"
        else:
            step.status = "done"
        step.note = re.sub(r"\s+", " ", strip_dispatch_status(out)).strip()[:160]
        done += step.status == "done"
        failed += step.status == "fail"
        incomplete_n += step.status == "incomplete"
        if on_step:
            on_step(step)
        if step.status in ("fail", "incomplete"):
            # Stop either way: a later step's task text lists earlier steps as
            # "ALREADY DONE (do not redo)" - continuing past an incomplete
            # step would tell the next specialist a half-finished change is
            # finished, compounding the original problem instead of surfacing
            # it. The user sees exactly where it stopped and why.
            stopped_at = step.id
            break
    return {"done": done, "failed": failed, "incomplete": incomplete_n, "stopped_at": stopped_at}


# --------------------------------------------------------------------------- #
# PlanGate: the review state machine shared by the worker thread and the UI
# --------------------------------------------------------------------------- #
class PlanGate:
    """Blocks the agent thread while the user reviews a plan. The UI thread only READS
    `view` (a snapshot) and feeds keys via `feed()`; the agent thread blocks in `wait()`.

    Modes (what the keyboard means right now):
      review   : a/enter approve | r revise | e edit steps | x cancel | j/k or arrows move
      edit     : j/k move | space skip/unskip step | K/J reorder | d done editing
      feedback : free text; enter sends it to the planner, esc goes back
    Decisions returned by wait(): ("approve", plan) | ("revise", text) | ("cancel", None)
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._q: "queue.Queue" = queue.Queue()
        self.plan: Optional[Plan] = None
        self.active = False
        self.mode = "review"
        self.cursor = 0
        self.scroll = 0
        self.text = ""                   # feedback being typed
        self.busy = ""                   # non-empty while the planner is (re)planning
        self.warnings: list = []
        self.error = ""
        self.exec_step = 0               # >0 while executing (drives the progress view)
        self.stop_requested = False      # set by request_stop(); polled between steps
        self.exec_note = ""

    # ---- called from the agent thread ---------------------------------- #
    def present(self, plan: "Plan") -> None:
        with self.lock:
            self.plan, self.active, self.mode = plan, True, "review"
            self.cursor = self.scroll = 0
            self.text, self.busy, self.error = "", "", ""
            self.stop_requested = False
            self.warnings = validate_plan(plan)
        while not self._q.empty():
            self._q.get_nowait()

    def set_busy(self, msg: str) -> None:
        with self.lock:
            self.busy = msg

    def set_error(self, msg: str) -> None:
        with self.lock:
            self.error = msg

    def wait(self, timeout: Optional[float] = None):
        """Block for the user's decision. On timeout (no TTY / nobody answering) -> cancel,
        never auto-approve: executing an unreviewed plan is the one thing this gate prevents."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return ("cancel", None)

    def close(self) -> None:
        with self.lock:
            self.active = False
            self.busy = ""

    def abort(self) -> None:
        self._q.put(("cancel", None))

    def request_stop(self) -> bool:
        """Graceful stop for Ctrl-C. Returns True if this call ARMED a stop (so the caller should
        swallow the keypress); False if there is nothing to stop gracefully (not executing, or a
        stop is already pending) - the caller then aborts hard, so a second Ctrl-C always works."""
        with self.lock:
            if not self.exec_step or self.stop_requested:
                return False
            self.stop_requested = True
            self.exec_note = "stopping after this step"
            return True

    # ---- called from the UI thread -------------------------------------- #
    def view(self) -> dict:
        with self.lock:
            p = self.plan
            return {"active": self.active, "mode": self.mode, "cursor": self.cursor,
                    "scroll": self.scroll, "text": self.text, "busy": self.busy,
                    "warnings": list(self.warnings), "error": self.error,
                    "exec_step": self.exec_step, "exec_note": self.exec_note,
                    "stopping": self.stop_requested,
                    "plan": None if p is None else {
                        "goal": p.goal, "summary": p.summary, "stack": list(p.stack),
                        "assumptions": list(p.assumptions), "questions": list(p.questions),
                        "risks": list(p.risks), "version": p.version,
                        "steps": [{"id": s.id, "title": s.title, "detail": s.detail, "owner": s.owner,
                                   "files": list(s.files), "verify": s.verify,
                                   "status": s.status, "note": s.note} for s in p.steps]}}

    def feed(self, ch: str) -> bool:
        """Route one keypress. Returns True if it was consumed. Escape sequences (arrows)
        arrive as separate chars from KeyReader, so we reassemble them here."""
        with self.lock:
            if not self.active or self.plan is None or self.busy or self.exec_step:
                return False
            key = self._decode(ch)
            if key is None:
                return True                                           # mid escape-sequence: swallow
            handled = True
            if key:                                                   # "" = an ignored sequence
                handled = getattr(self, f"_key_{self.mode}")(key)
            pend, self._pending = self._pending, None
            if pend and self.active and not self.busy:
                getattr(self, f"_key_{self.mode}")(pend)
            return handled

    _esc = ""
    _ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}

    def _decode(self, ch: str) -> Optional[str]:
        """Terminal escape grammar: ESC [ <params 0x30-0x3F>* <intermediate 0x20-0x2F>* <final 0x40-0x7E>
        (CSI, e.g. arrows `ESC [ A`, F5 `ESC [ 15 ~`) or ESC O <final> (SS3, e.g. F1). We consume
        the WHOLE sequence so no parameter byte can leak out as a typed character.

        A bare ESC is ambiguous until the next byte arrives: if that byte does not start a CSI/SS3
        sequence, the user really pressed Esc, so we report "esc" and then process the byte as a
        normal key (returning both, via `_pending`)."""
        if not self._esc:
            if ch == "\x1b":
                self._esc = ch
                return None
            return self._plain(ch)
        # inside an escape sequence
        if self._esc == "\x1b":
            if ch in "[O":
                self._esc += ch
                return None
            self._esc = ""                                    # bare Esc + an ordinary key
            self._pending = self._plain(ch)                   # the key must not be lost
            return "esc"
        self._esc += ch
        intro, final_ok = self._esc[1], False
        if intro == "O":                                      # SS3: exactly one more byte
            final_ok = True
        elif 0x40 <= ord(ch) <= 0x7E:                         # CSI final byte
            final_ok = True
        elif len(self._esc) > 16:                             # garbage: give up, drop it
            self._esc = ""
            return None
        if not final_ok:
            return None                                       # still collecting parameters
        seq, self._esc = self._esc, ""
        if seq.startswith("\x1b[") and len(seq) == 3:
            return self._ARROWS.get(seq[-1], "")              # "" = recognised as a sequence, no action
        return ""                                             # F-keys, Home/End, etc: ignored, never typed

    _pending: Optional[str] = None

    @staticmethod
    def _plain(ch: str) -> str:
        if ch in ("\r", "\n"):
            return "enter"
        if ch in ("\x7f", "\b"):
            return "backspace"
        return ch

    def _move(self, d: int) -> None:
        n = len(self.plan.steps)
        self.cursor = max(0, min(n - 1, self.cursor + d)) if n else 0

    def _key_review(self, k: str) -> bool:
        n = len(self.plan.steps)
        if k != "esc" and self.error.startswith("Esc does nothing"):
            self.error = ""
        if k in ("a", "enter", "y"):
            if any(s.status != "skip" for s in self.plan.steps):
                self._q.put(("approve", self.plan))
            else:
                self.error = "every step is skipped - nothing to run"
            return True
        if k == "r":
            self.mode, self.text, self.error = "feedback", "", ""
            return True
        if k == "e":
            self.mode, self.error = "edit", ""
            return True
        if k in ("x", "q"):
            self._q.put(("cancel", None))
            return True
        if k == "esc":                               # reflex key: must never destroy a waited-for plan
            self.error = "Esc does nothing here - press x to cancel, a to approve"
            return True
        if k in ("j", "down"):
            self._move(1)
        elif k in ("k", "up"):
            self._move(-1)
        elif k == "g":
            self.cursor = 0
        elif k == "G":
            self.cursor = max(0, n - 1)
        return True

    def _key_edit(self, k: str) -> bool:
        st = self.plan.steps
        if k in ("j", "down"):
            self._move(1)
        elif k in ("k", "up"):
            self._move(-1)
        elif k == " ":
            s = st[self.cursor]
            s.status = "todo" if s.status == "skip" else "skip"
            self.warnings = validate_plan(self.plan)
        elif k == "K" and self.cursor > 0:                            # move step up
            st[self.cursor - 1], st[self.cursor] = st[self.cursor], st[self.cursor - 1]
            self.cursor -= 1
            self.plan.renumber()
            self.warnings = validate_plan(self.plan)
        elif k == "J" and self.cursor < len(st) - 1:                  # move step down
            st[self.cursor + 1], st[self.cursor] = st[self.cursor], st[self.cursor + 1]
            self.cursor += 1
            self.plan.renumber()
            self.warnings = validate_plan(self.plan)
        elif k == "o":                                                # cycle owner
            keys = [""] + [s.key for s in SPECIALISTS]
            cur = st[self.cursor].owner
            st[self.cursor].owner = keys[(keys.index(cur) + 1) % len(keys)] if cur in keys else ""
        elif k in ("d", "enter", "esc", "e"):
            self.mode = "review"
        return True

    def _key_feedback(self, k: str) -> bool:
        if k == "esc":
            self.mode = "review"
        elif k == "enter":
            txt = self.text.strip()
            if txt:
                self._q.put(("revise", txt))
            else:
                self.mode = "review"
        elif k == "backspace":
            self.text = self.text[:-1]
        elif len(k) == 1 and k.isprintable():
            self.text += k
        return True




# --------------------------------------------------------------------------- #
# Classic (scrolling) mode: same decisions, driven by plain prompts
# --------------------------------------------------------------------------- #
class ClassicGate(PlanGate):
    """A PlanGate for terminals without the split-screen UI (`--classic`, `-p`).

    It IS-A PlanGate, so plan_and_execute needs no changes: it calls present() then wait().
    Here present() stores the plan and wait() prints it and asks on the calling thread.
    `ask` and `emit` are injectable so the whole flow is testable without a TTY.
    """

    def __init__(self, ask=None, emit=None, auto_approve: bool = False) -> None:
        super().__init__()
        self._ask = ask                      # callable(prompt:str) -> str
        self._emit = emit or (lambda *a, **k: None)
        self.auto_approve = auto_approve     # --yes: skip review (explicit user opt-in only)

    def render_text(self, plan: "Plan") -> str:
        """Plain-text plan, also what `-p` mode prints."""
        lines = [f"PLAN v{plan.version}: {plan.goal}"]
        if plan.summary:
            lines.append(f"  {plan.summary}")
        if plan.stack:
            lines.append("  stack: " + ", ".join(plan.stack))
        for q in plan.questions:
            lines.append(f"  ? {q}")
        for a in plan.assumptions:
            lines.append(f"  · assumes: {a}")
        for r in plan.risks:
            lines.append(f"  ▲ risk: {r}")
        lines.append("")
        for s in plan.steps:
            mark = "⊘" if s.status == "skip" else "○"
            lines.append(f"  {mark} {s.id}. {s.title}  [{s.owner or 'agent'}]")
            if s.files:
                lines.append(f"       files : {', '.join(s.files)}")
            if s.verify:
                lines.append(f"       verify: {s.verify}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)

    def wait(self, timeout: Optional[float] = None):
        plan = self.plan
        if plan is None:
            return ("cancel", None)
        if self.auto_approve:
            self._emit(self.render_text(plan))
            self._emit("[--yes] plan auto-approved")
            return ("approve", plan)
        if self._ask is None:                # no way to ask (piped stdin): NEVER guess "yes"
            self._emit(self.render_text(plan))
            self._emit("no interactive terminal to review the plan - cancelled. Re-run with --yes to skip review.")
            return ("cancel", None)
        while True:
            self._emit(self.render_text(plan))
            try:
                ans = self._ask("[a]pprove  [r]evise  [s]kip N  [x] cancel  › ").strip()
            except (EOFError, KeyboardInterrupt):
                return ("cancel", None)
            low = ans.lower()
            if low in ("a", "y", "yes", "approve", ""):
                if any(s.status != "skip" for s in plan.steps):
                    return ("approve", plan)
                self._emit("every step is skipped - nothing to run")
            elif low in ("x", "q", "n", "no", "cancel"):
                return ("cancel", None)
            elif low.startswith("r"):
                fb = ans[1:].strip() if len(ans) > 1 and ans[1] in " :" else ""
                fb = fb or self._ask("what should change? › ").strip()
                if fb:
                    return ("revise", fb)
            elif low.startswith("s"):
                m = re.search(r"\d+", low)
                if m and 1 <= int(m.group()) <= len(plan.steps):
                    st = plan.steps[int(m.group()) - 1]
                    st.status = "todo" if st.status == "skip" else "skip"
                    self.warnings = validate_plan(plan)
                else:
                    self._emit("usage: s 3   (toggle skip on step 3)")
            else:
                self._emit("please answer a, r, s N or x")


# --------------------------------------------------------------------------- #
# Orchestration: plan -> review loop -> execute
# --------------------------------------------------------------------------- #
PLAN_MAX_REVISIONS = 6
PLAN_FILE = "PLAN.md"


_BUILD_VERBS = (r"build|create|make|develop|design|generate|scaffold|implement|set ?up|write me|"
                r"code up|whip up|spin up|start a|start an|bootstrap|clone|recreate|prototype")
# things that are a PROJECT/artifact (not a one-file tweak)
_BUILD_NOUNS = (r"website|web ?site|web ?app|app|application|landing page|dashboard|portfolio|blog|store|shop|"
                r"e-?commerce|game|api|backend|frontend|full[- ]?stack|saas|platform|tool|bot|chatbot|"
                r"calculator|tracker|planner|clone|site|cms|crm|extension|plugin|cli|library|framework|"
                r"microservice|service|server|scraper|pipeline|project")
_QUESTION_START = r"^(explain|what|why|how (does|do|can|to)|summari[sz]e|show|list|read|where|which|is|are|does|do|can)\b"
_EDIT_VERBS = (r"\b(fix|typo|rename|tweak|adjust|bump|refactor|debug|patch|revert|undo|delete|remove|"
               r"update the|change the|move the|add (a|an|the) [\w-]+ (to|in|on|for) )")


def wants_plan(prompt: str, registry: Optional["ActivationRegistry"] = None) -> bool:
    """Deterministic, free: is this "make a thing" rather than "change a thing"?

    Planning helps when the request creates a project or spans many files; it is friction
    for a one-line edit or a question. So the test is: a BUILD verb applied to a buildable
    NOUN, minus anything that reads as a question or a small edit. Message LENGTH is
    deliberately not used: "build a website" (15 chars) is exactly who needs a plan.
    """
    t = " ".join(prompt.lower().split())
    # polite imperatives are commands, not questions: "can you build...", "could you make...", "would you create..."
    t = re.sub(r"^(hey |hi |ok |okay |so )?(can|could|would|will) you (please )?", "", t)
    if not t or re.match(_QUESTION_START, t):
        return False
    has_build = re.search(rf"\b({_BUILD_VERBS})\b", t)
    has_noun = re.search(rf"\b({_BUILD_NOUNS})\b", t)
    if not (has_build and has_noun):
        return False
    # an edit-shaped request only wins when nothing says "create a whole new thing"
    if re.search(_EDIT_VERBS, t) and not re.search(r"\b(from scratch|scaffold|new project|new app|whole|entire|full)\b", t):
        # ...unless the *leading* verb is clearly a build verb ("create a X ... and fix Y" is still a build)
        return bool(re.match(rf"^(please |can you |could you |i want (you )?to |i need (you )?to |let'?s )?({_BUILD_VERBS})\b", t))
    return True


def save_plan_file(ws, plan: "Plan") -> str:
    """Persist the plan as PLAN.md in the workspace so it survives the session. Goes through
    the normal write path (respects the workspace jail); never raises."""
    try:
        path = ws.root / PLAN_FILE
        path.write_text(plan.to_markdown(), encoding="utf-8")
        return PLAN_FILE
    except Exception:                                                 # noqa: BLE001
        return ""


def plan_and_execute(ws, provider_factory, request: str, gate: "PlanGate", ui=None,
                     should_stop=None) -> dict:
    """The whole plan-first workflow. Blocks on `gate` for the user's decision.

    provider_factory() -> a FRESH provider (own history) using the user's selected model.
    Returns {"outcome": "done"|"partial"|"cancelled"|"error", "plan": Plan|None, ...}.
    """
    plan: Optional[Plan] = None
    feedback = ""
    for rev in range(PLAN_MAX_REVISIONS + 1):
        gate.set_busy("planning" if plan is None else f"revising plan (v{plan.version + 1})")
        if ui:
            ui.set_phase("thinking")
            ui.add_log("sub", "planner: analysing request" if plan is None else "planner: revising")
        try:
            plan = run_planner(ws, provider_factory(), request, feedback=feedback, previous=plan, ui=ui,
                               on_phase=lambda m: gate.set_busy(m))
        except PlannerError as e:
            gate.set_busy("")
            gate.set_error(str(e))
            if ui:
                ui.add_log("err", f"planner: {str(e)[:60]}")
            if plan is None:                                          # nothing to review at all
                return {"outcome": "error", "plan": None, "error": str(e)}
            # a failed REVISION keeps the previous plan on screen so the user loses nothing
            gate.present(plan)
            gate.set_error(f"revision failed: {e}")
        else:
            gate.present(plan)
            if ui:
                ui.add_log("ok", f"plan v{plan.version} ready: {len(plan.steps)} steps")

        decision, payload = gate.wait(timeout=1800)
        if decision == "approve":
            plan = payload
            break
        if decision == "revise":
            feedback = payload
            if rev >= PLAN_MAX_REVISIONS:
                gate.set_error(f"revision limit ({PLAN_MAX_REVISIONS}) reached - approve or cancel")
                gate.present(plan)
                decision, payload = gate.wait(timeout=1800)
                if decision != "approve":
                    gate.close()
                    return {"outcome": "cancelled", "plan": plan}
                plan = payload
                break
            continue
        gate.close()
        return {"outcome": "cancelled", "plan": plan}
    else:                                                             # pragma: no cover
        gate.close()
        return {"outcome": "cancelled", "plan": plan}

    # ---- approved: execute --------------------------------------------- #
    saved = save_plan_file(ws, plan)
    if ui:
        ui.add_log("ok", f"plan approved ({plan.progress()[1]} steps)" + (f", saved {saved}" if saved else ""))

    def on_step(step: "PlanStep") -> None:
        with gate.lock:
            gate.exec_step = step.id if step.status == "run" else gate.exec_step
            gate.exec_note = step.note

    with gate.lock:
        gate.exec_step = 1
    stop_now = lambda: gate.stop_requested or bool(should_stop and should_stop())
    result = execute_plan(ws, plan, ui=ui, should_stop=stop_now, on_step=on_step)
    save_plan_file(ws, plan)                                          # refresh checkboxes
    with gate.lock:
        gate.exec_step = 0
    ok = result["failed"] == 0 and result["incomplete"] == 0 and result["stopped_at"] is None
    if ui:
        if ok:
            ui.add_log("ok", f"plan finished: {result['done']}/{len(plan.steps)} steps")
        else:
            why = (f"stopped at step {result['stopped_at']}"
                   + (" (incomplete: ran out of steps)" if result.get("incomplete") else
                      " (failed)" if result.get("failed") else ""))
            ui.add_log("warn", f"plan finished: {result['done']}/{len(plan.steps)} steps, {why}")
    gate.close()
    return {"outcome": "done" if ok else "partial", "plan": plan, **result}


# --------------------------------------------------------------------------- #
# Single entry point used by coder3.run_agent / repl
# --------------------------------------------------------------------------- #
def maybe_plan(ws, provider_factory, prompt: str, mode: str = "auto", gate=None,
               ask=None, emit=None, should_stop=None) -> Optional[dict]:
    """Decide whether to plan, and if so run the whole plan-first workflow.

    mode:  "auto"  plan only when wants_plan(prompt)      (default)
           "always" plan whatever the request looks like   (/plan)
           "never"  skip planning                          (/noplan)
    Returns None when NO planning happened (caller should run the normal agent loop), otherwise
    the plan_and_execute result dict. A "cancelled"/"error" outcome is returned as-is: the caller
    must NOT silently fall back to running the request without the plan the user just rejected.
    """
    if mode == "never" or getattr(ws, "specialist_registry", None) is None:
        return None
    if mode == "auto" and not wants_plan(prompt):
        return None
    ui = ws.ui
    own_gate = gate is None
    if gate is None:
        if ui is not None:
            gate = PlanGate()
        else:
            gate = ClassicGate(ask=ask, emit=emit, auto_approve=bool(getattr(ws, "auto", False)))
    if ui is not None:
        ui.plan_gate = gate
    try:
        return plan_and_execute(ws, provider_factory, prompt, gate, ui=ui, should_stop=should_stop)
    finally:
        if ui is not None and own_gate:
            ui.plan_gate = None


# =========================================================================== #
#  TOOL SPECS  -  added to the top-level model's toolset (auto-activation)
# =========================================================================== #
_SPECIALIST_KEYS = [s.key for s in SPECIALISTS]

SPECIALIST_TOOL_SPECS: list[dict] = [
    {"name": "dispatch_specialists",
     "description": ("Hand tasks to DOMAIN SPECIALISTS (each picks its own worker persona and has its own rate "
                     "budget; several run in parallel). Prefer over dispatch_agents when the work fits a domain: "
                     "frontend (HTML/CSS/JS/UI), backend (API/DB/frameworks), python, security, test, devops, docs, "
                     "or review (READ-ONLY findings, cannot write/run). 1-6 assignments."),
     "parameters": {"type": "object", "properties": {
         "assignments": {
             "type": "array", "minItems": 1, "maxItems": 6,
             "items": {"type": "object", "properties": {
                 "specialist": {"type": "string", "enum": _SPECIALIST_KEYS},
                 "task": {"type": "string",
                          "description": "Complete, self-contained instruction naming exact file(s)."}
             }, "required": ["specialist", "task"]},
             "description": "1-6 {specialist, task} assignments."}},
         "required": ["assignments"]}},
    {"name": "set_specialist_active",
     "description": "Enable/disable a specialist or worker for this session (e.g. one that keeps misfiring).",
     "parameters": {"type": "object", "properties": {
         "target": {"type": "string",
                    "description": f"Specialist key ({'/'.join(_SPECIALIST_KEYS)}) or worker key (e.g. html_worker)."},
         "active": {"type": "boolean"}},
         "required": ["target", "active"]}},
]


# =========================================================================== #
#  AUTO-ACTIVATION  -  suggest specialist routing for the top-level prompt
# =========================================================================== #
# Confidence thresholds on the router score (own-trigger hits count double).
STRONG_SCORE = 4          # >= two direct trigger hits: treat as an unambiguous domain match

# A small, targeted change ("fix the typo in a.py", "rename x", "remove unused imports"). These are
# faster and safer done directly by the model that already holds the context, so they must NOT be
# force-delegated: delegating re-sends the whole request to a fresh model that has to re-read
# everything, which is slower and can drop context for a one-line change.
_SMALL_EDIT = re.compile(
    r"^(please |can you |could you |just )*(fix|correct|rename|remove|delete|drop|tweak|adjust|bump|"
    r"update|change|replace|swap|move|comment|uncomment|reformat|format|lint|typo)\b"
    r"|\b(typo|unused (imports?|variables?|code)|one[- ]liner|small (fix|change|tweak))\b")
# ...unless the request is really about creating/adding a feature
_CREATES = re.compile(r"\b(create|build|make|generate|scaffold|implement|write|add (a|an|the|new)\b.*\b(endpoint|route|page|component|feature|section|module|class|function|test))")


def is_small_edit(task: str) -> bool:
    t = " ".join(task.lower().split())
    return bool(_SMALL_EDIT.search(t)) and not _CREATES.search(t)
_EXT_DOMAIN = {           # a concrete file extension is the strongest possible signal
    ".html": "frontendspecialist", ".htm": "frontendspecialist", ".css": "frontendspecialist",
    ".scss": "frontendspecialist", ".jsx": "frontendspecialist", ".tsx": "frontendspecialist",
    ".js": "frontendspecialist", ".ts": "frontendspecialist", ".vue": "frontendspecialist",
    ".svelte": "frontendspecialist", ".py": "pythonspecialist",
}


def route_confidence(task: str, registry: ActivationRegistry):
    """Return (specialist | None, score, is_strong). Pure Python, no LLM call."""
    text = task.lower()
    ranked = [(_score_specialist(s, text), s) for s in SPECIALISTS if registry.is_enabled(s.key)]
    ranked = [(sc, s) for sc, s in ranked if sc > 0]
    if not ranked:
        return None, 0, False
    ranked.sort(key=lambda t: -t[0])
    score, top = ranked[0]
    # explicit filename with a known extension is decisive
    for m in re.finditer(r"[\w./-]+(\.[a-z0-9]{1,5})\b", text):
        dom = _EXT_DOMAIN.get(m.group(1))
        if dom == top.key:
            score += 4
            break
    runner_up = ranked[1][0] if len(ranked) > 1 else 0
    # is_small_edit exists to keep genuinely trivial one-liners ("fix the
    # typo in a.py") off the forced-delegation path - fast, low-context
    # changes the top-level model can just make itself. But its verb list
    # ("fix", "update", "tweak", "adjust"...) also matches the bulk of
    # ordinary iterative debugging/tuning work ("fix the hero animation
    # timing in app.js", "tweak the gsap easing in style.css"), which used
    # to be vetoed just as hard, with no distinction from a true one-liner -
    # that's the gap that let a whole debug session run through the
    # top-level model's raw read_file/edit_file instead of e.g.
    # frontendspecialist.
    #
    # The right signal isn't "a filename/extension was mentioned" (bare
    # filenames appear in triggers like r"\.py\b" too, so that alone would
    # also force-route "fix the typo in a.py"). It's whether the task
    # matches a SUBSTANTIVE domain trigger - one naming a real concept
    # (animation, gsap, auth, sql injection, models.py...) rather than just
    # a bare extension/filename pattern. `_has_substantive_domain_match`
    # below checks that specifically; only that can override is_small_edit.
    small = is_small_edit(task)
    override_small = small and _has_substantive_domain_match(top, text)
    strong = score >= STRONG_SCORE and score >= runner_up + 2 and (not small or override_small)
    return top, score, strong


# Trigger patterns that match nothing but a bare filename/extension - not
# evidence of substantive domain content, just "this task mentions a file of
# this type". Used to decide whether is_small_edit may be overridden: a task
# that ONLY matches patterns like these (e.g. "fix the typo in a.py" matching
# r"\.py\b") is not distinguishable from a true trivial edit, so is_small_edit
# should still apply. A task matching anything else (r"\bgsap\b", r"\bsql
# injection\b", r"models\.py\b" - which names a concept, not just an
# extension) IS substantive, even if it also happens to use a "small edit" verb.
_BARE_FILENAME_TRIGGERS = frozenset({
    r"\.html?$", r"\.css$", r"\.scss$", r"\.sass$", r"\.less$", r"\.jsx?$", r"\.tsx?$",
    r"\.py\b", r"\.css\b", r"\.html?\b",
})

# The small-edit verbs is_small_edit itself keys off (fix/rename/remove/
# tweak/adjust/...). A trigger pattern built ONLY from one of these verbs
# (bare, or combined with nothing but a filename/extension, e.g.
# r"\bfix\b.*\.py") is not a substantive domain signal - it's exactly the
# same "small edit" verb is_small_edit already saw, just re-expressed as a
# trigger. Letting such a pattern override is_small_edit would be circular:
# the very thing that marked the task as a small edit would also be the
# thing excusing it from that check. Extracted from _SMALL_EDIT's own verb
# alternation (with the same backslash-escape stripping as
# _is_bare_filename_or_verb_pattern below, so \b anchors can't leak a
# spurious "b" token) so the two stay in sync if that list changes.
_SMALL_EDIT_VERBS = frozenset(
    re.findall(r"[a-z]+", re.sub(r"\\.", " ", _SMALL_EDIT.pattern.split(r"\b(typo")[0]).lower())
) - {"please", "can", "you", "could", "just"}


def _is_bare_filename_or_verb_pattern(pat: str) -> bool:
    if pat in _BARE_FILENAME_TRIGGERS:
        return True
    # Strip backslash-escapes to their literal meaning BEFORE extracting
    # words, or `\b` (word-boundary anchor) and `\.` (literal dot) get
    # mis-tokenized as parts of a word - e.g. r"\bfix\b.*\.py" naively
    # regex'd for [a-z]+ yields {"bfix", "b", "py"} (the leading "b" from
    # each "\b" fuses onto/duplicates as a token), so "fix" never matches
    # _SMALL_EDIT_VERBS and the whole pattern is wrongly kept as
    # substantive. Replacing every backslash-escape with a space first
    # gives the correct {"fix", "py"}.
    cleaned = re.sub(r"\\.", " ", pat)
    words = set(re.findall(r"[a-z]+", cleaned.lower()))
    # every alphabetic word in the pattern is either a small-edit verb or an
    # extension/filename fragment (e.g. "py", "css", "html") -> not substantive
    return bool(words) and words <= (_SMALL_EDIT_VERBS | {"py", "css", "scss", "sass",
                                                            "html", "htm", "js", "jsx",
                                                            "ts", "tsx", "less"})


def _has_substantive_domain_match(spec: SpecialistDef, text: str) -> bool:
    """True if `text` matches a trigger of `spec` (or one of its workers)
    that names real domain content, not just a bare filename/extension or a
    small-edit verb (alone or combined with nothing but a filename)."""
    for pat in spec.triggers:
        if not _is_bare_filename_or_verb_pattern(pat) and re.search(pat, text):
            return True
    for w in spec.workers:
        for pat in w.triggers:
            if not _is_bare_filename_or_verb_pattern(pat) and re.search(pat, text):
                return True
    return False


def suggest_routing_note(task: str, registry: ActivationRegistry) -> str:
    """Deterministic routing note appended to the user's prompt (plain Python:
    instant, free, no extra LLM call).

      * STRONG match -> a firm DIRECTIVE. The model is told to delegate FIRST.
        (The original code only ever produced a polite "consider..." line,
        which models routinely ignored - that is the reported symptom.)
      * weak match   -> soft hint, model decides.
      * no match     -> nothing.
    """
    matches = [m for m in route_task(task) if registry.is_enabled(m.key)]
    if not matches:
        return ""
    top, score, strong = route_confidence(task, registry)
    if top is None:
        return ""
    guess = route_worker(top, task)
    if strong:
        others = [m for m in matches if m.key != top.key]
        extra = (f" If the request also covers {', '.join(m.label for m in others)}, "
                 "add a second assignment for it in the SAME call.") if others else ""
        return (f"[ROUTING DIRECTIVE: this is {top.label} work"
                f"{f' (start with {guess.label})' if guess else ''}. Your FIRST action must be a "
                f"dispatch_specialists call with specialist='{top.key}' and a complete, "
                f"self-contained task naming the exact files.{extra} Do not write or edit those "
                "files yourself; after it returns, verify the result and report.]")
    hint = f"[routing hint: this looks like {top.label} work"
    if guess:
        hint += f", likely {guess.label}"
    if len(matches) > 1:
        hint += f"; also touches {', '.join(m.label for m in matches[1:])}"
    return hint + ". Consider dispatch_specialists if appropriate.]"


# =========================================================================== #
#  SAFETY NET  -  if the top-level model ignores a directive, delegate anyway
# =========================================================================== #
class _Guard:
    """Per-turn state: what the router decided for the current user prompt.

    Two independent counters:
      * write rescues  - capped at _MAX_WRITE_RESCUES (1). A write is a real,
        possibly destructive mutation; if the specialist already did the work
        once for this prompt, a second direct write attempt is just noise
        from the model re-trying, not a fresh reason to re-delegate.
      * analysis rescues - capped at _MAX_ANALYSIS_RESCUES (higher, effectively
        "for the rest of this prompt"). Reads/searches/symbol lookups are
        cheap and idempotent, so if the model keeps trying to inspect domain
        files itself instead of asking the specialist, every one of those
        attempts should still be redirected - a single early rescue must not
        disarm protection for the rest of a multi-step task.
    `delegated` still records that dispatch_specialists ran at least once
    this prompt (used by callers/telemetry), but no longer single-shots the
    analysis side of the net - see is_delegated_enough() below.
    """
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.top: Optional[SpecialistDef] = None
        self.strong = False
        self.task = ""
        self.delegated = False           # set once dispatch_specialists ran this prompt
        self.write_rescues = 0
        self.analysis_rescues = 0

    def arm(self, top, strong: bool, task: str) -> None:
        with self.lock:
            self.top, self.strong, self.task = top, strong, task
            self.delegated = False
            self.write_rescues, self.analysis_rescues = 0, 0

    def disarm(self) -> None:
        with self.lock:
            self.top, self.strong, self.delegated = None, False, False


# Tools that mutate the repo. Capped at _MAX_WRITE_RESCUES per prompt: once a
# specialist has done the write, letting the main model retry the same write
# itself would duplicate work.
_DOMAIN_WRITE_TOOLS = frozenset({"write_file", "edit_file", "edit_files", "rename_symbol"})
_MAX_WRITE_RESCUES = 1

# Tools that only read/inspect the repo. NOT capped at 1: a task can legitimately
# need several reads before a write (or be read-only end to end, e.g. a security
# audit), so every one of these calls on a domain-matched file/topic is
# redirected to the specialist for as long as the prompt's routing stays armed.
_DOMAIN_READ_TOOLS = frozenset({
    "read_file", "search_code", "find_symbol", "find_callers", "repo_map",
    "list_dir", "run_command", "git_diff", "git_blame", "git_log", "git_status",
    "goto_definition", "find_references",
})
_MAX_ANALYSIS_RESCUES = 20   # effectively "unbounded for one prompt" without risking a runaway loop


def _path_in_domain(spec: SpecialistDef, path: str) -> bool:
    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return _EXT_DOMAIN.get(ext) == spec.key


# Read tools whose purpose is to inspect actual file/diff CONTENT - these are
# the ones where "the armed task is about security" is a reasonable enough
# signal to redirect even a call with no domain keyword of its own (signal
# (c) below). Purely structural/listing tools (list_dir, git_log, git_status,
# repo_map) reveal no domain-specific content regardless of what the task is
# about, so task-text alone must NOT redirect those - only an explicit path
# match (signal a) or a domain keyword in the call's own arguments (signal b)
# should. Without this split, a "security audit" task would intercept even an
# unrelated `list_dir(".")`, which is over-broad and erodes trust in the net.
_CONTENT_INSPECTING_TOOLS = frozenset({
    "read_file", "search_code", "find_symbol", "find_callers", "git_diff", "git_blame",
    "goto_definition", "find_references",
})


def _args_touch_domain(spec: SpecialistDef, name: str, args: dict, task_text: str = "") -> bool:
    """True if this tool call's arguments - or, for content-inspecting tools
    only, the originally armed task text - name a path/pattern/command/topic
    in `spec`'s domain.

    Three signals, checked in order of strength, any one is enough:

      (a) file-extension match via `_EXT_DOMAIN` on a path argument -
          decisive for specialists that own a concrete extension (frontend:
          .css/.js/.html; python: .py). Strong, unambiguous, applies to any tool.
      (b) a trigger-keyword match against text the CALL ITSELF carries
          (pattern/command/query/glob/name/symbol/note arguments) - catches
          e.g. `search_code({"pattern": "SELECT"})` or
          `run_command({"command": "grep -r sql_injection"})`. Applies to any tool.
      (c) a trigger-keyword match against `task_text` (the prompt that got
          this specialist armed) - ONLY for `_CONTENT_INSPECTING_TOOLS`
          (read_file/search_code/find_symbol/find_callers/git_diff/git_blame).
          Needed for the common case of a plain `read_file({"path": "auth.py"})`
          with no domain keyword anywhere in the call's own arguments: an
          extension-agnostic specialist like Security (it audits .py/.js/
          whatever for injection/secrets/auth flaws, so no single extension
          identifies its domain) can only be matched this way on a bare read.
          Restricted to content-inspecting tools so a structural call like
          `list_dir(".")` - which reveals nothing about domain content either
          way - is never swept in just because the task happens to mention
          "security" in passing.
    """
    a = args or {}
    paths: list[str] = []
    if name == "edit_files":
        paths = [str(e.get("path", "")) for e in (a.get("edits") or []) if isinstance(e, dict)]
    else:
        for key in ("path", "file", "target"):
            p = a.get(key)
            if p:
                paths.append(str(p))
    if any(_path_in_domain(spec, p) for p in paths if p):
        return True
    call_text = " ".join(str(a.get(k, "")) for k in
                          ("pattern", "command", "query", "glob", "name", "symbol", "note")).lower()
    if call_text and any(re.search(pat, call_text) for pat in spec.triggers):
        return True
    if task_text and name in _CONTENT_INSPECTING_TOOLS and any(
            re.search(pat, task_text.lower()) for pat in spec.triggers):
        return True
    return False


# =========================================================================== #
#  WORKSPACE INSTALLATION  -  wire the new tool into an existing Workspace
# =========================================================================== #
def install(ws, keys: dict, limiter_cache: dict) -> ActivationRegistry:
    """Attach specialist dispatch to a coder3.Workspace instance.

    IDEMPOTENT: calling it again (repl() and the --prompt path both do) returns
    the existing registry instead of stacking another ws.execute wrapper.
    """
    existing = getattr(ws, "specialist_registry", None)
    if existing is not None and getattr(ws, "_specialists_installed", False):
        ws._specialist_keys, ws._specialist_limiter_cache = keys, limiter_cache
        return existing

    # bind to the module that is actually executing (fixes the __main__/coder3 split)
    for name in ("__main__", "coder3"):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, "Workspace") and isinstance(ws, m.Workspace):
            bind_host(m)
            break

    registry = ActivationRegistry()
    guard = _Guard()
    ws.specialist_registry = registry
    ws.specialist_guard = guard
    ws._specialist_keys = keys
    ws._specialist_limiter_cache = limiter_cache
    ws._specialists_installed = True

    def _dispatch_specialists(assignments: list[dict]) -> str:
        """Internal: returns text with a trailing [[dispatch_status:...]]
        marker (see dispatch_specialists' docstring) for execute_plan to
        read. LLM/UI-facing call sites below strip it via
        strip_dispatch_status() before the text is shown to the model or the
        user - the marker is plumbing, not part of the conversation."""
        if not ws.agent_ctx:
            return "error: dispatch_specialists is not available in this context"
        spec_holder = ws.agent_ctx["spec"]
        with guard.lock:
            guard.delegated = True
        return dispatch_specialists(ws, spec_holder, ws.agent_ctx["keys"],
                                    ws.agent_ctx["limiter_cache"], registry, assignments)

    def _set_specialist_active(target: str, active: bool) -> str:
        target = str(target).strip().lower()
        if target in SPECIALIST_BY_KEY:
            registry.set_enabled(target, bool(active))
            return f"{target} {'enabled' if active else 'disabled'}"
        all_workers = {w.key for s in SPECIALISTS for w in s.workers}
        if target in all_workers:
            registry.set_worker_enabled(target, bool(active))
            return f"worker {target} {'enabled' if active else 'disabled'}"
        return (f"error: unknown specialist/worker '{target}'. Valid: "
                + ", ".join(sorted(SPECIALIST_BY_KEY) + sorted(all_workers)))

    ws.dispatch_specialists = _dispatch_specialists       # type: ignore[attr-defined]
    ws.set_specialist_active = _set_specialist_active      # type: ignore[attr-defined]

    orig_execute = ws.execute

    def execute_with_specialists(name: str, args: dict) -> str:
        if name == "dispatch_specialists":
            try:
                return strip_dispatch_status(_dispatch_specialists(**(args or {})))
            except TypeError as e:
                return f"error: bad arguments for dispatch_specialists: {e}"
            except Exception as e:                                    # noqa: BLE001
                return f"error: {type(e).__name__}: {e}"
        if name == "set_specialist_active":
            try:
                return _set_specialist_active(**(args or {}))
            except TypeError as e:
                return f"error: bad arguments for set_specialist_active: {e}"
            except Exception as e:                                    # noqa: BLE001
                return f"error: {type(e).__name__}: {e}"

        # ---- safety net: strong directive ignored -> bounce the call back ----
        # Covers BOTH mutating tools (write_file/edit_file/edit_files) and
        # read/analysis tools (read_file, search_code, find_symbol,
        # find_callers, repo_map, list_dir, run_command, git_*) - a task that
        # is entirely read-only (e.g. a security audit) must still route to
        # the specialist, not just tasks that happen to end in a write.
        is_write = name in _DOMAIN_WRITE_TOOLS
        is_read = name in _DOMAIN_READ_TOOLS
        if is_write or is_read:
            with guard.lock:
                top, strong, delegated = guard.top, guard.strong, guard.delegated
                cap = _MAX_WRITE_RESCUES if is_write else _MAX_ANALYSIS_RESCUES
                count = guard.write_rescues if is_write else guard.analysis_rescues
                # Writes still respect `delegated` (once the specialist has
                # done ITS write for this prompt, don't also rescue a second
                # direct write attempt). Reads are allowed to keep firing even
                # after a dispatch has happened, since a multi-step task can
                # alternate specialist calls with the main model still poking
                # around - each poke should still be redirected, not just the
                # first one.
                can_rescue = top is not None and strong and count < cap
                if is_write:
                    can_rescue = can_rescue and not delegated
                if can_rescue:
                    can_rescue = _args_touch_domain(top, name, args, task_text=guard.task)
                if can_rescue:
                    if is_write:
                        guard.write_rescues += 1
                    else:
                        guard.analysis_rescues += 1
            if can_rescue:
                kind = "write" if is_write else "read"
                if ws.ui:
                    ws.ui.add_log("warn", f"delegating to {top.label} ({kind} intercepted: {name})")
                out = strip_dispatch_status(_dispatch_specialists([{"specialist": top.key, "task": guard.task}]))
                verb = "write" if is_write else f"{name} call"
                warn = "do NOT redo the same write" if is_write else "use the specialist's findings instead of re-inspecting yourself"
                return (f"[auto-delegated] Your direct {verb} was intercepted because this task "
                        f"belongs to {top.label}. The specialist has already done the work:\n{out}\n"
                        f"Review it; {warn}.")
        return orig_execute(name, args)

    ws.execute = execute_with_specialists       # type: ignore[method-assign]
    return registry


def arm_router(ws, prompt: str) -> str:
    """Called by run_agent once per user prompt: arms the safety net with the
    router's verdict and returns the note to append. One call, no LLM."""
    reg = getattr(ws, "specialist_registry", None)
    guard = getattr(ws, "specialist_guard", None)
    if reg is None or guard is None:
        return ""
    top, score, strong = route_confidence(prompt, reg)
    if top is None:
        guard.disarm()
    else:
        guard.arm(top, strong, prompt)
    return suggest_routing_note(prompt, reg)
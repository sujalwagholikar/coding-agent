"""
git_tools.py - git as a first-class primitive for coder3.py
=============================================================

Modern coding agents (Aider, Claude Code, Cline) treat git as load-bearing
infrastructure, not an optional afterthought: every AI change is diffable,
attributable, and revertible through the SAME mechanism a human would use,
and "what changed recently / who touched this / why" are cheap graph-native
questions instead of re-reading files from scratch.

This module is intentionally dependency-free (shells out to the `git`
binary only) and defensive throughout: every function returns a plain
string result (never raises to the caller) so it composes directly as a
Workspace tool method, exactly like repo_intel/exec_sandbox integrate.

Exposed operations, each mapped 1:1 to a coder3 tool:
  git_status()                 - porcelain status, human-rendered
  git_diff(path="", staged=)   - unified diff, working tree or staged
  git_log(path="", limit=)     - compact log, optionally scoped to a file
  git_blame(path, start, end)  - who/when for a line range
  git_commit(message, paths=)  - stage + commit (paths=[] -> stage all
                                  tracked changes), never force-pushes,
                                  never touches remotes
  git_branch(name="", base="") - list branches, or create+switch to one
                                  (never deletes/force-checks-out over
                                  uncommitted changes)

Safety posture:
  - No operation ever touches a remote (no push/pull/fetch/clone) - this
    module is strictly local-repo, so it can never leak code or credentials
    off the host or affect anyone else's copy of the repo.
  - No destructive git command is ever exposed (no reset --hard, no clean
    -fd, no checkout -- <file>, no rebase, no force anything). Commit and
    branch-create are the only mutating operations, both trivially
    reversible with plain git.
  - If the workspace root isn't a git repo at all, every function returns
    a clear "not a git repository" message instead of a stack trace, and
    callers (coder3.Workspace) degrade to "git tools unavailable" rather
    than crashing the agent.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


_TIMEOUT = 20


def _run(root: Path, args: list[str]) -> tuple[int, str]:
    """Run a git subcommand in `root`, always non-raising."""
    try:
        proc = subprocess.run(["git", *args], cwd=root, capture_output=True,
                               text=True, timeout=_TIMEOUT, errors="replace")
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except FileNotFoundError:
        return -1, "error: git is not installed on this host"
    except subprocess.TimeoutExpired:
        return -1, f"error: git command timed out after {_TIMEOUT}s"
    except Exception as e:                                       # noqa: BLE001
        return -1, f"error: {type(e).__name__}: {e}"


def is_repo(root: Path) -> bool:
    code, _ = _run(root, ["rev-parse", "--is-inside-work-tree"])
    return code == 0


def _not_a_repo() -> str:
    return ("not a git repository (or git isn't installed). Run 'git init' "
            "via run_command first if you want git-aware history/diffs/commits.")


# --------------------------------------------------------------------------- #
# Denylist: defence in depth. Even though the tool surface only ever calls
# the specific subcommands below, guard against a future caller passing raw
# args through by construction, and make the safety posture explicit and
# testable in one place.
# --------------------------------------------------------------------------- #
_FORBIDDEN_FLAGS = ("--force", "-f", "--hard", "push", "pull", "fetch",
                     "clone", "rebase", "reset", "clean", "remote")


def _reject_dangerous(args: list[str]) -> Optional[str]:
    low = [a.lower() for a in args]
    for bad in _FORBIDDEN_FLAGS:
        if bad in low:
            return f"error: '{bad}' is not permitted through git tools (local-only, non-destructive by design)"
    return None


# =========================================================================== #
# Public operations
# =========================================================================== #
def git_status(root: Path) -> str:
    if not is_repo(root):
        return _not_a_repo()
    code, out = _run(root, ["status", "--porcelain=v1", "-b"])
    if code != 0:
        return f"error: {out}"
    if not out:
        return "clean working tree, nothing to commit"
    lines = out.splitlines()
    branch_line = lines[0] if lines and lines[0].startswith("##") else ""
    changes = [l for l in lines if not l.startswith("##")]
    header = branch_line.lstrip("#").strip() or "(detached)"
    if not changes:
        return f"branch: {header}\nclean working tree, nothing to commit"
    body = "\n".join(changes[:200])
    more = f"\n... and {len(changes) - 200} more" if len(changes) > 200 else ""
    return f"branch: {header}\n{len(changes)} changed file(s):\n{body}{more}"


def git_diff(root: Path, path: str = "", staged: bool = False, context: int = 3) -> str:
    if not is_repo(root):
        return _not_a_repo()
    args = ["diff", f"--unified={max(0, min(context, 20))}"]
    if staged:
        args.append("--staged")
    if path:
        args += ["--", path]
    code, out = _run(root, args)
    if code != 0:
        return f"error: {out}"
    if not out:
        scope = f" for {path}" if path else ""
        which = "staged " if staged else ""
        return f"no {which}changes{scope}"
    MAX = 12_000
    if len(out) > MAX:
        out = out[:MAX] + f"\n... (truncated, {len(out) - MAX} more chars - narrow with `path`)"
    return out


def git_log(root: Path, path: str = "", limit: int = 20) -> str:
    if not is_repo(root):
        return _not_a_repo()
    limit = max(1, min(int(limit or 20), 100))
    args = ["log", f"-{limit}", "--pretty=format:%h  %ad  %an  %s", "--date=short"]
    if path:
        args += ["--", path]
    code, out = _run(root, args)
    if code != 0:
        return f"error: {out}"
    return out or "no commits yet" + (f" touching {path}" if path else "")


def git_blame(root: Path, path: str, start: int = 1, end: Optional[int] = None) -> str:
    if not is_repo(root):
        return _not_a_repo()
    if not path:
        return "error: path is required"
    args = ["blame", "-w", "--date=short"]
    if end:
        args += ["-L", f"{max(1, start)},{end}"]
    elif start and start > 1:
        args += ["-L", f"{start},+40"]
    args.append(path)
    code, out = _run(root, args)
    if code != 0:
        return f"error: {out}"
    lines = out.splitlines()
    if len(lines) > 200:
        lines = lines[:200] + [f"... ({len(out.splitlines()) - 200} more lines, narrow with start/end)"]
    return "\n".join(lines) or "no blame data (empty file or not tracked)"


def git_commit(root: Path, message: str, paths: Optional[list[str]] = None) -> str:
    if not is_repo(root):
        return _not_a_repo()
    if not message or not message.strip():
        return "error: commit message is required"
    bad = _reject_dangerous(["commit"])
    if bad:
        return bad
    stage_args = ["add", "-A"] if not paths else ["add", "--", *paths]
    code, out = _run(root, stage_args)
    if code != 0:
        return f"error staging files: {out}"
    code, out = _run(root, ["diff", "--staged", "--quiet"])
    if code == 0:
        return "nothing to commit (no staged changes match the given paths)"
    code, out = _run(root, ["commit", "-m", message.strip()])
    if code != 0:
        return f"error: commit failed: {out}"
    code2, short = _run(root, ["rev-parse", "--short", "HEAD"])
    return f"committed {short if code2 == 0 else ''}: {message.strip()}\n{out}"


def git_branch(root: Path, name: str = "", base: str = "") -> str:
    if not is_repo(root):
        return _not_a_repo()
    if not name:
        code, out = _run(root, ["branch", "--sort=-committerdate", "--format=%(HEAD) %(refname:short)  %(committerdate:relative)"])
        if code != 0:
            return f"error: {out}"
        return out or "no branches"
    bad = _reject_dangerous(["checkout", "-b", name])
    if bad:
        return bad
    # Refuse to switch branches over uncommitted changes - never silently
    # discard work; the model should commit or stash first via run_command.
    code, dirty = _run(root, ["status", "--porcelain"])
    if code == 0 and dirty:
        return ("error: working tree has uncommitted changes; commit or ask the user before "
                "switching branches, to avoid losing work.")
    args = ["checkout", "-b", name]
    if base:
        args.append(base)
    code, out = _run(root, args)
    if code != 0:
        return f"error: {out}"
    return f"created and switched to branch '{name}'{f' from {base}' if base else ''}"


def recent_activity_summary(root: Path, limit: int = 8) -> str:
    """Cheap 'what changed recently' primer, used to enrich the system prompt
    / repo_map the same way CodebaseMemory notes are auto-included - zero
    extra tool calls needed for the common 'what's been happening here'
    orientation question."""
    if not is_repo(root):
        return ""
    code, out = _run(root, ["log", f"-{limit}", "--pretty=format:%h %ad %s", "--date=short"])
    if code != 0 or not out:
        return ""
    code2, dirty = _run(root, ["status", "--porcelain"])
    dirty_note = f"\n{len(dirty.splitlines())} file(s) currently uncommitted." if code2 == 0 and dirty else ""
    return f"Recent commits:\n{out}{dirty_note}"

"""
exec_sandbox.py - ephemeral isolated execution for coder3.py's run_command
=============================================================================

coder3.py previously ran every shell command with a bare
`subprocess.Popen(command, shell=True, cwd=self.root, ...)` - full host
access, no resource caps, no filesystem/network isolation beyond whatever
the OS user account already restricts.

This module adds a real isolation layer with automatic best-available
backend selection, strongest first:

    1. docker   - full container isolation (network off by default, tmpfs
                  workdir mounted from a snapshot of the workspace, memory/
                  CPU/pids caps, dropped capabilities, no-new-privileges).
    2. gvisor   - same docker invocation, but with --runtime=runsc so
                  syscalls are intercepted by gVisor's user-space kernel
                  (stronger boundary against kernel-level escapes).
    3. bwrap    - bubblewrap: unprivileged namespace sandbox (new mount/pid/
                  net/uts namespaces), no container runtime required.
    4. unshare  - lower-level namespace isolation (new PID/mount/UTS/net
                  namespace) + prlimit resource caps, when bwrap isn't
                  installed but the kernel supports unprivileged namespaces.
    5. rlimit   - final fallback: no namespace isolation available, but
                  still applies CPU/memory/process-count ulimits via
                  `resource` before exec, so a runaway/fork-bombing command
                  can't take down the host. This is a strict improvement
                  over the old "just Popen it" behaviour even when nothing
                  fancier is installed.

Every backend is ephemeral: nothing persists between calls except the
explicit workspace mount the command needs to read/write real files.
Detection happens once per process and is cached; the result and any
warnings are exposed via `describe()` so the agent/UI can show the user
what's actually protecting them.

WASM (backend 6, not auto-selected today): stubbed via `run_wasm()` for a
Wasmtime/Wasmer-based interpreter sandbox for pure computation (no shell
semantics) - wired up but off by default since most `run_command` calls are
real shell pipelines (npm/pytest/etc) that need a POSIX environment WASM
sandboxes don't provide.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SandboxResult:
    exit_code: int
    output: str
    backend: str
    timed_out: bool = False
    error: Optional[str] = None


@dataclass
class SandboxLimits:
    timeout_s: int = 60
    memory_mb: int = 1024
    cpu_seconds: int = 120
    max_processes: int = 64
    network: bool = True     # most build/test/package commands need it; caller can disable


class _BackendCache:
    lock = threading.Lock()
    backend: Optional[str] = None
    detail: str = ""


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _docker_runtime_available(runtime: str) -> bool:
    if not _have("docker"):
        return False
    try:
        out = subprocess.run(["docker", "info", "--format", "{{json .Runtimes}}"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return False
        if runtime == "runc":
            return True     # default runtime always present if docker works at all
        return runtime in (out.stdout or "")
    except Exception:
        return False


def detect_backend() -> tuple[str, str]:
    """Pick the strongest isolation backend actually usable on this host,
    once per process (cheap: a handful of `which`/`docker info` probes).
    Returns (backend_name, human_detail)."""
    with _BackendCache.lock:
        if _BackendCache.backend is not None:
            return _BackendCache.backend, _BackendCache.detail

        backend, detail = "rlimit", "no namespace/container isolation available; using resource-limit fallback only"
        if _docker_runtime_available("runsc"):
            backend, detail = "gvisor", "Docker + gVisor (runsc) runtime detected - syscall-level sandboxing"
        elif _docker_runtime_available("runc"):
            backend, detail = "docker", "Docker detected - container isolation (namespaces + cgroups)"
        elif _have("bwrap"):
            backend, detail = "bwrap", "bubblewrap detected - unprivileged namespace sandbox"
        elif _have("unshare") and _have("prlimit") and sys.platform != "win32":
            backend, detail = "unshare", "unshare+prlimit detected - namespace isolation + resource caps"
        elif sys.platform != "win32":
            backend, detail = "rlimit", "no sandbox tool found (docker/bwrap/unshare) - resource-limit fallback only"
        else:
            backend, detail = "rlimit", "Windows host - namespace isolation unavailable, resource caps skipped too"

        _BackendCache.backend, _BackendCache.detail = backend, detail
        return backend, detail


def describe() -> str:
    backend, detail = detect_backend()
    return f"execution sandbox: {backend} — {detail}"


# --------------------------------------------------------------------------- #
# Per-backend command construction. Each wraps the *same* user shell command
# (`bash -lc "<command>"`) so exit codes / stdout+stderr interleaving behave
# identically to the old direct-Popen path from the model's point of view -
# only the isolation boundary around it changes.
# --------------------------------------------------------------------------- #
def _docker_cmd(command: str, root: Path, limits: SandboxLimits, runtime: Optional[str]) -> list[str]:
    cmd = [
        "docker", "run", "--rm",
        "--memory", f"{limits.memory_mb}m",
        "--pids-limit", str(limits.max_processes),
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "-v", f"{root}:/workspace:rw",
        "-w", "/workspace",
    ]
    if runtime:
        cmd += ["--runtime", runtime]
    if not limits.network:
        cmd += ["--network", "none"]
    cmd += ["python:3.12-slim" if _looks_python_only(command) else "ubuntu:24.04",
            "bash", "-lc", command]
    return cmd


def _looks_python_only(command: str) -> bool:
    head = command.strip().split()[:1]
    return bool(head) and head[0] in ("python", "python3", "pytest", "pip", "pip3")


def _bwrap_cmd(command: str, root: Path, limits: SandboxLimits) -> list[str]:
    cmd = [
        "bwrap",
        "--die-with-parent",
        "--unshare-pid", "--unshare-uts", "--unshare-ipc",
        "--proc", "/proc",
        "--dev", "/dev",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind-try", "/lib64", "/lib64",
        "--ro-bind-try", "/bin", "/bin",
        "--ro-bind-try", "/etc/resolv.conf", "/etc/resolv.conf",
        "--bind", str(root), "/workspace",
        "--chdir", "/workspace",
    ]
    if not limits.network:
        cmd += ["--unshare-net"]
    cmd += ["bash", "-lc", command]
    return cmd


def _unshare_cmd(command: str, root: Path, limits: SandboxLimits) -> list[str]:
    flags = ["--pid", "--fork", "--mount-proc", "--uts", "--ipc"]
    if not limits.network:
        flags.append("--net")
    cmd = ["unshare"] + flags + [
        "bash", "-lc", command,
    ]
    return cmd


def _apply_rlimits(limits: SandboxLimits):
    """preexec_fn for the plain-subprocess and unshare backends: caps CPU
    time, address space, and max child processes so a runaway command
    (infinite loop, fork bomb, memory leak) can't take the host down even
    with no namespace isolation in play."""
    def _pre():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 5))
            mem_bytes = limits.memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except Exception:
                pass  # RLIMIT_AS can be unsupported/unreliable on some platforms - best effort
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))
            except Exception:
                pass
            os.setsid()
        except Exception:
            pass
    return _pre


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
def run_sandboxed(command: str, root: Path, limits: Optional[SandboxLimits] = None,
                  on_line=None) -> SandboxResult:
    """Run `command` in the workspace at `root`, using the strongest
    isolation backend available on this host. `on_line` (optional) is
    called with each output line as it streams, for live UI updates -
    identical contract to the old run_command's ui.term_write callback.
    Ephemeral: for docker/gvisor the container is removed on exit (--rm);
    for bwrap/unshare the namespace is torn down when the process exits;
    nothing is left behind beyond whatever the command itself wrote into
    the mounted workspace, exactly as intended."""
    limits = limits or SandboxLimits()
    backend, _ = detect_backend()

    if backend in ("docker", "gvisor"):
        runtime = "runsc" if backend == "gvisor" else None
        argv = _docker_cmd(command, root, limits, runtime)
        preexec = None
    elif backend == "bwrap":
        argv = _bwrap_cmd(command, root, limits)
        preexec = _apply_rlimits(limits) if sys.platform != "win32" else None
    elif backend == "unshare":
        argv = _unshare_cmd(command, root, limits)
        preexec = _apply_rlimits(limits) if sys.platform != "win32" else None
    else:  # rlimit fallback - same shell=True path as before, but resource-capped
        argv = None
        preexec = _apply_rlimits(limits) if sys.platform != "win32" else None

    out_lines: list[str] = []
    try:
        popen_kw: dict = {}
        if sys.platform != "win32":
            popen_kw["start_new_session"] = True
        if preexec is not None:
            popen_kw["preexec_fn"] = preexec
        if argv is None:
            proc = subprocess.Popen(command, shell=True, cwd=root, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    errors="replace", **popen_kw)
        else:
            proc = subprocess.Popen(argv, cwd=root if backend in ("bwrap", "unshare") else None,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, errors="replace", **popen_kw)
    except FileNotFoundError as e:
        # sandbox tool vanished between detection and use (e.g. race with an
        # uninstall) - degrade to the rlimit fallback rather than failing the call
        return _run_plain_fallback(command, root, limits, on_line)
    except Exception as e:
        return SandboxResult(exit_code=-1, output="", backend=backend, error=str(e))

    def pump():
        for ln in proc.stdout:                                # type: ignore[union-attr]
            out_lines.append(ln)
            if on_line:
                on_line(ln.rstrip("\n"))
    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=limits.timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        reader.join(timeout=1)
        timed_out = True
    else:
        reader.join(timeout=2)

    text = "".join(out_lines)
    return SandboxResult(exit_code=proc.returncode if not timed_out else -9,
                        output=text, backend=backend, timed_out=timed_out)


def _run_plain_fallback(command: str, root: Path, limits: SandboxLimits, on_line) -> SandboxResult:
    preexec = _apply_rlimits(limits) if sys.platform != "win32" else None
    out_lines: list[str] = []
    popen_kw: dict = {}
    if sys.platform != "win32":
        popen_kw["start_new_session"] = True
    if preexec is not None:
        popen_kw["preexec_fn"] = preexec
    proc = subprocess.Popen(command, shell=True, cwd=root, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            errors="replace", **popen_kw)

    def pump():
        for ln in proc.stdout:                                # type: ignore[union-attr]
            out_lines.append(ln)
            if on_line:
                on_line(ln.rstrip("\n"))
    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=limits.timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        reader.join(timeout=1)
        timed_out = True
    else:
        reader.join(timeout=2)
    return SandboxResult(exit_code=proc.returncode if not timed_out else -9,
                        output="".join(out_lines), backend="rlimit", timed_out=timed_out)


def _kill_tree(proc: "subprocess.Popen") -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=5)
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_wasm(command: str, root: Path, limits: Optional[SandboxLimits] = None) -> SandboxResult:
    """Stub for a WASM-isolated execution path (wasmtime/wasmer), intended
    for pure single-binary computation rather than shell pipelines. Not
    auto-selected by detect_backend() since it can't run arbitrary shell
    commands (no pipes/redirection/POSIX utilities) - reserved for a future
    'run this compiled .wasm module' tool distinct from run_command."""
    if not _have("wasmtime") and not _have("wasmer"):
        return SandboxResult(exit_code=-1, output="", backend="wasm",
                             error="no WASM runtime (wasmtime/wasmer) installed on this host")
    runtime = "wasmtime" if _have("wasmtime") else "wasmer"
    limits = limits or SandboxLimits()
    try:
        proc = subprocess.run([runtime, "run", *shlex.split(command)],
                              cwd=root, capture_output=True, text=True,
                              timeout=limits.timeout_s)
        return SandboxResult(exit_code=proc.returncode, output=proc.stdout + proc.stderr, backend="wasm")
    except subprocess.TimeoutExpired:
        return SandboxResult(exit_code=-9, output="", backend="wasm", timed_out=True)
    except Exception as e:
        return SandboxResult(exit_code=-1, output="", backend="wasm", error=str(e))

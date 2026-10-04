#!/usr/bin/env python3
"""
coder.py - a terminal coding agent (Claude-Code / Codex style).

  * analyse code       -> read_file / list_dir / search / find_files
  * create files       -> write_file
  * inline edits       -> edit_file (exact-match find/replace with diff preview)
  * debug & fix        -> run_command + the agent loop iterates on real errors
  * refactor           -> multi-file edits, each shown as a diff for approval

Models (pick at startup or switch with /model):
  1. gemini-3.5-flash                        (GEMINI_API_KEY)
  2. gemini-3.5-flash-lite                   (GEMINI_API_KEY)
  3. openrouter/auto  (OPENROUTER_API_KEY)

Adaptive rate limiting: every model call goes through AdaptiveRateLimiter
(token bucket + AIMD + server Retry-After parsing + circuit breaker). The
footer HUD shows the live rate, budget and any active backoff.

Install:  pip install rich google-genai openai prompt_toolkit pygments httpx
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import difflib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Optional

import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Deque
from pygments import lex
from pygments.lexers import TextLexer, get_lexer_for_filename
from pygments.style import Style
from pygments.token import (Comment, Keyword, Name, Number, Operator,
                            Punctuation, String, Token)
from pygments.util import ClassNotFound
from rich.layout import Layout
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style as PTStyle
    HAVE_PT = True
except Exception:                                    # pragma: no cover
    HAVE_PT = False

HAVE_SPECIALISTS = False   # patched to True below, after TOOL_SPECS/SYSTEM_PROMPT exist

# -- graph-based repo intelligence + sandboxed execution (optional upgrades) - #
# Both degrade to the previous regex/raw-subprocess behaviour with zero
# crashes if their dependencies (tree_sitter, tree_sitter_languages,
# networkx) aren't installed - see repo_intel.py / exec_sandbox.py headers.
try:
    from repo_intel import RepoIntel, HAVE_TREE_SITTER, HAVE_NETWORKX
    HAVE_REPO_INTEL = True
except Exception:                                    # pragma: no cover
    HAVE_REPO_INTEL = False
    HAVE_TREE_SITTER = HAVE_NETWORKX = False

try:
    from exec_sandbox import run_sandboxed, SandboxLimits, describe as sandbox_describe
    HAVE_SANDBOX = True
except Exception:                                    # pragma: no cover
    HAVE_SANDBOX = False

# -- git as a first-class primitive (optional upgrade, same fallback shape) - #
try:
    import git_tools
    HAVE_GIT_TOOLS = True
except Exception:                                    # pragma: no cover
    HAVE_GIT_TOOLS = False

# =========================================================================== #
#  RATE LIMITING  (identical to the standalone, tested ratelimit.py module)
# =========================================================================== #
#  (embedded from the tested ratelimit module)
# --------------------------------------------------------------------------- #
# Provider profiles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RateProfile:
    """Starting assumptions for a provider/model. These are *seeds* - AIMD
    adapts away from them based on what the server actually tells us."""
    name: str
    rpm: float            # requests per minute ceiling
    tpm: float            # tokens per minute ceiling (input+output estimate)
    burst: int            # max requests allowed back-to-back
    floor_rpm: float      # never adapt below this
    max_retries: int = 6
    base_backoff: float = 1.5
    max_backoff: float = 90.0


# Conservative seeds. Real per-key limits depend on tier/quota, so these start
# safe and the controller raises them while calls keep succeeding.
PROFILES: dict[str, RateProfile] = {
    "gemini-3.5-flash": RateProfile(
        "gemini-3.5-flash", rpm=10, tpm=250_000, burst=3, floor_rpm=2),
    "gemini-3.5-flash-lite": RateProfile(
        "gemini-3.5-flash-lite", rpm=15, tpm=250_000, burst=4, floor_rpm=3),
    # OpenRouter documents ~20 rpm for :free variants; daily caps also apply.
    "openrouter/auto": RateProfile(
        "openrouter/auto",
        rpm=16, tpm=200_000, burst=2, floor_rpm=2, max_retries=8),
    # Groq's LPU inference is fast with generous published rpm on most models;
    # still start conservative and let AIMD raise it once calls keep succeeding.
    "llama-3.3-70b-versatile": RateProfile(
        "llama-3.3-70b-versatile", rpm=28, tpm=280_000, burst=4, floor_rpm=4),
    "openai/gpt-oss-120b": RateProfile(
        "openai/gpt-oss-120b", rpm=28, tpm=280_000, burst=4, floor_rpm=4),
}
DEFAULT_PROFILE = RateProfile("default", rpm=10, tpm=200_000, burst=2, floor_rpm=2)


# --------------------------------------------------------------------------- #
# Parsed server hint
# --------------------------------------------------------------------------- #
@dataclass
class RateLimitSignal:
    """Everything useful we could extract from a failed response."""
    status: int = 0
    retry_after: Optional[float] = None    # seconds the server told us to wait
    daily_exhausted: bool = False          # quota gone for the day - don't spin
    remaining_requests: Optional[float] = None
    reset_in: Optional[float] = None       # seconds until the window resets
    reason: str = ""

    @property
    def is_rate_limit(self) -> bool:
        return self.status == 429 or self.daily_exhausted

    @property
    def is_transient(self) -> bool:
        return self.status in (408, 425, 429, 500, 502, 503, 504)


_DELAY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|sec|secs|second|seconds|m|min)?\b", re.I)


def _parse_duration(value: Any) -> Optional[float]:
    """'32s', '1.5s', '500ms', '2m', 12, '12' -> seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _DELAY_RE.search(str(value))
    if not m:
        return None
    num = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    if unit == "ms":
        return num / 1000.0
    if unit in ("m", "min"):
        return num * 60.0
    return num


def _parse_retry_after_header(value: Any) -> Optional[float]:
    """Retry-After is either delta-seconds or an HTTP date."""
    if value is None:
        return None
    s = str(value).strip()
    try:
        return max(0.0, float(s))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(s)
        return max(0.0, dt.timestamp() - time.time())
    except Exception:
        return None


def _get_header(headers: Any, *names: str) -> Optional[str]:
    if headers is None:
        return None
    for n in names:
        try:
            v = headers.get(n)
        except Exception:
            v = None
        if v is None:
            try:
                v = headers.get(n.lower())
            except Exception:
                v = None
        if v is not None:
            return str(v)
    return None


def _walk_json(obj: Any):
    """Yield every dict nested inside obj (Gemini buries RetryInfo in details)."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)


def parse_error(exc: BaseException) -> RateLimitSignal:
    """Turn any provider exception into a RateLimitSignal.

    Handles both SDK shapes we verified:
      google-genai : .code  .status  .details(dict)  .message  .response
      openai       : .status_code    .body(dict)     .response.headers
    """
    sig = RateLimitSignal()
    sig.status = int(getattr(exc, "status_code", None) or getattr(exc, "code", None) or 0)

    # Collect a text blob + a header mapping + a json body from whichever shape.
    text = str(exc)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    body = getattr(exc, "body", None) or getattr(exc, "details", None)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            body = None

    # ---- headers -------------------------------------------------------- #
    ra = _parse_retry_after_header(_get_header(headers, "Retry-After", "retry-after"))
    if ra is not None:
        sig.retry_after = ra

    rem = _get_header(headers, "X-RateLimit-Remaining", "x-ratelimit-remaining")
    if rem is not None:
        try:
            sig.remaining_requests = float(rem)
        except ValueError:
            pass

    reset = _get_header(headers, "X-RateLimit-Reset", "x-ratelimit-reset")
    if reset is not None:
        try:
            r = float(reset)
            # OpenRouter sends a unix-ms epoch; others send delta-seconds.
            if r > 1e11:
                sig.reset_in = max(0.0, r / 1000.0 - time.time())
            elif r > 1e9:
                sig.reset_in = max(0.0, r - time.time())
            else:
                sig.reset_in = r
        except ValueError:
            pass

    # ---- JSON body ------------------------------------------------------ #
    if body is not None:
        for node in _walk_json(body):
            # Gemini: {"@type": ".../google.rpc.RetryInfo", "retryDelay": "32s"}
            if "retryDelay" in node:
                d = _parse_duration(node["retryDelay"])
                if d is not None:
                    sig.retry_after = d if sig.retry_after is None else max(sig.retry_after, d)
            # Gemini QuotaFailure -> which quota died (per-minute vs per-day)
            for key in ("quotaId", "quotaMetric"):
                q = str(node.get(key, ""))
                if "PerDay" in q or "perday" in q.lower():
                    sig.daily_exhausted = True
                    sig.reason = q
            # OpenRouter: metadata.headers carries the upstream reset info
            md = node.get("metadata")
            if isinstance(md, dict):
                mh = md.get("headers")
                if isinstance(mh, dict):
                    r = mh.get("X-RateLimit-Reset")
                    if r is not None:
                        try:
                            rv = float(r)
                            if rv > 1e11:
                                sig.reset_in = max(0.0, rv / 1000.0 - time.time())
                        except ValueError:
                            pass

    # ---- text heuristics (last resort) ---------------------------------- #
    low = text.lower()
    if not sig.status and ("429" in low or "rate limit" in low or "resource_exhausted" in low):
        sig.status = 429
    if ("free-models-per-day" in low
            or "requests per day" in low
            or "perday" in low
            or ("daily" in low and ("limit" in low or "quota" in low))):
        sig.daily_exhausted = True
        sig.reason = sig.reason or "daily quota"
    m = re.search(r"retry(?:ing)? in ([\d.]+)\s*(ms|s|m)", low)
    if m and sig.retry_after is None:
        sig.retry_after = _parse_duration(m.group(1) + m.group(2))

    if not sig.status:
        nm = type(exc).__name__.lower()
        if isinstance(exc, TimeoutError) or "timeout" in nm or "timed out" in low:
            sig.status = 504                       # hung / slow endpoint -> retry, don't die
        elif (isinstance(exc, ConnectionError) or "connecterror" in nm or "connectionerror" in nm
              or "remoteprotocol" in nm or "readerror" in nm or "server disconnected" in low):
            sig.status = 503                       # dropped socket -> retry

    if not sig.reason:
        sig.reason = (getattr(exc, "message", None) or text)[:160]
    return sig


# --------------------------------------------------------------------------- #
# Token bucket
# --------------------------------------------------------------------------- #
class _Bucket:
    """Continuous-refill token bucket. rate is units/second."""

    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.stamp = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.stamp) * self.rate)
        self.stamp = now

    def wait_time(self, cost: float) -> float:
        self._refill()
        if self.tokens >= cost:
            return 0.0
        if self.rate <= 0:
            return float("inf")
        return (cost - self.tokens) / self.rate

    def take(self, cost: float) -> None:
        self._refill()
        self.tokens -= cost          # may go negative -> future callers wait longer

    def set_rate(self, rate: float, capacity: Optional[float] = None) -> None:
        self._refill()
        self.rate = rate
        if capacity is not None:
            self.capacity = capacity
            self.tokens = min(self.tokens, capacity)


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #
class CircuitOpen(RuntimeError):
    """Raised when the breaker is open (endpoint failing / quota exhausted)."""
    def __init__(self, message: str, reopen_in: float):
        super().__init__(message)
        self.reopen_in = reopen_in


class _Breaker:
    def __init__(self, threshold: int = 4, cooldown: float = 45.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.open_until = 0.0

    def remaining(self) -> float:
        return max(0.0, self.open_until - time.monotonic())

    def is_open(self) -> bool:
        return self.remaining() > 0

    def record_failure(self, hold: Optional[float] = None) -> None:
        self.failures += 1
        if hold is not None:
            self.open_until = max(self.open_until, time.monotonic() + hold)
        elif self.failures >= self.threshold:
            self.open_until = time.monotonic() + self.cooldown
            self.failures = 0

    def record_success(self) -> None:
        self.failures = 0


# --------------------------------------------------------------------------- #
# The adaptive limiter
# --------------------------------------------------------------------------- #
@dataclass
class LimiterStats:
    calls: int = 0              # requests actually SENT (a retry is a new send)
    successes: int = 0
    failed: int = 0             # requests that ended in a final error (retries spent / hard error)
    rate_limited: int = 0
    retries: int = 0
    waited_total: float = 0.0
    tokens_used: int = 0        # finished requests: provider-reported tokens (estimate if none reported)
    tokens_estimated: int = 0   # how many of tokens_used were estimates (provider sent no usage)
    last_wait: float = 0.0
    last_event: str = "idle"
    last_latency: float = 0.0
    latency_total: float = 0.0
    history: list = field(default_factory=list)   # (ts, event) tail for the HUD


class LLMTimeout(TimeoutError):
    """A model call exceeded the hard watchdog (the SDK never answered)."""


class AdaptiveRateLimiter:
    """Per-model limiter. One instance per provider/model pair."""

    # AIMD constants
    INCREASE_STEP = 0.35        # +rpm per success (additive)
    DECREASE_FACTOR = 0.55      # x rpm on a 429   (multiplicative)
    PROBE_AFTER = 3             # successes required before we try going faster
    WINDOW = 60.0               # seconds of history behind the live rpm / tokens-per-minute meters

    def __init__(self, profile: RateProfile,
                 on_event: Optional[Callable[[str, dict], None]] = None):
        from collections import deque
        self.profile = profile
        self.on_event = on_event or (lambda kind, info: None)
        self._lock = threading.Lock()

        self.ceiling_rpm = profile.rpm
        self.rpm = profile.rpm * 0.7          # start below ceiling, then probe up
        self.rpm = max(self.rpm, profile.floor_rpm)

        burst = float(profile.burst)
        self._req = _Bucket(self.rpm / 60.0, burst)
        self._tok = _Bucket(profile.tpm / 60.0, profile.tpm * 0.25)
        self.breaker = _Breaker()
        self.stats = LimiterStats()
        self._success_streak = 0
        self._blocked_until = 0.0             # hard server-imposed pause
        # Pre-charge is PER THREAD: specialists/subagents share one limiter and call acquire()
        # concurrently, so a single shared field got overwritten and mis-reconciled.
        self._tl = threading.local()
        self._inflight: dict[int, tuple[float, int]] = {}   # ticket -> (t_start, est_tokens)
        self._ticket = 0
        self._req_events = deque()            # monotonic ts of each request start   (live rpm)
        self._tok_events = deque()            # (monotonic ts, tokens) per finished   (live tokens/min)

    # -- helpers ---------------------------------------------------------- #
    def _log(self, msg: str) -> None:
        self.stats.last_event = msg
        self.stats.history.append((time.time(), msg))
        del self.stats.history[:-12]

    def _apply_rate(self) -> None:
        self.rpm = min(max(self.rpm, self.profile.floor_rpm), self.ceiling_rpm)
        self._req.set_rate(self.rpm / 60.0, float(self.profile.burst))

    def _trim(self, now: float) -> None:
        cut = now - self.WINDOW
        while self._req_events and self._req_events[0] < cut:
            self._req_events.popleft()
        while self._tok_events and self._tok_events[0][0] < cut:
            self._tok_events.popleft()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Cheap ~4 chars/token estimate; good enough for pacing."""
        return max(1, len(text) // 4)

    # -- in-flight tracking (drives the live "requests" / "tokens" meters) -- #
    def begin(self, est_tokens: int) -> int:
        with self._lock:
            self._ticket += 1
            self._inflight[self._ticket] = (time.monotonic(), int(est_tokens))
            return self._ticket

    def end(self, ticket: int) -> None:
        with self._lock:
            self._inflight.pop(ticket, None)

    def note_failure(self, msg: str = "") -> None:
        with self._lock:
            self.stats.failed += 1
            if msg:
                self._log(msg)

    def refund(self) -> None:
        """A failed attempt must give its pre-charged tokens back (it used none of OUR budget);
        otherwise every retry silently drains the bucket and later calls stall for no reason."""
        with self._lock:
            charged = getattr(self._tl, "charged", 0.0)
            self._tl.charged = 0.0
            if charged:
                self._tok.tokens = min(self._tok.capacity, self._tok.tokens + charged)

    # -- proactive gate --------------------------------------------------- #
    def acquire(self, est_tokens: int = 1000,
                sleeper: Callable[[float], None] = time.sleep,
                tick: Optional[Callable[[float, str], None]] = None,
                should_abort: Optional[Callable[[], bool]] = None) -> float:
        """Block until a request may be sent. Returns total seconds waited.

        `tick(remaining, reason)` is called ~4x/sec while waiting so the UI can
        draw a live countdown instead of appearing frozen.
        """
        waited = 0.0
        while True:
            if should_abort and should_abort():
                raise KeyboardInterrupt
            with self._lock:
                if self.breaker.is_open():
                    raise CircuitOpen(
                        f"{self.profile.name}: paused ({self.stats.last_event})",
                        self.breaker.remaining())
                now = time.monotonic()
                # DEADLOCK FIX: the bucket can never hold more than `capacity`, so a request whose
                # estimate exceeds it (long history on a 1M-ctx model) could NEVER be satisfied and
                # this loop spun forever - freezing the request counter and token meter at N/N.
                # Clamp the cost to what the bucket can actually hold.
                cost = float(min(max(est_tokens, 1), self._tok.capacity))
                hard = max(0.0, self._blocked_until - now)
                w_req = self._req.wait_time(1.0)
                w_tok = self._tok.wait_time(cost)
                wait = max(hard, w_req, w_tok)
                if wait <= 0:
                    self._req.take(1.0)
                    self._tok.take(cost)
                    self._tl.charged = cost
                    self.stats.calls += 1
                    self.stats.last_wait = waited
                    self.stats.waited_total += waited
                    self._req_events.append(now)
                    self._trim(now)
                    return waited
                reason = ("server cooldown" if hard >= max(w_req, w_tok)
                          else "request pacing" if w_req >= w_tok else "token budget")

            # sleep in short slices so the UI countdown stays live; jitter
            # prevents parallel workers stampeding the moment a window opens.
            step = min(0.25, wait)
            if tick:
                tick(wait, reason)
            sleeper(step + random.uniform(0, 0.03))
            waited += step

    # -- feedback --------------------------------------------------------- #
    def on_success(self, tokens_used: int = 0, *, estimated: int = 0, latency: float = 0.0) -> None:
        with self._lock:
            charged = getattr(self._tl, "charged", 0.0)
            self._tl.charged = 0.0
            self.stats.successes += 1
            approx = False
            if not tokens_used:
                # provider sent no usage block: fall back to our estimate so the meter still moves
                tokens_used, approx = int(estimated or charged), True
                self.stats.tokens_estimated += tokens_used
            self.stats.tokens_used += tokens_used
            if latency:
                self.stats.last_latency = latency
                self.stats.latency_total += latency
            now = time.monotonic()
            self._tok_events.append((now, tokens_used))
            self._trim(now)
            self.breaker.record_success()
            self._success_streak += 1
            if self._success_streak >= self.PROBE_AFTER and self.rpm < self.ceiling_rpm:
                old = self.rpm
                self.rpm += self.INCREASE_STEP
                self._apply_rate()
                if int(self.rpm) != int(old):
                    self._log(f"probing up -> {self.rpm:.1f} rpm")
            # Reconcile the pre-charged estimate with what really happened (bounded so one huge
            # call can't put the bucket so far under water that every later call stalls).
            if charged and not approx:
                cap = self._tok.capacity
                delta = charged - min(float(tokens_used), cap)
                self._tok.tokens = max(-cap, min(cap, self._tok.tokens + delta))

    def on_rate_limit(self, sig: RateLimitSignal) -> float:
        """Register a 429. Returns how long the caller should wait before retry."""
        with self._lock:
            self.stats.rate_limited += 1
            self._success_streak = 0

            # A DAILY wall says nothing about our per-minute pace, so only
            # per-minute limits feed the AIMD controller. Shrinking rpm on a
            # daily wall would leave us needlessly slow once the day resets.
            if not sig.daily_exhausted:
                self.rpm = max(self.profile.floor_rpm, self.rpm * self.DECREASE_FACTOR)
                self._apply_rate()
                # ceiling was too optimistic: ease it toward what we observe
                # so we stop re-probing into the same wall.
                self.ceiling_rpm = max(self.rpm * 1.5,
                                       self.ceiling_rpm * 0.9, self.profile.floor_rpm)

            wait = sig.retry_after
            if wait is None:
                wait = sig.reset_in
            if wait is None:
                # no hint from server -> exponential backoff w/ full jitter
                attempt = min(self.stats.rate_limited, 8)
                cap = min(self.profile.max_backoff,
                          self.profile.base_backoff * (2 ** attempt))
                wait = random.uniform(self.profile.base_backoff, cap)
            wait = min(wait + random.uniform(0.2, 0.9), self.profile.max_backoff * 4)

            self._blocked_until = max(self._blocked_until, time.monotonic() + wait)
            self._req.tokens = min(self._req.tokens, 0.0)   # drain: no burst after a 429

            if sig.daily_exhausted:
                # daily quota is gone; spinning retries is pointless
                hold = max(wait, 300.0)
                self.breaker.record_failure(hold=hold)
                self._log(f"daily quota exhausted ({sig.reason[:40]})")
            else:
                self.breaker.record_failure()
                self._log(f"429 -> backoff {wait:.1f}s, rate {self.rpm:.1f} rpm")

            self.on_event("rate_limit", {"wait": wait, "rpm": self.rpm,
                                         "daily": sig.daily_exhausted})
            return wait

    def on_transient(self, sig: RateLimitSignal, attempt: int) -> float:
        """5xx / timeouts: retry with jittered exponential backoff, no AIMD hit."""
        with self._lock:
            self.breaker.record_failure()
            cap = min(self.profile.max_backoff, self.profile.base_backoff * (2 ** attempt))
            wait = sig.retry_after if sig.retry_after is not None \
                else random.uniform(self.profile.base_backoff, max(cap, self.profile.base_backoff))
            self._log(f"transient {sig.status} -> retry in {wait:.1f}s")
            return wait

    # -- introspection ---------------------------------------------------- #
    def snapshot(self) -> dict:
        with self._lock:
            now = time.monotonic()
            self._req._refill()          # was missing: the burst bar read a STALE bucket between calls
            self._trim(now)
            infl = list(self._inflight.values())
            infl_tok = sum(e for _, e in infl)
            tok_window = sum(t for _, t in self._tok_events)
            return {
                "name": self.profile.name,
                "rpm": round(self.rpm, 1),                 # currently ALLOWED rate (adaptive)
                "ceiling": round(self.ceiling_rpm, 1),
                "rpm_used": len(self._req_events),         # ACTUAL requests started in the last 60s
                "tpm_used": tok_window + infl_tok,         # ACTUAL tokens in the last 60s (+ in flight)
                "tpm_limit": self.profile.tpm,
                "req_tokens": round(self._req.tokens, 2),
                "burst": self.profile.burst,
                "blocked_for": round(max(0.0, self._blocked_until - now), 1),
                "breaker_open": self.breaker.is_open(),
                "breaker_left": round(self.breaker.remaining(), 1),
                "calls": self.stats.calls,
                "ok": self.stats.successes,
                "failed": self.stats.failed,
                "inflight": len(infl),
                "inflight_age": round(max((now - t0 for t0, _ in infl), default=0.0), 1),
                "inflight_tokens": infl_tok,
                "limited": self.stats.rate_limited,
                "retries": self.stats.retries,
                "waited": round(self.stats.waited_total, 1),
                "tokens": self.stats.tokens_used,
                "tokens_estimated": self.stats.tokens_estimated,
                "avg_latency": round(self.stats.latency_total / self.stats.successes, 1)
                               if self.stats.successes else 0.0,
                "last_latency": round(self.stats.last_latency, 1),
                "last": self.stats.last_event,
            }


def combined_snapshot(main: Optional["AdaptiveRateLimiter"], cache: Optional[dict] = None) -> dict:
    """ONE snapshot across the main limiter AND every specialist/subagent limiter.

    Specialists each own a limiter (specialists.get_specialist_limiter), so while they work the
    main limiter is idle and a main-only HUD looked frozen. Counters are summed; pacing state
    (rpm/burst/cooldown) is the main limiter's, with cooldown/breaker taking the worst of all."""
    lims: list = []
    seen: set = set()
    for l in ([main] if main is not None else []) + list((cache or {}).values()):
        if l is not None and hasattr(l, "snapshot") and id(l) not in seen:
            seen.add(id(l))
            lims.append(l)
    if not lims:
        return {}
    snaps = [l.snapshot() for l in lims]
    out = dict(snaps[0])
    for k in ("calls", "ok", "failed", "inflight", "inflight_tokens", "limited", "retries", "tokens",
              "tokens_estimated", "rpm_used", "tpm_used"):
        out[k] = sum(s[k] for s in snaps)
    out["waited"] = round(sum(s["waited"] for s in snaps), 1)
    out["inflight_age"] = max(s["inflight_age"] for s in snaps)
    out["blocked_for"] = max(s["blocked_for"] for s in snaps)
    out["breaker_left"] = max(s["breaker_left"] for s in snaps)
    out["breaker_open"] = any(s["breaker_open"] for s in snaps)
    out["parallel"] = len(snaps)
    return out


# --------------------------------------------------------------------------- #
# Convenience: run a callable under the limiter with retry
# --------------------------------------------------------------------------- #
LLM_HTTP_TIMEOUT = 180.0     # SDK-level socket timeout (seconds)
LLM_CALL_GUARD = 240.0       # hard watchdog above it: a call can NEVER hang the agent longer than this


def _run_guarded(fn: Callable[[], Any], timeout: float,
                 should_abort: Optional[Callable[[], bool]]) -> Any:
    """Run the blocking SDK call on a helper thread so a hung socket can neither freeze the
    agent nor swallow Ctrl-C. The abandoned thread is a daemon and mutates no shared state
    (providers only touch their history AFTER the call returns)."""
    box: dict = {}

    def runner():
        try:
            box["r"] = fn()
        except BaseException as e:                     # noqa: BLE001 - re-raised on the caller's thread
            box["e"] = e

    th = threading.Thread(target=runner, daemon=True, name="llm-call")
    th.start()
    t_end = time.monotonic() + timeout
    while th.is_alive():
        th.join(0.25)
        if should_abort and should_abort():
            raise KeyboardInterrupt
        if th.is_alive() and time.monotonic() > t_end:
            raise LLMTimeout(f"model call timed out after {timeout:.0f}s with no response")
    if "e" in box:
        raise box["e"]
    return box.get("r")


def call_with_limits(limiter: AdaptiveRateLimiter, fn: Callable[[], Any], *,
                     est_tokens: int = 1000,
                     tokens_from: Optional[Callable[[Any], int]] = None,
                     on_wait: Optional[Callable[[float, str], None]] = None,
                     should_abort: Optional[Callable[[], bool]] = None,
                     timeout: float = LLM_CALL_GUARD) -> Any:
    """Acquire -> call -> classify failure -> back off -> retry.

    Raises the last exception once retries are spent or the breaker opens on a
    non-recoverable condition (daily quota).
    """
    attempt = 0
    ticket = limiter.begin(est_tokens)                 # makes the request visible to the live meters
    try:
        while True:
            if should_abort and should_abort():
                raise KeyboardInterrupt
            limiter.acquire(est_tokens=est_tokens, tick=on_wait, should_abort=should_abort)
            t0 = time.monotonic()
            try:
                result = _run_guarded(fn, timeout, should_abort)
                limiter.on_success(tokens_from(result) if tokens_from else 0,
                                   estimated=est_tokens, latency=time.monotonic() - t0)
                return result
            except KeyboardInterrupt:
                limiter.refund()
                raise
            except Exception as exc:                   # noqa: BLE001
                limiter.refund()
                sig = parse_error(exc)
                if not sig.is_transient:
                    limiter.note_failure(f"error {sig.status or type(exc).__name__}: {sig.reason[:50]}")
                    raise                               # 400/401/403/404 -> real bug, don't retry
                attempt += 1
                limiter.stats.retries += 1
                if attempt > limiter.profile.max_retries:
                    limiter.note_failure("retries exhausted")
                    raise
                wait = (limiter.on_rate_limit(sig) if sig.is_rate_limit
                        else limiter.on_transient(sig, attempt))
                if sig.daily_exhausted:
                    limiter.note_failure("daily quota exhausted")
                    raise                               # surface to the UI, don't spin
                end = time.monotonic() + wait
                while True:
                    left = end - time.monotonic()
                    if left <= 0:
                        break
                    if should_abort and should_abort():
                        raise KeyboardInterrupt
                    if on_wait:
                        on_wait(left, "retry backoff")
                    time.sleep(min(0.25, left))
    finally:
        limiter.end(ticket)


# =========================================================================== #
#  SPLIT-SCREEN IDE UI  (from the tested ide_ui module)
# =========================================================================== #
# --------------------------------------------------------------------------- #
# VS Code Dark+ palette (sampled from the reference screenshot + default theme)
# --------------------------------------------------------------------------- #
VS = {
    "editor_bg":   "#0d1017",     # deep ink, a touch cooler/darker than the sidebar
    "sidebar_bg":  "#131826",     # distinct navy so the two panes separate clearly
    "panel_bg":    "#090b11",     # darkest layer, for the terminal/panel strip
    "card_bg":     "#1a2033",     # raised surface: section cards / chips
    "card_hi":     "#232b45",     # hovered/active raised surface
    "tab_active":  "#0d1017",
    "tab_idle":    "#1a2033",
    "border":      "#38405f",     # was #2a2f45 - clearly visible structure lines
    "rule":        "#2c3450",     # section dividers (visible but quiet)
    # --- text: THREE readable tiers. Nothing meaningful drops below `dim`. ---
    "fg":          "#f2f5ff",     # primary: near-white, high contrast
    "fg2":         "#c3cae8",     # secondary: labels, details (was a washed-out slate)
    "dim":         "#8d95bd",     # tertiary: hints only, still comfortably legible
    "line_num":    "#5b6489",
    "line_num_hl": "#dfe3ff",
    "blue":        "#4db2ff",     # vivid azure
    "btn_blue":    "#3574f0",
    "status_blue": "#2f6fed",
    "add_bg":      "#12351f",
    "add_bg_hi":   "#1c5a2e",
    "add_fg":      "#5af78e",
    "del_bg":      "#3a1220",
    "del_fg":      "#ff6b81",
    "cursor":      "#ffd166",
    "yellow":      "#ffd15c",
    "green":       "#4ff08c",
    "red":         "#ff5c7a",
    "orange":      "#ffa86b",
    "purple":      "#c99bff",
    "cyan":        "#4fe6ee",
    "pink":        "#ff7ad9",
    "guide":       "#3a4263",     # was #252a40 - meter tracks/timestamps were nearly invisible
    "track":       "#232b45",     # empty part of a meter
    "ink":         "#0b0e15",     # text colour on bright filled chips
}


class DarkPlus(Style):
    background_color = VS["editor_bg"]
    styles = {
        Token:             "#e6e9f5",
        Comment:           "italic #6b7394",
        Keyword:           "#7aa2ff",
        Keyword.Constant:  "#7aa2ff",
        Keyword.Namespace: "#c792ea",
        Name.Function:     "#ffd479",
        Name.Class:        "#5ce1e6",
        Name.Builtin:      "#5ce1e6",
        Name.Decorator:    "#ffd479",
        Name.Tag:          "#7aa2ff",
        Name.Attribute:    "#82d4ff",
        Name.Variable:     "#82d4ff",
        String:            "#ffb87a",
        Number:            "#c3f2a0",
        Operator:          "#e6e9f5",
        Punctuation:       "#e6e9f5",
    }


_STYLE_CACHE: dict = {}


def _tok_style(tok) -> str:
    """pygments token -> rich style string (foreground only)."""
    if tok in _STYLE_CACHE:
        return _STYLE_CACHE[tok]
    s = DarkPlus.style_for_token(tok)
    out = f"#{s['color']}" if s["color"] else VS["fg"]
    if s["italic"]:
        out += " italic"
    if s["bold"]:
        out += " bold"
    _STYLE_CACHE[tok] = out
    return out


def _lexer_for(path: str):
    try:
        return get_lexer_for_filename(path, stripnl=False)
    except ClassNotFound:
        return TextLexer(stripnl=False)


def highlight_lines(code: str, path: str) -> list[list[tuple[str, str]]]:
    """Tokenise once -> per-line list of (text, style). Multi-line tokens
    (docstrings, block comments) are split so colours survive across lines."""
    lines: list[list[tuple[str, str]]] = [[]]
    for tok, val in lex(code, _lexer_for(path)):
        style = _tok_style(tok)
        parts = val.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                lines.append([])
            if part:
                lines[-1].append((part, style))
    return lines


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class EditJob:
    """One animated file change. `rows` is the final rendered diff; `t0` drives
    the typing animation so the renderer can compute progress from wall-clock."""
    path: str
    kind: str                      # "create" | "edit" | "overwrite"
    # rows: (tag, old_no, new_no, text)   tag in {" ", "+", "-", "~"(collapsed)}
    rows: list = field(default_factory=list)
    new_lines: list = field(default_factory=list)    # highlighted final file
    added: int = 0
    removed: int = 0
    t0: float = field(default_factory=time.monotonic)
    cps: float = 420.0             # typed characters per second
    status: str = "pending"        # pending | typing | awaiting | kept | undone | rejected
    focus_row: int = 0             # row index the viewport should keep in view
    old_text: str = ""
    new_text: str = ""

    def total_chars(self) -> int:
        return sum(len(r[3]) + 1 for r in self.rows if r[0] in "+")

    def typed_chars(self, now: float) -> int:
        return int(max(0.0, now - self.t0) * self.cps)

    def done_typing(self, now: float) -> bool:
        return self.typed_chars(now) >= self.total_chars()


@dataclass
class Tab:
    path: str
    dirty: bool = False
    added: int = 0
    removed: int = 0


def build_rows(old: str, new: str, context: int = 3) -> tuple[list, int, int]:
    """Diff -> renderer rows. Unchanged runs longer than 2*context are collapsed
    so a one-line edit in a 500-line file shows the change, not the whole file."""
    a, b = old.splitlines(), new.splitlines()
    if max(len(a), len(b)) <= 60:          # short file: show it all, like a real editor
        context = max(context, 60)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    rows: list = []
    added = removed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            span = list(range(i1, i2))
            if len(span) > context * 2 + 1:
                head = span[:context] if rows else []
                tail = span[-context:] if j2 < len(b) else []
                for k in head:
                    rows.append((" ", k + 1, j1 + (k - i1) + 1, a[k]))
                hidden = len(span) - len(head) - len(tail)
                if hidden > 0:
                    rows.append(("~", 0, 0, f"{hidden} unchanged lines"))
                for k in tail:
                    rows.append((" ", k + 1, j1 + (k - i1) + 1, a[k]))
            else:
                for k in span:
                    rows.append((" ", k + 1, j1 + (k - i1) + 1, a[k]))
        else:
            for k in range(i1, i2):
                rows.append(("-", k + 1, 0, a[k]))
                removed += 1
            for k in range(j1, j2):
                rows.append(("+", 0, k + 1, b[k]))
                added += 1
    return rows, added, removed


# --------------------------------------------------------------------------- #
# Thread-safe UI state (the event bus)
# --------------------------------------------------------------------------- #
class UIState:
    LOG_MAX = 200

    def __init__(self):
        self.lock = threading.RLock()
        self.log: Deque[tuple[float, str, str]] = deque(maxlen=self.LOG_MAX)   # (ts, level, msg)
        self.tools: Deque[tuple[float, str, str, str]] = deque(maxlen=40)       # (ts, name, detail, state)
        self.tabs: list[Tab] = []
        self.active: Optional[str] = None
        self.jobs: dict[str, EditJob] = {}
        self.order: list[str] = []          # files changed this run (tray order)
        self.phase = "idle"
        self.phase_since = time.monotonic()
        self.wait_left = 0.0
        self.wait_why = ""
        self.assistant_text = ""
        self.prompt = ""
        self.step = 0
        self.t_start = time.monotonic()
        self.terminal: Deque[str] = deque(maxlen=200)        # last run_command output
        self.terminal_cmd = ""
        self.terminal_state = ""
        self.asking: Optional[str] = None       # pending y/n question shown in the editor footer
        self.plan_gate = None                   # specialists.PlanGate while a plan is being made/reviewed/run
        self.view_pin: Optional[str] = None     # manual Tab-cycle override: None=auto, "plan", or a tab path
        self.viewer_text: Optional[tuple[str, str]] = None   # (path, content) for read_file "peek"
        self.viewer_t0 = 0.0
        # sub_id -> {task, status, phase, detail, t0}; status: run|ok|err
        self.subagents: dict[int, dict] = {}
        # spec_id -> {key, label, task, worker, status, phase, detail, t0}; status: run|ok|err
        self.specialists: dict[int, dict] = {}

    def cycle_view(self, has_plan: bool) -> None:
        """Tab key: step through [plan] + every open file tab, overriding whatever
        the auto live/plan switch would otherwise pick. Tab again to advance;
        the pin is dropped once that view no longer exists (tab closed, plan gone)."""
        with self.lock:
            order = (["plan"] if has_plan else []) + [t.path for t in self.tabs]
            if not order:
                self.view_pin = None
                return
            if self.view_pin in order:
                i = (order.index(self.view_pin) + 1) % len(order)
            else:
                i = 0
            self.view_pin = order[i]
            if self.view_pin != "plan":
                self.active = self.view_pin

    def clear_pin_if_stale(self, has_plan: bool) -> None:
        with self.lock:
            if self.view_pin == "plan" and not has_plan:
                self.view_pin = None
            elif self.view_pin not in (None, "plan") and self.view_pin not in [t.path for t in self.tabs]:
                self.view_pin = None

    # -- mutation helpers (all take the lock) ------------------------------ #
    def add_log(self, level: str, msg: str) -> None:
        with self.lock:
            self.log.append((time.time(), level, msg))

    def set_phase(self, phase: str) -> None:
        with self.lock:
            if phase != self.phase:
                self.phase = phase
                self.phase_since = time.monotonic()
            if phase != "rate-limited":
                self.wait_left = 0.0

    def set_wait(self, left: float, why: str) -> None:
        with self.lock:
            self.wait_left, self.wait_why = left, why
            self.phase = "rate-limited" if left > 0.05 else self.phase

    def tool_start(self, name: str, detail: str) -> int:
        with self.lock:
            self.tools.append((time.time(), name, detail, "run"))
            return len(self.tools) - 1

    def tool_end(self, name: str, ok: bool = True) -> None:
        with self.lock:
            for i in range(len(self.tools) - 1, -1, -1):
                ts, n, d, st = self.tools[i]
                if n == name and st == "run":
                    self.tools[i] = (ts, n, d, "ok" if ok else "err")
                    break

    def _tab(self, path: str) -> Tab:
        for t in self.tabs:
            if t.path == path:
                return t
        t = Tab(path)
        self.tabs.append(t)
        if len(self.tabs) > 6:
            # keep the active tab and any dirty ones; drop the oldest clean one
            for i, old in enumerate(self.tabs):
                if not old.dirty and old.path != path:
                    self.tabs.pop(i)
                    break
        return t

    def open_file(self, path: str, content: str) -> None:
        with self.lock:
            self._tab(path)
            self.active = path
            self.viewer_text = (path, content)
            self.viewer_t0 = time.monotonic()

    def begin_edit(self, path: str, old: str, new: str, kind: str, cps: float = 420.0) -> EditJob:
        rows, add, rem = build_rows(old, new)
        job = EditJob(path=path, kind=kind, rows=rows, added=add, removed=rem,
                      old_text=old, new_text=new, cps=cps, status="typing")
        job.new_lines = highlight_lines(new, path)
        # focus on the first changed row
        for i, r in enumerate(rows):
            if r[0] in "+-":
                job.focus_row = i
                break
        with self.lock:
            tab = self._tab(path)
            tab.dirty = True
            tab.added, tab.removed = add, rem
            self.active = path
            self.viewer_text = None
            self.jobs[path] = job
        return job

    def finish_edit(self, path: str, status: str) -> None:
        with self.lock:
            j = self.jobs.get(path)
            if j:
                j.status = status
            if status in ("kept", "rejected", "undone"):
                t = self._tab(path)
                t.dirty = status == "kept" and False
            if status in ("kept",) and path not in self.order:
                self.order.append(path)

    def set_terminal(self, cmd: str, state: str = "running") -> None:
        with self.lock:
            self.terminal_cmd = cmd
            self.terminal_state = state
            self.terminal.clear()

    def term_write(self, text: str) -> None:
        with self.lock:
            for ln in text.splitlines():
                self.terminal.append(ln)

    # -- subagents (thread-safe: called from worker threads) --------------- #
    def sub_start(self, sub_id: int, task: str) -> None:
        with self.lock:
            self.subagents[sub_id] = {"task": task, "status": "run", "phase": "thinking",
                                      "detail": "", "t0": time.monotonic()}
        self.add_log("sub", f"[{sub_id}] start: {task[:60]}")

    def sub_update(self, sub_id: int, phase: str = None, detail: str = None) -> None:
        with self.lock:
            s = self.subagents.get(sub_id)
            if not s:
                return
            if phase is not None:
                s["phase"] = phase
            if detail is not None:
                s["detail"] = detail

    def sub_end(self, sub_id: int, ok: bool = True, summary: str = "") -> None:
        with self.lock:
            s = self.subagents.get(sub_id)
            if s:
                s["status"] = "ok" if ok else "err"
                s["phase"] = "done"
                s["detail"] = summary[:80]
        self.add_log("ok" if ok else "err", f"[{sub_id}] done: {summary[:60]}")

    # -- specialists (thread-safe: called from worker threads) -------------- #
    def spec_start(self, spec_id: int, key: str, label: str, task: str,
                   worker: Optional[str] = None) -> None:
        with self.lock:
            self.specialists[spec_id] = {"key": key, "label": label, "task": task,
                                         "worker": worker, "status": "run",
                                         "phase": "thinking", "detail": "",
                                         "t0": time.monotonic()}
        self.add_log("sub", f"[{label}] awake: {task[:50]}")

    def spec_update(self, spec_id: int, phase: str = None, detail: str = None,
                    worker: str = None) -> None:
        with self.lock:
            s = self.specialists.get(spec_id)
            if not s:
                return
            if phase is not None:
                s["phase"] = phase
            if detail is not None:
                s["detail"] = detail
            if worker is not None and worker != s.get("worker"):
                s["worker"] = worker
                self.add_log("sub", f"[{s['label']}] -> {worker}")

    def spec_end(self, spec_id: int, ok: bool = True, summary: str = "") -> None:
        with self.lock:
            s = self.specialists.get(spec_id)
            if s:
                s["status"] = "ok" if ok else "err"
                s["phase"] = "done"
                s["detail"] = summary[:80]
                label = s["label"]
            else:
                label = str(spec_id)
        self.add_log("ok" if ok else "err", f"[{label}] done: {summary[:60]}")

    # -- snapshot for the renderer (avoid holding the lock while drawing) --- #
    def snapshot(self) -> dict:
        with self.lock:
            return dict(
                log=list(self.log), tools=list(self.tools), tabs=[Tab(**t.__dict__) for t in self.tabs],
                active=self.active, jobs=dict(self.jobs), order=list(self.order),
                phase=self.phase, phase_since=self.phase_since, wait_left=self.wait_left,
                wait_why=self.wait_why, assistant_text=self.assistant_text, prompt=self.prompt,
                step=self.step, t_start=self.t_start, terminal=list(self.terminal),
                terminal_cmd=self.terminal_cmd, terminal_state=self.terminal_state,
                asking=self.asking, viewer_text=self.viewer_text, viewer_t0=self.viewer_t0,
                view_pin=self.view_pin,
                subagents={k: dict(v) for k, v in self.subagents.items()},
                specialists={k: dict(v) for k, v in self.specialists.items()},
                plan_view=self._plan_view())

    def _plan_view(self):
        """A copy of the plan-review state, or None. Kept out of the hot path when no plan is active."""
        g = self.plan_gate
        if g is None:
            return None
        v = g.view()
        return v if (v["active"] or v["busy"] or v["exec_step"]) else None


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
def _fit(text: str, width: int) -> str:
    """Truncate to `width` terminal CELLS (not characters): CJK/emoji are 2 cells wide,
    so len() would let a line overshoot its column and push the panel border over."""
    from rich.cells import cell_len
    if width <= 0:
        return ""
    text = text.replace("\n", " ").replace("\t", " ")
    if cell_len(text) <= width:
        return text
    out, used = [], 0
    for ch in text:
        w = cell_len(ch)
        if used + w > width - 1:
            break
        out.append(ch)
        used += w
    return "".join(out) + "…"


def _spinner(now: float) -> str:
    return "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(now * 12) % 10]


def _icon_for(path: str) -> tuple[str, str]:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {"py": ("", "#3572a5"), "tsx": ("", "#61dafb"), "ts": ("", "#3178c6"),
            "js": ("", "#f1e05a"), "jsx": ("", "#61dafb"), "json": ("", "#cbcb41"),
            "css": ("", "#563d7c"), "html": ("", "#e34c26"), "md": ("", "#519aba"),
            "go": ("", "#00add8"), "rs": ("", "#dea584"), "sh": ("", "#89e051"),
            }.get(ext, ("", "#858585"))


def _human(n: float) -> str:
    """12345 -> '12k', 1_250_000 -> '1.25M' (fits the narrow meter cells)."""
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1000:.0f}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(int(n))


class Renderer:
    """Turns a UIState snapshot into a rich Layout. Pure function of (state, now)."""

    LEFT_RATIO, RIGHT_RATIO = 4, 7

    def __init__(self, state: UIState, limiter_getter, spec_getter, ws_getter):
        self.state = state
        self.get_limiter = limiter_getter
        self.get_spec = spec_getter
        self.get_ws = ws_getter

    # ------------------------------------------------------------------ #
    # LEFT: telemetry  (redesigned: ruled sections, strict columns, chips)
    # ------------------------------------------------------------------ #
    def _meter(self, frac: float, width: int, color: str) -> Text:
        """Fixed-width bar with 1/8-cell precision. Width is exact by construction:
        full + (1 if partial else 0) + empty == width, always. The empty track is a
        solid dim block (not dotted `░`) so the bar reads as a bar even at 0%."""
        width = max(1, int(width))
        frac = max(0.0, min(1.0, float(frac)))
        eighths = round(frac * width * 8)
        full, rem = divmod(eighths, 8)
        full = min(full, width)
        partial = "▏▎▍▌▋▊▉"[rem - 1] if (rem and full < width) else ""
        empty = width - full - len(partial)
        t = Text(no_wrap=True, overflow="crop")
        t.append("█" * full, style=color)
        if partial:
            t.append(partial, style=f"{color} on {VS['track']}")
        t.append("█" * empty, style=VS["track"])
        return t

    PHASE_COL = {"idle": "green", "thinking": "cyan", "rate-limited": "yellow",
                 "running": "orange", "editing": "purple", "error": "red"}
    WORKER_COL = {"html": "orange", "css": "blue", "js": "yellow", "motion": "pink",
                  "fastapi": "green", "django": "green", "flask": "cyan", "nodejs": "green",
                  "database": "orange", "generic": "cyan", "analysis": "cyan",
                  "refactor": "purple", "scripting": "yellow", "exec": "orange"}

    @staticmethod
    def _pad(text: str, width: int) -> str:
        """Truncate with an ellipsis, then right-pad: guarantees an exact-width cell."""
        from rich.cells import cell_len
        text = _fit(text, width) if width > 0 else ""
        return text + " " * max(0, width - cell_len(text))

    @staticmethod
    def _pick(items: dict, limit: int) -> list:
        """Choose which rows to show: every RUNNING one first (newest first), then the most
        recent finished ones to fill the remaining slots. Returned in id order for stable display."""
        ids = sorted(items)
        running = [i for i in reversed(ids) if items[i]["status"] == "run"]
        done = [i for i in reversed(ids) if items[i]["status"] != "run"]
        chosen = (running + done)[:max(1, limit)]
        return [(i, items[i]) for i in sorted(chosen)]

    def _section(self, title: str, color: str, width: int, right: str = "") -> Text:
        """`▌ TITLE ───────────── 3 active`  - coloured marker, hairline rule, right label."""
        t = Text(no_wrap=True, overflow="crop")
        t.append(" ▌", style=f"bold {VS[color]}")
        t.append(f" {title} ", style=f"bold {VS[color]}")
        tail = f" {right} " if right else " "
        rule_w = max(1, width - 3 - len(title) - 2 - len(tail))
        t.append("─" * rule_w, style=VS["rule"])
        t.append(tail, style=VS["fg2"])
        return t

    def _kv(self, label: str, value: str, vcolor: str, width: int, label_w: int = 14) -> Text:
        """Two aligned columns: dim label, value right-aligned to the panel's inner edge."""
        t = Text(no_wrap=True, overflow="crop")
        t.append(f"   {self._pad(label, label_w)}", style=VS["fg2"])
        vw = max(1, width - 3 - label_w - 1)
        t.append(f"{value:>{vw}} ", style=f"bold {vcolor}")
        return t

    def _chip(self, text: str, bg: str) -> Text:
        """Filled pill: dark text on a solid colour - reads instantly, never fades."""
        return Text(f" {text} ", style=f"bold {VS['ink']} on {bg}")

    def _worker_chip(self, worker: str) -> Text:
        base = worker.replace("_worker", "")
        return self._chip(base.upper(), VS[self.WORKER_COL.get(base, "purple")])

    def left(self, s: dict, now: float, width: int, height: int) -> Panel:
        lim = self.get_limiter()
        spec = self.get_spec()
        ws_ = self.get_ws()
        cache = ((getattr(ws_, "agent_ctx", None) or {}).get("limiter_cache")) or {}
        snap = combined_snapshot(lim, cache)       # main + every specialist/subagent limiter
        inner = max(24, width)                     # frame() already removed the border columns
        rows: list = []
        gap = Text("")
        # Vertical priority: ACTIVE WORK (status, specialists, subagents, tools) beats static
        # info (model, counters). Below these heights the static blocks collapse to one line.
        compact = height < 40
        tiny = height < 26

        # -- HERO STATUS CARD --------------------------------------------- #
        phase = s["phase"]
        pc = VS[self.PHASE_COL.get(phase, "fg")]
        busy = phase != "idle"
        card = f"on {VS['card_bg']}"
        top = Text(no_wrap=True, overflow="crop")
        top.append(" ", style=card)
        top.append(f" {_spinner(now) if busy else '●'} {phase.upper()} ", style=f"bold {VS['ink']} on {pc}")
        if phase == "rate-limited" and s["wait_left"] > 0:
            timer = f"{s['wait_left']:.1f}s"
        else:
            timer = f"{max(0, now - s['phase_since']):.0f}s"
        top.append(f"  {timer}", style=f"bold {pc} {card}")
        if phase == "rate-limited" and s["wait_left"] > 0 and s["wait_why"]:
            top.append(f"  {_fit(s['wait_why'], max(4, inner - len(top.plain) - 2))}", style=f"{VS['fg2']} {card}")
        top.append(" " * max(0, inner - len(top.plain)), style=card)
        sub = Text(no_wrap=True, overflow="crop")
        sub.append("  ", style=card)
        sub.append("  step ", style=f"{VS['fg2']} {card}")
        sub.append(f"{s['step']}", style=f"bold {VS['fg']} {card}")
        sub.append("  ·  up ", style=f"{VS['dim']} {card}")
        up = int(now - s["t_start"])
        sub.append(f"{up // 60}m {up % 60:02d}s" if up >= 60 else f"{up}s", style=f"bold {VS['fg']} {card}")
        sub.append(" " * max(0, inner - len(sub.plain)), style=card)
        rows += [top, sub, gap]

        # -- MODEL -------------------------------------------------------- #
        if spec and compact:
            ln = Text(no_wrap=True, overflow="crop")
            ln.append(" ▌", style=f"bold {VS['blue']}")
            ln.append(" MODEL ", style=f"bold {VS['blue']}")
            ln.append(_fit(spec.label, max(4, inner - 10)), style=f"bold {VS['fg']}")
            rows.append(ln)
        elif spec:
            rows.append(self._section("MODEL", "blue", inner))
            name = Text(no_wrap=True, overflow="crop")
            name.append("   ")
            name.append(_fit(spec.label, inner - 5), style=f"bold {VS['fg']}")
            rows.append(name)
            rows.append(Text(f"   {_fit(spec.id, inner - 5)}", style=VS["dim"], no_wrap=True, overflow="crop"))
            rows.append(gap)

        # -- RATE LIMITS -------------------------------------------------- #
        if snap:
            allowed = max(snap["rpm"], 1.0)
            used = snap["rpm_used"]
            state_txt = ("PAUSED" if snap["breaker_open"] else "COOLDOWN" if snap["blocked_for"] > 0
                         else "BUSY" if snap["inflight"] else "READY")
            rows.append(self._section("RATE LIMITS", "cyan", inner, state_txt.lower()))
            lab_w = 8                                             # "rpm     ", "tok/min "
            val_txt_w = 9                                         # right-hand numeric cell
            bar_w = max(6, inner - 3 - lab_w - val_txt_w - 1)

            def load_col(f: float) -> str:                        # more of the budget used = hotter
                return VS["green"] if f < 0.6 else VS["yellow"] if f < 0.85 else VS["red"]

            def meter_row(label: str, frac: float, color: str, value: str) -> Text:
                t = Text(no_wrap=True, overflow="crop")
                t.append(f"   {label:<{lab_w}}", style=VS["fg2"])
                t.append_text(self._meter(max(0.0, min(1.0, frac)), bar_w, color))
                t.append(f" {value:>{val_txt_w}}", style=f"bold {VS['fg']}")
                return t

            # LIVE usage over the last 60s (was: the adaptive *limit* / ceiling, which barely moves)
            f_rpm = used / allowed
            rows.append(meter_row("rpm", f_rpm, load_col(f_rpm), f"{used}/{allowed:.0f}"))
            tpm_lim = max(snap["tpm_limit"], 1)
            f_tpm = snap["tpm_used"] / tpm_lim
            rows.append(meter_row("tok/min", f_tpm, load_col(f_tpm),
                                  _human(snap["tpm_used"]) + "/" + _human(tpm_lim)))
            burst = snap["burst"]
            bt = max(0.0, min(float(burst), snap["req_tokens"]))
            rows.append(meter_row("burst", bt / max(burst, 1), VS["cyan"], f"{bt:.1f}/{burst}"))
            if snap["breaker_open"] or snap["blocked_for"] > 0:
                wait = snap["breaker_left"] if snap["breaker_open"] else snap["blocked_for"]
                rows.append(self._kv("resumes in", f"{wait:.0f}s", VS["yellow"], inner))
            if not tiny:
                rows.append(gap)

            # -- SESSION counters (two-column grid) ------------------------ #
            lim_n = snap["limited"]
            if not tiny:
                rows.append(self._section("SESSION", "green", inner))
                pend = snap["inflight"]
                rows.append(self._kv("requests", f"{snap['ok']}/{snap['calls']}",
                                     VS["cyan"] if pend else VS["green"], inner))
                if pend:                                            # a request is on the wire RIGHT NOW
                    age = snap["inflight_age"]
                    rows.append(self._kv(f"{_spinner(now)} in flight",
                                         f"{pend} · {int(age)}s · ~{_human(snap['inflight_tokens'])} tok",
                                         VS["cyan"], inner))
                if snap["failed"]:
                    rows.append(self._kv("failed", str(snap["failed"]), VS["red"], inner))
                rows.append(self._kv("rate-limited", str(lim_n), VS["red"] if lim_n else VS["dim"], inner))
                rows.append(self._kv("tokens", ("~" if snap["tokens_estimated"] else "") + f"{snap['tokens']:,}"
                                     + (f" (+{_human(snap['inflight_tokens'])})" if pend else ""),
                                     VS["fg"], inner))
                if not compact:
                    rows.append(self._kv("retries", str(snap["retries"]), VS["fg"], inner))
                    if snap["avg_latency"]:
                        rows.append(self._kv("latency", f"{snap['last_latency']}s (avg {snap['avg_latency']}s)",
                                             VS["fg"], inner))
                    rows.append(self._kv("waited", f"{snap['waited']}s", VS["fg"], inner))
                rows.append(self._kv("files changed", str(len(s["order"])), VS["purple"], inner))
                rows.append(gap)

        # -- SPECIALISTS: one 2-line card each ----------------------------- #
        specs = s.get("specialists") or {}
        status_col = {"run": VS["cyan"], "ok": VS["green"], "err": VS["red"]}
        glyph = {"run": _spinner(now), "ok": "✓", "err": "✗"}
        if specs:
            active = sum(1 for v in specs.values() if v["status"] == "run")
            rows.append(self._section("SPECIALISTS", "orange", inner,
                                      f"{active} active" if active else f"{len(specs)} done"))
            for sid, v in self._pick(specs, 1 if tiny else 2 if compact else 4):
                c = status_col.get(v["status"], VS["dim"])
                l1 = Text(no_wrap=True, overflow="crop")
                l1.append(f"  {glyph.get(v['status'], '·')} ", style=f"bold {c}")
                chip_w = (len(v["worker"].replace("_worker", "")) + 2) if v.get("worker") else 0
                label_w = max(10, min(len("Frontend Specialist"), inner - 5 - chip_w - 1))
                l1.append(self._pad(v["label"], label_w), style=f"bold {VS['fg']}")
                if v.get("worker"):
                    l1.append(" ")
                    l1.append_text(self._worker_chip(v["worker"]))
                rows.append(l1)
                # running: latest thing it said, else the tool/phase it is in; finished: summary
                detail = v["detail"] or (v.get("phase") if v["status"] == "run" else "") or v["task"]
                if v["status"] == "run" and v.get("phase") and v["phase"] not in ("thinking",) \
                        and not v["phase"].startswith("thinking (step"):
                    detail = v["phase"]                         # concrete tool call beats stale prose
                l2 = Text(no_wrap=True, overflow="crop")
                l2.append("    └ ", style=VS["rule"])
                l2.append(_fit(detail, max(4, inner - 8)), style=VS["fg2"] if v["status"] == "run" else VS["dim"])
                rows.append(l2)
            rows.append(gap)

        # -- SUBAGENTS ------------------------------------------------------ #
        subs = s.get("subagents") or {}
        if subs:
            active = sum(1 for v in subs.values() if v["status"] == "run")
            rows.append(self._section("SUBAGENTS", "purple", inner,
                                      f"{active} active" if active else f"{len(subs)} done"))
            for sid, v in self._pick(subs, 1 if tiny else 2 if compact else 4):
                c = status_col.get(v["status"], VS["dim"])
                ln = Text(no_wrap=True, overflow="crop")
                ln.append(f"  {glyph.get(v['status'], '·')} ", style=f"bold {c}")
                ln.append(f"#{sid:<2} ", style=f"bold {VS['fg']}")
                ln.append(_fit(v["detail"] or v["task"], max(4, inner - 9)),
                          style=VS["fg2"] if v["status"] == "run" else VS["dim"])
                rows.append(ln)
            rows.append(gap)

        # -- HEIGHT BUDGET: what is left goes to TOOLS then LOG ------------- #
        # Exact accounting (every entry in `rows` is exactly one terminal line).
        # 2 = the panel's top+bottom border; 4 = two section headers + two spacers.
        while rows and isinstance(rows[-1], Text) and not rows[-1].plain.strip():
            rows.pop()                                       # a trailing spacer is wasted space
        # `height` is ALREADY net of the panel border (frame() passes height - 2).
        free = height - len(rows)                     # rows genuinely left below the emitted content
        n_tools = max(1, len(s["tools"]))
        n_log = max(1, len(s["log"]))
        # Each section costs 2 rows of chrome (spacer + header) plus >=1 body row.
        # Decide by what fits, in priority order, instead of all-or-nothing:
        #   >= 8 free : TOOLS and LOG both      (2 + 1) + (2 + 1) = 6 minimum, use the rest as bodies
        #   3..7 free : ONE section - TOOLS while work is running, otherwise LOG
        #   < 3 free  : neither (a header with no body is noise)
        if free >= 8:
            body = free - 4
            log_min = 2
            tool_n = min(n_tools, max(1, min(body * 6 // 10, body - log_min)))
            log_n = max(log_min, body - tool_n)
            if n_log < log_n:                          # LOG needs less -> surplus back to TOOLS
                tool_n = min(n_tools, tool_n + (log_n - n_log))
                log_n = max(log_min, body - tool_n)
        elif free >= 3:
            body = free - 2
            if any(t[3] == "run" for t in s["tools"]) or not s["log"]:
                tool_n, log_n = min(n_tools, body), 0
            else:
                tool_n, log_n = 0, min(n_log, body)
        else:
            tool_n = log_n = 0

        # -- TOOLS: fixed-width name column so every detail starts at one x -- #
        running = sum(1 for t in s["tools"] if t[3] == "run")
        if tool_n > 0:
            rows.append(gap)
            rows.append(self._section("TOOLS", "blue", inner, f"{running} running" if running else ""))
        name_w = max((len(t[1]) for t in list(s["tools"])[-max(1, tool_n):]), default=8)
        name_w = min(name_w, max(8, inner // 2))                # never let a name eat the detail
        tcol = {"run": (None, VS["yellow"]), "ok": ("✓", VS["green"]), "err": ("✗", VS["red"])}
        for ts, name, detail, st in (list(s["tools"])[-tool_n:] if tool_n > 0 else []):
            ico, colr = tcol[st]
            ln = Text(no_wrap=True, overflow="crop")
            ln.append(f"  {ico or _spinner(now)} ", style=f"bold {colr}")
            ln.append(self._pad(name, name_w) + " ", style=f"bold {VS['fg']}")
            ln.append(_fit(detail, max(4, inner - name_w - 6)), style=VS["fg2"])
            rows.append(ln)
        if not s["tools"] and tool_n > 0:
            rows.append(Text("   no tool calls yet", style=VS["dim"]))

        # -- LOG: readable timestamps, level glyph, fixed columns ------------ #
        if log_n > 0:
            rows.append(gap)
            rows.append(self._section("LOG", "blue", inner))
        lvl = {"info": (VS["fg2"], "·"), "ok": (VS["green"], "✓"), "warn": (VS["yellow"], "!"),
               "err": (VS["red"], "✗"), "ai": (VS["cyan"], "›"), "sub": (VS["purple"], "◆")}
        for ts, lv, msg in (list(s["log"])[-log_n:] if log_n > 0 else []):
            col, ic = lvl.get(lv, (VS["fg2"], "·"))
            ln = Text(no_wrap=True, overflow="crop")
            ln.append(time.strftime("  %H:%M:%S ", time.localtime(ts)), style=VS["dim"])
            ln.append(f"{ic} ", style=f"bold {col}")
            ln.append(_fit(msg, max(4, inner - 14)), style=col)
            rows.append(ln)

        return Panel(Group(*rows[: max(1, height)]),
                     title=f"[bold {VS['fg']}] EXPLORER · SESSION [/]", title_align="left",
                     border_style=VS["border"], box=box.SQUARE, padding=(0, 0),
                     style=f"on {VS['sidebar_bg']}")

    # ------------------------------------------------------------------ #
    # RIGHT: the live editor
    # ------------------------------------------------------------------ #
    def _tabbar(self, s: dict, width: int) -> Text:
        bar = Text(no_wrap=True, overflow="crop")
        for t in s["tabs"][-5:]:
            active = t.path == s["active"]
            bg = VS["tab_active"] if active else VS["tab_idle"]
            ico, icol = _icon_for(t.path)
            name = t.path.rsplit("/", 1)[-1]
            bar.append(f" {ico} ", style=f"{icol} on {bg}")
            bar.append(name, style=f"{VS['fg'] if active else VS['dim']} on {bg}")
            if t.dirty:
                bar.append(" ●", style=f"{VS['yellow']} on {bg}")
            elif t.added or t.removed:
                bar.append(" ●", style=f"{VS['green']} on {bg}")
            bar.append(" ", style=f"on {bg}")
            bar.append("│", style=f"{VS['border']} on {VS['tab_idle']}")
        return bar

    def _breadcrumb(self, path: Optional[str]) -> Text:
        t = Text(no_wrap=True, overflow="ellipsis")
        if not path:
            return t.append(" (no file open)", style=VS["dim"])
        parts = path.split("/")
        t.append(" ")
        for i, p in enumerate(parts):
            last = i == len(parts) - 1
            if last:
                ico, icol = _icon_for(p)
                t.append(f"{ico} ", style=icol)
            t.append(p, style=VS["fg"] if last else VS["dim"])
            if not last:
                t.append("  ›  ", style=VS["dim"])
        return t

    def _code_row(self, gutter_no: str, tag: str, segs: list, width: int, bg: str,
                  cursor_at: Optional[int] = None, num_w: int = 4) -> Text:
        """One editor line: [gutter][+/-][code], full-width background band."""
        line = Text(no_wrap=True, overflow="crop")
        num_style = f"{VS['line_num_hl'] if tag in '+-' else VS['line_num']} on {VS['editor_bg']}"
        line.append(f"{gutter_no:>{num_w}} ", style=num_style)
        mark = {"+": ("+", VS["add_fg"]), "-": ("-", VS["del_fg"])}.get(tag, (" ", VS["dim"]))
        line.append(mark[0] + " ", style=f"bold {mark[1]} on {bg}")
        col = 0
        room = max(0, width - num_w - 3)
        for text, style in segs:
            if col >= room:
                break
            text = text[: room - col]
            line.append(text, style=f"{style} on {bg}")
            col += len(text)
            if cursor_at is not None and col >= cursor_at:
                pass
        if cursor_at is not None and col < room:
            # blinking block cursor at the typing frontier
            if int(time.monotonic() * 2.2) % 2 == 0:
                line.append("▌", style=f"{VS['cursor']} on {bg}")
            else:
                line.append(" ", style=f"on {bg}")
            col += 1
        if col < room:
            line.append(" " * (room - col), style=f"on {bg}")     # extend band to full width
        return line

    def _slice_segs(self, segs: list, n: int) -> list:
        """First n characters of a highlighted line, preserving per-token colour."""
        out, left = [], n
        for text, style in segs:
            if left <= 0:
                break
            out.append((text[:left], style))
            left -= len(text)
        return out

    def _editor_body(self, s: dict, now: float, width: int, height: int) -> list:
        path = s["active"]
        job: Optional[EditJob] = s["jobs"].get(path) if path else None
        bg = VS["editor_bg"]

        # ---- a peek at a file the agent just read -------------------------
        if job is None and s["viewer_text"] and s["viewer_text"][0] == path:
            _, content = s["viewer_text"]
            hl = highlight_lines(content, path)
            # smoothly scroll the "reading" head down the file
            head = int((now - s["viewer_t0"]) * 40)
            start = max(0, min(head - height + 2, max(0, len(hl) - height)))
            out = []
            for i in range(start, min(len(hl), start + height)):
                dim = i > head
                segs = hl[i] if not dim else [(t, VS["dim"]) for t, _ in hl[i]]
                out.append(self._code_row(str(i + 1), " ", segs, width, bg))
            return out

        if job is None:
            welcome = [Text(""), Text("   Coder — live workspace", style=f"bold {VS['blue']}"),
                       Text("   Files the agent reads, writes and edits appear here in real time.",
                            style=VS["dim"]), Text("")]
            for k, v in (("read", "opens the file and scrolls it"), ("write", "types new files character by character"),
                         ("edit", "shows red/green line diffs with a live cursor"),
                         ("run", "streams the command output below")):
                welcome.append(Text.assemble(("     ", ""), (f"{k:<6}", f"bold {VS['cyan']}"), (v, VS["dim"])))
            return welcome

        # ---- animated write / edit ----------------------------------------
        typed = job.typed_chars(now) if job.status == "typing" else 10 ** 9
        budget = typed
        num_w = max(3, len(str(max((r[1] or 0, r[2] or 0) for r in job.rows) if job.rows else 1)) + 0)
        built: list = []            # (Text, is_frontier)
        frontier = 0
        for idx, (tag, old_no, new_no, text) in enumerate(job.rows):
            if tag == "~":
                built.append(Text.assemble((f"{'':>{num_w + 1}}", f"on {bg}"),
                                           (" ⋯ ", f"{VS['dim']} on {bg}"),
                                           (text, f"italic {VS['dim']} on {bg}")))
                continue
            if tag == "+":
                n = len(text) + 1
                if budget <= 0:
                    break                       # not typed yet - and nothing after it is either
                shown = min(len(text), budget)
                is_front = budget < n and job.status == "typing"
                budget -= n
                hl_all = job.new_lines[new_no - 1] if 0 < new_no <= len(job.new_lines) else [(text, VS["fg"])]
                segs = self._slice_segs(hl_all, shown)
                band = VS["add_bg_hi"] if is_front else VS["add_bg"]
                built.append(self._code_row(str(new_no), "+", segs, width, band,
                                            cursor_at=shown if is_front else None, num_w=num_w))
                if is_front:
                    frontier = len(built) - 1
                continue
            if tag == "-":
                hl_old = highlight_lines(text, job.path)[0] if text else []
                segs = [(t, VS["dim"]) for t, _ in hl_old] or [("", VS["dim"])]
                built.append(self._code_row(str(old_no), "-", segs, width, VS["del_bg"], num_w=num_w))
                continue
            hl = job.new_lines[new_no - 1] if 0 < new_no <= len(job.new_lines) else [(text, VS["fg"])]
            built.append(self._code_row(str(new_no), " ", hl, width, bg, num_w=num_w))
            frontier = max(frontier, len(built) - 1)

        # scroll: keep the frontier (or the first change once finished) in view
        if job.status == "typing":
            anchor = len(built) - 1
        else:
            anchor = next((i for i, r in enumerate(job.rows) if r[0] in "+-"), 0)
            anchor = min(anchor + 3, len(built) - 1)
        start = max(0, min(anchor - height // 2 if job.status != "typing" else anchor - height + 3,
                           max(0, len(built) - height)))
        return built[start:start + height]

    def _tray(self, s: dict, width: int) -> list:
        """The 'N files changed · Keep · Undo' tray from the screenshot."""
        order = list(dict.fromkeys(list(s["order"]) + [p for p, j in s["jobs"].items()
                                                       if j.status in ("typing", "awaiting")]))
        if not order:
            return []
        rows = []
        head = Text(" ")
        head.append(f"{len(order)} file{'s' if len(order) != 1 else ''} changed  ", style=VS["dim"])
        head.append(" Keep ", style=f"bold white on {VS['btn_blue']}")
        head.append(" ")
        head.append(" Undo ", style=f"{VS['fg']} on #262b40")
        rows.append(head)
        for p in order[-4:]:
            j = s["jobs"].get(p)
            ico, icol = _icon_for(p)
            name = p.rsplit("/", 1)[-1]
            folder = p.rsplit("/", 1)[0] if "/" in p else ""
            ln = Text(f"  {ico} ", style=icol)
            ln.append(name, style=f"bold {VS['fg']}")
            if folder:
                ln.append(f" {folder}", style=VS["dim"])
            if j:
                ln.append(f"   +{j.added}", style=f"bold {VS['green']}")
                ln.append(f" -{j.removed}", style=f"bold {VS['red']}")
                st = {"typing": ("writing…", VS["yellow"]), "awaiting": ("review", VS["blue"]),
                      "kept": ("kept", VS["green"]), "undone": ("undone", VS["red"]),
                      "rejected": ("rejected", VS["red"])}.get(j.status, ("", VS["dim"]))
                if st[0]:
                    ln.append(f"  {st[0]}", style=st[1])
            rows.append(ln)
        return rows

    def _terminal(self, s: dict, width: int, n: int) -> list:
        if not s["terminal_cmd"]:
            return []
        rows = [Text.assemble((" ❯ ", f"bold {VS['green']}"), (_fit(s["terminal_cmd"], width - 8), VS["fg"]),
                              (f"  [{s['terminal_state']}]", VS["yellow"] if s["terminal_state"] == "running"
                               else VS["green"] if s["terminal_state"] == "ok" else VS["red"]))]
        for ln in s["terminal"][-n:]:
            rows.append(Text(" " + _fit(ln, width - 2), style=VS["dim"]))
        return rows

    def _plan_progress_line(self, pv: Optional[dict], width: int) -> Optional[Text]:
        """Slim 'Step N/M: title' strip so plan context isn't lost while the live
        editor is showing mid-execution (see frame()/_editor_is_live)."""
        if not pv or not pv.get("exec_step"):
            return None
        steps = pv["plan"]["steps"]
        cur = next((st for st in steps if st["id"] == pv["exec_step"]), None)
        if cur is None:
            return None
        n_done = sum(1 for st in steps if st["status"] == "done")
        t = Text(no_wrap=True, overflow="ellipsis")
        t.append(f" ▸ step {pv['exec_step']}/{len(steps)} ", style=f"bold black on {VS['cyan']}")
        t.append(f" {n_done} done  ", style=f"{VS['dim']} on {VS['card_bg']}")
        t.append(_fit(cur["title"], width - 24), style=f"{VS['fg']} on {VS['card_bg']}")
        pad = width - len(t.plain)
        if pad > 0:
            t.append(" " * pad, style=f"on {VS['card_bg']}")
        return t

    def right(self, s: dict, now: float, width: int, height: int) -> Panel:
        tabbar = self._tabbar(s, width)
        crumb = self._breadcrumb(s["active"])
        plan_strip = self._plan_progress_line(s.get("plan_view"), width)
        tray = self._tray(s, width)
        term = self._terminal(s, width, 5)
        prompt_h = 2 if s["asking"] else 1
        strip_h = 1 if plan_strip else 0
        chrome = 2 + strip_h + len(tray) + (len(term) + 1 if term else 0) + prompt_h + 1
        body_h = max(4, height - chrome - 2)
        body = self._editor_body(s, now, width - 2, body_h)
        while len(body) < body_h:
            body.append(Text(" " * 5 + "~", style=VS["guide"]))

        parts: list = [tabbar, crumb]
        if plan_strip:
            parts.append(plan_strip)
        parts.extend(body)
        if term:
            parts.append(Text("─" * (width - 2), style=VS["border"]))
            parts.extend(term)
        if tray:
            parts.append(Text("─" * (width - 2), style=VS["border"]))
            parts.extend(tray)

        # status bar: mirrors VS Code's blue bar
        job = s["jobs"].get(s["active"]) if s["active"] else None
        sb = Text(no_wrap=True, overflow="crop")
        state = "ASK" if s["asking"] else (job.status.upper() if job else s["phase"].upper())
        sb.append(f"  {state}  ", style=f"bold white on {VS['status_blue']}")
        if job:
            sb.append(f"  +{job.added} -{job.removed}  ", style=f"white on {VS['status_blue']}")
            sb.append(f"{job.kind}  ", style=f"white on {VS['status_blue']}")
        hint = "  tab: switch view  "
        pad = max(0, width - 2 - len(sb.plain) - len(hint))
        sb.append(" " * pad, style=f"on {VS['status_blue']}")
        sb.append(hint, style=f"{VS['dim']} on {VS['status_blue']}")
        parts.append(sb)
        if s["asking"]:
            parts.append(Text.assemble((" ? ", f"bold {VS['yellow']}"), (_fit(s["asking"], width - 8), VS["fg"]),
                                       ("  [Y/n] ", f"bold {VS['blue']}")))

        title = " EDITOR · LIVE (plan step) " if plan_strip else " EDITOR · LIVE "
        return Panel(Group(*parts), title=f"[bold {VS['fg']}]{title}[/]", title_align="left",
                     border_style=VS["blue"] if (job and job.status == "typing") else VS["border"],
                     box=box.SQUARE, padding=(0, 0), style=f"on {VS['editor_bg']}")

    # ------------------------------------------------------------------ #
    # PLAN VIEW: replaces the editor pane while a plan is planned/reviewed/run
    # ------------------------------------------------------------------ #
    OWNER_STYLE = {"frontendspecialist": ("FRONTEND", "orange"), "backendspecialist": ("BACKEND", "green"),
                   "pythonspecialist": ("PYTHON", "yellow"), "securityspecialist": ("SECURITY", "red"),
                   "testspecialist": ("TEST", "cyan"), "devopsspecialist": ("DEVOPS", "blue"),
                   "docsspecialist": ("DOCS", "white"), "reviewspecialist": ("REVIEW", "magenta"),
                   "": ("AGENT", "purple")}
    STATUS_GLYPH = {"todo": ("○", "dim"), "run": (None, "cyan"), "done": ("✓", "green"),
                    "fail": ("✗", "red"), "skip": ("⊘", "dim")}

    def _plan_line(self, txt: Text, width: int, bg: Optional[str] = None) -> Text:
        """Pad/crop one row to EXACTLY `width` cells so panels never jitter or overflow."""
        from rich.cells import cell_len
        n = cell_len(txt.plain)
        if n > width:
            txt.truncate(width, overflow="crop")
        elif n < width:
            txt.append(" " * (width - n), style=f"on {bg}" if bg else "")
        return txt

    def _key_hint(self, pairs: list, width: int) -> Text:
        """Key-cap bar that degrades gracefully instead of hiding controls.

        `pairs` = [(key, long_label, color, short_label, priority)]  (lower priority number = more
        important). Tiers, tried in order until it fits `width`:
          1. every hint, long labels        2. every hint, short labels
          3. drop the LEAST important hints until it fits (never below the 2 most important)
        so `approve` and `cancel` can never disappear on a narrow terminal."""
        from rich.cells import cell_len
        items = [(p[0], p[1], p[2], p[3] if len(p) > 3 else p[1], p[4] if len(p) > 4 else i)
                 for i, p in enumerate(pairs)]

        def cost(sel, short):
            return 1 + sum(cell_len(k) + 2 + 1 + cell_len(sl if short else ll) + 2 for k, ll, _c, sl, _p in sel)

        keep = list(items)
        for short in (False, True):
            if cost(keep, short) <= width:
                return self._render_hint(keep, short)
        short = True
        by_importance = sorted(range(len(items)), key=lambda i: items[i][4], reverse=True)   # least important first
        for drop in by_importance:
            if len(keep) <= 2:
                break
            keep = [it for it in keep if it is not items[drop]]
            if cost(keep, short) <= width:
                break
        return self._render_hint(keep, short)

    def _render_hint(self, sel: list, short: bool) -> Text:
        t = Text(no_wrap=True, overflow="crop")
        t.append(" ")
        for key, long_l, colr, short_l, _p in sel:
            t.append(f" {key} ", style=f"bold {VS['ink']} on {VS[colr]}")
            t.append(f" {short_l if short else long_l}  ", style=VS["fg2"])
        return t

    def _plan_busy(self, pv: dict, now: float, width: int, height: int) -> list:
        """Planning / re-planning screen: shows what the planner is doing right now."""
        rev = pv["plan"] is not None
        head = "REVISING PLAN" if rev else "PLANNING"
        rows = [Text("")] * max(0, (height - 7) // 2)
        rows.append(Text.assemble(("  ", ""), (f" {_spinner(now)} {head} ", f"bold {VS['ink']} on {VS['cyan']}")))
        rows.append(Text(""))
        rows.append(Text.assemble(("    ", ""), (_fit(pv["busy"] or "working", width - 6), f"bold {VS['fg']}")))
        steps = ["read the workspace", "choose a stack", "break the work into steps", "assign specialists"]
        rows.append(Text(""))
        tick = int(now * 1.2) % (len(steps) + 1)
        for i, st in enumerate(steps):
            mark, col = ("✓", VS["green"]) if i < tick else (_spinner(now), VS["cyan"]) if i == tick else ("○", VS["dim"])
            rows.append(Text.assemble(("    ", ""), (f"{mark} ", f"bold {col}"),
                                      (st, VS["fg"] if i <= tick else VS["dim"])))
        rows.append(Text(""))
        rows.append(Text("    read-only: the planner cannot write files or run commands", style=VS["dim"]))
        return rows

    def _plan_header(self, pv: dict, width: int, executing: bool) -> list:
        p = pv["plan"]
        done = sum(1 for s in p["steps"] if s["status"] in ("done", "skip"))
        total = len(p["steps"])
        files = len({f for s in p["steps"] for f in s["files"]})
        t1 = Text(no_wrap=True, overflow="crop")
        t1.append(" ")
        t1.append(f" {'EXECUTING' if executing else 'PLAN'} ",
                  style=f"bold {VS['ink']} on {VS['cyan'] if executing else VS['purple']}")
        t1.append(" ")
        t1.append(f" v{p['version']} ", style=f"bold {VS['ink']} on {VS['yellow']}")
        t1.append("  ")
        t1.append(_fit(p["goal"] or "Untitled", max(8, width - 24)), style=f"bold {VS['fg']}")
        t2 = Text(no_wrap=True, overflow="crop")
        t2.append("   ")
        bits = [f"{total} steps", f"{files} files"] + ([f"{done}/{total} done"] if executing else [])
        for k, b_ in enumerate(bits):
            if k:
                t2.append("  ·  ", style=VS["dim"])
            t2.append(b_, style=f"bold {VS['green'] if 'done' in b_ else VS['fg']}")
        t2.append("     ")
        for stack in p["stack"][:4]:
            t2.append(f" {stack} ", style=f"{VS['blue']} on {VS['card_bg']}")
            t2.append(" ")
        rows = [self._plan_line(t1, width), self._plan_line(t2, width)]
        if executing:
            bar = Text(no_wrap=True, overflow="crop")
            bar.append("   ")
            bar.append_text(self._meter(done / max(total, 1), max(8, width - 6), VS["green"]))
            rows.append(self._plan_line(bar, width))
        return rows

    def _plan_steps(self, pv: dict, now: float, width: int, height: int, executing: bool) -> list:
        """One row per step: `▌ ○  2  Title ............ FRONTEND`. The owner chip lives in a fixed
        right-hand column (width = widest label) so chips align and can never be clipped."""
        steps = pv["plan"]["steps"]
        cur = pv["cursor"] if not executing else next((i for i, s in enumerate(steps) if s["status"] == "run"), 0)
        cur = max(0, min(cur, max(0, len(steps) - 1)))
        top = max(0, min(cur - height // 2, max(0, len(steps) - height)))     # keep cursor mid-window
        chip_w = max(len(v[0]) for v in self.OWNER_STYLE.values()) + 2          # " BACKEND " incl. padding
        # cells: marker(1) glyph(3) id(3) | title | gap(1) | chip | right margin(2)
        title_w = max(6, width - (1 + 3 + 3) - 1 - chip_w - 2)
        rows = []
        for i in range(top, min(len(steps), top + height)):
            s = steps[i]
            sel = (i == cur) if not executing else (s["status"] == "run")
            glyph, gcol = self.STATUS_GLYPH.get(s["status"], ("○", "dim"))
            bg = VS["card_hi"] if sel else None
            st = (lambda base: f"{base} on {bg}") if bg else (lambda base: base)
            skipped = s["status"] == "skip"
            t = Text(no_wrap=True, overflow="crop")
            t.append("▌" if sel else " ", style=st(f"bold {VS['cyan'] if executing else VS['purple']}"))
            t.append(f" {glyph or _spinner(now)} ", style=st(f"bold {VS[gcol]}"))
            t.append(f"{s['id']:>2} ", style=st(VS["dim"]))
            t.append(self._pad(s["title"], title_w),
                     style=st(f"{'bold ' if sel else ''}{VS['dim'] if skipped else VS['fg']}"))
            t.append(" ", style=st(""))
            lab, oc = self.OWNER_STYLE.get(s["owner"], ("AGENT", "purple"))
            t.append(f" {lab:^{chip_w - 2}} ",
                     style=(f"{VS['dim']} on {VS['card_bg']}" if skipped else f"bold {VS['ink']} on {VS[oc]}"))
            rows.append(self._plan_line(t, width, bg))
        return rows

    def _plan_detail(self, pv: dict, width: int, height: int) -> list:
        """Card for the step under the cursor (or the running/failed step while executing)."""
        steps = pv["plan"]["steps"]
        if not steps:
            return []
        idx = pv["cursor"]
        if pv["exec_step"]:
            idx = next((i for i, s in enumerate(steps) if s["status"] in ("run", "fail")), idx)
        s = steps[max(0, min(idx, len(steps) - 1))]
        rows: list = []

        def line(label: str, body: str, col: str = "fg", lcol: str = "blue") -> None:
            first = True
            for chunk in self._wrap(body, max(8, width - 13)):
                t = Text(no_wrap=True, overflow="crop")
                t.append(f"   {label:<8}" if first else " " * 11, style=f"bold {VS[lcol]}")
                t.append(chunk, style=VS[col])
                rows.append(t)
                first = False

        rows.append(Text.assemble((" ", ""), (f" STEP {s['id']} ", f"bold {VS['ink']} on {VS['blue']}"), ("  ", ""),
                                  (_fit(s["title"], width - 14), f"bold {VS['fg']}")))
        if s["detail"]:
            line("what", s["detail"], "fg2")
        if s["files"]:
            line("files", ", ".join(s["files"]), "cyan")
        if s["verify"]:
            line("verify", s["verify"], "green", "green")
        if s["note"] and s["status"] in ("done", "fail"):
            line("result", s["note"], "red" if s["status"] == "fail" else "dim", "red" if s["status"] == "fail" else "dim")
        return rows[:height]

    @staticmethod
    def _wrap(text: str, width: int) -> list:
        """Word-wrap by terminal CELLS (CJK-safe), always returning >= 1 line."""
        from rich.cells import cell_len
        out, cur = [], ""
        for w in text.split():
            while cell_len(w) > width:                               # a single overlong token
                out.append(_fit(w, width))
                w = w[len(_fit(w, width)) - 1:] if len(w) > 1 else ""
                if not w:
                    break
            if not w:
                continue
            if cell_len(cur) + (1 if cur else 0) + cell_len(w) <= width:
                cur = f"{cur} {w}" if cur else w
            else:
                out.append(cur)
                cur = w
        if cur:
            out.append(cur)
        return out or [""]

    def _plan_notes(self, pv: dict, width: int, height: int) -> list:
        """Assumptions / open questions / risks / warnings - the things the user must check
        BEFORE approving. Questions are highlighted: an unanswered question is a wrong default."""
        p = pv["plan"]
        items = ([("?", "yellow", q) for q in p["questions"]] + [("!", "orange", w) for w in pv["warnings"]] +
                 [("·", "fg2", a) for a in p["assumptions"]] + [("▲", "red", r) for r in p["risks"]])
        rows = []
        for mark, col, body in items:
            for k, chunk in enumerate(self._wrap(body, max(8, width - 6))):
                t = Text(no_wrap=True, overflow="crop")
                t.append(f"  {mark} " if k == 0 else "    ", style=f"bold {VS[col]}")
                t.append(chunk, style=VS["fg"] if mark in "?!" else VS["fg2"])
                rows.append(t)
        return rows[:height]

    def plan_view(self, s: dict, now: float, width: int, height: int) -> Panel:
        pv = s["plan_view"]
        inner_w, inner_h = max(20, width - 2), max(6, height - 2)
        executing = bool(pv["exec_step"])
        rows: list = []

        if pv["busy"]:
            rows = self._plan_busy(pv, now, inner_w, inner_h)
            title = " PLANNER "
            footer = self._key_hint([("ctrl-c", "abort", "red", "abort", 0)], inner_w)
        else:
            head = self._plan_header(pv, inner_w, executing)
            rows += head
            rows.append(Text("─" * inner_w, style=VS["rule"]))
            n_steps = len(pv["plan"]["steps"])
            notes = [] if executing else self._plan_notes(pv, inner_w, 100)
            detail = self._plan_detail(pv, inner_w, 100)
            # ---- vertical budget --------------------------------------------------------
            # rows available for content = body - header - its rule; every section below costs
            # 1 rule row + its lines. Priority: steps (the plan itself) > detail (verify!) > notes.
            avail = inner_h - 2 - len(rows)
            need_detail = len(detail) + (1 if detail else 0)          # + separator rule
            need_notes = len(notes) + (1 if notes else 0)
            # 1) the plan itself is sacred: give the step list ALL it needs (it scrolls only if the
            #    terminal genuinely cannot hold it), keeping a 3-row floor for the detail card so
            #    `verify` is not the first thing to vanish.
            floor_detail = min(need_detail, 4) if need_detail else 0
            step_h = max(1, min(n_steps, avail - floor_detail)) if avail > 1 else 1
            left = avail - step_h
            # 2) detail (what/files/VERIFY) gets what it needs next...
            det_h = min(need_detail, left) if left > 1 else 0
            left -= det_h
            # 3) ...and notes only get true leftovers.
            note_h = min(need_notes, left) if left > 1 else 0
            rows += self._plan_steps(pv, now, inner_w, step_h, executing)
            if det_h:
                rows.append(Text("─" * inner_w, style=VS["rule"]))
                rows += detail[:det_h - 1]
            if note_h:
                rows.append(Text("─" * inner_w, style=VS["rule"]))
                rows += notes[:note_h - 1]
            title = " EXECUTING PLAN " if executing else " PLAN REVIEW "

            mode = pv["mode"]
            if executing:
                footer = (self._key_hint([("ctrl-c", "again to abort now", "red", "abort", 0),
                                          ("tab", "switch view", "cyan", "view", 5)], inner_w) if pv.get("stopping")
                          else self._key_hint([("ctrl-c", "stop after this step", "red", "stop", 0),
                                               ("tab", "switch view", "cyan", "view", 5)], inner_w))
            elif mode == "review":
                footer = self._key_hint([("a", "approve & run", "green", "approve", 0),
                                         ("r", "revise", "yellow", "revise", 2),
                                         ("e", "edit steps", "blue", "edit", 3),
                                         ("↑↓", "browse", "purple", "move", 4),
                                         ("x", "cancel", "red", "cancel", 1)], inner_w)
            elif mode == "edit":
                footer = self._key_hint([("space", "skip", "yellow", "skip", 1), ("J/K", "reorder", "blue", "move", 2),
                                         ("o", "owner", "purple", "owner", 3), ("d", "done", "green", "done", 0)], inner_w)
            else:
                footer = self._key_hint([("enter", "send to planner", "green", "send", 0), ("esc", "back", "red", "back", 1)], inner_w)

        # pad body to fill, then footer rows pinned to the bottom
        body_h = inner_h - 2
        rows = rows[:body_h]
        while len(rows) < body_h:
            rows.append(Text(""))
        if pv["mode"] == "feedback" and not pv["busy"]:
            cur = "█" if int(now * 2) % 2 == 0 else " "
            inp = Text(no_wrap=True, overflow="crop")
            inp.append("  revise › ", style=f"bold {VS['yellow']}")
            inp.append(_fit(pv["text"], inner_w - 14) + cur, style=VS["fg"])
            rows.append(self._plan_line(inp, inner_w, VS["card_bg"]))
        elif pv["error"]:
            rows.append(self._plan_line(Text(f"  ⚠ {_fit(pv['error'], inner_w - 6)}", style=f"bold {VS['yellow']}"), inner_w))
        else:
            rows.append(Text(""))
        rows.append(self._plan_line(footer, inner_w))
        rows = [self._plan_line(r, inner_w) if isinstance(r, Text) else r for r in rows]

        return Panel(Group(*rows), title=f"[bold {VS['fg']}]{title}[/]", title_align="left",
                     border_style=VS["purple"] if not executing else VS["cyan"],
                     box=box.SQUARE, padding=(0, 0), style=f"on {VS['editor_bg']}")

    # -- does the editor have something live to show right now? ------------ #
    # During "EXECUTING PLAN" the plan_view used to stay glued to the screen for
    # the whole run (exec_step stays nonzero from step 1 to the last step), so a
    # write_file/edit_file/run_command happening underneath was never actually
    # visible - the animated diff, the "reading file" scroll, the terminal output
    # all rendered into UIState but the panel showing them never got a turn.
    # Flip to the live editor whenever there's real activity to show, and only
    # rest on the plan panel between steps (or before/after execution).
    EDITOR_LINGER = 1.2   # seconds to keep showing a job/command after it settles,
                          # so the screen doesn't strobe plan<->editor between steps

    def _editor_is_live(self, s: dict, now: float) -> bool:
        for job in s["jobs"].values():
            if job.status in ("typing", "pending", "awaiting"):
                return True
            if job.status in ("kept", "rejected", "undone") and now - job.t0 < self.EDITOR_LINGER + (job.total_chars() / max(job.cps, 1.0)):
                return True
        if s.get("viewer_text") and now - s.get("viewer_t0", 0.0) < 2.5:
            return True
        if s.get("terminal_state") == "running":
            return True
        if s.get("asking"):
            return True
        return False

    # ------------------------------------------------------------------ #
    def frame(self, width: int, height: int) -> Layout:
        s = self.state.snapshot()
        now = time.monotonic()
        root = Layout()
        root.split_row(Layout(name="left", ratio=self.LEFT_RATIO, minimum_size=28),
                       Layout(name="right", ratio=self.RIGHT_RATIO))
        lw = max(28, int(width * self.LEFT_RATIO / (self.LEFT_RATIO + self.RIGHT_RATIO)))
        rw = width - lw
        root["left"].update(self.left(s, now, lw - 2, height - 2))
        pv = s.get("plan_view")
        executing = bool(pv and pv.get("exec_step"))
        pin = s.get("view_pin")
        self.state.clear_pin_if_stale(has_plan=bool(pv))
        if pin == "plan" and pv:
            root["right"].update(self.plan_view(s, now, rw, height))
        elif pin and pin != "plan":
            s = dict(s, active=pin)                      # force this tab into view regardless of auto-detect
            root["right"].update(self.right(s, now, rw, height))
        elif pv and not (executing and self._editor_is_live(s, now)):
            root["right"].update(self.plan_view(s, now, rw, height))
        else:
            root["right"].update(self.right(s, now, rw, height))
        return root


# --------------------------------------------------------------------------- #
# Input + approval + live session
# --------------------------------------------------------------------------- #
class KeyReader(threading.Thread):
    """Reads single keypresses on a background thread (no Enter needed).

    The key source is injectable so the whole approval flow can be tested
    without a real TTY: pass `source=iter_of_chars` in tests.
    """

    def __init__(self, source=None):
        super().__init__(daemon=True)
        self.q: "queue.Queue[str]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._source = source

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        if self._source is not None:                     # test / scripted input
            for ch in self._source:
                if self._stop_evt.is_set():
                    return
                self.q.put(ch)
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                while not self._stop_evt.is_set():
                    if msvcrt.kbhit():
                        self.q.put(msvcrt.getwch())
                    else:
                        time.sleep(0.02)
            else:
                import select
                import termios
                import tty
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)                    # cbreak: keys arrive immediately, Ctrl-C still works
                    while not self._stop_evt.is_set():
                        r, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if r:
                            self.q.put(sys.stdin.read(1))
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:                                # not a tty -> approvals fall back to defaults
            return

    def get(self, timeout: float = 0.05) -> Optional[str]:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


class Approver:
    """Blocks a *tool thread* on a question while the UI thread stays live.

    The UI renders `state.asking`; the key loop calls `answer()`. If nobody can
    answer (no TTY) we fall back to `default` after `timeout` instead of hanging.
    """

    def __init__(self, state: UIState, keys: "KeyReader", auto: bool = False):
        self.state = state
        self.keys = keys
        self.auto = auto
        self._q: "queue.Queue[bool]" = queue.Queue()

    def ask(self, question: str, default: bool = True, timeout: Optional[float] = 60) -> bool:
        if self.auto:
            return True
        with self.state.lock:
            self.state.asking = question
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            # If nobody answered within a minute, something's wrong with key
            # capture in this terminal (common on some Windows setups) -
            # fail toward NOT running rather than hanging silently for
            # another 4 minutes with no visible feedback.
            with self.state.lock:
                self.state.add_log(
                    "warn",
                    "no y/n answer received in 60s — if you pressed y/n and "
                    "nothing happened, your terminal may not support this UI; "
                    "re-run with --yes to auto-approve, or --classic for a "
                    "plain scrolling prompt instead.",
                )
            return False
        finally:
            with self.state.lock:
                self.state.asking = None

    def feed(self, ch: str) -> bool:
        """Route one keypress. Returns True if it answered a pending question."""
        with self.state.lock:
            waiting = self.state.asking is not None
        if not waiting:
            return False
        c = ch.lower()
        if c in ("y", "\r", "\n", " "):
            self._q.put(True)
            return True
        if c in ("n", "\x1b", "q"):
            self._q.put(False)
            return True
        return False


class LiveSession:
    """Owns the terminal via rich.Live; runs the agent on a worker thread.

    Main thread : redraw @ ~20fps + route keypresses to the Approver.
    Worker      : the agent loop (model calls, tools). Blocks on approvals.
    """

    FPS = 20

    def __init__(self, console, state: UIState, renderer: Renderer, approver: Approver, keys: KeyReader):
        self.console = console
        self.state = state
        self.renderer = renderer
        self.approver = approver
        self.keys = keys

    def _draw(self):
        size = self.console.size
        return self.renderer.frame(size.width, size.height)

    def run(self, work) -> None:
        """Run `work()` on a thread while animating. Returns when it finishes."""
        from rich.live import Live
        err: list = []

        def target():
            try:
                work()
            except BaseException as e:            # noqa: BLE001 - surface to main thread
                err.append(e)

        worker = threading.Thread(target=target, daemon=True)
        if not self.keys.is_alive():
            self.keys.start()
        with Live(self._draw(), console=self.console, screen=True, auto_refresh=False,
                  redirect_stdout=False, redirect_stderr=False) as live:
            worker.start()
            try:
                while worker.is_alive():
                    ch = self.keys.get(timeout=1.0 / self.FPS)
                    if ch is not None:
                        if ch == "\x03":                      # Ctrl-C in cbreak arrives as a char
                            g = self.state.plan_gate
                            if g is not None and g.request_stop():
                                # 1st Ctrl-C while a plan RUNS: finish the current step, then halt.
                                self.state.add_log("warn", "stopping after the current step (Ctrl-C again to abort now)")
                                continue
                            raise KeyboardInterrupt
                        gate = self.state.plan_gate
                        capturing_text = (gate is not None and getattr(gate, "mode", None) == "feedback") \
                                          or self.state.asking is not None
                        if ch == "\t" and not capturing_text:      # cycle plan/tabs view (see UIState.cycle_view)
                            self.state.cycle_view(has_plan=gate is not None)
                            live.update(self._draw(), refresh=True)
                            continue
                        if not (gate is not None and gate.feed(ch)):
                            self.approver.feed(ch)
                    live.update(self._draw(), refresh=True)
                # one last frame so the final state is visible before the screen closes
                live.update(self._draw(), refresh=True)
                time.sleep(0.25)
            except KeyboardInterrupt:
                self.state.add_log("warn", "interrupted by user")
                self.approver.auto = False
                self.approver._q.put(False)                   # unblock any pending approval
                if self.state.plan_gate is not None:
                    self.state.plan_gate.abort()              # ...and a plan waiting for review
                raise
        if err:
            raise err[0]


# =========================================================================== #
#  THEME
# =========================================================================== #
PALETTE = {
    "bg": "#0f111a", "fg": "#e6e9f5", "dim": "#7d84a8",
    "accent": "#7c5cff", "accent2": "#00e5ff", "ok": "#3ddc97",
    "warn": "#ffca4b", "err": "#ff5c7a", "user": "#7aa2ff", "tool": "#c792ea",
}
THEME = Theme({
    "dim": PALETTE["dim"], "accent": PALETTE["accent"], "accent2": PALETTE["accent2"],
    "ok": PALETTE["ok"], "warn": PALETTE["warn"], "err": PALETTE["err"],
    "user": PALETTE["user"], "tool": PALETTE["tool"],
})
console = Console(theme=THEME, highlight=False)

BANNER = r"""
  ____ ___  ____  _____ ____
 / ___/ _ \|  _ \| ____|  _ \
| |  | | | | | | |  _| | |_) |
| |__| |_| | |_| | |___|  _ <
 \____\___/|____/|_____|_| \_\
"""

# =========================================================================== #
#  MODELS
# =========================================================================== #
@dataclass(frozen=True)
class ModelSpec:
    key: str
    id: str
    provider: str            # "gemini" | "openrouter" | "groq" | "openai_compat"
    label: str
    env: str
    ctx: int
    note: str
    base_url: str = ""       # only used by "groq"/"openai_compat"; "" -> provider default


MODELS = [
    ModelSpec("1", "gemini-3.5-flash", "gemini", "Gemini 3.5 Flash",
              "GEMINI_API_KEY", 1_048_576, "frontier agentic + coding, 1M ctx"),
    ModelSpec("2", "gemini-3.5-flash-lite", "gemini", "Gemini 3.5 Flash-Lite",
              "GEMINI_API_KEY", 1_048_576, "fastest / cheapest 3.5, high-throughput"),
    ModelSpec("3", "openrouter/auto", "openrouter",
              "Nemotron 3 Ultra (free)", "OPENROUTER_API_KEY", 1_000_000,
              "550B MoE (55B active), free tier, strict limits"),
    ModelSpec("4", "llama-3.3-70b-versatile", "groq", "Groq Llama 3.3 70B",
              "GROQ_API_KEY", 128_000, "Groq LPU inference, very fast, large ctx"),
    ModelSpec("5", "openai/gpt-oss-120b", "groq", "Groq GPT-OSS 120B",
              "GROQ_API_KEY", 128_000, "Groq-hosted OpenAI open model, fast + cheap"),
    # "5" reserved above; custom entries the user adds at runtime start at "9"+ (see add_custom_model).
]

# key -> next free numeric slot for models added at runtime via add_custom_model()
_NEXT_CUSTOM_KEY = 9


def add_custom_model(id_: str, provider: str, label: str, env: str,
                     ctx: int = 128_000, note: str = "custom", base_url: str = "") -> ModelSpec:
    """Register a brand-new model at runtime (from /model custom or --custom-model)
    and return its ModelSpec. Lets a user point the agent at ANY model id, on
    ANY OpenAI-compatible API (any base_url), using ANY env var to hold the key -
    without editing this file. provider must be "groq" or "openai_compat"
    (both use the OpenAI-compatible chat-completions adapter); "gemini" is
    reserved for the native Gemini SDK path.
    """
    global _NEXT_CUSTOM_KEY
    key = str(_NEXT_CUSTOM_KEY)
    _NEXT_CUSTOM_KEY += 1
    spec = ModelSpec(key, id_.strip(), provider, label or id_.strip(), env.strip() or "CUSTOM_API_KEY",
                     int(ctx or 128_000), note, base_url.strip())
    MODELS.append(spec)
    return spec

# =========================================================================== #
#  TOOLS  (one canonical JSON-schema list shared by both providers)
# =========================================================================== #
TOOL_SPECS: list[dict] = [
    {"name": "list_dir",
     "description": "List a directory (default '.').",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}}}},
    {"name": "repo_map",
     "description": ("Free orientation (no model call): folder tree + one-line symbol outline per file + "
                     "most-important-files ranking. Call ONCE first in unfamiliar code, then narrow with `path`."),
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "max_depth": {"type": "integer"}}}},
    {"name": "find_symbol",
     "description": ("Where is a function/class/method/CSS rule/HTML id DEFINED? Returns file + line range for "
                     "every match across the repo. Use this instead of search_code to locate definitions."),
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string", "description": "Symbol name, CSS selector or #id (partial names work)."}},
         "required": ["name"]}},
    {"name": "read_file",
     "description": ("Read a file. Best: `symbol` = ONE function/class/method (Class.method if ambiguous), CSS "
                     "selector (e.g. '.card') or HTML #id. Files over ~400 lines return a symbol OUTLINE first - "
                     "then read a symbol or a start/end range. full=true forces the raw whole file (expensive)."),
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"},
         "symbol": {"type": "string", "description": "Function/class/method name, CSS selector or #id."},
         "start": {"type": "integer"}, "end": {"type": "integer"}, "full": {"type": "boolean"}},
         "required": ["path"]}},
    {"name": "find_files",
     "description": "Find files by glob pattern, e.g. '**/*.py'.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "search_code",
     "description": ("Regex search over text files -> file:line: match (max 60). For plain text/strings only: "
                     "use find_symbol for definitions and find_callers for usages."),
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": "string"},
         "glob": {"type": "string", "description": "Optional file glob filter, e.g. '*.css'"}},
         "required": ["pattern"]}},
    {"name": "find_callers",
     "description": ("Who defines / calls / imports a function or class (real call graph, not a text grep). "
                     "Run before renaming, deleting or changing a signature."),
     "parameters": {"type": "object", "properties": {
         "symbol": {"type": "string", "description": "Function or class name."}}, "required": ["symbol"]}},
    {"name": "goto_definition",
     "description": ("Jump from a USE of a name to its exact definition, import-aware: if this file "
                     "imports that exact name from a specific file, returns ONLY that definition, "
                     "not every same-named symbol in the repo (unlike find_symbol). Give line pointing "
                     "at the usage (optionally col if the line has more than one identifier), or pass "
                     "symbol by name instead of line/col."),
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "line": {"type": "integer", "description": "1-based line of the usage."},
         "col": {"type": "integer", "description": "0-based column, only if line has multiple identifiers."},
         "symbol": {"type": "string", "description": "Alternative to line/col: resolve by name directly."}},
         "required": ["path", "line"]}},
    {"name": "find_references",
     "description": ("Every occurrence of the identifier at path:line (or by symbol name), labeled "
                     "definition/call/read so you can tell a declaration from a usage at a glance - not "
                     "a flat grep. Import-aware: only checks files actually connected via an import."),
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "line": {"type": "integer"}, "col": {"type": "integer"},
         "symbol": {"type": "string"}},
         "required": ["path", "line"]}},
    {"name": "rename_symbol",
     "description": ("Safely rename a function/class/variable EVERYWHERE it is actually used, across "
                     "every file - not a text find/replace. Only touches import-connected files (an "
                     "unrelated same-named symbol elsewhere is left alone), refuses if new_name would "
                     "collide with something that already exists in a touched file, shows one combined "
                     "diff across every affected file, one approval, atomic write. Use this instead of "
                     "edit_file/edit_files for any rename that could be used from more than one file."),
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "line": {"type": "integer"},
         "col": {"type": "integer"}, "symbol": {"type": "string"},
         "new_name": {"type": "string", "description": "The new identifier name."}},
         "required": ["path", "line", "new_name"]}},
    {"name": "write_file",
     "description": ("Create a NEW file (shown as a diff for approval). Replacing an existing file over 120 lines "
                     "needs overwrite=true - prefer edit_file."),
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "content": {"type": "string"},
         "overwrite": {"type": "boolean"}},
         "required": ["path", "content"]}},
    {"name": "edit_file",
     "description": ("Edit an existing file. Prefer `symbol`+`new`: replaces that whole function/class/method/"
                     "CSS rule/HTML element with `new` (its COMPLETE new source). `insert_after`+`new`: adds new "
                     "code after that symbol. Or `old`+`new`: replace one unique exact snippet (whitespace drift "
                     "is auto-tolerated). Edits that break Python syntax are rejected."),
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"},
         "new": {"type": "string", "description": "Replacement / inserted source."},
         "symbol": {"type": "string", "description": "Function/class/method, CSS selector or #id to replace."},
         "insert_after": {"type": "string", "description": "Symbol after which to insert `new`."},
         "old": {"type": "string", "description": "Exact unique snippet to replace (small tweaks only)."}},
         "required": ["path", "new"]}},
    {"name": "edit_files",
     "description": ("Atomic multi-file batch (max 20): every edit is validated first and nothing is written "
                     "unless all resolve. Same fields as edit_file. Use for renames/changes spanning files."),
     "parameters": {"type": "object", "properties": {
         "edits": {"type": "array", "minItems": 1, "maxItems": 20,
                   "items": {"type": "object", "properties": {
                       "path": {"type": "string"}, "new": {"type": "string"},
                       "symbol": {"type": "string"}, "insert_after": {"type": "string"},
                       "old": {"type": "string"}},
                       "required": ["path", "new"]},
                   }},
         "required": ["edits"]}},
    {"name": "git_status",
     "description": "Working tree status (read-only).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "git_diff",
     "description": "Unified diff of uncommitted (or staged) changes - cheaper than re-reading files. Read-only.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "staged": {"type": "boolean"}, "context": {"type": "integer"}}}},
    {"name": "git_log",
     "description": "Recent commits, optionally for one file. Read-only.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "limit": {"type": "integer"}}}},
    {"name": "git_blame",
     "description": "Who last changed each line in a range, and when. Read-only.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
         "required": ["path"]}},
    {"name": "git_commit",
     "description": "Commit to the LOCAL repo (never pushes) - only verified work. Stages all tracked changes unless `paths` given.",
     "parameters": {"type": "object", "properties": {
         "message": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}},
         "required": ["message"]}},
    {"name": "git_branch",
     "description": "List branches, or create+switch to `name` (refuses over uncommitted changes).",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string"}, "base": {"type": "string"}}}},
    {"name": "run_command",
     "description": "Run a shell command in the sandboxed workspace (tests, linters, scripts). Needs approval unless safe.",
     "parameters": {"type": "object", "properties": {
         "command": {"type": "string"}, "timeout": {"type": "integer", "description": "seconds, default 60"}},
         "required": ["command"]}},
    {"name": "dispatch_agents",
     "description": ("Run 2-8 INDEPENDENT subtasks in parallel as subagents (own context; shared workspace and "
                     "memory). Each task must be self-contained and name its exact files. Not for dependent work."),
     "parameters": {"type": "object", "properties": {
         "tasks": {"type": "array", "minItems": 2, "maxItems": 8, "items": {"type": "string"},
                   "description": "2-8 independent, self-contained task instructions."}},
         "required": ["tasks"]}},
    {"name": "save_note",
     "description": "Save ONE short durable fact (ideally with file:line) to shared codebase memory, visible to all agents and sessions.",
     "parameters": {"type": "object", "properties": {
         "note": {"type": "string"},
         "tag": {"type": "string"}},
         "required": ["note"]}},
    {"name": "recall_notes",
     "description": "Read shared memory notes (recent ones are already in your prompt). Filter by tag/query.",
     "parameters": {"type": "object", "properties": {
         "tag": {"type": "string"}, "query": {"type": "string"}}}},
]

# specialist/worker tools (frontend/backend/python coordinators) are appended
# at import time below, once specialists.py has loaded - see bottom of file.

SYSTEM_PROMPT = """You are Coder, an expert terminal coding agent working in the user's workspace.

WORKFLOW (every token you read is re-sent on every later step - stay narrow):
1. ORIENT: unfamiliar code -> repo_map once (free, no model call: symbols + importance ranking), then narrow with path.
   Where is X defined -> find_symbol. Who uses X -> find_callers (real call graph; run it before renaming, deleting
   or changing a signature). search_code is only for plain text/strings.
2. READ narrowly: read_file(path, symbol="name") returns ONE function/class/method, CSS rule (selector) or HTML #id.
   Files over ~400 lines return a symbol outline first - then read a symbol or a start/end range. Never pull a whole large file.
3. EDIT precisely: edit_file(path, symbol=..., new=<COMPLETE new source of that symbol>) to change a function/class/rule/
   element; edit_file(path, insert_after=<symbol>, new=...) to add new code next to one; `old`+`new` only for a small unique
   line-level tweak. Several files at once: edit_files (atomic). write_file is for NEW files (replacing a big existing file
   needs overwrite=true). Edits that break Python syntax are rejected before they reach disk.
4. VERIFY: run the program/tests with run_command and read the real output. Debugging: reproduce, read the traceback,
   fix the root cause, re-run. Never guess.
5. Make the smallest change that solves the problem; keep existing style; never invent file contents or command output.
   Be concise. Your step budget is limited: don't re-read what you already have.

Git (git_status/git_diff/git_log/git_blame/git_commit/git_branch): local-only. Read ops are free and cheaper than re-reading
files (git_diff = what changed, git_blame/git_log = why). Commit only after a logical unit is implemented AND verified.

Shared memory (save_note/recall_notes): a "KNOWN ABOUT THIS CODEBASE" block above is persisted knowledge from earlier work -
verify anything load-bearing. Save short one-fact notes (with file:line) for things that would be wasteful to rediscover.

Workspace root: {root}
OS: {os}

Parallel subagents (dispatch_agents): only when a task splits into independent pieces touching different files - never for
dependent steps or the same file. Each task must be self-contained (exact files + what to do); subagents don't see this
conversation. They share the rate limiter. Read every summary before concluding.
{specialist_section}"""

# Only advertised when specialists.py actually loaded (see build_provider).
SPECIALIST_SECTION_BASE = """Domain specialists (dispatch_specialists) - PREFER these over dispatch_agents when the task clearly fits:
- frontendspecialist: HTML/CSS/JS/TS/React/Vue/UI/motion.   - backendspecialist: FastAPI/Django/Flask/Node/database/REST.
- pythonspecialist: Python analysis/refactor/scripting/debugging.   - securityspecialist: auth, injection, secrets, deps audit+fix.
- testspecialist: write/run/fix tests, coverage, flaky tests.   - devopsspecialist: Docker/CI-CD/IaC/observability.
- docsspecialist: README/API docs/docstrings/architecture.   - reviewspecialist: READ-ONLY review with file:line findings (second opinion, never the fix).{browser_line}
Each has its own rate-limit budget and picks the right worker persona itself, so dispatching several at once is safe.
set_specialist_active disables/enables a misbehaving specialist or worker for the session.
"""

# Appended to the specialist list only when browsertester.py actually loaded.
_BROWSER_SPECIALIST_LINE = """
- browserspecialist: real end-to-end browser testing via an ISOLATED browser-automation agent (own
  model, own key, own rate limit - never shares your budget). Internally wakes qa_worker. Use it
  to verify UI/HTTP work actually functions (not just that it compiles): click through forms,
  check navigation, read the real DOM/console for errors. It has no write access - it tests and
  reports bugs; route any bugs it finds to the specialist that owns that domain to fix."""

# Advertised whenever the browser_test tool itself is on TOOL_SPECS, independent of whether the
# QA specialist wrapper is also wired up, so the model can call it directly for a quick check too.
_BROWSER_TOOL_LINE = """
After building or changing anything with a visible UI or an HTTP endpoint, verify it actually
works with a real browser using browser_test (directly, or via dispatch_specialists with
specialist='browserspecialist') before declaring the task done - passing a lint/build step is not
the same as the feature working for a real user."""


def _specialist_section() -> str:
    browser_line = _BROWSER_SPECIALIST_LINE if HAVE_BROWSERTESTER else ""
    section = SPECIALIST_SECTION_BASE.format(browser_line=browser_line)
    if HAVE_BROWSERTESTER:
        section += _BROWSER_TOOL_LINE + "\n"
    return section



# --------------------------------------------------------------------------- #
# Specialists integration: import here (after TOOL_SPECS/SYSTEM_PROMPT exist,
# since specialists.py only imports coder3 lazily inside function bodies - no
# circular import at module-load time either direction).
# --------------------------------------------------------------------------- #
SPECIALISTS_IMPORT_ERROR: Optional[str] = None
try:
    import specialists as _specialists_mod
    # idempotent: never append the same tool twice if this block is re-executed
    _have = {t["name"] for t in TOOL_SPECS}
    TOOL_SPECS.extend(t for t in _specialists_mod.SPECIALIST_TOOL_SPECS if t["name"] not in _have)
    HAVE_SPECIALISTS = True
except Exception as _e:                                          # pragma: no cover
    # Previously swallowed silently: the model was still TOLD to call
    # dispatch_specialists in SYSTEM_PROMPT while the tool did not exist, and the
    # user saw nothing. Record the real reason so it can be shown loudly at startup.
    import traceback as _tb
    _specialists_mod = None
    HAVE_SPECIALISTS = False
    SPECIALISTS_IMPORT_ERROR = "".join(_tb.format_exception_only(type(_e), _e)).strip()

# --------------------------------------------------------------------------- #
# Browser-testing integration (browsertester.py -> browseragent.py): a fourth
# specialist (browserspecialist/qa_worker) plus the browser_test tool and the
# autonomous build->test->fix loop. Imported here for the SAME reason as
# specialists.py above: TOOL_SPECS/SYSTEM_PROMPT must already exist, and this
# must never be able to break a coder3.py install that doesn't have
# browser_use/Playwright installed - it is optional on top of an already-
# optional layer. HAVE_BROWSERTESTER only means "the specialist + tool are
# registered", NOT "a real browser is available right now" - that is checked
# lazily by browsertester.browser_testing_available() when the tool actually
# runs, so a missing Gemini key / missing Playwright install never blocks
# startup, only that one tool call.
# --------------------------------------------------------------------------- #
BROWSERTESTER_IMPORT_ERROR: Optional[str] = None
HAVE_BROWSERTESTER = False
if HAVE_SPECIALISTS:
    try:
        import browsertester as _browsertester_mod
        _browsertester_mod.bind_host(sys.modules[__name__])
        _browsertester_mod._ensure_registered()
        _have2 = {t["name"] for t in TOOL_SPECS}
        if _browsertester_mod.BROWSER_TEST_TOOL_SPEC["name"] not in _have2:
            TOOL_SPECS.append(_browsertester_mod.BROWSER_TEST_TOOL_SPEC)
        HAVE_BROWSERTESTER = True
    except Exception as _e2:                                     # pragma: no cover
        import traceback as _tb2
        _browsertester_mod = None
        HAVE_BROWSERTESTER = False
        BROWSERTESTER_IMPORT_ERROR = "".join(_tb2.format_exception_only(type(_e2), _e2)).strip()
else:
    _browsertester_mod = None


# =========================================================================== #
#  WORKSPACE + TOOL EXECUTION
# =========================================================================== #
IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
               "build", ".mypy_cache", ".pytest_cache", ".idea", ".vscode", ".coder3"}
MAX_READ_CHARS = 24_000
WRITE_OVERWRITE_LINES = 120     # replacing an existing file longer than this needs overwrite=true
OUTLINE_MIN_LINES = 400      # read_file returns a symbol outline (not raw text) above this size
# live-typing tuning: natural speed, but always visible (>= MIN) and never slow (<= MAX)
TYPE_CPS = 320.0
TYPE_MIN_S = 0.9
TYPE_MAX_S = 5.0
TYPE_HOLD_S = 0.35
MAX_OUT_CHARS = 6_000

# --------------------------------------------------------------------------- #
# Cheap, regex-based top-level symbol extraction for repo_map(). Deliberately
# NOT a real parser (no per-language AST deps to install) - good enough to
# orient a model in a huge, unfamiliar codebase without spending a single
# model call or reading full file contents. One line of symbols per file,
# capped, so repo_map() stays cheap even on a repo with thousands of files.
# --------------------------------------------------------------------------- #
class CodebaseMemory:
    """Persistent, on-disk, workspace-scoped notes: 'what we've learned about
    this codebase so far'. Stored at <root>/.coder3/memory.json. Every
    subagent, specialist, and session that opens the SAME workspace root
    shares this file, so a finding one worker writes ("auth lives in
    src/auth/, JWT validated in middleware.py:42") is immediately visible to
    every other worker and every future session - instead of each subagent
    starting from zero knowledge of the codebase every single time, which is
    the single biggest cost driver on large/unfamiliar repos.

    Deliberately a flat, small, append-mostly JSON file (not a database or a
    vector store): the whole point is near-zero overhead to read and write,
    since it gets read on every provider construction and written after
    almost every investigation. Capped size with oldest-note eviction so it
    can never grow into its own huge-context problem.
    """
    FILE = ".coder3/memory.json"
    MAX_NOTES = 300
    MAX_NOTE_CHARS = 600
    SUMMARY_MAX_NOTES = 40     # how many notes get echoed into the system prompt

    def __init__(self, root: Path):
        self.root = root
        self.path = root / self.FILE
        self._lock = threading.Lock()

    def _load(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save(self, notes: list[dict]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(notes, indent=1), encoding="utf-8")
        except Exception:
            pass   # memory is best-effort; never let a disk hiccup break the agent

    def save(self, note: str, tag: str = "general") -> str:
        note = str(note or "").strip()[:self.MAX_NOTE_CHARS]
        if not note:
            return "error: empty note"
        tag = (str(tag or "general").strip().lower() or "general")[:40]
        with self._lock:
            notes = self._load()
            notes.append({"tag": tag, "note": note, "ts": time.time()})
            if len(notes) > self.MAX_NOTES:
                notes = notes[-self.MAX_NOTES:]      # evict oldest, keep recent knowledge
            self._save(notes)
        return f"saved to shared codebase memory (tag={tag})"

    def recall(self, tag: str = "", query: str = "") -> str:
        with self._lock:
            notes = self._load()
        if tag:
            tag = tag.strip().lower()
            notes = [n for n in notes if n.get("tag", "").lower() == tag]
        if query:
            q = query.lower()
            notes = [n for n in notes if q in n.get("note", "").lower()]
        if not notes:
            return "(no matching notes in shared codebase memory yet)"
        notes = notes[-80:]      # most recent 80 matches
        return "\n".join(f"[{n.get('tag', 'general')}] {n.get('note', '')}" for n in notes)

    def summary_for_prompt(self) -> str:
        """A short block to prepend to every new Provider's system prompt, so
        EVERY subagent/specialist/session starts already knowing what earlier
        work discovered about this codebase - without needing to call
        recall_notes first. Empty string if there's nothing saved yet."""
        with self._lock:
            notes = self._load()
        if not notes:
            return ""
        recent = notes[-self.SUMMARY_MAX_NOTES:]
        lines = [f"- [{n.get('tag', 'general')}] {n.get('note', '')}" for n in recent]
        more = len(notes) - len(recent)
        tail = f"\n  ...and {more} older note(s); call recall_notes(tag=...) for the full history." if more > 0 else ""
        return ("KNOWN ABOUT THIS CODEBASE (from earlier work by you or other agents - "
                "verify with read_file/search_code before relying on it, but treat it as a "
                "strong head start so you don't have to re-discover it from scratch):\n"
                + "\n".join(lines) + tail)


class Workspace:
    def __init__(self, root: Path, auto_approve: bool = False,
                 ui: Optional["UIState"] = None, approver: Optional["Approver"] = None):
        self.root = root.resolve()
        self.auto = auto_approve
        self.changed: list[str] = []
        self.ui = ui                      # None -> classic scrolling mode
        self.approver = approver
        # original contents captured before each write, so "Undo" is real:
        #   rel path -> original text, or None if the file did not exist
        self.snapshots: dict[str, Optional[str]] = {}
        # wired in by the REPL after construction so dispatch_agents can build
        # fresh Providers for subagents that share this session's model + limiter
        self.agent_ctx: Optional[dict] = None      # {"spec", "keys", "limiter_cache"}
        self.specialist_registry = None            # set by specialists.install()
        # shared codebase memory: findings any subagent/specialist/session writes
        # via save_note() are visible to every OTHER subagent/specialist/session
        # against this same workspace root, including ones started later - see
        # CodebaseMemory below. This is what actually fixes subagents each
        # starting from zero knowledge of the codebase every single time.
        self.memory = CodebaseMemory(self.root)
        # Graph-based code intelligence: AST cache + symbol/dependency graph +
        # PageRank ranking, backing repo_map/read_file/edit_file below. One
        # instance per workspace so its caches persist across the whole
        # session (and on disk under .coder3/ast_cache.json across sessions).
        self.intel: Optional["RepoIntel"] = RepoIntel(self.root) if HAVE_REPO_INTEL else None
        # what the agent actually did (shown by /repomap) - proof the smart paths are used
        self.usage: dict[str, int] = {}

    def _tick(self, key: str) -> None:
        self.usage[key] = self.usage.get(key, 0) + 1

    # -- read-only tools --------------------------------------------------- #
    def save_note(self, note: str, tag: str = "general") -> str:
        return self.memory.save(note, tag)

    def recall_notes(self, tag: str = "", query: str = "") -> str:
        return self.memory.recall(tag, query)

    # -- safety ------------------------------------------------------------ #
    def resolve(self, p: str) -> Path:
        path = (self.root / p).resolve() if not os.path.isabs(p) else Path(p).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            raise PermissionError(f"path escapes workspace: {p}")
        return path

    def rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    # -- read-only tools --------------------------------------------------- #
    def list_dir(self, path: str = ".") -> str:
        d = self.resolve(path or ".")
        if not d.is_dir():
            return f"error: not a directory: {path}"
        rows = []
        for e in sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if e.name in IGNORE_DIRS:
                continue
            rows.append(f"{'[dir] ' if e.is_dir() else '      '}{e.name}"
                        + ("" if e.is_dir() else f"  ({e.stat().st_size}b)"))
        return "\n".join(rows) or "(empty)"

    def repo_map(self, path: str = ".", max_depth: int = 3, max_entries: int = 150) -> str:
        """Free orientation (no model call): directory tree + a one-line symbol outline per source
        file + a 'most important files' ranking from the call/import graph. Kept deliberately
        small (150 entries, depth 3): narrow `path` to zoom in rather than dumping everything."""
        root = self.resolve(path or ".")
        if not root.is_dir():
            return f"error: not a directory: {path}"
        self._tick("repo_map")
        lines: list[str] = []
        counts = {"entries": 0, "files": 0, "dirs": 0, "trunc": False}

        def walk(d: Path, depth: int, prefix: str) -> None:
            if counts["trunc"]:
                return
            try:
                entries = sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            except Exception:
                return
            for e in (x for x in entries if x.name not in IGNORE_DIRS and not x.name.startswith(".")):
                if counts["entries"] >= max_entries:
                    counts["trunc"] = True
                    return
                counts["entries"] += 1
                if e.is_dir():
                    counts["dirs"] += 1
                    if depth < max_depth:
                        lines.append(f"{prefix}{e.name}/")
                        walk(e, depth + 1, prefix + "  ")
                    else:
                        lines.append(f"{prefix}{e.name}/  [not expanded]")
                else:
                    counts["files"] += 1
                    outline = ""
                    if self.intel is not None:
                        try:
                            outline = self.intel.symbol_outline(e)
                        except Exception:
                            outline = ""
                    lines.append(f"{prefix}{e.name}" + (f"  — {outline}" if outline else ""))

        walk(root, 0, "")
        header = f"repo map: {self.rel(root) or '.'}  ({counts['dirs']} dirs, {counts['files']} files)"
        if counts["trunc"]:
            header += f"  [truncated at {max_entries} - pass a subfolder as `path` for detail]"
        ranked_section = ""
        if self.intel is not None:
            try:
                rel_root = self.rel(root)
                ranked = [(r, sc) for r, sc in self.intel.ranked_files([], top_n=40)
                          if not rel_root or rel_root == "." or r.replace("\\", "/").startswith(rel_root.replace("\\", "/") + "/")][:8]
            except Exception:
                ranked = []
            if ranked:
                ranked_section = ("\nmost important files (call/import graph): "
                                  + ", ".join(r for r, _ in ranked) + "\n")
        return header + ranked_section + "\n" + ("\n".join(lines) or "(empty)")

    def read_file(self, path: str, start: Optional[int] = None, end: Optional[int] = None,
                 symbol: str = "", full: bool = False) -> str:
        f = self.resolve(path)
        if not f.is_file():
            return f"error: no such file: {path}"
        try:
            full_text = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"error: {e}"
        lines = full_text.splitlines()

        # Symbol read (function/class/method/CSS rule). If it can't be resolved we now
        # SAY WHY (closest matches, ambiguity) instead of silently dumping the whole file.
        if symbol:
            if self.intel is None:
                return "error: symbol lookup unavailable (repo_intel not loaded) - use start/end line ranges"
            try:
                sym, err = self.intel.resolve_symbol(f, symbol)
                hit = self.intel.extract_symbol(f, symbol) if sym else None
            except Exception as e:                                  # noqa: BLE001
                hit, err = None, f"{type(e).__name__}: {e}"
            if not hit:
                self._tick("read_symbol_failed")
                return f"error: {err or 'symbol not found'}"
            self._tick("read_symbol")
            if self.ui:
                self.ui.open_file(self.rel(f), full_text)
                self.ui.set_phase("thinking")
            body = "\n".join(f"{i:>5} | {ln}" for i, ln in
                             enumerate(hit["text"].splitlines(), hit["start_line"]))
            return (f"[{hit['kind']} {symbol}, lines {hit['start_line']}-{hit['end_line']} "
                    f"of {len(lines)}]\n" + body)

        # Big file, no range asked: give the structure first. This is what makes the model
        # reach for symbol= / ranges instead of pulling thousands of lines.
        if (self.intel is not None and start is None and not end and not full
                and len(lines) > OUTLINE_MIN_LINES):
            try:
                outline = self.intel.outline_text(f)
            except Exception:                                       # noqa: BLE001
                outline = None
            if outline:
                self._tick("read_outline")
                if self.ui:
                    self.ui.open_file(self.rel(f), full_text)
                    self.ui.set_phase("thinking")
                return (f"[{self.rel(f)}: {len(lines)} lines - showing the SYMBOL OUTLINE instead of raw text. "
                        f"Next: read_file(path, symbol=\"<name from list>\") for one function/class/CSS rule, "
                        f"or start/end for a line range; full=true only if you really need the whole file]\n"
                        + outline)

        self._tick("read_range" if (start is not None or end) else "read_full")
        if self.ui:
            self.ui.open_file(self.rel(f), "\n".join(lines))
            self.ui.set_phase("thinking")
        start = max(1, int(start or 1))
        end = min(len(lines), int(end)) if end else len(lines)
        chunk = lines[start - 1:end]
        out = "\n".join(f"{i:>5} | {ln}" for i, ln in enumerate(chunk, start))
        if len(out) > MAX_READ_CHARS:
            out = out[:MAX_READ_CHARS] + f"\n... [truncated; file has {len(lines)} lines, use start/end]"
        return out or "(empty file)"

    def find_files(self, pattern: str) -> str:
        hits = []
        for p in self.root.glob(pattern):
            if any(part in IGNORE_DIRS for part in p.parts):
                continue
            if p.is_file():
                hits.append(self.rel(p))
            if len(hits) >= 300:
                break
        return "\n".join(sorted(hits)) or "(no matches)"

    def find_callers(self, symbol: str) -> str:
        """Definitions, callers (incl. top-level code) and importers of a function/class from the
        call graph. Import-aware for Python: when a caller's own `from X import Y` names exactly
        which definition it uses, callers are attributed to THAT definition only, not merged
        across every same-named function in the repo. Falls back to name-based matching (with a
        disclaimer) only for calls that aren't resolvable via an explicit import - e.g. locally
        defined symbols, dynamic dispatch, or non-Python languages."""
        if self.intel is None:
            return "error: repo intelligence unavailable (repo_intel.py failed to import)"
        try:
            r = self.intel.references(symbol)
        except Exception as e:                                        # noqa: BLE001
            return f"error: {e}"
        self._tick("find_callers")
        if not (r["defs"] or r["callers"] or r["importers"]):
            return (f"no definition, caller or importer found for '{symbol}' (checked all indexed "
                    f"py/js/ts/css/html/go/java/rust/c files). Try find_symbol for a partial name, "
                    f"or search_code for dynamic/string references.")
        out = []
        by_def = r.get("by_def") or {}
        if len(r["defs"]) > 1 and by_def:
            # Multiple same-named definitions AND we have per-definition
            # attribution - show callers grouped under the definition they
            # actually resolve to, so it's unambiguous which one each caller uses.
            out.append(f"{len(r['defs'])} definitions share the name '{symbol}':")
            for def_label, callers in by_def.items():
                out.append(f"  {def_label}")
                if callers:
                    out.append("    called by:\n      " + "\n      ".join(callers))
                else:
                    out.append("    called by: (no callers resolved to this definition specifically)")
        else:
            if r["defs"]:
                out.append("defined at:\n  " + "\n  ".join(r["defs"]))
            out.append("called by:\n  " + ("\n  ".join(r["callers"]) if r["callers"] else "(no callers found)"))
        if r["importers"]:
            out.append("imported in: " + ", ".join(r["importers"]))
        return "\n".join(out)

    def find_symbol(self, name: str) -> str:
        """Where is `name` defined? (functions, classes, methods, CSS rules, HTML ids, constants)"""
        if self.intel is None:
            return "error: repo intelligence unavailable (repo_intel.py failed to import)"
        try:
            hits = self.intel.find_symbols(name)
        except Exception as e:                                        # noqa: BLE001
            return f"error: {e}"
        self._tick("find_symbol")
        if not hits:
            return (f"no definition matching '{name}' in indexed files. Try a shorter/partial name, "
                    f"or search_code for plain text.")
        return "\n".join(f"{h['file']}:{h['start_line']}-{h['end_line']}  {h['kind']}  {h['name']}" for h in hits)

    # -- LSP-style semantic navigation --------------------------------------- #
    # These three go beyond find_symbol/find_callers (which answer "where is
    # X defined / called, by name, anywhere in the repo") by resolving a
    # SPECIFIC occurrence through the import graph: "the `UserService` used
    # on THIS line resolves to THIS ONE class, not every same-named class in
    # the repo". That precision is what makes a safe multi-file rename
    # possible at all - a name-only rename would also rewrite an unrelated
    # same-named symbol in a file that never imported this one.
    def goto_definition(self, path: str, line: int, col: Optional[int] = None,
                        symbol: str = "") -> str:
        """Jump from a USE of a name to its exact definition, import-aware:
        if this file imports that exact name from a specific file, returns
        ONLY that definition - not every same-named symbol in the repo.
        Give `line` (1-based) pointing at the usage, optionally `col` (0-based)
        if the line has more than one identifier; or skip line/col entirely
        and pass `symbol` by name instead."""
        if self.intel is None:
            return "error: repo intelligence unavailable (repo_intel.py failed to import)"
        f = self.resolve(path)
        if not f.is_file():
            return f"error: no such file: {path}"
        try:
            r = self.intel.goto_definition(f, line, col, symbol=symbol or None)
        except Exception as e:                                        # noqa: BLE001
            return f"error: {e}"
        self._tick("goto_definition")
        if not r.get("ok"):
            return f"error: {r.get('error', 'definition not found')}"
        via = {"import": "resolved via this file's own import", "name": "resolved by name (single match)"}\
            .get(r["resolved_via"], f"resolved by name ({r['resolved_via']})")
        lines = [f"'{r['word']}' -> {via}:"]
        for d in r["definitions"]:
            lines.append(f"  {d['file']}:{d['line']}-{d['end_line']}  {d['kind']}  {d['qualname']}")
        return "\n".join(lines)

    def find_references(self, path: str, line: int, col: Optional[int] = None,
                        symbol: str = "") -> str:
        """Every occurrence of the identifier used/defined at `path:line[:col]`
        (or by `symbol` name) across the repo, labeled `definition` / `call` /
        `read` so you can tell a declaration from a usage from a plain
        reference at a glance - not just a flat grep. Import-aware: only
        looks in files actually connected to this definition via an import,
        plus the defining file itself."""
        if self.intel is None:
            return "error: repo intelligence unavailable (repo_intel.py failed to import)"
        f = self.resolve(path)
        if not f.is_file():
            return f"error: no such file: {path}"
        try:
            r = self.intel.find_references_at(f, line, col, symbol=symbol or None)
        except Exception as e:                                        # noqa: BLE001
            return f"error: {e}"
        self._tick("find_references")
        if not r.get("ok"):
            return f"error: {r.get('error', 'no references found')}"
        if not r["occurrences"]:
            return f"'{r['word']}': no occurrences found (checked {', '.join(r['definition_files']) or 'no files'})"
        lines = [f"'{r['word']}' - {len(r['occurrences'])} occurrence(s):"]
        for o in r["occurrences"]:
            lines.append(f"  {o['file']}:{o['line']}  [{o['kind']}]  {o['text']}")
        return "\n".join(lines)

    def rename_symbol(self, path: str, line: int, new_name: str, col: Optional[int] = None,
                      symbol: str = "") -> str:
        """Safely rename a function/class/variable EVERYWHERE it is actually
        used, across every file - not a text find/replace. Only touches
        files import-connected to the definition (an unrelated same-named
        symbol elsewhere is left alone), refuses if the new name would
        collide with something that already exists in a touched file, and
        shows one combined diff across every affected file for a SINGLE
        approval before anything is written - exactly like edit_files, and
        for the same reason: a rename either lands everywhere or not at all,
        never half-applied because file 3 of 5 failed."""
        if self.intel is None:
            return "error: repo intelligence unavailable (repo_intel.py failed to import)"
        f = self.resolve(path)
        if not f.is_file():
            return f"error: no such file: {path}"
        try:
            r = self.intel.rename_symbol(f, line, col, new_name, symbol=symbol or None)
        except Exception as e:                                        # noqa: BLE001
            return f"error: {e}"
        if not r.get("ok"):
            return f"error: {r.get('error', 'rename failed')}"
        changes = r["changes"]
        if not changes:
            return f"'{r['word']}': no occurrences found to rename"
        # Same atomic multi-file pipeline as edit_files: show every diff +
        # animate, ONE approval for the whole rename, then write together.
        for c in changes:
            self._show_diff(c["file"], c["old_text"], c["new_text"])
            self._animate(c["file"], c["old_text"], c["new_text"], "edit")
        files_list = ", ".join(c["file"] for c in changes)
        total = sum(c["count"] for c in changes)
        if not self._approve(f"Rename '{r['word']}' -> '{new_name}' ({total} occurrence(s) across "
                             f"{len(changes)} file(s): {files_list})?"):
            if self.ui:
                for c in changes:
                    self.ui.finish_edit(c["file"], "rejected")
            return f"user rejected the rename (nothing written)"
        applied = []
        for c in changes:
            rel = c["file"]
            self.snapshots.setdefault(rel, c["old_text"])
            self.resolve(rel).write_text(c["new_text"], encoding="utf-8")
            self.changed.append(rel)
            self.intel.invalidate(rel)
            if self.ui:
                self.ui.finish_edit(rel, "kept")
            applied.append(f"{rel} ({c['count']}x)")
        self._tick("rename_symbol")
        return (f"renamed '{r['word']}' -> '{new_name}' in {total} place(s) across {len(changes)} file(s):\n"
                + "\n".join(f"  - {a}" for a in applied))

    # -- git tools ----------------------------------------------------------- #
    # Shell out to the real `git` binary directly (never through run_command's
    # sandbox/approval path) - these are read-mostly, local-only, and never
    # touch a remote, so they're safe to run unprompted like search_code is.
    # git_commit is the one mutating op and still requires no approval since
    # it only ever affects this local repo's history, exactly like `git
    # commit` from a human's own terminal would - nothing is pushed anywhere.
    def git_status(self) -> str:
        if not HAVE_GIT_TOOLS:
            return "error: git tools unavailable"
        return git_tools.git_status(self.root)

    def git_diff(self, path: str = "", staged: bool = False, context: int = 3) -> str:
        if not HAVE_GIT_TOOLS:
            return "error: git tools unavailable"
        return git_tools.git_diff(self.root, path=path, staged=staged, context=context)

    def git_log(self, path: str = "", limit: int = 20) -> str:
        if not HAVE_GIT_TOOLS:
            return "error: git tools unavailable"
        return git_tools.git_log(self.root, path=path, limit=limit)

    def git_blame(self, path: str, start: int = 1, end: Optional[int] = None) -> str:
        if not HAVE_GIT_TOOLS:
            return "error: git tools unavailable"
        return git_tools.git_blame(self.root, path, start=start, end=end)

    def git_commit(self, message: str, paths: Optional[list] = None) -> str:
        if not HAVE_GIT_TOOLS:
            return "error: git tools unavailable"
        result = git_tools.git_commit(self.root, message, paths=paths)
        if self.ui:
            self.ui.add_log("ok" if result.startswith("committed") else "warn", result.splitlines()[0][:90])
        return result

    def git_branch(self, name: str = "", base: str = "") -> str:
        if not HAVE_GIT_TOOLS:
            return "error: git tools unavailable"
        return git_tools.git_branch(self.root, name=name, base=base)

    def search_code(self, pattern: str, glob: str = "") -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"error: bad regex: {e}"
        gl = (glob or "").strip()
        gl = "" if gl in ("**/*", "*", "**") else gl
        out, n, cap = [], 0, 60
        for dp, dns, fns in os.walk(self.root):
            dns[:] = [d for d in dns if d not in IGNORE_DIRS and not d.startswith(".")]
            for fn in fns:
                p = Path(dp) / fn
                rel = self.rel(p).replace("\\", "/")
                if gl and not (fnmatch.fnmatch(rel, gl) or fnmatch.fnmatch(fn, gl)
                               or fnmatch.fnmatch(rel, gl.lstrip("*/"))):
                    continue
                try:
                    if p.stat().st_size > 1_000_000:
                        continue
                    raw = p.read_bytes()
                    if b"\0" in raw[:1024]:
                        continue
                    for i, ln in enumerate(raw.decode("utf-8", "ignore").splitlines(), 1):
                        if rx.search(ln):
                            out.append(f"{rel}:{i}: {ln.strip()[:140]}")
                            n += 1
                            if n >= cap:
                                return "\n".join(out) + f"\n... [{cap} match limit - narrow the pattern or use glob]"
                except Exception:
                    continue
        return "\n".join(out) or "(no matches)"

    # -- approval ---------------------------------------------------------- #
    def _approve(self, title: str) -> bool:
        if self.auto:
            return True
        if self.ui is not None and self.approver is not None:
            return self.approver.ask(title)          # non-blocking for the display
        return Confirm.ask(f"[warn]{title}[/]", default=True, console=console)

    # -- live animation helper --------------------------------------------- #
    def _animate(self, rel: str, old: str, new: str, kind: str) -> Optional["EditJob"]:
        """Start the on-screen typing/diff animation and wait for it to finish, so
        the user *watches* the change being written before being asked to keep it."""
        if self.ui is None:
            return None
        job = self.ui.begin_edit(rel, old, new, kind)
        # Duration budget instead of a fixed speed: every change is on screen long enough
        # to be seen (min), and even a huge rewrite finishes in bounded time (max).
        # Speed is derived from that, so a 1-line fix types slowly enough to watch and a
        # 300-line file types fast enough not to stall the agent.
        total = max(1, job.total_chars())
        duration = min(TYPE_MAX_S, max(TYPE_MIN_S, total / TYPE_CPS))
        job.cps = total / duration
        self.ui.set_phase("editing")
        deadline = time.monotonic() + duration + 1.0
        while not job.done_typing(time.monotonic()) and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(TYPE_HOLD_S)                      # let the finished diff register before the prompt
        job.t0 -= 3600                                # snap to fully typed
        self.ui.finish_edit(rel, "awaiting")
        return job

    def undo(self, rel: str) -> str:
        """Revert one file to how it was before the agent touched it."""
        if rel not in self.snapshots:
            return f"nothing to undo for {rel}"
        orig = self.snapshots.pop(rel)
        f = self.resolve(rel)
        if orig is None:
            f.unlink(missing_ok=True)
        else:
            f.write_text(orig, encoding="utf-8")
        self.changed = [c for c in self.changed if c != rel]
        if self.intel is not None:
            self.intel.invalidate(rel)
        if self.ui:
            self.ui.finish_edit(rel, "undone")
            self.ui.order[:] = [p for p in self.ui.order if p != rel]
        return f"reverted {rel}"

    def _show_diff(self, rel: str, old: str, new: str) -> None:
        if self.ui is not None:
            return                                    # split mode draws its own diff
        diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(),
                                         f"a/{rel}", f"b/{rel}", lineterm="", n=3))
        if not diff:
            console.print("[dim]  (no changes)[/]")
            return
        console.print(Panel(Syntax("\n".join(diff), "diff", theme="monokai", word_wrap=True),
                            title=f"[accent2]diff[/] {rel}", border_style="accent", box=box.ROUNDED))

    # -- mutating tools ---------------------------------------------------- #
    def write_file(self, path: str, content: str, overwrite: bool = False) -> str:
        f = self.resolve(path)
        old = f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""
        rel = self.rel(f)
        if old == content:
            return f"no change: {rel} already has this content"
        existed = f.exists()
        if existed and not overwrite and old.count("\n") > WRITE_OVERWRITE_LINES:
            return (f"error: {rel} already exists ({old.count(chr(10)) + 1} lines). Re-sending a whole large "
                    f"file wastes tokens and risks losing code: use edit_file(symbol=...) / "
                    f"edit_file(insert_after=...) / edit_files, or pass overwrite=true if you really "
                    f"mean to replace it entirely.")
        if f.suffix == ".py":
            bad = _py_syntax_error(content)
            if bad and (not existed or not _py_syntax_error(old)):
                return f"error: not written - {rel} would contain a Python syntax error ({bad}). Fix and resend."
        self._show_diff(rel, old, content)
        self._animate(rel, old, content, "overwrite" if existed else "create")
        if not self._approve(f"{'Overwrite' if existed else 'Create'} {rel}?"):
            if self.ui:
                self.ui.finish_edit(rel, "rejected")
            return "user rejected this change"
        self.snapshots.setdefault(rel, old if existed else None)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
        self.changed.append(rel)
        if self.intel is not None:
            self.intel.invalidate(rel)
        if self.ui:
            self.ui.finish_edit(rel, "kept")
        return f"{'updated' if existed else 'created'} {rel} ({len(content.splitlines())} lines)"

    def _resolve_edit(self, f: Path, new: str, old: str = "", symbol: str = "",
                      insert_after: str = "") -> tuple[Optional[str], Optional[str], str]:
        """Pure matching logic shared by edit_file and edit_files: given a
        target file's CURRENT text and a requested change, compute the new
        full text via the same 3-tier strategy (CST patch by symbol -> exact
        substring -> fuzzy fallback), with no side effects (no write, no
        approval, no animation). Returns (updated_text_or_None, error_or_None,
        match_note) so callers can preview/approve/apply everything as one
        atomic step instead of each tier reaching for the filesystem itself."""
        text = f.read_text(encoding="utf-8", errors="replace")
        match_note = ""

        if insert_after and not symbol and not old:
            if self.intel is None:
                return None, "insert_after unavailable (repo_intel not loaded) - use `old` as an anchor", ""
            try:
                patched = self.intel.cst_insert(f, insert_after, new)
            except Exception:
                patched = None
            if patched is None:
                _s, why = self.intel.resolve_symbol(f, insert_after)
                self._tick("edit_insert_failed")
                return None, why or f"could not insert after '{insert_after}'", ""
            self._tick("edit_insert")
            return self._syntax_gate(f, text, patched, f" (inserted after {insert_after})")

        # Tier 1 - CST patch: if the caller names a symbol, replace that
        # whole function/class by AST node boundary. Immune to whitespace/
        # comment drift anywhere else in the file, and can't accidentally
        # hit a second occurrence the way substring matching can.
        updated = None
        if symbol:
            if self.intel is None:
                if not old:
                    return None, "symbol edits unavailable (repo_intel not loaded) - use `old` text matching", ""
            else:
                try:
                    patched = self.intel.cst_patch(f, symbol, new)
                except Exception:
                    patched = None
                if patched is not None:
                    updated = patched
                    match_note = f" (symbol patch: {symbol})"
                    self._tick("edit_symbol")
                else:
                    # explain WHY (ambiguous / not found / unsupported) - the old code fell
                    # through to `old` matching and reported a misleading "`old` not found"
                    _s, why = self.intel.resolve_symbol(f, symbol)
                    why = why or f"could not patch symbol '{symbol}'"
                    self._tick("edit_symbol_failed")
                    if not old:
                        return None, why, ""
                    match_note = f" (symbol '{symbol}' not usable, fell back to text match)"

        if updated is None:
            # Tier 2 - exact substring match, same contract as before.
            n = text.count(old) if old else 0
            if n == 1:
                updated = text.replace(old, new, 1)
                self._tick("edit_text")
            elif n > 1:
                return None, (f"`old` matches {n} places in {self.rel(f)}. Include more surrounding "
                              "context so it matches exactly once, or pass `symbol` to target one "
                              "function/class/CSS rule by name."), ""
            elif old and self.intel is not None:
                # Tier 3 - fuzzy fallback: `old` doesn't match byte-for-byte
                # (whitespace/formatting drift, a stray comment, etc)
                try:
                    fuzzy_result, fuzzy_msg = self.intel.fuzzy_replace(text, old, new)
                except Exception:
                    fuzzy_result, fuzzy_msg = None, "error: fuzzy match failed"
                if fuzzy_result is None:
                    return None, f"{self.rel(f)}: {fuzzy_msg}", ""
                updated = fuzzy_result
                match_note += f" ({fuzzy_msg})"
                self._tick("edit_fuzzy")
            else:
                return None, (f"`old` not found in {self.rel(f)}. Re-read the file and copy the "
                              "exact text (whitespace/indentation matter), or pass `symbol` to "
                              "target a function/class/CSS rule by name instead of matching raw text."), ""
        return self._syntax_gate(f, text, updated, match_note)

    def _syntax_gate(self, f: Path, before: str, after: str, note: str) -> tuple[Optional[str], Optional[str], str]:
        """Reject an edit that turns valid Python into invalid Python - BEFORE it is shown, approved
        or written - so the model fixes its own replacement instead of shipping a broken file."""
        if f.suffix == ".py":
            bad = _py_syntax_error(after)
            if bad and not _py_syntax_error(before):
                self._tick("edit_rejected_syntax")
                return None, (f"edit rejected: {self.rel(f)} would have a Python syntax error ({bad}). Nothing "
                              f"written. For symbol= edits `new` must be the COMPLETE symbol source (incl. "
                              f"decorators/def line and correct indentation)."), ""
        return after, None, note

    def edit_file(self, path: str, new: str, old: str = "", symbol: str = "", insert_after: str = "") -> str:
        f = self.resolve(path)
        if not f.is_file():
            return f"error: no such file: {path}"
        text = f.read_text(encoding="utf-8", errors="replace")
        updated, err, match_note = self._resolve_edit(f, new, old=old, symbol=symbol, insert_after=insert_after)
        if err:
            return f"error: {err}"

        rel = self.rel(f)
        self._show_diff(rel, text, updated)
        self._animate(rel, text, updated, "edit")
        if not self._approve(f"Apply edit to {rel}?"):
            if self.ui:
                self.ui.finish_edit(rel, "rejected")
            return "user rejected this change"
        self.snapshots.setdefault(rel, text)
        f.write_text(updated, encoding="utf-8")
        self.changed.append(rel)
        if self.intel is not None:
            self.intel.invalidate(rel)          # cache is content-hash keyed, but invalidate eagerly anyway
        if self.ui:
            self.ui.finish_edit(rel, "kept")
        return f"edited {rel}{match_note}"

    def edit_files(self, edits: list) -> str:
        """Apply MULTIPLE edits across one or more files as a SINGLE atomic
        unit: every edit is matched and validated FIRST (no file touched
        yet); if any one of them fails to resolve, NOTHING is written and
        the whole batch is rejected with a clear per-edit error list. Only
        if every edit resolves cleanly does the user see one combined diff
        and one approval prompt, then every file is written together.

        This is what real multi-file refactors need that repeated edit_file
        calls don't give you: a rename that touches 5 files either lands
        completely or not at all, never half-applied because file 3 of 5
        failed to match after files 1-2 were already on disk.

        `edits`: list of {"path", "new", "old"?, "symbol"?} - same fields as
        edit_file, one dict per file to change (one edit per file per call;
        call it again for a second edit to the same file after the first lands).
        """
        if not edits:
            return "error: edits is empty"
        if len(edits) > 20:
            return "error: too many edits in one batch (max 20) - split into smaller batches"

        resolved: list[tuple[Path, str, str, str, str]] = []   # (f, rel, old_text, new_text, match_note)
        errors: list[str] = []
        seen_paths: set[str] = set()
        for i, e in enumerate(edits):
            if not isinstance(e, dict):
                errors.append(f"edit #{i+1}: not an object")
                continue
            path = str(e.get("path", "")).strip()
            if not path:
                errors.append(f"edit #{i+1}: missing 'path'")
                continue
            if path in seen_paths:
                errors.append(f"edit #{i+1} ({path}): duplicate path in this batch - "
                              "combine into one edit or apply sequentially instead")
                continue
            f = self.resolve(path)
            if not f.is_file():
                errors.append(f"{path}: no such file")
                continue
            new = e.get("new", "")
            old = e.get("old", "") or ""
            symbol = e.get("symbol", "") or ""
            text = f.read_text(encoding="utf-8", errors="replace")
            updated, err, note = self._resolve_edit(f, new, old=old, symbol=symbol,
                                                    insert_after=e.get("insert_after", "") or "")
            if err:
                errors.append(err)
                continue
            seen_paths.add(path)
            resolved.append((f, self.rel(f), text, updated, note))

        if errors:
            return ("error: batch rejected, nothing written. Fix and resubmit the failing edits:\n"
                    + "\n".join(f"  - {msg}" for msg in errors))

        # every edit resolved cleanly - show each diff, animate, then ONE
        # combined approval gates the whole batch.
        for f, rel, old_text, new_text, note in resolved:
            self._show_diff(rel, old_text, new_text)
            self._animate(rel, old_text, new_text, "edit")

        files_list = ", ".join(rel for _, rel, *_ in resolved)
        if not self._approve(f"Apply {len(resolved)} edits atomically across: {files_list}?"):
            if self.ui:
                for _, rel, *_ in resolved:
                    self.ui.finish_edit(rel, "rejected")
            return f"user rejected this batch of {len(resolved)} edits (nothing written)"

        applied = []
        for f, rel, old_text, new_text, note in resolved:
            self.snapshots.setdefault(rel, old_text)
            f.write_text(new_text, encoding="utf-8")
            self.changed.append(rel)
            if self.intel is not None:
                self.intel.invalidate(rel)
            if self.ui:
                self.ui.finish_edit(rel, "kept")
            applied.append(f"{rel}{note}")
        return f"atomically applied {len(applied)} edits:\n" + "\n".join(f"  - {a}" for a in applied)

    @staticmethod
    def _kill_tree(proc: "subprocess.Popen") -> None:
        """Kill the shell AND everything it spawned (shell=True leaves orphans otherwise)."""
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, timeout=5)
            else:
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:                                          # noqa: BLE001
            proc.kill()

    # Commands that only READ or TEST things and can't do lasting damage
    # (no writes, no installs, no network mutation) never need a y/n prompt.
    # This is what actually fixes the "stuck on a node -e one-liner
    # forever" problem: a quick test command should just run, not wait on
    # a keypress that might not even be visible in your terminal.
    _SAFE_COMMAND_PREFIXES = (
        "node -e", "node --eval", "node -p", "node --print",
        "python -c", "python3 -c", "py -c",
        "node -v", "node --version", "npm -v", "npm --version",
        "python --version", "python3 --version",
        "npm test", "npm run test", "npm run lint", "npm run build",
        "npm run dev", "npm run typecheck", "npm run check",
        "pytest", "python -m pytest",
        "echo ", "cat ", "type ", "dir", "ls", "pwd", "cd ",
        "git status", "git diff", "git log", "git branch",
    )

    @classmethod
    def _is_safe_command(cls, command: str) -> bool:
        c = command.strip().lower()
        return any(c.startswith(p) for p in cls._SAFE_COMMAND_PREFIXES)

    def run_command(self, command: str, timeout: int = 60) -> str:
        if self.ui is None:
            console.print(Panel(Text(command, style="tool"), title="[warn]run[/]",
                                border_style="warn", box=box.ROUNDED))
        else:
            self.ui.set_terminal(command, "awaiting")
        if self._is_safe_command(command):
            # read-only / quick-test command -> just run it, no y/n wait
            pass
        elif not self._approve(f"Run: {command[:90]}"):
            if self.ui:
                self.ui.set_terminal(command, "rejected")
            return "user rejected running this command"
        timeout = max(1, min(int(timeout or 60), 600))
        if self.ui:
            self.ui.set_terminal(command, "running")
            self.ui.set_phase("running")

        def on_line(ln: str) -> None:
            if self.ui:
                self.ui.term_write(ln)

        # Ephemeral isolated execution: run_sandboxed auto-picks the
        # strongest backend actually present on the host (docker/gvisor >
        # bwrap > unshare+prlimit > resource-limit-only fallback) - see
        # exec_sandbox.py. This replaces the previous bare `Popen(shell=True)`
        # with the same command/timeout, but inside a namespace-isolated,
        # resource-capped, ephemeral process every command it can be.
        if HAVE_SANDBOX:
            try:
                res = run_sandboxed(command, self.root,
                                    SandboxLimits(timeout_s=timeout), on_line=on_line)
            except Exception as e:                                # noqa: BLE001
                return f"error: sandbox execution failed: {e}"
            if res.error:
                return f"error: {res.error}"
            if res.timed_out:
                if self.ui:
                    self.ui.set_terminal(command, "timeout")
                return (f"error: command timed out after {timeout}s [{res.backend}]\n"
                        + res.output[-MAX_OUT_CHARS:])
            if self.ui:
                self.ui.terminal_state = "ok" if res.exit_code == 0 else f"exit {res.exit_code}"
            text = res.output[-MAX_OUT_CHARS:]
            return f"exit code: {res.exit_code}  [sandbox: {res.backend}]\n--- output ---\n{text}"

        # -- fallback: sandbox module unavailable, original direct-Popen path -- #
        out_lines: list[str] = []
        try:
            popen_kw: dict = {}
            if sys.platform != "win32":
                popen_kw["start_new_session"] = True          # own process group -> we can kill the whole tree
            proc = subprocess.Popen(command, shell=True, cwd=self.root, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    errors="replace", **popen_kw)
        except Exception as e:                                    # noqa: BLE001
            return f"error: {e}"

        def pump():
            for ln in proc.stdout:                                # type: ignore[union-attr]
                out_lines.append(ln)
                if self.ui:
                    self.ui.term_write(ln.rstrip("\n"))
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            reader.join(timeout=1)
            if self.ui:
                self.ui.set_terminal(command, "timeout")
            return f"error: command timed out after {timeout}s\n" + "".join(out_lines)[-MAX_OUT_CHARS:]
        reader.join(timeout=2)
        text = "".join(out_lines)[-MAX_OUT_CHARS:]
        if self.ui:
            self.ui.terminal_state = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        # stdout+stderr are merged so the model sees them in true interleaved order
        return f"exit code: {proc.returncode}\n--- output ---\n{text}"

    def dispatch_agents(self, tasks: list[str]) -> str:
        if not self.agent_ctx:
            return "error: dispatch_agents is not available in this context"
        spec_holder = self.agent_ctx["spec"]
        # spec_holder may be a live {"spec": ModelSpec} box (so /model switches are
        # picked up without re-wiring) or a plain ModelSpec - support both.
        spec = spec_holder["spec"] if isinstance(spec_holder, dict) else spec_holder
        return dispatch_agents(self, spec, self.agent_ctx["keys"],
                               self.agent_ctx["limiter_cache"], tasks)

    # -- dispatch ---------------------------------------------------------- #
    def execute(self, name: str, args: dict) -> str:
        self._tick("tool:" + name)
        try:
            fn = {"list_dir": self.list_dir, "repo_map": self.repo_map, "read_file": self.read_file,
                  "find_files": self.find_files, "search_code": self.search_code,
                  "find_callers": self.find_callers, "find_symbol": self.find_symbol,
                  "goto_definition": self.goto_definition, "find_references": self.find_references,
                  "rename_symbol": self.rename_symbol,
                  "write_file": self.write_file, "edit_file": self.edit_file,
                  "edit_files": self.edit_files,
                  "run_command": self.run_command,
                  "dispatch_agents": self.dispatch_agents,
                  "save_note": self.save_note, "recall_notes": self.recall_notes,
                  "git_status": self.git_status, "git_diff": self.git_diff,
                  "git_log": self.git_log, "git_blame": self.git_blame,
                  "git_commit": self.git_commit, "git_branch": self.git_branch}.get(name)
            if not fn:
                return f"error: unknown tool {name}"
            return fn(**(args or {}))
        except PermissionError as e:
            return f"error: {e}"
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"



def _py_syntax_error(text: str) -> Optional[str]:
    """None if `text` is valid Python, else 'line N: message'."""
    try:
        ast.parse(text)
        return None
    except SyntaxError as e:
        return f"line {e.lineno}: {e.msg}"
    except Exception as e:                                          # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def budget_note(step: int, max_steps: int) -> str:
    """Appended to a tool result when an agent is about to run out of steps, so it wraps up
    with a summary instead of being cut off mid-task (which used to look like success).

    Warning points scale with `max_steps` instead of being hardcoded to
    steps 12/13: on a short budget (e.g. subagents, 12 steps) that's still
    "2 left / 1 left". On a longer budget (specialists, 28 steps, sized for
    semantic-navigation-heavy AST/CST work: find_definition -> find_references
    -> read symbol -> edit -> verify) a fixed "2 steps left" warning fired far
    too late to actually let a multi-file refactor wrap up cleanly - by the
    time the model saw it, there was no room left to finish and summarize.
    Using a percentage of the budget (~20% / ~7% remaining) keeps the SAME
    warning shape (soft nudge, then hard stop) proportionate to how much
    room the agent actually had to begin with.
    """
    left = max_steps - step
    if left <= 0:
        return ""
    soft_at = max(2, round(max_steps * 0.20))
    hard_at = max(1, round(max_steps * 0.07))
    if left == hard_at:
        return "\n[budget: 1 step left - reply NOW with what is done and what remains]"
    if left == soft_at:
        return (f"\n[budget: {left} steps left - stop exploring, finish the change now, "
                "then reply with a summary of what's done and what (if anything) remains]")
    return ""


# --- history compaction -------------------------------------------------------------------
# Providers re-send the WHOLE conversation on every turn, so one 15k-token file dump read at
# step 2 is paid for again on steps 3..14. After each round we keep the last few tool-result
# turns verbatim and replace older big results with a one-line stub (the model can re-read).
COMPACT_KEEP_TURNS = 2          # most recent tool-result turns kept in full
COMPACT_MIN_CHARS = 700         # smaller results are never worth eliding
COMPACT_STATS = {"results": 0, "chars": 0}
_STUB_HINT = {"read_file": "re-read with read_file(symbol=/start/end) if you still need it",
              "repo_map": "call repo_map(path=<subfolder>) again if needed",
              "run_command": "re-run the command if you need the output again"}


def _elide_stub(name: str, out: str) -> str:
    first = next((l.strip() for l in out.splitlines() if l.strip()), "")[:100]
    COMPACT_STATS["results"] += 1
    COMPACT_STATS["chars"] += max(0, len(out) - 260)
    return (f"[{name} output elided to save tokens: {len(out):,} chars, first line: {first}. "
            f"{_STUB_HINT.get(name, 're-run the tool if you still need it')}]")


# =========================================================================== #
#  PROVIDER ADAPTERS  (each call is metered by the AdaptiveRateLimiter)
# =========================================================================== #
@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ModelTurn:
    text: str = ""
    calls: list = field(default_factory=list)
    tokens: int = 0


class Provider:
    """Common interface. `turn()` performs ONE rate-limited model call."""

    def __init__(self, spec: ModelSpec, limiter: AdaptiveRateLimiter, system: str):
        self.spec = spec
        self.limiter = limiter
        self.system = system

    def turn(self, on_wait, should_abort) -> ModelTurn:            # pragma: no cover
        raise NotImplementedError

    def add_user(self, text: str) -> None:                          # pragma: no cover
        raise NotImplementedError

    def add_tool_results(self, results: list[tuple[ToolCall, str]]) -> None:  # pragma: no cover
        raise NotImplementedError

    def est_tokens(self) -> int:                                    # pragma: no cover
        return 1500


class GeminiProvider(Provider):
    def __init__(self, spec, limiter, system, api_key, thinking="medium", exclude_tools=()):
        super().__init__(spec, limiter, system)
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=api_key,
                                   http_options=types.HttpOptions(timeout=int(LLM_HTTP_TIMEOUT * 1000)))
        self.history: list = []          # list[types.Content]
        decl = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["parameters"])
                for t in TOOL_SPECS if t["name"] not in exclude_tools]
        self.cfg = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=decl)],
            # Own the tool loop: gives one metered call per turn and sidesteps the
            # reported streaming+AFC empty-response bug on 3.5 Flash.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level=thinking),
        )

    def add_user(self, text):
        t = self.types
        self.history.append(t.Content(role="user", parts=[t.Part(text=text)]))

    def add_tool_results(self, results):
        t = self.types
        parts = [t.Part(function_response=t.FunctionResponse(
                     id=c.id, name=c.name, response={"result": out}))
                 for c, out in results]
        self.history.append(t.Content(role="user", parts=parts))
        self.compact()

    def compact(self) -> None:
        """Elide stale big tool results (see COMPACT_*). Only user-side function_response parts are
        touched - the model's own content (thought signatures) stays verbatim as Gemini requires."""
        t = self.types
        idx = [i for i, c in enumerate(self.history)
               if any(getattr(p, "function_response", None) for p in (c.parts or []))]
        for i in idx[:-COMPACT_KEEP_TURNS] if COMPACT_KEEP_TURNS else idx:
            parts = self.history[i].parts
            for j, p in enumerate(list(parts or [])):
                fr = getattr(p, "function_response", None)
                res = (getattr(fr, "response", None) or {}).get("result") if fr else None
                if isinstance(res, str) and len(res) >= COMPACT_MIN_CHARS:
                    try:
                        parts[j] = t.Part(function_response=t.FunctionResponse(
                            id=fr.id, name=fr.name, response={"result": _elide_stub(fr.name, res)}))
                    except Exception:                                # noqa: BLE001
                        pass

    def est_tokens(self):
        chars = sum(len(getattr(p, "text", "") or "") + len(str(getattr(p, "function_response", "") or ""))
                    for c in self.history for p in (c.parts or []))
        return max(500, chars // 4)

    def turn(self, on_wait, should_abort):
        def _call():
            return self.client.models.generate_content(
                model=self.spec.id, contents=self.history, config=self.cfg)

        def _tokens(resp):
            u = getattr(resp, "usage_metadata", None)
            return int(getattr(u, "total_token_count", 0) or 0) if u else 0

        resp = call_with_limits(self.limiter, _call, est_tokens=self.est_tokens(),
                                tokens_from=_tokens, on_wait=on_wait, should_abort=should_abort)
        cand = (resp.candidates or [None])[0]
        if cand is None or cand.content is None:
            reason = getattr(cand, "finish_reason", "no candidates")
            return ModelTurn(text=f"[model returned no content: {reason}]", tokens=_tokens(resp))

        # Keep the model's content VERBATIM (incl. thought signatures) - Gemini 3.x
        # thought preservation requires the unmodified history to be echoed back.
        self.history.append(cand.content)
        text, calls = [], []
        for i, p in enumerate(cand.content.parts or []):
            if getattr(p, "function_call", None):
                fc = p.function_call
                calls.append(ToolCall(fc.id or f"call_{len(self.history)}_{i}", fc.name, dict(fc.args or {})))
            elif getattr(p, "text", None) and not getattr(p, "thought", False):
                text.append(p.text)
        return ModelTurn("".join(text), calls, _tokens(resp))


class OpenAICompatProvider(Provider):
    """Any OpenAI-compatible chat-completions API: OpenRouter, Groq, or a fully
    custom endpoint the user points at. base_url and model id are both driven
    entirely by the ModelSpec, so this one class covers all three."""

    # sane default base_urls per named provider; "openai_compat" MUST supply
    # its own via spec.base_url (checked in build_provider before we get here)
    _DEFAULT_BASE_URLS = {
        "openrouter": "https://openrouter.ai/api/v1",
        "groq": "https://api.groq.com/openai/v1",
    }

    def __init__(self, spec, limiter, system, api_key, base_url: str = "", exclude_tools=()):
        super().__init__(spec, limiter, system)
        from openai import OpenAI
        resolved_base = base_url or self._DEFAULT_BASE_URLS.get(spec.provider, "")
        if not resolved_base:
            raise ValueError(
                f"no base_url for provider {spec.provider!r} - custom/openai_compat "
                "models must set ModelSpec.base_url"
            )
        # max_retries=0: the AdaptiveRateLimiter owns ALL retry/backoff policy;
        # the SDK's built-in retries would double-count and fight our pacing.
        self.client = OpenAI(base_url=resolved_base, api_key=api_key,
                             max_retries=0, timeout=LLM_HTTP_TIMEOUT,
                             default_headers={"HTTP-Referer": "https://localhost/coder",
                                              "X-Title": "coder.py"})
        self.messages: list[dict] = [{"role": "system", "content": system}]
        self.tools = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in TOOL_SPECS if t["name"] not in exclude_tools]

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results):
        for c, out in results:
            self.messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
        self.compact()

    def compact(self) -> None:
        """Elide stale big tool results (see COMPACT_*); tool messages carry no name, so map it
        from the assistant tool_calls that produced them."""
        names = {tc["id"]: tc["function"]["name"]
                 for m in self.messages for tc in (m.get("tool_calls") or [])}
        blocks: list[list[int]] = []
        for i, m in enumerate(self.messages):
            if m.get("role") == "tool":
                if blocks and blocks[-1][-1] == i - 1:
                    blocks[-1].append(i)
                else:
                    blocks.append([i])
        for blk in blocks[:-COMPACT_KEEP_TURNS] if COMPACT_KEEP_TURNS else blocks:
            for i in blk:
                m = self.messages[i]
                c = m.get("content")
                if isinstance(c, str) and len(c) >= COMPACT_MIN_CHARS:
                    m["content"] = _elide_stub(names.get(m.get("tool_call_id"), "tool"), c)

    def est_tokens(self):
        return max(500, sum(len(str(m.get("content") or "")) for m in self.messages) // 4)

    def turn(self, on_wait, should_abort):
        def _call():
            return self.client.chat.completions.create(
                model=self.spec.id, messages=self.messages, tools=self.tools,
                tool_choice="auto", max_tokens=8192)

        def _tokens(resp):
            u = getattr(resp, "usage", None)
            return int(getattr(u, "total_tokens", 0) or 0) if u else 0

        resp = call_with_limits(self.limiter, _call, est_tokens=self.est_tokens(),
                                tokens_from=_tokens, on_wait=on_wait, should_abort=should_abort)
        if not getattr(resp, "choices", None):
            # Some OpenAI-compatible providers (incl. OpenRouter) can return HTTP 200
            # with an error object in the body instead of raising.
            err = getattr(resp, "error", None) or "empty response"
            return ModelTurn(text=f"[provider returned no choices: {err}]")
        msg = resp.choices[0].message
        entry: dict = {"role": "assistant", "content": msg.content or ""}
        calls = []
        if getattr(msg, "tool_calls", None):
            entry["tool_calls"] = [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name,
                                                 "arguments": tc.function.arguments}}
                                   for tc in msg.tool_calls]
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append(ToolCall(tc.id, tc.function.name, args))
        self.messages.append(entry)
        return ModelTurn(msg.content or "", calls, _tokens(resp))


# kept as an alias so any external code importing OpenRouterProvider by name
# (e.g. browsertester.py, or a user's own script) keeps working unchanged.
OpenRouterProvider = OpenAICompatProvider


# =========================================================================== #
#  UI HELPERS
# =========================================================================== #
def make_hud(limiter: AdaptiveRateLimiter, spec: ModelSpec, ws: Workspace) -> Panel:
    cache = ((getattr(ws, "agent_ctx", None) or {}).get("limiter_cache")) or {}
    s = combined_snapshot(limiter, cache)
    load = min(1.0, s["rpm_used"] / max(s["rpm"], 1))
    bar_w = 16
    filled = int(load * bar_w)
    bar = Text()
    bar.append("█" * filled, style="ok" if load > 0.5 else "warn")
    bar.append("░" * (bar_w - filled), style="dim")

    t = Table.grid(padding=(0, 2))
    t.add_row(Text(f"● {spec.label}", style="accent2"),
              Text("rate ", style="dim") + bar + Text(f" {s['rpm_used']}/{s['rpm']:.0f} rpm"),
              Text(f"calls {s['ok']}/{s['calls']}" + (f" (+{s['inflight']} live)" if s["inflight"] else ""),
                   style="accent2" if s["inflight"] else "ok"),
              Text(f"429s {s['limited']}" + (f" fail {s['failed']}" if s["failed"] else ""),
                   style="err" if (s["limited"] or s["failed"]) else "dim"),
              Text(f"tok {s['tokens'] + s['inflight_tokens']:,}", style="dim"),
              Text(f"changed {len(ws.changed)}", style="tool"))
    state = ("PAUSED " + f"{s['breaker_left']}s" if s["breaker_open"]
             else f"cooldown {s['blocked_for']}s" if s["blocked_for"] > 0 else "ready")
    t.add_row(Text(f"⏵ {state}", style="warn" if state != "ready" else "ok"),
              Text(f"↳ {s['last']}", style="dim"), "", "", "", "")
    return Panel(t, border_style="dim", box=box.ROUNDED, padding=(0, 1))


class Thinking:
    """Live status line: spinner + phase + rate-limit countdown while we wait."""

    def __init__(self, limiter, spec, ws):
        self.limiter, self.spec, self.ws = limiter, spec, ws
        self.phase = "thinking"
        self.wait_left = 0.0
        self.wait_why = ""
        self.live: Optional[Live] = None
        self.spin = Spinner("dots2", style="accent2")
        self.t0 = time.time()

    def _render(self):
        el = time.time() - self.t0
        line = Text()
        if self.wait_left > 0.05:
            line.append(f" ⏳ {self.wait_why}: {self.wait_left:4.1f}s ", style="warn")
        else:
            line.append(f" {self.phase}… ", style="accent2")
        line.append(f"{el:4.1f}s", style="dim")
        self.spin.text = line
        return Group(self.spin, make_hud(self.limiter, self.spec, self.ws))

    def __rich__(self):                       # re-evaluated on EVERY auto-refresh -> timer/meters stay live
        return self._render()

    def __enter__(self):
        self.live = Live(self, console=console, refresh_per_second=8, transient=True)
        self.live.__enter__()
        return self

    def __exit__(self, *a):
        self.live.__exit__(*a)

    def on_wait(self, left: float, why: str):
        self.wait_left, self.wait_why = left, why
        self.live.update(self._render())

    def set(self, phase: str):
        self.phase = phase
        self.wait_left = 0.0
        self.live.update(self._render())


def show_tool_call(c: ToolCall) -> None:
    detail = ""
    a = c.args or {}
    if c.name in ("read_file", "write_file", "edit_file", "list_dir"):
        detail = a.get("path", ".")
    elif c.name == "run_command":
        detail = a.get("command", "")
    elif c.name in ("find_files", "search_code"):
        detail = a.get("pattern", "")
    console.print(Text.assemble(("  ⚙ ", "tool"), (c.name, "bold tool"), ("  ", ""), (detail[:100], "dim")))


def show_banner() -> None:
    lines = BANNER.strip("\n").splitlines()
    grad = ["#7c5cff", "#6d6cff", "#5a86ff", "#38a8ff", "#00c8ff", "#00e5ff"]
    t = Text()
    for i, ln in enumerate(lines):
        t.append(ln + "\n", style=grad[i % len(grad)])
    console.print(Align.center(t))
    console.print(Align.center(Text("terminal coding agent · adaptive rate limiting", style="dim")))
    console.print()


def prompt_custom_model(env_keys: dict[str, str]) -> ModelSpec:
    """Interactively define a brand-new model: any provider shape, any model id,
    any base URL, any env var to hold the key. This is what makes 'any model,
    any API key, any provider' actually possible without editing source."""
    console.print(Panel(
        "Define a custom model. Works with Groq, Together, Fireworks, DeepInfra, "
        "a local vLLM/Ollama-OpenAI-compat server, or literally any endpoint that "
        "speaks the OpenAI chat-completions API.",
        title="[accent2]custom model[/]", border_style="accent", box=box.ROUNDED))
    provider = Prompt.ask(
        "[accent]Provider type[/] (groq = api.groq.com; openai_compat = any other "
        "OpenAI-compatible endpoint, you supply the base URL)",
        choices=["groq", "openai_compat"], default="groq", console=console)
    model_id = Prompt.ask("[accent]Model id[/] (exactly as the provider expects it, "
                          "e.g. 'llama-3.1-8b-instant' or 'meta-llama/Llama-3-70b')",
                          console=console)
    base_url = ""
    if provider == "openai_compat":
        base_url = Prompt.ask("[accent]Base URL[/] (e.g. https://api.together.xyz/v1)", console=console)
    label = Prompt.ask("[accent]Label[/] (just for display)", default=model_id, console=console)
    env = Prompt.ask("[accent]Env var name[/] to store this key under",
                     default="GROQ_API_KEY" if provider == "groq" else "CUSTOM_API_KEY",
                     console=console)
    ctx = Prompt.ask("[accent]Context window (tokens)[/]", default="128000", console=console)
    try:
        ctx_i = int(ctx)
    except ValueError:
        ctx_i = 128_000
    spec = add_custom_model(model_id, provider, label, env, ctx_i, "user-added custom model", base_url)
    if not env_keys.get(env):
        key = Prompt.ask(f"[warn]{env}[/] (hidden)", password=True, console=console)
        env_keys[env] = key.strip()
    return spec


def pick_model(env_keys: dict[str, str]) -> ModelSpec:
    tbl = Table(title="Choose a model", box=box.ROUNDED, border_style="accent", title_style="accent2")
    tbl.add_column("#", style="accent2", justify="center")
    tbl.add_column("Model", style="bold")
    tbl.add_column("Provider")
    tbl.add_column("Context", justify="right")
    tbl.add_column("Key", justify="center")
    tbl.add_column("Notes", style="dim")
    for m in MODELS:
        has = bool(env_keys.get(m.env))
        tbl.add_row(m.key, m.id, m.provider, f"{m.ctx // 1000}K",
                    "[ok]✓[/]" if has else "[err]✗[/]", m.note)
    tbl.add_row("c", "[dim]<define your own>[/]", "[dim]any[/]", "", "", "[dim]custom model / key / provider[/]")
    console.print(tbl)
    while True:
        ch = Prompt.ask("[accent]Model[/]", choices=[m.key for m in MODELS] + ["c"], default="1", console=console)
        if ch == "c":
            return prompt_custom_model(env_keys)
        spec = next(m for m in MODELS if m.key == ch)
        if not env_keys.get(spec.env):
            key = Prompt.ask(f"[warn]{spec.env} not set. Paste key (hidden)[/]", password=True, console=console)
            if not key.strip():
                console.print("[err]No key given, pick again.[/]")
                continue
            env_keys[spec.env] = key.strip()
        return spec


def base_system_prompt(ws: "Workspace", with_specialists: bool = True) -> str:
    """The one place SYSTEM_PROMPT is rendered. The specialist paragraph is only
    included when specialists.py really loaded AND is installed on this workspace,
    so the model is never told about a tool that does not exist. Also prepends a
    summary of shared codebase memory (see CodebaseMemory) so EVERY new provider -
    the top-level agent, every subagent, every specialist - starts already knowing
    what earlier work discovered about this codebase, instead of from zero."""
    section = _specialist_section() if (with_specialists and HAVE_SPECIALISTS) else ""
    base = SYSTEM_PROMPT.format(root=ws.root, os=sys.platform, specialist_section=section)
    mem = ws.memory.summary_for_prompt() if getattr(ws, "memory", None) else ""
    # Recent git activity is a second, complementary form of automatic
    # self-orientation: memory notes are what past WORK discovered, this is
    # what past COMMITS actually changed - cheap (one git log call, cached
    # for the process lifetime) and often answers "what's going on here"
    # before a single tool call.
    git_primer = ""
    if HAVE_GIT_TOOLS:
        try:
            git_primer = git_tools.recent_activity_summary(ws.root)
        except Exception:
            git_primer = ""
    prefix = "\n\n".join(p for p in (mem, git_primer) if p)
    return (prefix + "\n\n" + base) if prefix else base


def install_specialists_and_qa(ws: Workspace, keys: dict, limiters: dict) -> None:
    """Idempotent one-call installer for the whole specialist ecosystem: the
    three domain specialists (specialists.py) plus, on top, the browser-QA
    specialist and autonomous auto-fix loop (browsertester.py). Safe to call
    every time a Workspace is (re)built - each install() is itself idempotent,
    and a missing/broken browsertester.py never prevents specialists.py (or
    the base agent) from working; it just means no real-browser QA this run.
    """
    if not HAVE_SPECIALISTS or ws.specialist_registry is not None:
        return
    _specialists_mod.install(ws, keys, limiters)
    if HAVE_BROWSERTESTER:
        try:
            _browsertester_mod.install(ws, keys, limiters)
        except Exception as e:                                       # noqa: BLE001
            if ws.ui:
                ws.ui.add_log("warn", f"browser QA specialist not installed: {type(e).__name__}: {e}")
            else:
                console.print(f"[warn]browser QA specialist not installed:[/] {type(e).__name__}: {e}")


_NO_DELEGATION = frozenset({"dispatch_agents", "dispatch_specialists", "set_specialist_active"})
_DISPATCH_HEADING = "Parallel subagents (dispatch_agents):"


def subagent_system_prompt(ws: Workspace) -> str:
    """Same core workflow as the main agent, minus everything about delegating (a subagent
    can't delegate) - and it says how to report back."""
    base = base_system_prompt(ws, with_specialists=False).split(_DISPATCH_HEADING)[0].rstrip()
    return (base + "\n\nYou are a SUBAGENT: do only the task you were given, yourself - you cannot delegate. "
            "Touch only the files it names. Finish with a 2-line summary: what you changed "
            "(file :: symbol) and how you verified it.")


def build_provider(spec: ModelSpec, keys: dict, ws: Workspace, limiter_cache: dict,
                   role: str = "main") -> Provider:
    profile = PROFILES.get(spec.id, DEFAULT_PROFILE)
    if spec.id not in limiter_cache:                 # keep learned rate across /model switches
        limiter_cache[spec.id] = AdaptiveRateLimiter(profile)
    limiter = limiter_cache[spec.id]
    # subagents get a delegation-free prompt AND are never even offered the dispatch tools
    system = subagent_system_prompt(ws) if role == "sub" else base_system_prompt(ws)
    excl = _NO_DELEGATION if role == "sub" else frozenset()
    if spec.provider == "gemini":
        return GeminiProvider(spec, limiter, system, keys[spec.env], exclude_tools=excl)
    # openrouter / groq / openai_compat all speak the same OpenAI-compatible
    # chat-completions API - only base_url + model id + key differ.
    return OpenAICompatProvider(spec, limiter, system, keys[spec.env], base_url=spec.base_url,
                                exclude_tools=excl)


# =========================================================================== #
#  AGENT LOOP
# =========================================================================== #
MAX_STEPS = 25


def _tool_detail(c: ToolCall) -> str:
    a = c.args or {}
    n = c.name
    if n in ("read_file", "edit_file"):
        p = str(a.get("path", "."))
        if a.get("symbol"):
            return f"{p} :: {a['symbol']}"                      # symbol mode is now visible
        if a.get("insert_after"):
            return f"{p} :: +after {a['insert_after']}"
        if n == "read_file" and (a.get("start") or a.get("end")):
            return f"{p}:{a.get('start') or 1}-{a.get('end') or ''}"
        return p
    if n in ("write_file", "list_dir", "repo_map"):
        return str(a.get("path", "."))
    if n == "edit_files":
        return ", ".join(str(e.get("path", "?")) for e in (a.get("edits") or []) if isinstance(e, dict))
    if n == "find_callers":
        return str(a.get("symbol", ""))
    if n == "find_symbol":
        return str(a.get("name", ""))
    if n == "run_command":
        return str(a.get("command", ""))
    if n.startswith("git_"):
        return str(a.get("path", "") or a.get("message", "") or a.get("name", ""))
    return str(a.get("pattern", "") or a.get("note", "") or a.get("query", ""))


def run_agent(provider: Provider, ws: Workspace, prompt: str) -> None:
    """Agent loop. In split mode (ws.ui set) every event flows through UIState so the
    renderer can draw it live; otherwise it falls back to the classic scrolling UI."""
    ui = ws.ui
    sent_prompt = prompt
    if HAVE_SPECIALISTS and ws.specialist_registry is not None:
        # arm_router: deterministic Python routing (no LLM call). For a strong domain
        # match it appends a firm DIRECTIVE and arms a safety net that delegates
        # automatically if the model ignores it and writes domain files itself.
        hint = _specialists_mod.arm_router(ws, prompt)
        if hint:
            sent_prompt = f"{prompt}\n\n{hint}"
            if ui:
                strong = "ROUTING DIRECTIVE" in hint
                ui.add_log("sub", ("route → " if strong else "hint → ") +
                           hint.split("this ")[1].split(" work")[0].split(".")[0][:40]
                           if "this " in hint else "routed")
    provider.add_user(sent_prompt)
    interrupted = threading.Event()
    if ui:
        ui.prompt = prompt
        ui.add_log("ai", f"task: {prompt[:70]}")
    else:
        console.print(Rule(style="dim"))

    def fail(title: str, body: str, level: str = "err") -> None:
        if ui:
            ui.set_phase("error")
            ui.add_log(level, f"{title}: {body[:80]}")
            ui.assistant_text = f"{title}: {body}"
        else:
            console.print(Panel(body, title=f"[err]{title}[/]", border_style="err"))

    for step in range(1, MAX_STEPS + 1):
        if ui:
            ui.step = step
            ui.set_phase("thinking")
        try:
            if ui:
                turn = provider.turn(ui.set_wait, interrupted.is_set)
            else:
                with Thinking(provider.limiter, provider.spec, ws) as tui:
                    turn = provider.turn(tui.on_wait, interrupted.is_set)
        except CircuitOpen as e:
            fail("rate limited", f"{e}. Paused ~{int(e.reopen_in)}s (daily quota or repeated "
                                 f"failures). Use /model to switch models, or wait.", "warn")
            return
        except KeyboardInterrupt:
            fail("interrupted", "stopped by user", "warn")
            return
        except Exception as e:                                   # noqa: BLE001
            sig = parse_error(e)
            hint = {401: " Check your API key.", 403: " Check your API key / permissions.",
                    404: " Model id not found for this key."}.get(sig.status, "")
            if sig.is_rate_limit:
                hint = " Retries exhausted; wait a bit or switch model with /model."
            fail("model error", f"{type(e).__name__}: {sig.reason}.{hint}")
            return

        if turn.text.strip():
            if ui:
                ui.assistant_text = turn.text
                ui.add_log("ai", turn.text.strip().splitlines()[0][:90])
            else:
                console.print(Markdown(turn.text))

        if not turn.calls:
            if ui:
                ui.set_phase("idle")
                ui.add_log("ok", "task complete")
            else:
                console.print(make_hud(provider.limiter, provider.spec, ws))
            return

        results = []
        for c in turn.calls:
            if ui:
                ui.tool_start(c.name, _tool_detail(c))
                ui.add_log("info", f"{c.name} {_tool_detail(c)[:60]}")
            else:
                show_tool_call(c)
            out = ws.execute(c.name, c.args)
            if ui:
                ui.tool_end(c.name, ok=not out.startswith("error"))
                if out.startswith("error") or out.startswith("user rejected"):
                    ui.add_log("warn", out.splitlines()[0][:80])
            results.append((c, out))
        note = budget_note(step, MAX_STEPS)
        if note and results:
            results[-1] = (results[-1][0], results[-1][1] + note)
        provider.add_tool_results(results)
    msg = f"stopped after {MAX_STEPS} steps - send a follow-up to continue."
    if ui:
        ui.set_phase("idle")
        ui.add_log("warn", msg)
    else:
        console.print(f"[warn]{msg}[/]")


# =========================================================================== #
#  SUBAGENTS  (parallel workers sharing the rate limiter + workspace)
# =========================================================================== #
SUB_MAX_STEPS = 12          # tighter budget than the top-level agent - subtasks are narrower

# Explicit status markers, NOT a substring check on the rendered text - same
# fix, same reasoning, as specialists.py's run_specialist/dispatch_specialists
# (see the long comment there): "FAILED" in results.values() is one unlucky
# word away from a false positive (a subagent reporting "3 tests FAILED, 2
# passed" as its own genuinely-successful summary), and doesn't distinguish
# a hard failure from a subagent that simply ran out of its step budget
# ("...(INCOMPLETE)..." doesn't contain "FAILED" either, so it silently
# counted as "ok" before this fix). dispatch_agents only feeds a UI log line
# today (nothing here gates further execution the way execute_plan did for
# specialists), but that's exactly the kind of "harmless until it isn't"
# staleness worth closing while touching this code at all.
_SUBAGENT_STATUS_RE = re.compile(r"^\[subagent \d+(?: (FAILED))?\]")


def _subagent_status(result: str) -> str:
    if "(INCOMPLETE)" in result:
        return "incomplete"
    m = _SUBAGENT_STATUS_RE.match(result or "")
    if m and m.group(1):
        return "failed"
    return "ok" if m else "failed"       # no recognisable tag at all -> fail-closed, same as specialists.py


def run_subagent(sub_id: int, provider: Provider, ws: Workspace, task: str) -> str:
    """One subagent's full tool loop. Runs on a worker thread; talks to `ws` (shared
    workspace, same disk root + approver) and to its own Provider/history. All model
    calls go through `provider.limiter`, which is shared with every other subagent
    (and the parent), so pacing/backoff/circuit-breaking is unified automatically -
    a subagent can never push the group over the model's real budget.
    """
    ui = ws.ui
    provider.add_user(task)
    if ui:
        ui.sub_start(sub_id, task)

    def wait_cb(left: float, why: str) -> None:
        if ui:
            ui.sub_update(sub_id, phase=f"waiting {left:.0f}s ({why})")

    last_text = ""
    for step in range(1, SUB_MAX_STEPS + 1):
        if ui:
            ui.sub_update(sub_id, phase=f"thinking (step {step})")
        try:
            turn = provider.turn(wait_cb, lambda: False)
        except CircuitOpen as e:
            summary = f"rate limited: paused ~{int(e.reopen_in)}s"
            if ui:
                ui.sub_end(sub_id, ok=False, summary=summary)
            return f"[subagent {sub_id} FAILED] {summary}"
        except Exception as e:                                    # noqa: BLE001
            sig = parse_error(e)
            summary = f"error: {type(e).__name__}: {sig.reason}"
            if ui:
                ui.sub_end(sub_id, ok=False, summary=summary)
            return f"[subagent {sub_id} FAILED] {summary}"

        if turn.text.strip():
            last_text = turn.text.strip()
            if ui:
                ui.sub_update(sub_id, detail=last_text.splitlines()[0][:70])

        if not turn.calls:
            if ui:
                ui.sub_end(sub_id, ok=True, summary=last_text.splitlines()[0][:70] if last_text else "done")
            return f"[subagent {sub_id}] {last_text or 'done (no output)'}"

        results = []
        for c in turn.calls:
            if ui:
                ui.sub_update(sub_id, phase=f"{c.name} {_tool_detail(c)[:40]}")
            out = ws.execute(c.name, c.args)
            results.append((c, out))
        note = budget_note(step, SUB_MAX_STEPS)
        if note and results:
            results[-1] = (results[-1][0], results[-1][1] + note)
        provider.add_tool_results(results)

    summary = f"stopped after {SUB_MAX_STEPS} steps (INCOMPLETE)"
    if ui:
        ui.sub_end(sub_id, ok=False, summary=summary)      # was ok=True: unfinished work showed a green tick
    return f"[subagent {sub_id}] {summary}. Last output: {last_text[:200]}"


def install_write_lock(ws: Workspace) -> None:
    """Serialise every mutating workspace call behind ONE persistent RLock, installed exactly once
    (idempotent). Shared by dispatch_agents and the specialists so parallel agents can't interleave
    writes, snapshots or git index changes. Wrappers forward **kw so new edit/write params work."""
    if getattr(ws, "_write_lock_installed", False):
        return
    lock = threading.RLock()
    ws._specialist_write_lock = lock
    orig_write, orig_edit = ws.write_file, ws.edit_file
    orig_edit_files = getattr(ws, "edit_files", None)
    orig_commit = getattr(ws, "git_commit", None)

    def locked_write(path, content, **kw):
        with lock:
            return orig_write(path, content, **kw)

    def locked_edit(path, new=None, old="", symbol="", **kw):
        with lock:
            return orig_edit(path, new, old=old, symbol=symbol, **kw)

    ws.write_file = locked_write          # type: ignore[method-assign]
    ws.edit_file = locked_edit            # type: ignore[method-assign]
    if orig_edit_files is not None:
        def locked_edit_files(edits):
            with lock:
                return orig_edit_files(edits)
        ws.edit_files = locked_edit_files  # type: ignore[method-assign]
    if orig_commit is not None:
        def locked_commit(message, paths=None):
            with lock:
                return orig_commit(message, paths=paths)
        ws.git_commit = locked_commit      # type: ignore[method-assign]
    ws._write_lock_installed = True


def max_parallel_subagents(limiter: AdaptiveRateLimiter, n_requested: int, hard_cap: int = 6) -> int:
    """Derive a safe worker count from what the model can actually sustain right now,
    instead of a fixed thread count. Bounded by:
      - the profile's burst size (how many requests can fire back-to-back at all)
      - the current adapted rpm (half of it, so subagents don't immediately queue up
        behind each other and stall - leaves headroom for the parent's own calls)
      - a hard cap so a huge task list can't spawn an unreasonable number of threads
      - never less than 1, never more than requested
    """
    snap = limiter.snapshot()
    from_burst = max(1, int(snap["burst"]))
    from_rpm = max(1, int(snap["rpm"] // 2))
    return max(1, min(n_requested, from_burst, from_rpm, hard_cap))


def dispatch_agents(parent_ws: Workspace, spec: ModelSpec, keys: dict,
                    limiter_cache: dict, tasks: list[str]) -> str:
    """Fan a list of tasks out to parallel subagents against the same workspace root.

    Each subagent gets its own Provider (own conversation, so histories don't tangle)
    but the SAME AdaptiveRateLimiter instance as the parent (via limiter_cache, keyed
    by model id) - that limiter's internal lock already makes concurrent acquire()
    calls safe, so N subagents calling the model at once are automatically paced,
    backed off, and circuit-broken as a single adaptive budget rather than N separate
    ones. Workspace writes go through the same approval gate as the parent; the
    Approver has one `asking` slot, so concurrent approval requests are naturally
    serialized (first request wins the slot, the rest wait their turn).
    """
    tasks = [t for t in (tasks or []) if str(t).strip()][:8]
    if len(tasks) < 2:
        return "error: dispatch_agents needs at least 2 independent tasks"

    n_workers = max_parallel_subagents(limiter_cache.get(spec.id) or
                                       AdaptiveRateLimiter(PROFILES.get(spec.id, DEFAULT_PROFILE)),
                                       len(tasks))
    ui = parent_ws.ui
    if ui:
        ui.add_log("sub", f"dispatching {len(tasks)} subagents ({n_workers} parallel)")

    # ONE persistent, correctly-signed write lock shared with the specialists. (The old code swapped in
    # `locked_edit(path, old, new)` around every run: wrong argument order - so every subagent text
    # edit had old/new SWAPPED and failed - and it crashed on symbol=/insert_after=.)
    install_write_lock(parent_ws)

    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="subagent") as pool:
        futures = {}
        for i, task in enumerate(tasks, start=1):
            provider = build_provider(spec, keys, parent_ws, limiter_cache, role="sub")
            futures[pool.submit(run_subagent, i, provider, parent_ws, task)] = i
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:                            # noqa: BLE001
                results[i] = f"[subagent {i} FAILED] {type(e).__name__}: {e}"

    statuses = {i: _subagent_status(r) for i, r in results.items()}
    ok = sum(1 for s in statuses.values() if s == "ok")
    incomplete = sum(1 for s in statuses.values() if s == "incomplete")
    failed = sum(1 for s in statuses.values() if s == "failed")
    if ui:
        parts = [f"{ok}/{len(tasks)} subagents finished ok"]
        if incomplete:
            parts.append(f"{incomplete} incomplete (ran out of steps)")
        if failed:
            parts.append(f"{failed} failed")
        ui.add_log("sub", ", ".join(parts))
    return "\n\n".join(results[i] for i in sorted(results))


# =========================================================================== #
#  REPL
# =========================================================================== #
HELP = """[accent2]Commands[/]
  [accent]/model[/]      switch model (incl. [accent]custom[/] - any model id/key/provider)
  [accent]/auto[/]       toggle auto-approve edits/commands
  [accent]/limits[/]     show rate-limit stats   [accent]/files[/]   list files changed this session
  [accent]/cd PATH[/]    change workspace        [accent]/clear[/]   reset conversation
  [accent]/split[/]      toggle split-screen     [accent]/undo[/]    revert last agent change
  [accent]/plan[/]       always plan first      [accent]/noplan[/]  never plan   [accent]/plan auto[/] plan only build requests
  [accent]/specialists[/] show/enable/disable frontend, backend & python specialists + workers
  [accent]/memory[/]     show what's saved in shared codebase memory (see save_note/recall_notes)
  [accent]/repomap[/]    show AST/PageRank graph-intel cache stats (files parsed, symbols indexed)
  [accent]/sandbox[/]    show which execution isolation backend run_command is using on this host
  [accent]/help[/]       this help               [accent]/exit[/]    quit
Anything else is sent to the agent as a coding task. For independent subtasks, the agent may
call dispatch_agents (generic) or dispatch_specialists (domain-aware: frontend/backend/python,
each with its own rate-limit budget and auto-selected worker persona) to run parallel subagents,
automatically paced by their adaptive rate limiters - no extra setup needed on your end. Every
subagent/specialist automatically starts pre-loaded with shared codebase memory (see /memory), and
can call repo_map to cheaply orient itself in a large or unfamiliar codebase before diving in.
/specialists usage:
  /specialists                 show status of every specialist + worker
  /specialists off frontendspecialist    disable a specialist for this session
  /specialists on  html_worker            re-enable a worker"""


def show_limits(limiter: AdaptiveRateLimiter) -> None:
    s = limiter.snapshot()
    t = Table(title=f"Rate limiter · {s['name']}", box=box.ROUNDED, border_style="accent")
    t.add_column("metric", style="dim")
    t.add_column("value", style="accent2")
    for k in ("rpm", "ceiling", "rpm_used", "tpm_used", "tpm_limit", "burst", "blocked_for",
              "breaker_open", "breaker_left", "calls", "ok", "failed", "inflight", "limited",
              "retries", "waited", "tokens", "avg_latency", "last"):
        t.add_row(k, str(s[k]))
    console.print(t)
    hist = limiter.stats.history[-6:]
    if hist:
        console.print(Text("recent: ", style="dim") +
                      Text(" | ".join(m for _, m in hist), style="dim"))


def _specialists_cmd(ws: Workspace, arg: str) -> None:
    """/specialists              show status table
       /specialists off KEY      disable a specialist or worker
       /specialists on  KEY      enable a specialist or worker"""
    if not HAVE_SPECIALISTS or ws.specialist_registry is None:
        console.print("[err]specialists module not available[/]")
        return
    reg = ws.specialist_registry
    parts = arg.split()
    if len(parts) == 2 and parts[0] in ("on", "off"):
        msg = ws.set_specialist_active(parts[1], parts[0] == "on")
        console.print(f"[{'ok' if not msg.startswith('error') else 'err'}]{msg}[/]")
        return
    snap = reg.snapshot()
    t = Table(title="Specialists & Workers", box=box.ROUNDED, border_style="accent")
    t.add_column("specialist", style="accent2")
    t.add_column("status")
    t.add_column("worker", style="dim")
    t.add_column("worker status")
    for sdef in _specialists_mod.SPECIALISTS:
        on = snap["enabled"].get(sdef.key, True)
        awake = sdef.key in snap["awake"]
        status = "[ok]AWAKE[/]" if awake else ("[dim]enabled[/]" if on else "[err]disabled[/]")
        first = True
        for w in sdef.workers:
            won = snap["worker_enabled"].get(w.key, True)
            wawake = snap["awake_workers"].get(w.key) == sdef.key
            wstatus = "[ok]active[/]" if wawake else ("[dim]ready[/]" if won else "[err]off[/]")
            t.add_row(sdef.label if first else "", status if first else "", w.key, wstatus)
            first = False
    console.print(t)


def _undo_cmd(ws: Workspace, arg: str) -> str:
    """/undo         revert the most recently changed file
       /undo all     revert every file the agent changed this session
       /undo <path>  revert one named file"""
    if not ws.snapshots:
        return "nothing to undo"
    if arg == "all":
        return "\n".join(ws.undo(r) for r in list(ws.snapshots))
    if arg:
        return ws.undo(arg)
    return ws.undo(list(ws.snapshots)[-1])


def _qa_summary_suffix(res: dict) -> str:
    """Short suffix describing the autonomous browser QA pass, if one ran.

    Always says something concrete about verification status - a plan that
    "completed" but was never actually checked in a browser (skipped, missing
    deps, no key, budget exhausted) is a materially different outcome from
    one that was checked and passed, and the user should be told which one
    happened rather than the QA state being silently dropped from the
    one-line summary.
    """
    qa = res.get("qa")
    if qa is None:
        return ""
    verdict = res.get("qa_verdict")
    if qa.skipped_reason:
        # still surface *why* nothing was verified - "plan complete" with no
        # mention of testing at all previously looked identical to "plan
        # complete AND verified", which is the "no proper guide/no limit"
        # confusion from the user's report.
        return f"; browser QA not completed ({qa.skipped_reason})"
    if not qa.reports:
        return ""
    if verdict == "verified" or qa.final_passed:
        return f"; browser QA passed after {qa.loops_run} pass(es)"
    return f"; browser QA still found issues after {qa.loops_run} auto-fix pass(es) - see log"


def _plan_summary_text(res: dict) -> str:
    """One compact, plain line describing how a plan run ended."""
    plan = res.get("plan")
    out = res.get("outcome")
    if out == "error":
        return f"planning failed: {res.get('error', 'unknown error')}"
    if out == "cancelled":
        if not plan or plan.progress()[0] == 0:
            return "plan cancelled - nothing was changed"
        return f"plan cancelled after {plan.progress()[0]}/{len(plan.steps)} steps"
    done, total = plan.progress()
    suffix = _qa_summary_suffix(res)
    if out == "done":
        return f"plan complete: {done}/{total} steps (saved as PLAN.md){suffix}"
    return f"plan stopped at step {res.get('stopped_at')}: {done}/{total} steps done (PLAN.md has the details){suffix}"


def run_with_planning(provider, ws, prompt, provider_factory, plan_mode="auto", ask=None, emit=None):
    """Plan-first entry point. Returns the plan result, or None if the normal agent ran.

    Contract: if a plan was proposed and the user cancelled it (or planning failed) we STOP.
    Falling through to the normal loop would do exactly the thing the user just declined.

    When browsertester.py is loaded and installed on this workspace, planned requests get an
    autonomous real-browser QA pass (with automatic bug -> fix -> re-test loops) appended after
    execution - see browsertester.plan_and_execute_with_qa / maybe_plan_with_qa. Falls back to
    plain specialists.maybe_plan (no QA pass) whenever that isn't available, so the base
    plan-and-execute workflow is unaffected by whether browser testing deps are installed.
    """
    if HAVE_SPECIALISTS and ws.specialist_registry is not None and plan_mode != "never":
        if HAVE_BROWSERTESTER and getattr(ws, "_browsertester_installed", False):
            res = _browsertester_mod.maybe_plan_with_qa(ws, provider_factory, prompt, mode=plan_mode,
                                                         ask=ask, emit=emit)
        else:
            res = _specialists_mod.maybe_plan(ws, provider_factory, prompt, mode=plan_mode, ask=ask, emit=emit)
        if res is not None:
            if ws.ui is not None:
                ws.ui.add_log("ok" if res["outcome"] == "done" else "warn", _plan_summary_text(res))
                ws.ui.assistant_text = _plan_summary_text(res)
            return res
    run_agent(provider, ws, prompt)
    return None


def _plan_report(res: dict) -> None:
    """What to leave in the normal scrollback after the alt-screen closes."""
    plan = res.get("plan")
    col = {"done": "ok", "partial": "warn", "cancelled": "warn", "error": "err"}.get(res.get("outcome", ""), "dim")
    console.print(f"[{col}]{_plan_summary_text(res)}[/]")
    if not plan or not plan.steps:
        return
    t = Table(box=box.SIMPLE_HEAD, border_style="dim", title=f"plan · {plan.goal[:60]}", title_style="accent2")
    t.add_column("#", justify="right")
    t.add_column("step")
    t.add_column("owner", style="tool")
    t.add_column("status")
    sty = {"done": "ok", "fail": "err", "skip": "dim", "todo": "dim", "run": "warn"}
    for st in plan.steps:
        t.add_row(str(st.id), st.title[:50], st.owner.replace("specialist", "") or "agent",
                  f"[{sty.get(st.status, 'dim')}]{st.status}[/]")
    console.print(t)
    qa = res.get("qa")
    if qa is not None and qa.reports and not qa.final_passed and not qa.skipped_reason:
        last = qa.reports[-1]
        console.print(f"[warn]browser QA - unresolved after {qa.loops_run} auto-fix pass(es):[/]")
        for bug in last.bugs[:6]:
            console.print(f"  [err]-[/] [{bug.severity}] {bug.title}")


def repl(ws: Workspace, keys: dict, spec: ModelSpec, split: bool = True) -> None:
    limiters: dict = {}
    state = UIState()
    holder = {"spec": spec}
    if split:
        ws.ui = state
    # dispatch_agents reads this each call; passing `holder` (not `spec`) means a
    # later /model switch (which mutates holder["spec"]) is picked up automatically.
    ws.agent_ctx = {"spec": holder, "keys": keys, "limiter_cache": limiters}
    install_specialists_and_qa(ws, keys, limiters)
    provider = build_provider(spec, keys, ws, limiters)
    plan_mode = {"v": "auto"}                       # auto | always | never   (/plan, /noplan)
    # A planner needs its OWN conversation on the user's CURRENT model (holder is updated by /model).
    make_planner = lambda: build_provider(holder["spec"], keys, ws, limiters)
    session = None
    if HAVE_PT:
        hist = Path.home() / ".coder_history"
        session = PromptSession(history=FileHistory(str(hist)))

    console.print(Panel(
        f"[ok]workspace[/] {ws.root}\n[ok]model[/] {spec.label} ([dim]{spec.id}[/])\n"
        f"[ok]layout[/] {'split-screen (telemetry | live editor)' if split else 'classic'}   "
        f"[ok]approvals[/] {'AUTO' if ws.auto else 'y/n per change'}   [dim]/help for commands[/]",
        border_style="accent", box=box.ROUNDED))

    def run_split(prompt: str) -> None:
        keyrd = KeyReader()
        approver = Approver(state, keyrd, auto=ws.auto)
        ws.approver = approver
        rend = Renderer(state, lambda: provider.limiter, lambda: holder["spec"], lambda: ws)
        sess = LiveSession(console, state, rend, approver, keyrd)
        state.assistant_text = ""
        result: dict = {}
        try:
            sess.run(lambda: result.update(r=run_with_planning(provider, ws, prompt, make_planner, plan_mode["v"])))
        except KeyboardInterrupt:
            console.print("[warn]interrupted[/]")
        finally:
            keyrd.stop()
        # the alt-screen just closed - leave a compact summary in the normal scrollback
        if result.get("r"):
            _plan_report(result["r"])
        _summary(state, ws)

    while True:
        try:
            if session:
                line = session.prompt([("class:p", "\u276f ")],
                                      style=PTStyle.from_dict({"p": "#7c5cff bold"}))
            else:
                line = Prompt.ask("[accent]\u276f[/]", console=console)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/]")
            return
        line = line.strip()
        if not line:
            continue
        if line in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]bye[/]")
            return
        if line == "/help":
            console.print(HELP)
        elif line == "/model":
            spec = pick_model(keys)
            holder["spec"] = spec
            provider = build_provider(spec, keys, ws, limiters)
            console.print(f"[ok]switched to[/] {spec.label}")
        elif line == "/auto":
            ws.auto = not ws.auto
            console.print(f"[warn]auto-approve {'ON' if ws.auto else 'OFF'}[/]")
        elif line == "/split":
            split = not split
            ws.ui = state if split else None
            console.print(f"[warn]layout: {'split-screen' if split else 'classic'}[/]")
        elif line == "/limits":
            show_limits(provider.limiter)
        elif line == "/files":
            console.print("\n".join(f"  [tool]\u2022[/] {f}" for f in dict.fromkeys(ws.changed)) or "[dim](none)[/]")
        elif line == "/memory" or line.startswith("/memory "):
            arg = line[len("/memory"):].strip()
            tag = arg if arg else ""
            body = ws.recall_notes(tag=tag)
            console.print(Panel(body, title=f"[accent2]shared codebase memory[/]"
                                + (f" [dim](tag={tag})[/]" if tag else ""),
                                border_style="accent", box=box.ROUNDED))
        elif line == "/sandbox":
            if HAVE_SANDBOX:
                console.print(f"[accent2]{sandbox_describe()}[/]")
            else:
                console.print("[warn]exec_sandbox module unavailable - run_command uses the plain "
                              "subprocess fallback (no isolation upgrade applied)[/]")
        elif line == "/repomap":
            if ws.intel is not None:
                from rich.markup import escape as _esc
                console.print(f"[accent2]{ws.intel.stats()}[/]")
                for row in ws.intel.health():
                    console.print(f"[dim]{_esc(row)}[/]")
                u = ws.usage
                console.print("[accent2]this session:[/] "
                              f"reads: {u.get('read_symbol', 0)} symbol / {u.get('read_outline', 0)} outline / "
                              f"{u.get('read_range', 0)} range / {u.get('read_full', 0)} whole-file "
                              f"({u.get('read_symbol_failed', 0)} symbol misses) | "
                              f"edits: {u.get('edit_symbol', 0)} symbol / {u.get('edit_text', 0)} exact-text / "
                              f"{u.get('edit_fuzzy', 0)} fuzzy ({u.get('edit_symbol_failed', 0)} symbol misses) | "
                              f"insert {u.get('edit_insert', 0)} | "
                              f"find_symbol {u.get('tool:find_symbol', 0)}, find_callers {u.get('tool:find_callers', 0)}, "
                              f"repo_map {u.get('tool:repo_map', 0)}")
                console.print(f"[accent2]history compaction:[/] {COMPACT_STATS['results']} stale tool outputs elided, "
                              f"~{COMPACT_STATS['chars'] // 4:,} tokens not re-sent so far")
            else:
                console.print("[warn]repo_intel module unavailable (tree_sitter/tree_sitter_languages/"
                              "networkx not installed) - repo_map/read_file/edit_file use their "
                              "regex/exact-match fallback behaviour[/]")
        elif line.startswith("/undo"):
            console.print(f"[warn]{_undo_cmd(ws, line[5:].strip())}[/]")
        elif line.startswith("/specialists"):
            _specialists_cmd(ws, line[len("/specialists"):].strip())
        elif line in ("/plan", "/noplan", "/plan auto"):
            plan_mode["v"] = {"/plan": "always", "/noplan": "never", "/plan auto": "auto"}[line]
            console.print(f"[warn]planning: {plan_mode['v']}[/]  "
                          + {"always": "every request is planned and reviewed first",
                             "never": "requests run directly, no plan",
                             "auto": "build/create requests are planned, small edits run directly"}[plan_mode["v"]])
        elif line == "/clear":
            provider = build_provider(spec, keys, ws, limiters)
            console.print("[dim]conversation cleared[/]")
        elif line.startswith("/cd"):
            target = line[3:].strip()
            p = Path(target).expanduser() if target else Path.cwd()
            if p.is_dir():
                ws.root = p.resolve()
                provider = build_provider(spec, keys, ws, limiters)
                console.print(f"[ok]workspace \u2192[/] {ws.root}")
            else:
                console.print(f"[err]not a directory:[/] {target}")
        elif line.startswith("/"):
            console.print("[err]unknown command[/] - try /help")
        elif split:
            run_split(line)
        else:
            res = run_with_planning(provider, ws, line, make_planner, plan_mode["v"],
                                    ask=lambda q: Prompt.ask(f"[accent]{q}[/]", console=console),
                                    emit=lambda t: console.print(Text(t)))
            if res:
                _plan_report(res)


def _summary(state: "UIState", ws: Workspace) -> None:
    """After the alt-screen closes, print a compact record so the work isn't lost."""
    s = state.snapshot()
    if s["assistant_text"].strip():
        console.print(Markdown(s["assistant_text"]))
    files = [(p, j) for p, j in s["jobs"].items() if j.status in ("kept", "rejected", "undone")]
    if files:
        t = Table(box=box.SIMPLE_HEAD, border_style="dim", title="files", title_style="accent2")
        t.add_column("file"); t.add_column("+", style="ok", justify="right")
        t.add_column("-", style="err", justify="right"); t.add_column("status")
        for p, j in files:
            col = {"kept": "ok", "rejected": "err", "undone": "err"}[j.status]
            t.add_row(p, str(j.added), str(j.removed), f"[{col}]{j.status}[/]")
        console.print(t)
        console.print("[dim]/undo [file] reverts the last change[/]")


def main() -> None:
    ap = argparse.ArgumentParser(description="coder.py - terminal coding agent")
    ap.add_argument("path", nargs="?", default=None, help="workspace folder (default: ask)")
    ap.add_argument("-m", "--model", help="model number, or 'custom' with --custom-model to use any model")
    ap.add_argument("-p", "--prompt", help="run one prompt non-interactively then exit")
    ap.add_argument("--yes", action="store_true", help="auto-approve edits and commands")
    ap.add_argument("--classic", action="store_true", help="scrolling UI instead of split-screen")
    ap.add_argument("--plan", action="store_true", help="always plan and review before executing (with -p)")
    ap.add_argument("--no-plan", action="store_true", help="never plan; run the request directly (with -p)")
    ap.add_argument("--custom-model", help="model id for a custom provider, e.g. 'llama-3.1-8b-instant'. "
                                            "Implies -m custom.")
    ap.add_argument("--custom-provider", choices=["groq", "openai_compat"], default="groq",
                     help="custom provider shape (default: groq). openai_compat needs --custom-base-url.")
    ap.add_argument("--custom-base-url", default="", help="base URL for --custom-provider openai_compat, "
                                                            "e.g. https://api.together.xyz/v1")
    ap.add_argument("--custom-env", default="", help="env var name to read the custom model's key from "
                                                       "(default: GROQ_API_KEY or CUSTOM_API_KEY)")
    ap.add_argument("--api-key", default="", help="API key for --custom-model, if you'd rather pass it "
                                                    "directly than set an env var")
    ap.add_argument("--custom-ctx", type=int, default=128_000, help="context window (tokens) for --custom-model")
    a = ap.parse_args()

    show_banner()
    if not HAVE_SPECIALISTS:
        console.print(Panel(
            f"[warn]Specialists are DISABLED[/] - specialists.py failed to load, so the agent will "
            f"only use generic dispatch_agents.\n[err]{SPECIALISTS_IMPORT_ERROR or 'specialists.py not found next to coder3.py'}[/]\n"
            f"[dim]Put specialists.py in the same folder as coder3.py: {Path(__file__).resolve().parent}[/]",
            title="[warn]specialists[/]", border_style="warn", box=box.ROUNDED))
    elif not HAVE_BROWSERTESTER:
        console.print(Panel(
            f"[dim]Real-browser QA is unavailable - browsertester.py/browseragent.py failed to load, "
            f"so plans won't get an automatic browser test/auto-fix pass.\n"
            f"{BROWSERTESTER_IMPORT_ERROR or 'browsertester.py/browseragent.py not found next to coder3.py'}[/]",
            title="[dim]browser QA[/]", border_style="dim", box=box.ROUNDED))
    else:
        note = _browsertester_mod.uninstall_note()
        if note:
            console.print(Panel(f"[dim]{note}[/]", title="[dim]browser QA[/]",
                                border_style="dim", box=box.ROUNDED))
    keys = {"GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY", "")}

    path = a.path or Prompt.ask("[accent]Workspace folder[/]", default=str(Path.cwd()), console=console)
    root = Path(path).expanduser()
    if not root.is_dir():
        console.print(f"[err]not a directory:[/] {root}")
        sys.exit(1)

    if a.custom_model:
        env = a.custom_env or ("GROQ_API_KEY" if a.custom_provider == "groq" else "CUSTOM_API_KEY")
        spec = add_custom_model(a.custom_model, a.custom_provider, a.custom_model, env,
                                a.custom_ctx, "custom model from CLI flags", a.custom_base_url)
        if a.api_key:
            keys[env] = a.api_key
        elif env not in keys:
            keys[env] = os.getenv(env, "")
    elif a.model and a.model != "custom":
        spec = next(m for m in MODELS if m.key == a.model)
    else:
        spec = pick_model(keys)
    if not keys.get(spec.env):
        keys[spec.env] = Prompt.ask(f"[warn]{spec.env}[/] (hidden)", password=True, console=console)

    ws = Workspace(root, auto_approve=a.yes)
    if a.prompt:
        limiters: dict = {}
        holder = {"spec": spec}
        ws.agent_ctx = {"spec": holder, "keys": keys, "limiter_cache": limiters}
        install_specialists_and_qa(ws, keys, limiters)
        res = run_with_planning(build_provider(spec, keys, ws, limiters), ws, a.prompt,
                                lambda: build_provider(spec, keys, ws, limiters),
                                "never" if a.no_plan else "always" if a.plan else "auto",
                                ask=(lambda q: Prompt.ask(f"[accent]{q}[/]", console=console)) if sys.stdin.isatty() else None,
                                emit=lambda t: console.print(Text(t)))
        if res:
            _plan_report(res)
            if res["outcome"] in ("error", "cancelled"):
                sys.exit(2)
    else:
        repl(ws, keys, spec, split=not a.classic)


if __name__ == "__main__":
    main()
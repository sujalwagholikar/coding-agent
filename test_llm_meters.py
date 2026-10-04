"""Regression tests for the 'stuck at N/N requests + frozen token meter' bug."""
import sys, time, threading, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import coder3 as c

FAST = c.RateProfile("t", rpm=600, tpm=250_000, burst=50, floor_rpm=60,
                     max_retries=2, base_backoff=0.05, max_backoff=0.2)
passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond: passed += 1; print("  ok  ", name)
    else:    failed += 1; print("  FAIL", name)

def run_with_deadline(fn, secs):
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", fn()), daemon=True); t.start(); t.join(secs)
    return ("r" in box), box.get("r")

print("1. token-bucket deadlock")
lim = c.AdaptiveRateLimiter(c.PROFILES["gemini-3.5-flash"])
done, _ = run_with_deadline(lambda: lim.acquire(est_tokens=900_000), 3)   # >> 62.5k capacity
check("acquire() with estimate far above bucket capacity returns", done)
w = lim._tok.wait_time(min(900_000, lim._tok.capacity))                   # bucket now ~empty
check(f"next giant request is PACED with a finite wait ({w:.1f}s <= full refill {lim._tok.capacity/lim._tok.rate:.0f}s), not infinite",
      0 < w <= lim._tok.capacity / lim._tok.rate + 0.5)

print("2. token meter always moves")
lim = c.AdaptiveRateLimiter(FAST)
c.call_with_limits(lim, lambda: "x", est_tokens=1234, tokens_from=lambda r: 0)       # provider sent no usage
s = lim.snapshot()
check("no provider usage -> estimate used, meter still moves", s["tokens"] == 1234 and s["tokens_estimated"] == 1234)
c.call_with_limits(lim, lambda: "x", est_tokens=10, tokens_from=lambda r: 777)
check("provider usage is added exactly", lim.snapshot()["tokens"] == 1234 + 777)
check("calls==ok==2, none failed", (lambda s: s["calls"] == s["ok"] == 2 and s["failed"] == 0)(lim.snapshot()))

print("3. in-flight request is visible while it runs (counter no longer looks frozen)")
lim = c.AdaptiveRateLimiter(FAST); seen = {}
def slow():
    time.sleep(0.6); return 1
th = threading.Thread(target=lambda: c.call_with_limits(lim, slow, est_tokens=5000, tokens_from=lambda r: 4000), daemon=True)
th.start(); time.sleep(0.3); mid = lim.snapshot(); th.join(3); end = lim.snapshot()
check("mid-call: inflight==1, calls=1, ok=0, ~tokens shown", mid["inflight"] == 1 and mid["calls"] == 1 and mid["ok"] == 0 and mid["inflight_tokens"] == 5000)
check("mid-call: live tok/min already counts the in-flight estimate", mid["tpm_used"] >= 5000)
check("after: inflight==0, ok==1, tokens==4000", end["inflight"] == 0 and end["ok"] == 1 and end["tokens"] == 4000)
check("live rpm counts the request", end["rpm_used"] == 1)

print("4. hung SDK call can't hang the agent")
lim = c.AdaptiveRateLimiter(FAST)
def hang(): time.sleep(30)
t0 = time.time()
def go():
    try: c.call_with_limits(lim, hang, est_tokens=100, timeout=0.4)
    except Exception as e: return type(e).__name__
done, name = run_with_deadline(go, 8)
check("hung call is cut off and retried, then raised (LLMTimeout) instead of freezing", done and name == "LLMTimeout")
s = lim.snapshot()
check("retries recorded, failed==1, inflight cleared", s["retries"] >= 2 and s["failed"] == 1 and s["inflight"] == 0)
check("failed attempts refunded their token pre-charge (no phantom drain)", lim._tok.tokens > lim._tok.capacity - 1)

print("5. timeouts / dropped sockets classified as retryable")
class ReadTimeout(Exception): pass
class RemoteProtocolError(Exception): pass
check("timeout -> transient 504", c.parse_error(ReadTimeout("boom")).is_transient)
check("dropped connection -> transient", c.parse_error(RemoteProtocolError("Server disconnected")).is_transient)
check("auth error still NOT retried", not c.parse_error(type("E", (Exception,), {"status_code": 401})("bad key")).is_transient)

print("6. burst meter refreshes without any request")
lim = c.AdaptiveRateLimiter(c.RateProfile("b", rpm=600, tpm=250_000, burst=10, floor_rpm=60))
for _ in range(10): lim.acquire(est_tokens=10)
a = lim.snapshot()["req_tokens"]; time.sleep(0.6); b = lim.snapshot()["req_tokens"]
check(f"burst bucket refills in snapshot ({a} -> {b})", b > a)

print("7. parallel workers keep separate pre-charge (thread-local)")
lim = c.AdaptiveRateLimiter(FAST); errs = []
def worker(n):
    lim.acquire(est_tokens=n); time.sleep(0.05); lim.on_success(n // 2)   # each really used HALF its estimate
ths = [threading.Thread(target=worker, args=(n,)) for n in (1000, 2000, 3000, 4000)]
[t.start() for t in ths]; [t.join() for t in ths]
lim._tok._refill()
exp = lim._tok.capacity - 5000          # (1000+2000+3000+4000)/2 really consumed
check(f"each thread refunded ITS OWN over-estimate (bucket {lim._tok.tokens:.0f} ~ {exp:.0f})", abs(lim._tok.tokens - exp) < 900)

print("8. combined snapshot sums main + specialist limiters")
m = c.AdaptiveRateLimiter(FAST); sp = c.AdaptiveRateLimiter(FAST)
c.call_with_limits(m, lambda: 1, est_tokens=1, tokens_from=lambda r: 100)
for _ in range(3): c.call_with_limits(sp, lambda: 1, est_tokens=1, tokens_from=lambda r: 50)
cs = c.combined_snapshot(m, {"__specialist__x": sp, "dup": m})
check("ok/calls/tokens summed, main not double-counted", cs["ok"] == 4 and cs["calls"] == 4 and cs["tokens"] == 250)

print(f"\n{passed} passed, {failed} failed"); sys.exit(1 if failed else 0)

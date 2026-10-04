# Fix: LLM request counter stuck at N/N + frozen token meter

## Root causes (all in coder3.py)
1. **Deadlock in `AdaptiveRateLimiter.acquire()`** - the token bucket holds at most `tpm*0.25`
   (62.5k tokens for Gemini Flash) but the request cost was the raw history estimate. Once the
   conversation grew past that, `wait_time()` could never reach 0 -> infinite loop -> counter frozen
   at N/N and tokens frozen. (Reproduced; cost is now clamped to bucket capacity.)
2. **Meters read stale/static data**: `snapshot()` never refilled the burst bucket; the "rpm" bar
   showed the adaptive *limit* not usage; the UI only read the MAIN limiter while every specialist /
   subagent worker has its own (so the meters froze whenever specialists worked); the non-split HUD
   handed `Live` a static object.
3. **No hard timeout on Gemini calls** (OpenAI path had 180s); timeouts / dropped sockets were not
   classified as retryable, so they either hung or killed the run.
4. **Token pre-charge leaked**: failed attempts never refunded their estimate, and the pre-charge was
   one shared field clobbered by parallel workers.
5. Tokens stayed at 0 whenever a provider returned no usage block.

## What changed
- `acquire()`: clamped cost, abortable (Ctrl-C works while waiting), per-thread pre-charge.
- `call_with_limits()`: watchdog thread (`LLM_CALL_GUARD`=240s), refund on failure, in-flight tracking,
  `failed` counter. Gemini client now has `HttpOptions(timeout=180s)`.
- `parse_error()`: timeouts -> 504, dropped connections -> 503 (retried with backoff).
- New live metrics: requests in flight (count/age/est tokens), real rpm + tokens/min over a rolling 60s,
  latency, failed count, token fallback to estimate (shown with `~`).
- `combined_snapshot()` aggregates main + all specialist/subagent limiters for the UI (both HUDs, /limits).
- New tests: `python test_llm_meters.py` (18 checks). Existing suites unchanged and passing.

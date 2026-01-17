# Slow Test Investigation Report

## Executive Summary

The test suite (`make test`) takes approximately **3-4 minutes** to run. This investigation identified the root causes and provides recommendations for significant speedups.

## Timing Breakdown

From pytest durations analysis (top 15 slowest tests):

| Test | Duration | Root Cause |
|------|----------|------------|
| `test_eval_loop_never_dlq_for_excluded_error` | **60.08s** | Visibility backoff timeout |
| `test_end_to_end_evaluation` | **40.01s** | Visibility backoff timeout |
| `test_eval_loop_handles_expired_receipt_on_nack` | **20.01s** | Long poll wait_time_seconds |
| `test_round_trip_with_nested_collections` | **6-8s** | Hypothesis generation |
| `test_main_loop_shutdown_timeout_returns_false` | **5.00s** | Intentional timeout test |
| Various lifecycle/health tests | **0.5-1.2s each** | time.sleep waits |

**Total impact of top 3 tests alone: ~120 seconds (50% of test runtime)**

## Root Cause Analysis

### 1. EvalLoop Visibility Timeout Backoff (Critical - 120s)

**Location**: `src/weakincentives/evals/_loop.py:355`

```python
def _handle_failure(self, msg, error):
    # ...
    backoff = min(60 * msg.delivery_count, 900)  # 60 SECONDS per retry!
    msg.nack(visibility_timeout=backoff)
```

When a message fails and is nacked, the test waits for the visibility timeout to expire before the message becomes available again. The backoff formula uses **60 seconds per delivery count**.

**Affected tests**:
- `test_eval_loop_never_dlq_for_excluded_error`: 5 iterations × 12s avg = 60s
- `test_end_to_end_evaluation`: Multiple samples × wait_time_seconds
- `test_eval_loop_handles_expired_receipt_on_nack`: 20s long poll timeout

### 2. InMemoryMailbox Reaper Interval

**Location**: `src/weakincentives/runtime/mailbox/_in_memory.py:172`

```python
def _reaper_loop(self) -> None:
    while not self._stop_reaper.wait(timeout=0.1):  # 100ms polling
        self._reap_expired()
```

The reaper checks every 100ms for expired messages. Combined with visibility timeouts, this adds latency.

### 3. time.sleep() Anti-patterns

Found **68 instances** of `time.sleep` across test files:

| File | Count | Typical Values |
|------|-------|----------------|
| `test_lifecycle.py` | 28 | 0.01-0.15s |
| `test_redis_mailbox_invariants.py` | 15 | 0.1-1.5s |
| `test_watchdog.py` | 10 | 0.1-0.2s |
| `test_mailbox.py` | 6 | 0.1-1.2s |
| `test_dlq.py` | 1 | 0.1s |
| `test_lease_extender.py` | 2 | 0.1-0.25s |
| Others | 6 | Various |

**Common anti-patterns**:
- Waiting for threads to start: `time.sleep(0.05)` before assertions
- Waiting for async operations: `time.sleep(0.1)` after state changes
- Waiting for timeouts: `time.sleep(0.15)` for heartbeat stall detection

### 4. Long Poll Wait Times

**Location**: `src/weakincentives/evals/_loop.py:148`

```python
def run(self, *, wait_time_seconds: int = 20, ...):
```

Tests that don't override `wait_time_seconds` may block for up to 20 seconds on empty queues.

## Recommendations

### High Impact (Would save ~120s)

#### 1. Make EvalLoop backoff configurable for tests

```python
# In EvalLoopConfig
backoff_base: int = 60  # Seconds per retry (default for production)

# In tests
config = EvalLoopConfig(lease_extender=None, backoff_base=0)
eval_loop = EvalLoop(loop=main_loop, evaluator=evaluator, config=config)
```

#### 2. Add test-mode to InMemoryMailbox with instant visibility

```python
class InMemoryMailbox:
    def __init__(self, name: str, *, test_mode: bool = False):
        self._test_mode = test_mode

    def _nack(self, receipt_handle: str, visibility_timeout: int) -> None:
        if self._test_mode:
            visibility_timeout = 0  # Instant requeue in tests
        # ... rest of implementation
```

#### 3. Override wait_time_seconds in slow tests

```python
# Instead of default 20s wait
eval_loop.run(max_iterations=1, wait_time_seconds=0)
```

### Medium Impact (Would save ~15-30s)

#### 4. Replace time.sleep() with event-based synchronization

```python
# Before (anti-pattern)
thread.start()
time.sleep(0.05)  # Hope thread started
assert loop.running

# After (proper synchronization)
started = threading.Event()
def run_with_signal():
    started.set()
    loop.run()

thread = threading.Thread(target=run_with_signal)
thread.start()
assert started.wait(timeout=1.0)
assert loop.running
```

#### 5. Use mock time for heartbeat/watchdog tests

```python
# Using freezegun or unittest.mock
with patch('time.monotonic') as mock_time:
    mock_time.return_value = 0.0
    hb = Heartbeat()

    mock_time.return_value = 0.15  # Simulate 150ms passing
    assert hb.elapsed() >= 0.1
```

#### 6. Use condition variables instead of polling

```python
# In InMemoryMailbox
def _reap_expired(self) -> None:
    # Instead of fixed 0.1s polling, wake on nack with timeout
    with self._lock:
        if self._pending_nacks:
            self._condition.wait(timeout=min_visibility_timeout)
```

### Low Impact (Best practices)

#### 7. Use pytest-timeout to catch slow tests early

```toml
[tool.pytest.ini_options]
timeout = 5  # Fail tests that take > 5 seconds
```

#### 8. Mark intentionally slow tests

```python
@pytest.mark.slow
def test_main_loop_shutdown_timeout_returns_false():
    """This test intentionally waits for a timeout."""
    ...
```

```bash
# Run fast tests only
pytest -m "not slow"
```

## Implementation Priority

1. **Immediate** (would cut test time in half):
   - Fix `test_eval_loop_never_dlq_for_excluded_error` visibility timeout
   - Fix `test_end_to_end_evaluation` visibility timeout
   - Fix `test_eval_loop_handles_expired_receipt_on_nack` wait_time

2. **Short-term**:
   - Add `test_mode` to InMemoryMailbox
   - Replace `time.sleep(0.05)` patterns with Event synchronization

3. **Medium-term**:
   - Add configurable backoff to EvalLoopConfig
   - Use mock time for watchdog/heartbeat tests

## Files Requiring Refactoring

Priority order based on impact:

1. `tests/evals/test_loop.py` - 3 critical slow tests
2. `tests/runtime/test_lifecycle.py` - 28 sleep calls
3. `tests/runtime/test_mailbox.py` - visibility timeout waits
4. `tests/runtime/test_watchdog.py` - heartbeat timing tests
5. `tests/contrib/mailbox/test_redis_mailbox_*.py` - Redis-specific timing

## Appendix: All time.sleep Locations

```
tests/runtime/test_dlq.py:778
tests/runtime/test_lease_extender.py:88,174
tests/adapters/test_throttling.py:45,311,313
tests/contrib/mailbox/test_redis_mailbox_stress.py:113,167,210
tests/test_thread_safety.py:303
tests/contrib/mailbox/test_redis_mailbox_ttl.py:327,365
tests/runtime/test_mailbox.py:334,390,456,875,966
tests/contrib/mailbox/test_redis_mailbox_invariants.py:58,81,122,198,290,307,314,352,367,371,413,450,461
tests/cli/test_wink_debug_app.py:285
tests/runtime/test_lifecycle.py:99,181,183,397,417,435,457,486,519,552,581,607,634,939,980,1041,1078,1379,1411,1438,1457,1518,1546,1583,1615,1619
tests/contrib/mailbox/test_redis_mailbox_properties.py:338
tests/runtime/test_watchdog.py:41,49,98,111,130,148,376,403
tests/adapters/claude_agent_sdk/test_log_aggregator.py:825,875,886,919,997,1025
```

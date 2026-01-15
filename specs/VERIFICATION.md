# Redis Mailbox Verification Specification

Formal verification framework for Redis mailbox implementation.

**Source:** `src/weakincentives/contrib/mailbox/_redis.py`

## Overview

Two complementary verification layers:

1. **Embedded TLA+ Specification**: Formal model via `@formal_spec` decorator, checked by TLC
2. **Property-Based Testing**: Hypothesis stateful tests against real implementation

## Key Invariants

| ID | Name | Description |
|----|------|-------------|
| INV-1 | State Exclusivity | Message in exactly one state: pending ⊕ invisible ⊕ deleted |
| INV-2 | Handle Freshness | Each delivery generates unique receipt handle |
| INV-3 | Stale Handle Rejection | Operations with old handles must fail |
| INV-4 | Delivery Count Monotonicity | Delivery count never decreases |
| INV-5 | No Message Loss | Every sent message eventually acknowledged or remains queued |
| INV-6 | Visibility Timeout | Expired invisible messages return to pending |
| INV-7 | Handle Uniqueness | Each delivery gets unique handle |
| INV-8 | Pending No Duplicates | No duplicate message IDs in pending queue |
| INV-9 | Data Integrity | Every queued message has associated data |

## TLA+ Specification

Embedded via `@formal_spec` decorator:

```python
@formal_spec(
    module="RedisMailbox",
    state_vars=[StateVar("pending", ...), StateVar("invisible", ...), ...],
    actions=[Action("Send", ...), Action("Receive", ...), ...],
    invariants=[Invariant("INV-1", "StateExclusivity", ...), ...],
    constants={"MaxMessages": 3, "MaxDeliveries": 3},
)
class RedisMailbox[T, R]: ...
```

### State Space

| Variable | Purpose |
|----------|---------|
| `pending` | Message IDs in pending list (FIFO) |
| `invisible` | Messages held by consumers |
| `data` | Message bodies |
| `handles` | Current valid receipt handle per message |
| `deleted` | Acknowledged message IDs |
| `deliveryCounts` | Delivery count per message |

### Verification

```bash
make verify-mailbox  # Extract TLA+ and run TLC
```

## Property-Based Testing

Hypothesis stateful tests verify actual implementation:

```python
class RedisMailboxStateMachine:
    """Random operation sequences with invariant checks."""

def check_inv_1_exclusivity(model, redis_state):
    for msg_id in model.data:
        states = [s for s in ["pending", "invisible", "deleted"] if msg_id in getattr(model, s)]
        assert len(states) == 1
```

### Running Tests

```bash
make test-redis-properties  # Full property suite
pytest tests/contrib/mailbox/test_redis_mailbox_properties.py -k smoke
```

## Redis Guarantees Assumed

- Lua script atomicity (no interleaving)
- LIST/HASH/ZSET operations atomic
- Monotonic timestamps from Redis TIME

## Verification Checklist

Before merging `RedisMailbox` changes:

- [ ] `@formal_spec` updated to match Lua script changes
- [ ] `make verify-mailbox` passes
- [ ] `make test-redis-properties` passes
- [ ] Regression tests for discovered bugs

## Limitations

- TLC uses bounded model checking (MaxMessages=3)
- TLC cannot verify liveness properties (covered by Hypothesis)
- Single Redis instance assumed (cluster requires slot locality)

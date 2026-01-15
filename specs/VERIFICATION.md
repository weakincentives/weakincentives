# Redis Mailbox Formal Verification Specification

## Purpose

This specification defines a formal verification framework for the Redis mailbox
implementation. The Redis mailbox uses complex algorithms with Lua scripts for
atomic multi-key operations, visibility timeout management, and receipt handle
validation. Bugs in these algorithms can lead to critical data loss or duplicate
processing.

The verification framework provides two complementary layers:

1. **Embedded TLA+ Specification**: A formal model embedded directly in the Python
   implementation using the `@formal_spec` decorator. The spec is extracted and
   exhaustively checked by the TLC model checker for safety invariants.

2. **Property-Based Testing**: Hypothesis-based stateful tests that verify the
   actual Python implementation against the same invariants, and also verify
   liveness properties (e.g., eventual requeue) that TLC cannot check.

Together, these layers provide high confidence that:

- The algorithm design is correct (TLA+ extracted from `@formal_spec`)
- The implementation matches the design (Hypothesis)
- The spec and code cannot drift (co-located in the same file)

## Scope

This specification covers verification of `RedisMailbox` in
`src/weakincentives/contrib/mailbox/_redis.py`. The following operations are
modeled:

| Operation | Lua Script | Verification Priority |
| ----------------- | ------------------ | --------------------- |
| `send()` | `_LUA_SEND` | Medium |
| `receive()` | `_LUA_RECEIVE` | Critical |
| `acknowledge()` | `_LUA_ACKNOWLEDGE` | Critical |
| `nack()` | `_LUA_NACK` | High |
| `extend()` | `_LUA_EXTEND` | High |
| `_reap_expired()` | `_LUA_REAP` | Critical |

## Key Invariants

These invariants must hold in all reachable states:

### INV-1: Message State Exclusivity

A message must be in exactly one of three states:

```
∀ msg ∈ Messages:
    (msg ∈ Pending ⊕ msg ∈ Invisible ⊕ msg ∈ Deleted)
```

**Violation scenario**: Race between `receive()` and `_reap_expired()` could
leave a message in both pending and invisible sets.

**Implementation guarantee**: Lua script atomicity ensures RPOP + ZADD execute
as a single operation.

### INV-2: Receipt Handle Freshness

Each delivery of a message generates a unique receipt handle. Old handles become
invalid after redelivery:

```
∀ msg ∈ Invisible:
    ∀ d1, d2 ∈ DeliveryHistory(msg):
        d1.seq ≠ d2.seq ⟹ d1.handle ≠ d2.handle
```

**Violation scenario**: If receipt handles were reused or predictable, a slow
consumer could acknowledge a message that was already redelivered to another
consumer.

**Implementation guarantee**: Each `receive()` generates a new UUID suffix.
`_LUA_REAP` deletes the old handle before requeueing.

### INV-3: Stale Handle Rejection

Operations with stale receipt handles must fail:

```
∀ msg ∈ Invisible:
    LET current = CurrentHandle(msg)
    IN ∀ stale ∈ PreviousHandles(msg):
        Acknowledge(msg, stale) = FAILURE ∧
        Nack(msg, stale) = FAILURE ∧
        Extend(msg, stale) = FAILURE
```

**Violation scenario**: If stale handles were accepted, duplicate processing
could occur.

**Implementation guarantee**: Lua scripts compare `HGET ... :handle` with the
provided suffix before proceeding.

### INV-4: Delivery Count Monotonicity

Delivery count never decreases for a given message:

```
∀ msg ∈ Messages:
    ∀ d1, d2 ∈ DeliveryHistory(msg):
        d1.seq < d2.seq ⟹ d1.count < d2.count
```

**Implementation guarantee**: `HINCRBY` in `_LUA_RECEIVE` atomically increments.

### INV-5: No Message Loss

Every sent message is eventually acknowledged or remains in the queue:

```
∀ msg ∈ SentMessages:
    ◇(msg ∈ Acknowledged ∨ msg ∈ Pending ∨ msg ∈ Invisible)
```

This is a liveness property requiring fairness assumptions about consumers.

### INV-6: Visibility Timeout Correctness

Messages with expired visibility timeout eventually return to pending:

```
∀ msg ∈ Invisible:
    msg.expiresAt < Now ⟹ ◇(msg ∈ Pending)
```

**Implementation guarantee**: Background reaper thread runs `_LUA_REAP` every
`reaper_interval` seconds.

### INV-7: Handle Uniqueness

Each delivery of a message gets a unique receipt handle:

```
∀ msg ∈ Messages:
    ∀ d1, d2 ∈ DeliveryHistory(msg):
        d1.seq ≠ d2.seq ⟹ d1.handle ≠ d2.handle
```

**Violation scenario**: If handles were reused, a consumer could accidentally
acknowledge a message that was redelivered to another consumer.

**Implementation guarantee**: Each `receive()` generates a new UUID suffix stored
in the `:meta` hash. Handle validation compares against the current suffix.

### INV-8: Pending No Duplicates

The pending queue contains no duplicate message IDs:

```
∀ i, j ∈ 1..Len(pending):
    i ≠ j ⟹ pending[i] ≠ pending[j]
```

**Violation scenario**: A bug in `Nack` or `ReapOne` could accidentally add a
message ID to pending without first removing it, resulting in duplicate
deliveries of the same message from distinct queue positions.

**Implementation guarantee**: Lua scripts always remove from one state before
adding to another. `_LUA_NACK` calls `ZREM` on invisible before `LPUSH` to
pending. `_LUA_REAP` similarly removes before requeueing.

### INV-9: Data Integrity

Every message in pending or invisible has associated data:

```
∀ msgId ∈ Messages:
    (msgId ∈ Pending ∨ msgId ∈ Invisible) ⟹ msgId ∈ DOMAIN data
```

**Violation scenario**: A partial write could add a message ID to pending
without storing its body in the data hash, or a corruption could delete data
while the message is still queued.

**Implementation guarantee**: `_LUA_SEND` atomically stores data with `HSET`
before adding the message ID to pending with `LPUSH`. Data is only removed
by `_LUA_ACKNOWLEDGE` after the message is removed from invisible. This
invariant complements INV-5 (NoMessageLoss) which checks the reverse direction.

### FIFO Ordering (Structural Property)

Messages are delivered in send order (within visibility constraints). This is a
structural property of the Redis LIST data structure, not a numbered invariant:

```
∀ m1, m2 ∈ Messages:
    SendTime(m1) < SendTime(m2) ∧
    m1 ∈ Pending ∧ m2 ∈ Pending ⟹
        ReceiveTime(m1) < ReceiveTime(m2)
```

**Implementation guarantee**: Redis LIST with LPUSH/RPOP provides FIFO.

## TLA+ Specification

### Embedded Specification

The TLA+ specification is embedded directly in the `RedisMailbox` class using
the `@formal_spec` decorator (see `specs/FORMAL_VERIFICATION.md` for complete
documentation on the decorator system).

```python
# src/weakincentives/contrib/mailbox/_redis.py

from weakincentives.formal import formal_spec, StateVar, Action, Invariant

@formal_spec(
    module="RedisMailbox",
    extends=("Integers", "Sequences", "FiniteSets", "TLC"),
    constants={"MaxMessages": 3, "MaxDeliveries": 3, ...},
    state_vars=[StateVar("pending", ...), ...],
    actions=[Action("Send", ...), ...],
    invariants=[Invariant("INV-1", ...), ...],
)
class RedisMailbox[T, R]:
    ...
```

The decorator generates extracted TLA+ files:

```
specs/tla/extracted/
├── RedisMailbox.tla           # Generated TLA+ module
└── RedisMailbox.cfg           # Generated TLC config
```

### State Space

The TLA+ model tracks:

| State Variable | Purpose |
| --------------- | ----------------------------------------------- |
| `pending` | Sequence of message IDs in pending list (FIFO) |
| `invisible` | Messages currently held by consumers |
| `data` | Message bodies (persists until acknowledged) |
| `handles` | Current valid receipt handle per message |
| `deleted` | Set of acknowledged message IDs |
| `deliveryCounts` | Delivery count per message (monotonic) |
| `deliveryHistory` | Delivery history for INV-4 verification |
| `now` | Abstract time counter for timeout modeling |

See `src/weakincentives/contrib/mailbox/_redis.py` for the complete embedded
specification.

### Actions

The TLA+ model includes actions for all mailbox operations:

- **Send**: Atomically add message to pending and store data
- **Receive**: Move message from pending to invisible with new handle
- **Acknowledge**: Remove message from invisible and mark deleted
- **Nack**: Return message to pending (immediate or delayed)
- **Extend**: Extend visibility timeout for held message
- **ReapExpired**: Move expired messages back to pending
- **TimeAdvance**: Model time passage for timeout testing

Each action models the corresponding Lua script's atomicity guarantees.

### Verification

The embedded specification is verified using TLC (TLA+ model checker):

```bash
make verify-mailbox
```

This extracts the TLA+ spec and runs TLC with bounded model checking:

- **MaxMessages**: 3 (state space explosion limited)
- **MaxDeliveries**: 3 (tests redelivery scenarios)
- **NumConsumers**: 2 (tests concurrent access)
- **VisibilityTimeout**: 5 abstract time units

TLC exhaustively checks all invariants (INV-1 through INV-9) over the entire
reachable state space.

**Limitations**: TLC cannot verify liveness properties or unbounded scenarios.
These are covered by property-based testing.

## Property-Based Testing with Hypothesis

Property-based tests complement TLC verification by:

1. Testing the actual Python implementation (not just the model)
2. Verifying liveness properties (eventual redelivery, no deadlocks)
3. Running unbounded scenarios (hundreds of messages, long-running workers)
4. Testing integration with real Redis (network failures, script errors)

### Test Structure

See `tests/contrib/mailbox/test_redis_mailbox_properties.py` for the complete
test suite.

**Key components**:

- **Reference Model** (`MailboxModel`): Pure Python model that mirrors expected
  behavior. Tests compare Redis state against this reference after each
  operation.

- **Stateful Tests** (`RedisMailboxStateMachine`): Hypothesis stateful testing
  that generates random sequences of operations (send, receive, ack, nack,
  extend, reap) and verifies invariants after each step.

- **Concurrent Tests**: Multi-consumer scenarios with thread pools to stress
  test Lua script atomicity under concurrent load.

### Invariant Verification

Each of the 9 invariants has a corresponding assertion in the property tests:

```python
# Example: INV-1 (Message State Exclusivity)
def check_inv_1_exclusivity(model: MailboxModel, redis_state: dict) -> None:
    """Every message is in exactly one state."""
    for msg_id in model.data:
        states = []
        if msg_id in model.pending: states.append("pending")
        if msg_id in model.invisible: states.append("invisible")
        if msg_id in model.deleted: states.append("deleted")
        assert len(states) == 1, f"{msg_id} in multiple states: {states}"
```

See the test file for complete implementations of all 9 invariant checks.

### Running Property Tests

```bash
# Full property-based test suite (slow, thorough)
make test-redis-properties

# Quick smoke test
pytest tests/contrib/mailbox/test_redis_mailbox_properties.py -k smoke

# Stress test with 1000 examples
pytest tests/contrib/mailbox/test_redis_mailbox_properties.py --hypothesis-seed=random --hypothesis-profile=stress
```

### Test Coverage

| Test Category | Purpose | Example Count |
| --------------------- | --------------------------------------------- | ------------- |
| Stateful tests | Random operation sequences | 100 |
| Targeted invariants | Specific invariant regression tests | 50 each |
| Concurrent stress | Multi-consumer race conditions | 20 |
| Liveness | Eventual redelivery, no stuck messages | 50 |
| Error injection | Network failures, script errors | 30 |
| **Total** | **Comprehensive coverage** | **~600** |

## Verification Workflow

### Development Workflow

1. **Design changes**: Update embedded `@formal_spec` in `_redis.py`
2. **Extract TLA+**: Run `make extract-tla` to generate `.tla` files
3. **Model check**: Run `make verify-mailbox` (TLC checks invariants)
4. **Property tests**: Run `make test-redis-properties` (Hypothesis checks implementation)
5. **Full test suite**: Run `make test` (includes unit tests + property tests)

### CI Pipeline

GitHub Actions runs:

```bash
# Extract and verify TLA+ spec
make extract-tla
make verify-mailbox

# Run property-based tests
make test-redis-properties

# Full test suite with coverage
make test
```

See `.github/workflows/verify.yml` for complete CI configuration.

## Maintenance Guidelines

### When to Update TLA+ Spec

Update the embedded `@formal_spec` when:

1. **Lua script changes**: Any modification to `_LUA_SEND`, `_LUA_RECEIVE`, etc.
2. **State structure changes**: Adding/removing Redis keys, changing data types
3. **Invariant changes**: New invariants or relaxed constraints
4. **Atomicity changes**: Operations that were separate become atomic or vice versa

**Critical**: The embedded spec must stay in sync with Lua script implementation.

### When to Update Property Tests

Update `test_redis_mailbox_properties.py` when:

1. **New operations**: Added mailbox methods (e.g., `peek()`, `purge()`)
2. **New invariants**: Corresponding property checks needed
3. **Bug fixes**: Add regression test for the specific failure case
4. **Performance changes**: Timeout values, reaper intervals

### Verification Checklist

Before merging changes to `RedisMailbox`:

- [ ] Embedded `@formal_spec` updated to match Lua script changes
- [ ] `make verify-mailbox` passes (TLC finds no invariant violations)
- [ ] `make test-redis-properties` passes (Hypothesis finds no failures)
- [ ] New property tests added for new operations/invariants
- [ ] Regression test added for any discovered bugs
- [ ] CHANGELOG.md updated with verification status

## Assumptions

### Redis Guarantees

The verification relies on these Redis guarantees:

1. **Lua script atomicity**: Scripts execute atomically, no interleaving
2. **LIST operations**: LPUSH/RPOP are atomic
3. **HSET/HGET/HDEL**: Hash operations are atomic
4. **ZADD/ZREM/ZRANGEBYSCORE**: Sorted set operations are atomic
5. **No clock skew**: Redis TIME returns monotonic timestamps

### Cluster Mode Assumptions

**Current scope**: Verification assumes single Redis instance or Redis Sentinel.

**Redis Cluster limitations**:
- Multi-key operations require keys to hash to same slot
- Current implementation uses a common prefix (`{mailbox:queue_id}`) to ensure
  slot locality

### Timing Assumptions

- **Reaper frequency**: Background reaper runs every `reaper_interval` seconds
- **Clock monotonicity**: `time.monotonic()` never goes backward
- **Timeout precision**: Visibility timeouts have ±1 second precision

### Consumer Assumptions

- **Bounded processing**: Consumers complete or fail within visibility timeout
- **No malicious consumers**: Consumers don't intentionally corrupt state
- **Handle secrecy**: Receipt handles are not shared between consumers

### Model Limitations

TLC verification uses bounded model checking:

- **Small state space**: MaxMessages=3, MaxDeliveries=3
- **No time**: Abstract time counter, not wall clock
- **Perfect consumers**: No failures, crashes, or Byzantine behavior

Property tests handle unbounded scenarios and failure injection.

## References

- **TLA+ Specification**: `src/weakincentives/contrib/mailbox/_redis.py` (embedded)
- **Extracted TLA+**: `specs/tla/extracted/RedisMailbox.tla` (generated)
- **Property Tests**: `tests/contrib/mailbox/test_redis_mailbox_properties.py`
- **Formal Verification Guide**: `specs/FORMAL_VERIFICATION.md`
- **Redis Lua Scripting**: https://redis.io/docs/manual/programmability/eval-intro/
- **TLA+ Resources**: https://learntla.com/
- **Hypothesis Documentation**: https://hypothesis.readthedocs.io/

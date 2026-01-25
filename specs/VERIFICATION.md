# Redis Mailbox Verification Specification

## Purpose

Formal verification framework for Redis mailbox using embedded TLA+ specification
and property-based testing. Verifies critical invariants for message queue
correctness.

**Implementation:**

- TLA+ spec: `src/weakincentives/contrib/mailbox/_redis.py` (embedded)
- Property tests: `tests/contrib/mailbox/test_redis_mailbox_properties.py`

## Key Invariants

### TLA+ Invariants (Model-Checked)

| ID | Name | Description |
|----|------|-------------|
| INV-1 | MessageStateExclusive | Message in exactly one state (pending/invisible/deleted) |
| INV-2-3 | HandleValidity | Valid handle required; stale handles rejected |
| INV-4 | DeliveryCountMonotonic | Delivery count never decreases |
| INV-4b | DeliveryCountPersistence | Counts persist across redelivery |
| INV-5 | NoMessageLoss | Every message with data is either pending or invisible |
| INV-7 | HandleUniqueness | Each delivery gets unique handle |
| INV-8 | PendingNoDuplicates | No duplicate IDs in pending queue |
| INV-9 | DataIntegrity | Every queued message has associated data |

### Liveness Properties (Python-Tested)

| ID | Name | Description |
|----|------|-------------|
| INV-6 | VisibilityTimeoutCorrectness | Expired invisible messages return to pending |

Note: INV-6 is a liveness property that cannot be expressed as a TLA+ invariant
because it quantifies over state variable domains. It is verified through
property-based tests in `tests/contrib/mailbox/test_redis_mailbox_invariants.py`.

## TLA+ State Variables

| Variable | Purpose |
|----------|---------|
| `pending` | Messages in pending list (FIFO) |
| `invisible` | Messages held by consumers (msg_id -> {expiresAt, handle}) |
| `data` | Message bodies (msg_id -> body) |
| `handles` | Current valid receipt handle per message |
| `deleted` | Acknowledged message IDs |
| `deliveryCounts` | Delivery count per message |
| `deliveryHistory` | Delivery history per message (for INV-4) |
| `now` | Abstract time counter |
| `nextMsgId` | Counter for generating message IDs |
| `nextHandle` | Counter for generating handle suffixes |
| `consumerState` | Consumer state (consumer_id -> {holding, handle}) |

## TLA+ Actions

| Action | Description |
|--------|-------------|
| `Send` | Add message to pending |
| `Receive` | Move pending to invisible with new handle |
| `Acknowledge` | Remove from invisible, mark deleted |
| `AcknowledgeFail` | Ack fails with stale handle |
| `Nack` | Return to pending with optional delay |
| `NackFail` | Nack fails with stale handle |
| `Extend` | Extend visibility timeout |
| `ExtendFail` | Extend fails with stale/invalid handle |
| `ReapOne` | Move one expired message back to pending |
| `Tick` | Advance abstract time |

## Property-Based Testing

Complements TLC by testing actual Python implementation:

- Stateful tests with random operation sequences
- Concurrent multi-consumer scenarios
- Liveness verification (eventual redelivery)
- Error injection (network failures)

## Running Verification

```bash
make verify-mailbox      # TLC model check + property tests
make verify-formal       # TLC model check only
make property-tests      # Property tests only
```

## Redis Guarantees Assumed

- Lua script atomicity
- LIST, HASH, ZSET operations atomic
- No clock skew

## Related Specifications

- `specs/FORMAL_VERIFICATION.md` - @formal_spec decorator
- `specs/MAILBOX.md` - Mailbox protocol

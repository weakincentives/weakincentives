# Redis Mailbox Formal Verification Specification

## Purpose

This specification defines a formal verification framework for the Redis mailbox
implementation. The Redis mailbox uses complex algorithms with Lua scripts for
atomic multi-key operations, visibility timeout management, and receipt handle
validation. Bugs in these algorithms can lead to critical data loss or duplicate
processing.

The verification framework provides two complementary layers:

1. **TLA+ Specification**: A formal model of the mailbox state machine that can
   be exhaustively checked by the TLC model checker for safety and liveness
   properties.

1. **Property-Based Testing**: Hypothesis-based stateful tests that verify the
   actual Python implementation against the same invariants.

Together, these layers provide high confidence that:

- The algorithm design is correct (TLA+)
- The implementation matches the design (Hypothesis)

## Scope

This specification covers verification of `RedisMailbox` in
`src/weakincentives/contrib/mailbox/_redis.py`. The following operations are
modeled:

| Operation | Lua Script | Verification Priority |
|-----------|------------|----------------------|
| `send()` | Pipeline | Medium |
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

### INV-7: FIFO Ordering

Messages are delivered in send order (within visibility constraints):

```
∀ m1, m2 ∈ Messages:
    SendTime(m1) < SendTime(m2) ∧
    m1 ∈ Pending ∧ m2 ∈ Pending ⟹
        ReceiveTime(m1) < ReceiveTime(m2)
```

**Implementation guarantee**: Redis LIST with LPUSH/RPOP provides FIFO.

## TLA+ Specification

### File Structure

```
specs/tla/
├── RedisMailbox.tla           # Main state machine
├── RedisMailboxMC.tla         # Model checking configuration
├── RedisMailboxMC.cfg         # TLC config file
└── README.md                  # Running instructions
```

### State Variables

```tla
---------------------------- MODULE RedisMailbox ----------------------------
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    MaxMessages,        \* Maximum messages to model (e.g., 3)
    MaxDeliveries,      \* Maximum deliveries per message (e.g., 3)
    NumConsumers,       \* Number of concurrent consumers (e.g., 2)
    VisibilityTimeout   \* Timeout value in abstract time units

VARIABLES
    pending,            \* Sequence of message IDs in pending list
    invisible,          \* Function: msg_id -> {expiresAt, handle}
    data,               \* Function: msg_id -> body (or NULL if deleted)
    handles,            \* Function: msg_id -> current valid handle suffix
    deleted,            \* Set of deleted message IDs
    now,                \* Abstract time counter
    nextMsgId,          \* Counter for generating message IDs
    nextHandle,         \* Counter for generating handle suffixes
    consumerState,      \* Function: consumer_id -> {holding, handle}
    deliveryCounts,     \* Function: msg_id -> count (persists across requeue)
    deliveryHistory     \* Function: msg_id -> Sequence of (count, handle) for INV-4

vars == <<pending, invisible, data, handles, deleted, now, nextMsgId,
          nextHandle, consumerState, deliveryCounts, deliveryHistory>>
```

### Initial State

```tla
Init ==
    /\ pending = <<>>
    /\ invisible = [m \in {} |-> [expiresAt |-> 0, handle |-> 0]]
    /\ data = [m \in {} |-> ""]
    /\ handles = [m \in {} |-> 0]
    /\ deleted = {}
    /\ now = 0
    /\ nextMsgId = 1
    /\ nextHandle = 1
    /\ consumerState = [c \in 1..NumConsumers |-> [holding |-> NULL, handle |-> 0]]
    /\ deliveryCounts = [m \in {} |-> 0]
    /\ deliveryHistory = [m \in {} |-> <<>>]
```

### Actions

#### Send

```tla
Send(body) ==
    /\ nextMsgId <= MaxMessages
    /\ LET msgId == nextMsgId
       IN /\ pending' = Append(pending, msgId)
          /\ data' = data @@ (msgId :> body)
          /\ deliveryCounts' = deliveryCounts @@ (msgId :> 0)
          /\ deliveryHistory' = deliveryHistory @@ (msgId :> <<>>)
          /\ nextMsgId' = nextMsgId + 1
    /\ UNCHANGED <<invisible, handles, deleted, now, nextHandle, consumerState>>
```

#### Receive (Atomic Lua Script)

```tla
Receive(consumer) ==
    /\ Len(pending) > 0
    /\ consumerState[consumer].holding = NULL
    /\ LET msgId == Head(pending)
           newHandle == nextHandle
           newExpiry == now + VisibilityTimeout
           newCount == deliveryCounts[msgId] + 1
       IN /\ pending' = Tail(pending)
          /\ invisible' = invisible @@
                (msgId :> [expiresAt |-> newExpiry, handle |-> newHandle])
          /\ handles' = handles @@ (msgId :> newHandle)
          /\ deliveryCounts' = [deliveryCounts EXCEPT ![msgId] = newCount]
          /\ deliveryHistory' = [deliveryHistory EXCEPT
                ![msgId] = Append(@, [count |-> newCount, handle |-> newHandle])]
          /\ nextHandle' = nextHandle + 1
          /\ consumerState' = [consumerState EXCEPT
                ![consumer] = [holding |-> msgId, handle |-> newHandle]]
    /\ UNCHANGED <<data, deleted, now, nextMsgId>>
```

#### Acknowledge (Atomic Lua Script)

```tla
Acknowledge(consumer) ==
    /\ consumerState[consumer].holding /= NULL
    /\ LET msgId == consumerState[consumer].holding
           providedHandle == consumerState[consumer].handle
       IN /\ msgId \in DOMAIN handles
          /\ handles[msgId] = providedHandle  \* Handle validation
          /\ msgId \in DOMAIN invisible       \* Still in invisible
          /\ invisible' = [m \in (DOMAIN invisible) \ {msgId} |-> invisible[m]]
          /\ data' = [m \in (DOMAIN data) \ {msgId} |-> data[m]]
          /\ handles' = [m \in (DOMAIN handles) \ {msgId} |-> handles[m]]
          /\ deliveryCounts' = [m \in (DOMAIN deliveryCounts) \ {msgId} |->
                                deliveryCounts[m]]
          /\ deleted' = deleted \cup {msgId}
          /\ consumerState' = [consumerState EXCEPT
                ![consumer] = [holding |-> NULL, handle |-> 0]]
    /\ UNCHANGED <<pending, now, nextMsgId, nextHandle, deliveryHistory>>

\* Acknowledge fails if handle is stale
AcknowledgeFail(consumer) ==
    /\ consumerState[consumer].holding /= NULL
    /\ LET msgId == consumerState[consumer].holding
           providedHandle == consumerState[consumer].handle
       IN \/ msgId \notin DOMAIN handles
          \/ handles[msgId] /= providedHandle
          \/ msgId \notin DOMAIN invisible
    /\ consumerState' = [consumerState EXCEPT
            ![consumer] = [holding |-> NULL, handle |-> 0]]
    /\ UNCHANGED <<pending, invisible, data, handles, deleted, now,
                   nextMsgId, nextHandle, deliveryCounts, deliveryHistory>>
```

#### Nack

The Python implementation (`_LUA_NACK`) ALWAYS deletes the handle on nack,
regardless of visibility_timeout. With delayed nack, the message stays in
invisible but has no valid handle until redelivery.

```tla
Nack(consumer, newTimeout) ==
    /\ consumerState[consumer].holding /= NULL
    /\ LET msgId == consumerState[consumer].holding
           providedHandle == consumerState[consumer].handle
       IN /\ msgId \in DOMAIN handles
          /\ handles[msgId] = providedHandle
          /\ msgId \in DOMAIN invisible
          /\ IF newTimeout = 0
             THEN \* Immediate requeue to pending
                  /\ pending' = Append(pending, msgId)
                  /\ invisible' = [m \in (DOMAIN invisible) \ {msgId} |->
                                   invisible[m]]
             ELSE \* Delayed requeue: stays in invisible with new expiry
                  /\ invisible' = [invisible EXCEPT
                        ![msgId].expiresAt = now + newTimeout,
                        ![msgId].handle = 0]  \* No valid handle
                  /\ UNCHANGED pending
          \* Handle is ALWAYS invalidated on nack (matches _LUA_NACK line 84)
          /\ handles' = [m \in (DOMAIN handles) \ {msgId} |-> handles[m]]
          /\ consumerState' = [consumerState EXCEPT
                ![consumer] = [holding |-> NULL, handle |-> 0]]
    /\ UNCHANGED <<data, deleted, now, nextMsgId, nextHandle,
                   deliveryCounts, deliveryHistory>>

\* Nack fails if handle is stale
NackFail(consumer) ==
    /\ consumerState[consumer].holding /= NULL
    /\ LET msgId == consumerState[consumer].holding
           providedHandle == consumerState[consumer].handle
       IN \/ msgId \notin DOMAIN handles
          \/ handles[msgId] /= providedHandle
          \/ msgId \notin DOMAIN invisible
    /\ consumerState' = [consumerState EXCEPT
            ![consumer] = [holding |-> NULL, handle |-> 0]]
    /\ UNCHANGED <<pending, invisible, data, handles, deleted, now,
                   nextMsgId, nextHandle, deliveryCounts, deliveryHistory>>
```

#### Extend (Atomic Lua Script)

Extends visibility timeout for a message the consumer is holding.

```tla
Extend(consumer, newTimeout) ==
    /\ consumerState[consumer].holding /= NULL
    /\ LET msgId == consumerState[consumer].holding
           providedHandle == consumerState[consumer].handle
       IN /\ msgId \in DOMAIN handles
          /\ handles[msgId] = providedHandle
          /\ msgId \in DOMAIN invisible
          \* Update expiry time (ZADD XX updates existing only)
          /\ invisible' = [invisible EXCEPT
                ![msgId].expiresAt = now + newTimeout]
    \* Handle and consumer state remain valid
    /\ UNCHANGED <<pending, data, handles, deleted, now, nextMsgId,
                   nextHandle, consumerState, deliveryCounts, deliveryHistory>>

\* Extend fails if handle is stale or message not in invisible
ExtendFail(consumer) ==
    /\ consumerState[consumer].holding /= NULL
    /\ LET msgId == consumerState[consumer].holding
           providedHandle == consumerState[consumer].handle
       IN \/ msgId \notin DOMAIN handles
          \/ handles[msgId] /= providedHandle
          \/ msgId \notin DOMAIN invisible
    /\ consumerState' = [consumerState EXCEPT
            ![consumer] = [holding |-> NULL, handle |-> 0]]
    /\ UNCHANGED <<pending, invisible, data, handles, deleted, now,
                   nextMsgId, nextHandle, deliveryCounts, deliveryHistory>>
```

#### Reap Expired (Atomic Lua Script)

```tla
ReapOne ==
    /\ \E msgId \in DOMAIN invisible:
        /\ invisible[msgId].expiresAt < now
        /\ pending' = Append(pending, msgId)
        /\ invisible' = [m \in (DOMAIN invisible) \ {msgId} |-> invisible[m]]
        \* Handle deleted by reaper (matches _LUA_REAP line 107)
        /\ handles' = [m \in (DOMAIN handles) \ {msgId} |-> handles[m]]
        \* Invalidate any consumer holding this message
        /\ consumerState' = [c \in DOMAIN consumerState |->
            IF consumerState[c].holding = msgId
            THEN [holding |-> NULL, handle |-> 0]
            ELSE consumerState[c]]
    \* deliveryCounts persists - this is the critical fix
    /\ UNCHANGED <<data, deleted, now, nextMsgId, nextHandle,
                   deliveryCounts, deliveryHistory>>
```

#### Time Advance

```tla
Tick ==
    /\ now' = now + 1
    /\ UNCHANGED <<pending, invisible, data, handles, deleted, nextMsgId,
                   nextHandle, consumerState, deliveryCounts, deliveryHistory>>
```

### Next State Relation

```tla
Next ==
    \/ \E body \in {"a", "b", "c"}: Send(body)
    \/ \E c \in 1..NumConsumers: Receive(c)
    \/ \E c \in 1..NumConsumers: Acknowledge(c)
    \/ \E c \in 1..NumConsumers: AcknowledgeFail(c)
    \/ \E c \in 1..NumConsumers, t \in 0..VisibilityTimeout: Nack(c, t)
    \/ \E c \in 1..NumConsumers: NackFail(c)
    \/ \E c \in 1..NumConsumers, t \in 1..VisibilityTimeout: Extend(c, t)
    \/ \E c \in 1..NumConsumers: ExtendFail(c)
    \/ ReapOne
    \/ Tick
```

### Invariants

```tla
\* INV-1: Message State Exclusivity
MessageStateExclusive ==
    \A msgId \in 1..nextMsgId-1:
        LET inPending == \E i \in 1..Len(pending): pending[i] = msgId
            inInvisible == msgId \in DOMAIN invisible
            inDeleted == msgId \in deleted
        IN (inPending /\ ~inInvisible /\ ~inDeleted) \/
           (~inPending /\ inInvisible /\ ~inDeleted) \/
           (~inPending /\ ~inInvisible /\ inDeleted)

\* INV-2 & INV-3: Handle Validity
HandleValidity ==
    \A c \in 1..NumConsumers:
        LET state == consumerState[c]
        IN state.holding /= NULL =>
            (state.holding \in DOMAIN handles =>
                handles[state.holding] = state.handle)

\* INV-4: Delivery Count Monotonicity
\* Uses deliveryHistory to verify counts are strictly increasing
DeliveryCountMonotonic ==
    \A msgId \in DOMAIN deliveryHistory:
        LET history == deliveryHistory[msgId]
        IN \A i \in 1..Len(history)-1:
            history[i].count < history[i+1].count

\* INV-4b: Delivery counts persist across requeue
\* After reap, the next receive must have count = previous + 1
DeliveryCountPersistence ==
    \A msgId \in DOMAIN deliveryCounts:
        \A i \in 1..Len(deliveryHistory[msgId]):
            deliveryHistory[msgId][i].count = i

\* INV-5: No Message Loss (Safety part)
NoMessageLoss ==
    \A msgId \in DOMAIN data:
        LET inPending == \E i \in 1..Len(pending): pending[i] = msgId
            inInvisible == msgId \in DOMAIN invisible
        IN inPending \/ inInvisible

\* INV-7: Handle Uniqueness across deliveries
HandleUniqueness ==
    \A msgId \in DOMAIN deliveryHistory:
        LET history == deliveryHistory[msgId]
        IN \A i, j \in 1..Len(history):
            i /= j => history[i].handle /= history[j].handle

\* All invariants combined
TypeInvariant ==
    /\ MessageStateExclusive
    /\ HandleValidity
    /\ DeliveryCountMonotonic
    /\ DeliveryCountPersistence
    /\ NoMessageLoss
    /\ HandleUniqueness
```

### Liveness Properties

```tla
\* Fairness: consumers eventually make progress
Fairness ==
    /\ WF_vars(Tick)
    /\ WF_vars(ReapOne)
    /\ \A c \in 1..NumConsumers:
        /\ WF_vars(Receive(c))
        /\ WF_vars(Acknowledge(c))

\* INV-6: Expired messages eventually requeued
EventualRequeue ==
    \A msgId \in DOMAIN invisible:
        invisible[msgId].expiresAt < now ~>
            (\E i \in 1..Len(pending): pending[i] = msgId)

Spec == Init /\ [][Next]_vars /\ Fairness
```

### Model Checking Configuration

```tla
---------------------------- MODULE RedisMailboxMC ----------------------------
EXTENDS RedisMailbox

\* Small model for exhaustive checking
CONSTANTS
    MaxMessages = 3,
    MaxDeliveries = 3,
    NumConsumers = 2,
    VisibilityTimeout = 2
=============================================================================
```

**RedisMailboxMC.cfg:**

```
SPECIFICATION Spec

CONSTANTS
    MaxMessages = 3
    MaxDeliveries = 3
    NumConsumers = 2
    VisibilityTimeout = 2

INVARIANTS
    TypeInvariant
    MessageStateExclusive
    HandleValidity
    DeliveryCountMonotonic
    DeliveryCountPersistence
    NoMessageLoss
    HandleUniqueness

PROPERTIES
    EventualRequeue

\* Enable simulation mode for larger models
\* SIMULATION
\*     NumSimulations = 1000
\*     TraceLength = 100
```

### Running TLC

```bash
# Install TLA+ tools (or use make setup-tlaplus)
brew install tlaplus  # macOS
# or download from https://github.com/tlaplus/tlaplus/releases

# Run model checker (simulation mode - fast, for CI)
make tlaplus-check

# Run exhaustive model checker (slow, for thorough verification)
make tlaplus-check-exhaustive

# Expected output (no errors):
# Model checking completed. No error has been found.
```

### TLC Limitations and Workarounds

**State Space Constraint**: The TLA+ specification uses `StateConstraint` to bound
`now <= 5`, preventing infinite state exploration from the unbounded `Tick` action.
This is sufficient for safety property verification but limits liveness checking.

**Liveness Property Limitation**: The `EventualRequeue` temporal property:

```tla
EventualRequeue ==
    \A msgId \in DOMAIN invisible:
        invisible[msgId].expiresAt < now ~>
            InPending(msgId)
```

Cannot be checked by TLC because it quantifies over the domain of the `invisible`
state variable. TLC reports: "TLC cannot handle temporal formulas that quantify
over state variable domains."

**Workaround**: Liveness properties are verified through Hypothesis property-based
tests in `tests/contrib/mailbox/test_redis_mailbox_invariants.py`:

- `TestEventualRequeue.test_expired_message_eventually_requeued` - Single message
- `TestEventualRequeue.test_multiple_messages_eventually_requeued` - Multiple messages

These tests verify the actual implementation behavior with real timing, complementing
the TLA+ safety verification.

## Property-Based Testing with Hypothesis

**Important**: Hypothesis is required for complete verification coverage. While tests
gracefully skip when Hypothesis is not installed, this significantly reduces test
coverage. For verification purposes, ensure Hypothesis is installed:

```bash
uv sync --all-extras  # Installs hypothesis as dev dependency
```

### Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    # ... existing deps ...
    "hypothesis>=6.100.0",
]
```

### Test File Structure

```
tests/contrib/mailbox/
├── test_redis_mailbox_properties.py   # Stateful property tests
├── test_redis_mailbox_invariants.py   # Specific invariant tests
└── conftest.py                        # Redis fixtures
```

### Fixtures

```python
# tests/contrib/mailbox/conftest.py

import pytest
from uuid import uuid4

from redis import Redis
from weakincentives.contrib.mailbox import RedisMailbox

@pytest.fixture
def redis_client():
    """Fresh Redis connection for each test."""
    client = Redis(host="localhost", port=6379, db=15)
    yield client
    client.flushdb()
    client.close()

@pytest.fixture
def mailbox(redis_client):
    """Fresh RedisMailbox for each test."""
    mb = RedisMailbox(
        name=f"test-{uuid4().hex[:8]}",
        client=redis_client,
        reaper_interval=0.1,  # Fast reaper for testing
    )
    yield mb
    mb.close()
    mb.purge()
```

### Reference Model

The reference model tracks expected state for comparison with Redis:

```python
# tests/contrib/mailbox/test_redis_mailbox_properties.py

from dataclasses import dataclass, field
from collections import deque
from typing import Any
from uuid import uuid4

@dataclass
class MessageState:
    """Tracks a message through its lifecycle."""
    id: str
    body: Any
    delivery_count: int = 0
    current_handle: str | None = None
    expires_at: float | None = None

@dataclass
class MailboxModel:
    """Reference model for RedisMailbox state."""

    pending: deque[str] = field(default_factory=deque)
    invisible: dict[str, MessageState] = field(default_factory=dict)
    data: dict[str, MessageState] = field(default_factory=dict)
    deleted: set[str] = field(default_factory=set)
    delivery_history: dict[str, list[tuple[int, str]]] = field(
        default_factory=dict
    )

    def send(self, msg_id: str, body: Any) -> None:
        """Model a send operation."""
        state = MessageState(id=msg_id, body=body)
        self.data[msg_id] = state
        self.pending.append(msg_id)

    def receive(
        self, msg_id: str, handle: str, expires_at: float
    ) -> None:
        """Model a receive operation."""
        if msg_id in self.pending:
            self.pending.remove(msg_id)

        state = self.data[msg_id]
        state.delivery_count += 1
        state.current_handle = handle
        state.expires_at = expires_at
        self.invisible[msg_id] = state

        # Track delivery history
        if msg_id not in self.delivery_history:
            self.delivery_history[msg_id] = []
        self.delivery_history[msg_id].append(
            (state.delivery_count, handle)
        )

    def acknowledge(self, msg_id: str, handle: str) -> bool:
        """Model an acknowledge. Returns True if successful."""
        if msg_id not in self.invisible:
            return False
        if self.invisible[msg_id].current_handle != handle:
            return False

        del self.invisible[msg_id]
        del self.data[msg_id]
        self.deleted.add(msg_id)
        return True

    def nack(
        self, msg_id: str, handle: str, visibility_timeout: int, now: float
    ) -> bool:
        """Model a nack. Returns True if successful."""
        if msg_id not in self.invisible:
            return False
        if self.invisible[msg_id].current_handle != handle:
            return False

        state = self.invisible.pop(msg_id)
        state.current_handle = None

        if visibility_timeout <= 0:
            self.pending.append(msg_id)
            state.expires_at = None
        else:
            state.expires_at = now + visibility_timeout
            self.invisible[msg_id] = state

        return True

    def reap(self, now: float) -> list[str]:
        """Model reaper. Returns list of requeued message IDs."""
        requeued = []
        for msg_id, state in list(self.invisible.items()):
            if state.expires_at is not None and state.expires_at <= now:
                del self.invisible[msg_id]
                state.current_handle = None
                state.expires_at = None
                self.pending.append(msg_id)
                requeued.append(msg_id)
        return requeued

    def extend(
        self, msg_id: str, handle: str, new_timeout: int, now: float
    ) -> bool:
        """Model an extend. Returns True if successful."""
        if msg_id not in self.invisible:
            return False
        if self.invisible[msg_id].current_handle != handle:
            return False

        self.invisible[msg_id].expires_at = now + new_timeout
        return True

    def is_handle_valid(self, msg_id: str, handle: str) -> bool:
        """Check if a handle is currently valid."""
        if msg_id not in self.invisible:
            return False
        return self.invisible[msg_id].current_handle == handle

    def get_pending_count(self) -> int:
        return len(self.pending)

    def get_invisible_count(self) -> int:
        return len(self.invisible)

    def total_count(self) -> int:
        return len(self.data)
```

### Stateful Property-Based Tests

```python
# tests/contrib/mailbox/test_redis_mailbox_properties.py

import time
from hypothesis import settings, HealthCheck
from hypothesis.stateful import (
    RuleBasedStateMachine,
    Bundle,
    rule,
    invariant,
    precondition,
    initialize,
)
from hypothesis import strategies as st

from weakincentives.contrib.mailbox import RedisMailbox
from weakincentives.runtime.mailbox import ReceiptHandleExpiredError


class RedisMailboxStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based tests for RedisMailbox.

    This state machine exercises the mailbox through random sequences
    of operations while maintaining a reference model. Invariants are
    checked after each step.
    """

    # Bundles track values across rules
    sent_ids = Bundle("sent_ids")
    received = Bundle("received")  # (msg_id, receipt_handle) tuples

    def __init__(self):
        super().__init__()
        self.model = MailboxModel()
        self.start_time = time.time()

    @initialize()
    def setup(self):
        """Create fresh mailbox for each test run."""
        from redis import Redis
        from uuid import uuid4

        self.client = Redis(host="localhost", port=6379, db=15)
        self.client.flushdb()

        self.mailbox = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=self.client,
            reaper_interval=0.05,  # 50ms for fast testing
        )

    def teardown(self):
        """Clean up after test run."""
        if hasattr(self, "mailbox"):
            self.mailbox.close()
            self.mailbox.purge()
        if hasattr(self, "client"):
            self.client.close()

    @rule(target=sent_ids, body=st.binary(min_size=1, max_size=100))
    def send_message(self, body):
        """Send a message and track it."""
        msg_id = self.mailbox.send(body)
        self.model.send(msg_id, body)
        return msg_id

    @rule(
        target=received,
        timeout=st.integers(min_value=1, max_value=10),
    )
    @precondition(lambda self: self.model.get_pending_count() > 0)
    def receive_message(self, timeout):
        """Receive a message if any are pending."""
        msgs = self.mailbox.receive(
            visibility_timeout=timeout,
            wait_time_seconds=0,
        )

        if msgs:
            msg = msgs[0]
            expires_at = time.time() + timeout
            self.model.receive(msg.id, msg.receipt_handle, expires_at)
            return (msg.id, msg.receipt_handle)

        return None

    @rule(receipt=received)
    def acknowledge_message(self, receipt):
        """Acknowledge a received message."""
        if receipt is None:
            return

        msg_id, handle = receipt
        suffix = handle.split(":", 1)[1] if ":" in handle else handle

        try:
            self.mailbox._acknowledge(msg_id, suffix)
            assert self.model.acknowledge(msg_id, handle), \
                "Model predicted failure but implementation succeeded"
        except ReceiptHandleExpiredError:
            assert not self.model.is_handle_valid(msg_id, handle), \
                "Model predicted success but implementation failed"

    @rule(
        receipt=received,
        new_timeout=st.integers(min_value=0, max_value=5),
    )
    def nack_message(self, receipt, new_timeout):
        """Return a message to the queue."""
        if receipt is None:
            return

        msg_id, handle = receipt
        suffix = handle.split(":", 1)[1] if ":" in handle else handle

        try:
            self.mailbox._nack(msg_id, suffix, new_timeout)
            expected = self.model.nack(
                msg_id, handle, new_timeout, time.time()
            )
            assert expected, \
                "Model predicted failure but implementation succeeded"
        except ReceiptHandleExpiredError:
            assert not self.model.is_handle_valid(msg_id, handle), \
                "Model predicted success but implementation failed"

    @rule(
        receipt=received,
        new_timeout=st.integers(min_value=1, max_value=30),
    )
    def extend_message(self, receipt, new_timeout):
        """Extend visibility timeout for a received message."""
        if receipt is None:
            return

        msg_id, handle = receipt
        suffix = handle.split(":", 1)[1] if ":" in handle else handle

        try:
            self.mailbox._extend(msg_id, suffix, new_timeout)
            expected = self.model.extend(
                msg_id, handle, new_timeout, time.time()
            )
            assert expected, \
                "Model predicted failure but implementation succeeded"
        except ReceiptHandleExpiredError:
            assert not self.model.is_handle_valid(msg_id, handle), \
                "Model predicted success but implementation failed"

    @rule()
    def advance_time(self):
        """
        Wait briefly to allow visibility timeouts to expire.
        Also syncs model reaper state.
        """
        time.sleep(0.1)
        self.model.reap(time.time())

    # =========================================================================
    # Invariants - checked after every rule
    # =========================================================================

    @invariant()
    def message_count_matches(self):
        """Total message count matches between model and implementation."""
        expected = self.model.total_count()
        actual = self.mailbox.approximate_count()
        # Allow for timing differences in invisible set
        assert abs(expected - actual) <= 1, \
            f"Count mismatch: model={expected}, redis={actual}"

    @invariant()
    def no_messages_lost(self):
        """
        Every message in the model exists in Redis.
        Messages are either pending, invisible, or deleted.
        """
        for msg_id in self.model.data:
            if msg_id in self.model.deleted:
                continue

            # Check it exists somewhere in Redis
            in_pending = self._msg_in_pending(msg_id)
            in_invisible = self._msg_in_invisible(msg_id)
            has_data = self.client.hexists(
                self.mailbox._keys.data, msg_id
            )

            assert has_data, f"Message {msg_id} data lost"
            assert in_pending or in_invisible, \
                f"Message {msg_id} not in pending or invisible"

    @invariant()
    def message_state_exclusive(self):
        """Each message is in exactly one location."""
        # Get all known message IDs
        all_pending = self._get_all_pending()
        all_invisible = self._get_all_invisible()

        # Check no overlap
        overlap = set(all_pending) & set(all_invisible)
        assert not overlap, \
            f"Messages in both pending and invisible: {overlap}"

    @invariant()
    def deleted_messages_gone(self):
        """Deleted messages have no remaining state."""
        for msg_id in self.model.deleted:
            assert not self._msg_in_pending(msg_id), \
                f"Deleted message {msg_id} in pending"
            assert not self._msg_in_invisible(msg_id), \
                f"Deleted message {msg_id} in invisible"

    @invariant()
    def delivery_count_monotonic(self):
        """Delivery counts are strictly increasing for each message."""
        for msg_id, history in self.model.delivery_history.items():
            counts = [count for count, _ in history]
            for i in range(1, len(counts)):
                assert counts[i] > counts[i-1], \
                    f"Non-monotonic delivery count for {msg_id}: {counts}"

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _msg_in_pending(self, msg_id: str) -> bool:
        """Check if message is in pending list."""
        pending = self.client.lrange(self.mailbox._keys.pending, 0, -1)
        return msg_id.encode() in pending

    def _msg_in_invisible(self, msg_id: str) -> bool:
        """Check if message is in invisible set."""
        score = self.client.zscore(self.mailbox._keys.invisible, msg_id)
        return score is not None

    def _get_all_pending(self) -> list[str]:
        """Get all message IDs in pending."""
        return [
            m.decode()
            for m in self.client.lrange(self.mailbox._keys.pending, 0, -1)
        ]

    def _get_all_invisible(self) -> list[str]:
        """Get all message IDs in invisible."""
        return [
            m.decode()
            for m in self.client.zrange(self.mailbox._keys.invisible, 0, -1)
        ]


# Configure Hypothesis settings
TestRedisMailbox = RedisMailboxStateMachine.TestCase
TestRedisMailbox.settings = settings(
    max_examples=100,
    stateful_step_count=50,
    deadline=None,  # Disable deadline for I/O operations
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
    ],
)
```

### Targeted Invariant Tests

```python
# tests/contrib/mailbox/test_redis_mailbox_invariants.py

"""
Targeted tests for specific invariants.

These tests focus on edge cases and race conditions that are
critical for correctness.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import pytest
from hypothesis import given, settings, strategies as st
from redis import Redis

from weakincentives.contrib.mailbox import RedisMailbox
from weakincentives.runtime.mailbox import ReceiptHandleExpiredError


class TestReceiptHandleFreshness:
    """Tests for INV-2: Receipt Handle Freshness."""

    def test_redelivery_generates_new_handle(self, mailbox):
        """Each delivery of the same message gets a unique handle."""
        mailbox.send(b"test")

        # First receive
        msgs1 = mailbox.receive(visibility_timeout=1)
        handle1 = msgs1[0].receipt_handle

        # Wait for timeout
        time.sleep(1.5)

        # Second receive (redelivery)
        msgs2 = mailbox.receive(visibility_timeout=30)
        handle2 = msgs2[0].receipt_handle

        assert handle1 != handle2, "Redelivery must generate new handle"
        assert msgs1[0].id == msgs2[0].id, "Same message ID"

    def test_old_handle_rejected_after_redelivery(self, mailbox):
        """Stale handles from previous delivery are rejected."""
        mailbox.send(b"test")

        # First consumer receives
        msgs1 = mailbox.receive(visibility_timeout=1)
        old_handle = msgs1[0].receipt_handle
        msg_id = msgs1[0].id

        # Wait for timeout and redelivery
        time.sleep(1.5)

        # Second consumer receives
        msgs2 = mailbox.receive(visibility_timeout=30)
        assert len(msgs2) == 1

        # First consumer tries to ack with old handle
        old_suffix = old_handle.split(":", 1)[1]
        with pytest.raises(ReceiptHandleExpiredError):
            mailbox._acknowledge(msg_id, old_suffix)

        # Second consumer can still ack
        msgs2[0].acknowledge()

    @given(st.integers(min_value=2, max_value=5))
    @settings(max_examples=10, deadline=None)
    def test_handle_unique_across_n_deliveries(self, redis_client, n):
        """Handles are unique across N deliveries of the same message."""
        mailbox = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.05,
        )

        try:
            mailbox.send(b"test")
            handles = []

            for i in range(n):
                msgs = mailbox.receive(visibility_timeout=1)
                if msgs:
                    handles.append(msgs[0].receipt_handle)
                time.sleep(0.2)  # Allow reaper to run

            # All handles should be unique
            assert len(handles) == len(set(handles)), \
                f"Duplicate handles found: {handles}"
        finally:
            mailbox.close()
            mailbox.purge()


class TestMessageStateExclusivity:
    """Tests for INV-1: Message State Exclusivity."""

    def test_receive_atomic_transition(self, mailbox, redis_client):
        """Receive atomically moves message from pending to invisible."""
        mailbox.send(b"test")

        # Receive
        msgs = mailbox.receive(visibility_timeout=30)
        msg_id = msgs[0].id

        # Check state
        in_pending = redis_client.lrange(mailbox._keys.pending, 0, -1)
        in_invisible = redis_client.zscore(mailbox._keys.invisible, msg_id)

        assert msg_id.encode() not in in_pending
        assert in_invisible is not None

    def test_concurrent_receive_no_duplicates(self, mailbox):
        """Parallel receives never return the same message twice."""
        # Send 100 messages
        for i in range(100):
            mailbox.send(f"msg-{i}".encode())

        received = []
        lock = threading.Lock()

        def worker():
            local_received = []
            for _ in range(50):
                msgs = mailbox.receive(visibility_timeout=60)
                for m in msgs:
                    local_received.append(m.id)
            with lock:
                received.extend(local_received)

        # Run 4 concurrent workers
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No duplicates
        assert len(received) == len(set(received)), \
            "Duplicate messages received"

    def test_reap_and_receive_race(self, mailbox, redis_client):
        """
        Reaper and receive don't cause duplicate state.

        This tests a potential race where:
        1. Message expires in invisible
        2. Reaper starts moving it to pending
        3. Consumer calls receive
        """
        # Send and receive with short timeout
        mailbox.send(b"test")
        msgs = mailbox.receive(visibility_timeout=1)
        msg_id = msgs[0].id

        # Wait for expiry
        time.sleep(1.2)

        # Check invariant: message in exactly one place
        in_pending = msg_id.encode() in redis_client.lrange(
            mailbox._keys.pending, 0, -1
        )
        in_invisible = redis_client.zscore(
            mailbox._keys.invisible, msg_id
        ) is not None

        assert in_pending != in_invisible, \
            f"Message in invalid state: pending={in_pending}, invisible={in_invisible}"


class TestNoMessageLoss:
    """Tests for INV-5: No Message Loss."""

    @given(st.lists(st.binary(min_size=1, max_size=50), min_size=1, max_size=20))
    @settings(max_examples=20, deadline=None)
    def test_all_messages_accounted_for(self, redis_client, bodies):
        """Every sent message is acked or remains in queue."""
        mailbox = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.1,
        )

        try:
            # Send all messages
            sent_ids = [mailbox.send(body) for body in bodies]

            # Receive and ack half
            acked = set()
            for _ in range(len(bodies) // 2):
                msgs = mailbox.receive(visibility_timeout=60)
                if msgs:
                    msgs[0].acknowledge()
                    acked.add(msgs[0].id)

            # Verify: acked + remaining = total
            remaining = mailbox.approximate_count()
            assert len(acked) + remaining == len(bodies), \
                f"Message loss: sent={len(bodies)}, acked={len(acked)}, remaining={remaining}"
        finally:
            mailbox.close()
            mailbox.purge()

    def test_crash_recovery_no_loss(self, redis_client):
        """Messages survive mailbox close/reopen."""
        name = f"test-{uuid4().hex[:8]}"

        # First mailbox instance sends messages
        mb1 = RedisMailbox(name=name, client=redis_client)
        for i in range(10):
            mb1.send(f"msg-{i}".encode())
        mb1.close()

        # Second instance sees all messages
        mb2 = RedisMailbox(name=name, client=redis_client)
        try:
            assert mb2.approximate_count() == 10
        finally:
            mb2.close()
            mb2.purge()


class TestDeliveryCountMonotonicity:
    """Tests for INV-4: Delivery Count Monotonicity."""

    def test_delivery_count_increments(self, mailbox):
        """Each delivery increments the count."""
        mailbox.send(b"test")

        counts = []
        for _ in range(3):
            msgs = mailbox.receive(visibility_timeout=1)
            if msgs:
                counts.append(msgs[0].delivery_count)
            time.sleep(0.2)  # Allow reaper

        # Counts should be strictly increasing
        assert counts == sorted(counts), f"Counts not monotonic: {counts}"
        assert len(set(counts)) == len(counts), f"Duplicate counts: {counts}"

    def test_delivery_count_survives_redelivery(self, mailbox):
        """Delivery count persists across timeout and requeue."""
        mailbox.send(b"test")

        # First delivery
        msgs1 = mailbox.receive(visibility_timeout=1)
        assert msgs1[0].delivery_count == 1

        # Let it timeout and get requeued
        time.sleep(1.5)

        # Second delivery - count should be 2, not reset to 1
        msgs2 = mailbox.receive(visibility_timeout=1)
        assert msgs2[0].delivery_count == 2, \
            "Delivery count was reset after requeue!"

        # Let it timeout again
        time.sleep(1.5)

        # Third delivery
        msgs3 = mailbox.receive(visibility_timeout=30)
        assert msgs3[0].delivery_count == 3, \
            "Delivery count was reset after second requeue!"

    def test_delivery_count_survives_nack(self, mailbox):
        """Delivery count persists across nack and requeue."""
        mailbox.send(b"test")

        # First delivery
        msgs1 = mailbox.receive(visibility_timeout=30)
        assert msgs1[0].delivery_count == 1
        msgs1[0].nack(visibility_timeout=0)  # Immediate requeue

        # Second delivery after nack
        msgs2 = mailbox.receive(visibility_timeout=30)
        assert msgs2[0].delivery_count == 2, \
            "Delivery count was reset after nack!"


class TestVisibilityTimeout:
    """Tests for INV-6: Visibility Timeout Correctness."""

    def test_message_requeued_after_timeout(self, mailbox):
        """Expired messages return to pending."""
        mailbox.send(b"test")

        # Receive with short timeout
        msgs = mailbox.receive(visibility_timeout=1)
        assert len(msgs) == 1

        # Immediately try to receive again - should be empty
        msgs2 = mailbox.receive(visibility_timeout=30, wait_time_seconds=0)
        assert len(msgs2) == 0

        # Wait for timeout
        time.sleep(1.5)

        # Now should be available
        msgs3 = mailbox.receive(visibility_timeout=30)
        assert len(msgs3) == 1
        assert msgs3[0].id == msgs[0].id

    def test_extend_prevents_requeue(self, mailbox):
        """Extended visibility prevents timeout requeue."""
        mailbox.send(b"test")

        msgs = mailbox.receive(visibility_timeout=1)
        msg = msgs[0]

        # Extend before timeout
        time.sleep(0.5)
        msg.extend_visibility(10)

        # Wait past original timeout
        time.sleep(1.0)

        # Should still be invisible (not requeued)
        msgs2 = mailbox.receive(visibility_timeout=30, wait_time_seconds=0)
        assert len(msgs2) == 0, "Message was requeued despite extension"

        # Original handle should still work
        msg.acknowledge()


class TestFIFOOrdering:
    """Tests for INV-7: FIFO Ordering."""

    def test_messages_received_in_send_order(self, mailbox):
        """Messages are delivered in FIFO order."""
        # Send in order
        ids = []
        for i in range(10):
            msg_id = mailbox.send(f"msg-{i}".encode())
            ids.append(msg_id)

        # Receive and verify order
        received_ids = []
        while True:
            msgs = mailbox.receive(visibility_timeout=60)
            if not msgs:
                break
            received_ids.append(msgs[0].id)
            msgs[0].acknowledge()

        assert received_ids == ids, "Messages received out of order"
```

### Concurrent Stress Tests

```python
# tests/contrib/mailbox/test_redis_mailbox_stress.py

"""
Stress tests for concurrent access patterns.

These tests verify correctness under high concurrency.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import pytest
from redis import Redis

from weakincentives.contrib.mailbox import RedisMailbox
from weakincentives.runtime.mailbox import ReceiptHandleExpiredError


class TestConcurrentStress:
    """High-concurrency stress tests."""

    @pytest.mark.slow
    def test_producer_consumer_stress(self, redis_client):
        """
        Multiple producers and consumers operating concurrently.

        Verifies:
        - No message loss
        - No duplicate processing
        - Correct final count
        """
        mailbox = RedisMailbox(
            name=f"stress-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.1,
        )

        num_producers = 4
        num_consumers = 4
        messages_per_producer = 100
        total_messages = num_producers * messages_per_producer

        sent = []
        received = []
        acked = []
        sent_lock = threading.Lock()
        received_lock = threading.Lock()
        acked_lock = threading.Lock()
        stop_consumers = threading.Event()

        def producer(producer_id):
            for i in range(messages_per_producer):
                body = f"p{producer_id}-m{i}".encode()
                msg_id = mailbox.send(body)
                with sent_lock:
                    sent.append(msg_id)

        def consumer(consumer_id):
            while not stop_consumers.is_set():
                msgs = mailbox.receive(
                    visibility_timeout=30,
                    wait_time_seconds=1,
                )
                for msg in msgs:
                    with received_lock:
                        received.append(msg.id)
                    try:
                        msg.acknowledge()
                        with acked_lock:
                            acked.append(msg.id)
                    except ReceiptHandleExpiredError:
                        pass  # Expected if redelivered

        try:
            # Start consumers first
            consumer_threads = [
                threading.Thread(target=consumer, args=(i,))
                for i in range(num_consumers)
            ]
            for t in consumer_threads:
                t.start()

            # Then producers
            producer_threads = [
                threading.Thread(target=producer, args=(i,))
                for i in range(num_producers)
            ]
            for t in producer_threads:
                t.start()
            for t in producer_threads:
                t.join()

            # Wait for consumers to drain
            time.sleep(2)
            stop_consumers.set()
            for t in consumer_threads:
                t.join(timeout=5)

            # Verify
            remaining = mailbox.approximate_count()

            # All messages accounted for
            assert len(sent) == total_messages
            assert len(set(acked)) + remaining == total_messages, \
                f"Message loss: sent={total_messages}, acked={len(set(acked))}, remaining={remaining}"

            # No duplicate acks (set size equals list size)
            assert len(acked) == len(set(acked)), \
                f"Duplicate acks: {len(acked)} total, {len(set(acked))} unique"

        finally:
            stop_consumers.set()
            mailbox.close()
            mailbox.purge()

    @pytest.mark.slow
    def test_reaper_under_load(self, redis_client):
        """
        Reaper correctly handles messages expiring under load.

        Simulates consumers that are slower than visibility timeout.
        """
        mailbox = RedisMailbox(
            name=f"reaper-stress-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.1,
        )

        num_messages = 50
        visibility_timeout = 1  # Short timeout

        try:
            # Send messages
            for i in range(num_messages):
                mailbox.send(f"msg-{i}".encode())

            # Receive but don't ack (let them expire)
            received_handles = []
            for _ in range(num_messages):
                msgs = mailbox.receive(visibility_timeout=visibility_timeout)
                if msgs:
                    received_handles.append((msgs[0].id, msgs[0].receipt_handle))

            # Wait for expiry
            time.sleep(visibility_timeout + 0.5)

            # All messages should be back in pending
            count = mailbox.approximate_count()
            assert count == num_messages, \
                f"Expected {num_messages} messages, got {count}"

            # Old handles should be rejected
            for msg_id, old_handle in received_handles[:5]:
                suffix = old_handle.split(":", 1)[1]
                with pytest.raises(ReceiptHandleExpiredError):
                    mailbox._acknowledge(msg_id, suffix)

        finally:
            mailbox.close()
            mailbox.purge()
```

## Makefile Integration

Add these targets to the project Makefile:

```makefile
# =============================================================================
# Formal Verification
# =============================================================================

.PHONY: tlaplus-check
tlaplus-check: ## Run TLC model checker on Redis mailbox spec
	@echo "Running TLC model checker..."
	@if command -v tlc >/dev/null 2>&1; then \
		cd specs/tla && tlc RedisMailboxMC.tla -config RedisMailboxMC.cfg -workers auto; \
	else \
		echo "TLC not installed. Install with: brew install tlaplus"; \
		exit 1; \
	fi

.PHONY: property-tests
property-tests: ## Run Hypothesis property-based tests
	uv run pytest tests/contrib/mailbox/test_redis_mailbox_properties.py \
		tests/contrib/mailbox/test_redis_mailbox_invariants.py \
		-v --hypothesis-show-statistics

.PHONY: stress-tests
stress-tests: ## Run concurrent stress tests
	uv run pytest tests/contrib/mailbox/test_redis_mailbox_stress.py \
		-v -m slow --timeout=120

.PHONY: verify-mailbox
verify-mailbox: tlaplus-check property-tests ## Run all mailbox verification
	@echo "All mailbox verification checks passed"
```

## CI Integration

Add verification to CI pipeline:

```yaml
# .github/workflows/verify.yml

name: Formal Verification

on:
  push:
    paths:
      - 'src/weakincentives/contrib/mailbox/_redis.py'
      - 'specs/tla/**'
      - 'tests/contrib/mailbox/test_redis_mailbox_*.py'
  pull_request:
    paths:
      - 'src/weakincentives/contrib/mailbox/_redis.py'
      - 'specs/tla/**'
      - 'tests/contrib/mailbox/test_redis_mailbox_*.py'

jobs:
  tlaplus:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Java
        uses: actions/setup-java@v4
        with:
          distribution: 'temurin'
          java-version: '17'

      - name: Cache TLA+ Tools
        id: cache-tlaplus
        uses: actions/cache@v4
        with:
          path: /usr/local/lib/tla2tools.jar
          key: tlaplus-v1.8.0

      - name: Install TLA+ Tools
        if: steps.cache-tlaplus.outputs.cache-hit != 'true'
        run: |
          wget -q https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
          sudo mv tla2tools.jar /usr/local/lib/

      - name: Run TLC
        run: |
          cd specs/tla
          java -jar /usr/local/lib/tla2tools.jar \
            -config RedisMailboxMC.cfg \
            RedisMailboxMC.tla \
            -workers auto

  property-tests:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run property tests
        run: |
          uv run pytest \
            tests/contrib/mailbox/test_redis_mailbox_properties.py \
            tests/contrib/mailbox/test_redis_mailbox_invariants.py \
            -v --hypothesis-show-statistics

      - name: Run stress tests
        run: |
          uv run pytest \
            tests/contrib/mailbox/test_redis_mailbox_stress.py \
            -v -m slow --timeout=120
```

## Maintenance Guidelines

### When to Update TLA+ Spec

Update `specs/tla/RedisMailbox.tla` when:

1. Adding new operations to RedisMailbox
1. Changing Lua script logic
1. Modifying state transitions
1. Adding new invariants

### When to Update Property Tests

Update property tests when:

1. Adding new public methods
1. Changing visibility timeout behavior
1. Modifying receipt handle generation
1. Adding new error conditions

### Verification Checklist

Before merging changes to `_redis.py`:

- [ ] `make tlaplus-check` passes
- [ ] `make property-tests` passes
- [ ] `make stress-tests` passes
- [ ] TLA+ spec updated if algorithm changed
- [ ] Property tests updated if new invariants added

## Assumptions

The formal verification is based on these assumptions about the runtime
environment. If any assumption is violated, the invariants may not hold.

### Redis Guarantees

1. **Lua Script Atomicity**: Redis executes Lua scripts atomically. No other
   command can interleave during script execution. This is the foundation
   for all multi-key atomic operations.

1. **FIFO List Ordering**: Redis LIST operations (LPUSH/RPOP) maintain FIFO
   order. Messages pushed first are popped first.

1. **Sorted Set Score Ordering**: ZRANGEBYSCORE returns members in score order.
   The reaper correctly finds expired messages by querying `score <= now`.

1. **Single-Threaded Execution**: Redis is single-threaded for command
   execution. This prevents race conditions between concurrent commands.

### Cluster Mode Assumptions

5. **Hash Tag Co-location**: All keys for a queue use the same hash tag
   (`{queue:name}`), ensuring they reside on the same shard.

1. **No Cross-Shard Transactions**: Operations are atomic only within a single
   queue. Cross-queue operations are not atomic.

1. **Eventual Replication**: During failover, recently written data may be lost
   if not yet replicated. Configure `min-replicas-to-write` for durability.

### Timing Assumptions

8. **Reaper Fairness**: The background reaper thread runs at least once every
   `reaper_interval` seconds. Under normal operation, this is 1 second.

1. **Clock Monotonicity**: `time.time()` returns monotonically increasing
   values. Clock skew or NTP adjustments could affect visibility timeouts.

1. **Visibility Timeout > Processing Time**: The visibility timeout should
   exceed the maximum expected processing time, or consumers should call
   `extend_visibility()` periodically.

### Consumer Assumptions

11. **Handle Secrecy**: Receipt handles are not shared between consumers. A
    consumer uses only handles from its own receive calls.

01. **Single Acknowledgment**: A consumer attempts to acknowledge each message
    at most once. The implementation is idempotent, but redundant acks waste
    resources.

### Model Limitations

The TLA+ model makes these simplifications:

1. **Finite State Space**: The model uses small constants (MaxMessages=3,
   NumConsumers=2) for exhaustive checking. Larger models require simulation.

1. **Abstract Time**: Time advances in discrete ticks, not continuous. This
   may miss timing-dependent edge cases.

1. **No Network Partitions**: The model assumes reliable Redis connectivity.
   Network failures are not modeled.

1. **No Message Body**: Message bodies are abstracted to simple identifiers.
   Serialization errors are not modeled.

## References

- [TLA+ Home Page](https://lamport.azurewebsites.net/tla/tla.html)
- [Learn TLA+](https://learntla.com/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Hypothesis Stateful Testing](https://hypothesis.readthedocs.io/en/latest/stateful.html)
- [Redis Lua Scripting](https://redis.io/docs/interact/programmability/eval-intro/)

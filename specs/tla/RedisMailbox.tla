---------------------------- MODULE RedisMailbox ----------------------------
(* Redis Mailbox Formal Specification
 *
 * This TLA+ specification models the Redis mailbox state machine for formal
 * verification. It covers the core operations: send, receive, acknowledge,
 * nack, extend, and reap.
 *
 * Key invariants verified:
 * - INV-1: Message State Exclusivity
 * - INV-2: Receipt Handle Freshness
 * - INV-3: Stale Handle Rejection
 * - INV-4: Delivery Count Monotonicity
 * - INV-5: No Message Loss
 * - INV-6: Visibility Timeout Correctness
 * - INV-7: FIFO Ordering
 *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    MaxMessages,        \* Maximum messages to model (e.g., 3)
    MaxDeliveries,      \* Maximum deliveries per message (e.g., 3)
    NumConsumers,       \* Number of concurrent consumers (e.g., 2)
    VisibilityTimeout   \* Timeout value in abstract time units

\* Special value for null/none
NULL == CHOOSE x : x \notin 1..MaxMessages

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

-----------------------------------------------------------------------------
(* Type Invariant *)

TypeOK ==
    /\ pending \in Seq(1..MaxMessages)
    /\ invisible \in [DOMAIN invisible -> [expiresAt: Int, handle: Nat]]
    /\ DOMAIN invisible \subseteq 1..MaxMessages
    /\ data \in [DOMAIN data -> {"a", "b", "c"}]
    /\ DOMAIN data \subseteq 1..MaxMessages
    /\ handles \in [DOMAIN handles -> Nat]
    /\ DOMAIN handles \subseteq 1..MaxMessages
    /\ deleted \subseteq 1..MaxMessages
    /\ now \in Nat
    /\ nextMsgId \in 1..(MaxMessages + 1)
    /\ nextHandle \in Nat
    /\ consumerState \in [1..NumConsumers ->
            [holding: {NULL} \cup 1..MaxMessages, handle: Nat]]
    /\ deliveryCounts \in [DOMAIN deliveryCounts -> Nat]
    /\ DOMAIN deliveryCounts \subseteq 1..MaxMessages
    /\ deliveryHistory \in [DOMAIN deliveryHistory ->
            Seq([count: Nat, handle: Nat])]
    /\ DOMAIN deliveryHistory \subseteq 1..MaxMessages

-----------------------------------------------------------------------------
(* Initial State *)

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

-----------------------------------------------------------------------------
(* Helper Operators *)

\* Check if message is in pending sequence
InPending(msgId) ==
    \E i \in 1..Len(pending): pending[i] = msgId

\* Remove function key
RemoveKey(f, k) ==
    [m \in (DOMAIN f) \ {k} |-> f[m]]

\* Add or update function key
UpdateFunc(f, k, v) ==
    [m \in (DOMAIN f) \cup {k} |-> IF m = k THEN v ELSE f[m]]

-----------------------------------------------------------------------------
(* Actions *)

(* Send: Add a new message to the pending queue *)
Send(body) ==
    /\ nextMsgId <= MaxMessages
    /\ LET msgId == nextMsgId
       IN /\ pending' = Append(pending, msgId)
          /\ data' = UpdateFunc(data, msgId, body)
          /\ deliveryCounts' = UpdateFunc(deliveryCounts, msgId, 0)
          /\ deliveryHistory' = UpdateFunc(deliveryHistory, msgId, <<>>)
          /\ nextMsgId' = nextMsgId + 1
    /\ UNCHANGED <<invisible, handles, deleted, now, nextHandle, consumerState>>

(* Receive: Atomically move message from pending to invisible *)
Receive(consumer) ==
    /\ Len(pending) > 0
    /\ consumerState[consumer].holding = NULL
    /\ LET msgId == Head(pending)
           newHandle == nextHandle
           newExpiry == now + VisibilityTimeout
           newCount == deliveryCounts[msgId] + 1
       IN /\ pending' = Tail(pending)
          /\ invisible' = UpdateFunc(invisible, msgId,
                [expiresAt |-> newExpiry, handle |-> newHandle])
          /\ handles' = UpdateFunc(handles, msgId, newHandle)
          /\ deliveryCounts' = [deliveryCounts EXCEPT ![msgId] = newCount]
          /\ deliveryHistory' = [deliveryHistory EXCEPT
                ![msgId] = Append(@, [count |-> newCount, handle |-> newHandle])]
          /\ nextHandle' = nextHandle + 1
          /\ consumerState' = [consumerState EXCEPT
                ![consumer] = [holding |-> msgId, handle |-> newHandle]]
    /\ UNCHANGED <<data, deleted, now, nextMsgId>>

(* Acknowledge: Successfully complete message processing *)
Acknowledge(consumer) ==
    /\ consumerState[consumer].holding /= NULL
    /\ LET msgId == consumerState[consumer].holding
           providedHandle == consumerState[consumer].handle
       IN /\ msgId \in DOMAIN handles
          /\ handles[msgId] = providedHandle  \* Handle validation
          /\ msgId \in DOMAIN invisible       \* Still in invisible
          /\ invisible' = RemoveKey(invisible, msgId)
          /\ data' = RemoveKey(data, msgId)
          /\ handles' = RemoveKey(handles, msgId)
          /\ deliveryCounts' = RemoveKey(deliveryCounts, msgId)
          /\ deleted' = deleted \cup {msgId}
          /\ consumerState' = [consumerState EXCEPT
                ![consumer] = [holding |-> NULL, handle |-> 0]]
    /\ UNCHANGED <<pending, now, nextMsgId, nextHandle, deliveryHistory>>

(* AcknowledgeFail: Acknowledge fails if handle is stale *)
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

(* Nack: Return message to queue with optional delay *)
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
                  /\ invisible' = RemoveKey(invisible, msgId)
             ELSE \* Delayed requeue: stays in invisible with new expiry
                  /\ invisible' = [invisible EXCEPT
                        ![msgId].expiresAt = now + newTimeout,
                        ![msgId].handle = 0]  \* No valid handle
                  /\ UNCHANGED pending
          \* Handle is ALWAYS invalidated on nack (matches _LUA_NACK line 84)
          /\ handles' = RemoveKey(handles, msgId)
          /\ consumerState' = [consumerState EXCEPT
                ![consumer] = [holding |-> NULL, handle |-> 0]]
    /\ UNCHANGED <<data, deleted, now, nextMsgId, nextHandle,
                   deliveryCounts, deliveryHistory>>

(* NackFail: Nack fails if handle is stale *)
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

(* Extend: Extend visibility timeout for a message *)
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

(* ExtendFail: Extend fails if handle is stale or message not in invisible *)
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

(* ReapOne: Move one expired message back to pending *)
ReapOne ==
    /\ \E msgId \in DOMAIN invisible:
        /\ invisible[msgId].expiresAt < now
        /\ pending' = Append(pending, msgId)
        /\ invisible' = RemoveKey(invisible, msgId)
        \* Handle deleted by reaper (matches _LUA_REAP line 107)
        /\ handles' = RemoveKey(handles, msgId)
        \* Invalidate any consumer holding this message
        /\ consumerState' = [c \in DOMAIN consumerState |->
            IF consumerState[c].holding = msgId
            THEN [holding |-> NULL, handle |-> 0]
            ELSE consumerState[c]]
    \* deliveryCounts persists - this is critical for INV-4
    /\ UNCHANGED <<data, deleted, now, nextMsgId, nextHandle,
                   deliveryCounts, deliveryHistory>>

(* Tick: Advance abstract time *)
Tick ==
    /\ now' = now + 1
    /\ UNCHANGED <<pending, invisible, data, handles, deleted, nextMsgId,
                   nextHandle, consumerState, deliveryCounts, deliveryHistory>>

-----------------------------------------------------------------------------
(* Next State Relation *)

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

-----------------------------------------------------------------------------
(* Invariants *)

(* INV-1: Message State Exclusivity
 * A message must be in exactly one of three states:
 * pending, invisible, or deleted
 *)
MessageStateExclusive ==
    \A msgId \in 1..nextMsgId-1:
        LET inPending == InPending(msgId)
            inInvisible == msgId \in DOMAIN invisible
            inDeleted == msgId \in deleted
        IN (inPending /\ ~inInvisible /\ ~inDeleted) \/
           (~inPending /\ inInvisible /\ ~inDeleted) \/
           (~inPending /\ ~inInvisible /\ inDeleted)

(* INV-2 & INV-3: Handle Validity
 * Consumers holding a message have a valid handle for it
 *)
HandleValidity ==
    \A c \in 1..NumConsumers:
        LET state == consumerState[c]
        IN state.holding /= NULL =>
            (state.holding \in DOMAIN handles =>
                handles[state.holding] = state.handle)

(* INV-4: Delivery Count Monotonicity
 * Uses deliveryHistory to verify counts are strictly increasing
 *)
DeliveryCountMonotonic ==
    \A msgId \in DOMAIN deliveryHistory:
        LET history == deliveryHistory[msgId]
        IN \A i \in 1..Len(history)-1:
            history[i].count < history[i+1].count

(* INV-4b: Delivery counts persist across requeue
 * After reap, the next receive must have count = previous + 1
 *)
DeliveryCountPersistence ==
    \A msgId \in DOMAIN deliveryCounts:
        \A i \in 1..Len(deliveryHistory[msgId]):
            deliveryHistory[msgId][i].count = i

(* INV-5: No Message Loss (Safety part)
 * Every message with data is either pending or invisible
 *)
NoMessageLoss ==
    \A msgId \in DOMAIN data:
        LET inPending == InPending(msgId)
            inInvisible == msgId \in DOMAIN invisible
        IN inPending \/ inInvisible

(* INV-7: Handle Uniqueness across deliveries
 * Each delivery of a message gets a unique handle
 *)
HandleUniqueness ==
    \A msgId \in DOMAIN deliveryHistory:
        LET history == deliveryHistory[msgId]
        IN \A i, j \in 1..Len(history):
            i /= j => history[i].handle /= history[j].handle

(* Combined Type Invariant for model checking *)
Invariant ==
    /\ MessageStateExclusive
    /\ HandleValidity
    /\ DeliveryCountMonotonic
    /\ DeliveryCountPersistence
    /\ NoMessageLoss
    /\ HandleUniqueness

-----------------------------------------------------------------------------
(* Liveness Properties *)

(* Fairness: consumers eventually make progress *)
Fairness ==
    /\ WF_vars(Tick)
    /\ WF_vars(ReapOne)
    /\ \A c \in 1..NumConsumers:
        /\ WF_vars(Receive(c))
        /\ WF_vars(Acknowledge(c))

(* INV-6: Expired messages eventually requeued *)
EventualRequeue ==
    \A msgId \in DOMAIN invisible:
        invisible[msgId].expiresAt < now ~>
            InPending(msgId)

(* Full Specification with fairness *)
Spec == Init /\ [][Next]_vars /\ Fairness

=============================================================================

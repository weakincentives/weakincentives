---------------------------- MODULE RedisMailbox ----------------------------
(* Generated from Python formal specification metadata *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    MaxMessages,
    MaxDeliveries,
    NumConsumers,
    VisibilityTimeout

VARIABLES
    pending,  \* Sequence of message IDs in pending list
    invisible,  \* msg_id -> {expiresAt, handle}
    data,  \* msg_id -> body (or NULL if deleted)
    handles,  \* msg_id -> current valid handle suffix
    deleted,  \* Set of deleted message IDs
    now,  \* Abstract time counter
    nextMsgId,  \* Counter for generating message IDs
    nextHandle,  \* Counter for generating handle suffixes
    consumerState,  \* consumer_id -> {holding, handle}
    deliveryCounts,  \* msg_id -> count (persists across requeue)
    deliveryHistory  \* msg_id -> Seq of (count, handle) for INV-4

vars == <<pending, invisible, data, handles, deleted, now, nextMsgId, nextHandle, consumerState, deliveryCounts, deliveryHistory>>

-----------------------------------------------------------------------------
(* Helper Operators *)

NULL ==
    0

InPending(msgId) ==
    \E i \in 1..Len(pending): pending[i] = msgId

RemoveKey(f, k) ==
    [m \in (DOMAIN f) \ {k} |-> f[m]]

UpdateFunc(f, k, v) ==
    [m \in (DOMAIN f) \cup {k} |-> IF m = k THEN v ELSE f[m]]

-----------------------------------------------------------------------------
(* Initial State *)

Init ==
    pending = <<>>
    /\ invisible = [x \in {} |-> 0]
    /\ data = [x \in {} |-> 0]
    /\ handles = [x \in {} |-> 0]
    /\ deleted = {}
    /\ now = 0
    /\ nextMsgId = 0
    /\ nextHandle = 0
    /\ consumerState = [c \in 1..NumConsumers |-> [holding |-> NULL, handle |-> 0]]
    /\ deliveryCounts = [x \in {} |-> 0]
    /\ deliveryHistory = [x \in {} |-> 0]

-----------------------------------------------------------------------------
(* Actions *)

(* Add a new message to the pending queue (immediate visibility) *)
Send(body) ==
    /\ nextMsgId <= MaxMessages
    /\ pending' = Append(pending, nextMsgId)
    /\ data' = UpdateFunc(data, nextMsgId, body)
    /\ deliveryCounts' = UpdateFunc(deliveryCounts, nextMsgId, 0)
    /\ deliveryHistory' = UpdateFunc(deliveryHistory, nextMsgId, <<>>)
    /\ nextMsgId' = nextMsgId + 1
    /\ UNCHANGED <<invisible, handles, deleted, now, nextHandle, consumerState>>

(* Atomically move message from pending to invisible *)
Receive(consumer) ==
    /\ Len(pending) > 0
    /\ consumerState[consumer].holding = NULL
    /\ pending' = Tail(pending)
    /\ invisible' = UpdateFunc(invisible, Head(pending), [expiresAt |-> now + VisibilityTimeout, handle |-> nextHandle])
    /\ handles' = UpdateFunc(handles, Head(pending), nextHandle)
    /\ deliveryCounts' = [deliveryCounts EXCEPT ![Head(pending)] = @ + 1]
    /\ deliveryHistory' = [deliveryHistory EXCEPT ![Head(pending)] = Append(@, [count |-> deliveryCounts[Head(pending)] + 1, handle |-> nextHandle])]
    /\ nextHandle' = nextHandle + 1
    /\ consumerState' = [consumerState EXCEPT ![consumer] = [holding |-> Head(pending), handle |-> nextHandle]]
    /\ UNCHANGED <<data, deleted, now, nextMsgId>>

(* Successfully complete message processing *)
Acknowledge(consumer) ==
    /\ consumerState[consumer].holding /= NULL
    /\ consumerState[consumer].holding \in DOMAIN handles
    /\ handles[consumerState[consumer].holding] = consumerState[consumer].handle
    /\ consumerState[consumer].holding \in DOMAIN invisible
    /\ invisible' = RemoveKey(invisible, consumerState[consumer].holding)
    /\ data' = RemoveKey(data, consumerState[consumer].holding)
    /\ handles' = RemoveKey(handles, consumerState[consumer].holding)
    /\ deliveryCounts' = RemoveKey(deliveryCounts, consumerState[consumer].holding)
    /\ deleted' = deleted \cup {consumerState[consumer].holding}
    /\ consumerState' = [consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]
    /\ UNCHANGED <<pending, now, nextMsgId, nextHandle, deliveryHistory>>

(* Acknowledge fails if handle is stale *)
AcknowledgeFail(consumer) ==
    /\ consumerState[consumer].holding /= NULL
    /\ consumerState[consumer].holding \notin DOMAIN handles \/ handles[consumerState[consumer].holding] /= consumerState[consumer].handle \/ consumerState[consumer].holding \notin DOMAIN invisible
    /\ consumerState' = [consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]
    /\ UNCHANGED <<pending, invisible, data, handles, deleted, now, nextMsgId, nextHandle, deliveryCounts, deliveryHistory>>

(* Return message to queue with optional delay *)
Nack(consumer, newTimeout) ==
    /\ consumerState[consumer].holding /= NULL
    /\ consumerState[consumer].holding \in DOMAIN handles
    /\ handles[consumerState[consumer].holding] = consumerState[consumer].handle
    /\ consumerState[consumer].holding \in DOMAIN invisible
    /\ pending' = IF newTimeout = 0 THEN Append(pending, consumerState[consumer].holding) ELSE pending
    /\ invisible' = IF newTimeout = 0 THEN RemoveKey(invisible, consumerState[consumer].holding) ELSE [invisible EXCEPT ![consumerState[consumer].holding].expiresAt = now + newTimeout, ![consumerState[consumer].holding].handle = 0]
    /\ handles' = RemoveKey(handles, consumerState[consumer].holding)
    /\ consumerState' = [consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]
    /\ UNCHANGED <<data, deleted, now, nextMsgId, nextHandle, deliveryCounts, deliveryHistory>>

(* Nack fails if handle is stale *)
NackFail(consumer) ==
    /\ consumerState[consumer].holding /= NULL
    /\ consumerState[consumer].holding \notin DOMAIN handles \/ handles[consumerState[consumer].holding] /= consumerState[consumer].handle \/ consumerState[consumer].holding \notin DOMAIN invisible
    /\ consumerState' = [consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]
    /\ UNCHANGED <<pending, invisible, data, handles, deleted, now, nextMsgId, nextHandle, deliveryCounts, deliveryHistory>>

(* Extend visibility timeout for a message *)
Extend(consumer, newTimeout) ==
    /\ consumerState[consumer].holding /= NULL
    /\ consumerState[consumer].holding \in DOMAIN handles
    /\ handles[consumerState[consumer].holding] = consumerState[consumer].handle
    /\ consumerState[consumer].holding \in DOMAIN invisible
    /\ invisible' = [invisible EXCEPT ![consumerState[consumer].holding].expiresAt = now + newTimeout]
    /\ UNCHANGED <<pending, data, handles, deleted, now, nextMsgId, nextHandle, consumerState, deliveryCounts, deliveryHistory>>

(* Extend fails if handle is stale or message not in invisible *)
ExtendFail(consumer) ==
    /\ consumerState[consumer].holding /= NULL
    /\ consumerState[consumer].holding \notin DOMAIN handles \/ handles[consumerState[consumer].holding] /= consumerState[consumer].handle \/ consumerState[consumer].holding \notin DOMAIN invisible
    /\ consumerState' = [consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]
    /\ UNCHANGED <<pending, invisible, data, handles, deleted, now, nextMsgId, nextHandle, deliveryCounts, deliveryHistory>>

(* Move one expired message back to pending *)
ReapOne ==
    /\ \E msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now
    /\ pending' = Append(pending, CHOOSE msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now)
    /\ invisible' = RemoveKey(invisible, CHOOSE msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now)
    /\ handles' = RemoveKey(handles, CHOOSE msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now)
    /\ consumerState' = [c \in DOMAIN consumerState |-> IF consumerState[c].holding = (CHOOSE msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now) THEN [holding |-> NULL, handle |-> 0] ELSE consumerState[c]]
    /\ UNCHANGED <<data, deleted, now, nextMsgId, nextHandle, deliveryCounts, deliveryHistory>>

(* Advance abstract time *)
Tick ==
    /\ now' = now + 1
    /\ UNCHANGED <<pending, invisible, data, handles, deleted, nextMsgId, nextHandle, consumerState, deliveryCounts, deliveryHistory>>

-----------------------------------------------------------------------------
(* Next State *)

Next ==
    \/ \E body \in 1..MaxMessages : Send(body)
    \/ \E consumer \in 1..NumConsumers : Receive(consumer)
    \/ \E consumer \in 1..NumConsumers : Acknowledge(consumer)
    \/ \E consumer \in 1..NumConsumers : AcknowledgeFail(consumer)
    \/ \E consumer \in 1..NumConsumers, newTimeout \in 0..VisibilityTimeout*2 : Nack(consumer, newTimeout)
    \/ \E consumer \in 1..NumConsumers : NackFail(consumer)
    \/ \E consumer \in 1..NumConsumers, newTimeout \in 1..VisibilityTimeout*2 : Extend(consumer, newTimeout)
    \/ \E consumer \in 1..NumConsumers : ExtendFail(consumer)
    \/ ReapOne
    \/ Tick

-----------------------------------------------------------------------------
(* Specification *)

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Invariants *)

(* INV-1: A message must be in exactly one state: pending, invisible, or deleted *)
MessageStateExclusive ==
    \A msgId \in 1..nextMsgId-1:
        LET inPending == InPending(msgId)
            inInvisible == msgId \in DOMAIN invisible
            inDeleted == msgId \in deleted
        IN (inPending /\ ~inInvisible /\ ~inDeleted) \/
           (~inPending /\ inInvisible /\ ~inDeleted) \/
           (~inPending /\ ~inInvisible /\ inDeleted)

(* INV-2-3: Consumers holding a message have a valid handle for it *)
HandleValidity ==
    \A c \in 1..NumConsumers:
        LET state == consumerState[c]
        IN state.holding /= NULL =>
            (state.holding \in DOMAIN handles =>
                handles[state.holding] = state.handle)

(* INV-4: Delivery counts are strictly increasing *)
DeliveryCountMonotonic ==
    \A msgId \in DOMAIN deliveryHistory:
        LET history == deliveryHistory[msgId]
        IN \A i \in 1..Len(history)-1:
            history[i].count < history[i+1].count

(* INV-4b: Delivery counts persist across requeue *)
DeliveryCountPersistence ==
    \A msgId \in DOMAIN deliveryCounts:
        \A i \in 1..Len(deliveryHistory[msgId]):
            deliveryHistory[msgId][i].count = i

(* INV-5: Every message with data is either pending or invisible *)
NoMessageLoss ==
    \A msgId \in DOMAIN data:
        LET inPending == InPending(msgId)
            inInvisible == msgId \in DOMAIN invisible
        IN inPending \/ inInvisible

(* INV-7: Each delivery of a message gets a unique handle *)
HandleUniqueness ==
    \A msgId \in DOMAIN deliveryHistory:
        LET history == deliveryHistory[msgId]
        IN \A i, j \in 1..Len(history):
            i /= j => history[i].handle /= history[j].handle

-----------------------------------------------------------------------------
(* State Constraint *)

StateConstraint == now <= 1

=============================================================================
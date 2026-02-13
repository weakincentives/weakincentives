# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TLA+ formal specification for RedisMailbox."""

from __future__ import annotations

from typing import Any

from weakincentives.formal import Action, ActionParameter, Invariant, StateVar

REDIS_MAILBOX_SPEC_KWARGS: dict[str, Any] = {
    "module": "RedisMailbox",
    "extends": ("Integers", "Sequences", "FiniteSets", "TLC"),
    "constants": {
        "MaxMessages": 2,
        "MaxDeliveries": 2,
        "NumConsumers": 2,
        "VisibilityTimeout": 2,
    },
    "state_vars": [
        StateVar(
            "pending", "Seq(MessageId)", "Sequence of message IDs in pending list"
        ),
        StateVar("invisible", "Function", "msg_id -> {expiresAt, handle}"),
        StateVar("data", "Function", "msg_id -> body (or NULL if deleted)"),
        StateVar("handles", "Function", "msg_id -> current valid handle suffix"),
        StateVar("deleted", "Set", "Set of deleted message IDs"),
        StateVar("now", "Nat", "Abstract time counter"),
        StateVar(
            "nextMsgId",
            "Nat",
            "Counter for generating message IDs",
            initial_value="1",
        ),
        StateVar(
            "nextHandle",
            "Nat",
            "Counter for generating handle suffixes",
            initial_value="1",
        ),
        StateVar(
            "consumerState",
            "Function",
            "consumer_id -> {holding, handle}",
            initial_value="[c \\in 1..NumConsumers |-> [holding |-> NULL, handle |-> 0]]",
        ),
        StateVar(
            "deliveryCounts",
            "Function",
            "msg_id -> count (persists across requeue)",
        ),
        StateVar(
            "deliveryHistory",
            "Function",
            "msg_id -> Seq of (count, handle) for INV-4",
        ),
    ],
    "helpers": {
        "NULL": "0",
        "InPending(msgId)": r"\E i \in 1..Len(pending): pending[i] = msgId",
        "RemoveKey(f, k)": r"[m \in (DOMAIN f) \ {k} |-> f[m]]",
        "UpdateFunc(f, k, v)": r"[m \in (DOMAIN f) \cup {k} |-> IF m = k THEN v ELSE f[m]]",
    },
    "actions": [
        Action(
            name="Send",
            parameters=(ActionParameter("body", "1..MaxMessages"),),
            preconditions=("nextMsgId <= MaxMessages",),
            updates={
                "pending": "Append(pending, nextMsgId)",
                "data": "UpdateFunc(data, nextMsgId, body)",
                "deliveryCounts": "UpdateFunc(deliveryCounts, nextMsgId, 0)",
                "deliveryHistory": "UpdateFunc(deliveryHistory, nextMsgId, <<>>)",
                "nextMsgId": "nextMsgId + 1",
            },
            description="Add a new message to the pending queue (immediate visibility)",
        ),
        Action(
            name="Receive",
            parameters=(ActionParameter("consumer", "1..NumConsumers"),),
            preconditions=(
                "Len(pending) > 0",
                "consumerState[consumer].holding = NULL",
            ),
            updates={
                "pending": "Tail(pending)",
                "invisible": "UpdateFunc(invisible, Head(pending), [expiresAt |-> now + VisibilityTimeout, handle |-> nextHandle])",
                "handles": "UpdateFunc(handles, Head(pending), nextHandle)",
                "deliveryCounts": "[deliveryCounts EXCEPT ![Head(pending)] = @ + 1]",
                "deliveryHistory": "[deliveryHistory EXCEPT ![Head(pending)] = Append(@, [count |-> deliveryCounts[Head(pending)] + 1, handle |-> nextHandle])]",
                "nextHandle": "nextHandle + 1",
                "consumerState": "[consumerState EXCEPT ![consumer] = [holding |-> Head(pending), handle |-> nextHandle]]",
            },
            description="Atomically move message from pending to invisible",
        ),
        Action(
            name="Acknowledge",
            parameters=(ActionParameter("consumer", "1..NumConsumers"),),
            preconditions=(
                "consumerState[consumer].holding /= NULL",
                r"consumerState[consumer].holding \in DOMAIN handles",
                "handles[consumerState[consumer].holding] = consumerState[consumer].handle",
                r"consumerState[consumer].holding \in DOMAIN invisible",
            ),
            updates={
                "invisible": "RemoveKey(invisible, consumerState[consumer].holding)",
                "data": "RemoveKey(data, consumerState[consumer].holding)",
                "handles": "RemoveKey(handles, consumerState[consumer].holding)",
                "deliveryCounts": "RemoveKey(deliveryCounts, consumerState[consumer].holding)",
                r"deleted": r"deleted \cup {consumerState[consumer].holding}",
                "consumerState": "[consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]",
            },
            description="Successfully complete message processing",
        ),
        Action(
            name="AcknowledgeFail",
            parameters=(ActionParameter("consumer", "1..NumConsumers"),),
            preconditions=(
                "consumerState[consumer].holding /= NULL",
                r"consumerState[consumer].holding \notin DOMAIN handles \/ handles[consumerState[consumer].holding] /= consumerState[consumer].handle \/ consumerState[consumer].holding \notin DOMAIN invisible",
            ),
            updates={
                "consumerState": "[consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]",
            },
            description="Acknowledge fails if handle is stale",
        ),
        Action(
            name="Nack",
            parameters=(
                ActionParameter("consumer", "1..NumConsumers"),
                ActionParameter("newTimeout", "0..VisibilityTimeout"),
            ),
            preconditions=(
                "consumerState[consumer].holding /= NULL",
                r"consumerState[consumer].holding \in DOMAIN handles",
                "handles[consumerState[consumer].holding] = consumerState[consumer].handle",
                r"consumerState[consumer].holding \in DOMAIN invisible",
            ),
            updates={
                "pending": "IF newTimeout = 0 THEN Append(pending, consumerState[consumer].holding) ELSE pending",
                "invisible": "IF newTimeout = 0 THEN RemoveKey(invisible, consumerState[consumer].holding) ELSE [invisible EXCEPT ![consumerState[consumer].holding].expiresAt = now + newTimeout, ![consumerState[consumer].holding].handle = 0]",
                "handles": "RemoveKey(handles, consumerState[consumer].holding)",
                "consumerState": "[consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]",
            },
            description="Return message to queue with optional delay",
        ),
        Action(
            name="NackFail",
            parameters=(ActionParameter("consumer", "1..NumConsumers"),),
            preconditions=(
                "consumerState[consumer].holding /= NULL",
                r"consumerState[consumer].holding \notin DOMAIN handles \/ handles[consumerState[consumer].holding] /= consumerState[consumer].handle \/ consumerState[consumer].holding \notin DOMAIN invisible",
            ),
            updates={
                "consumerState": "[consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]",
            },
            description="Nack fails if handle is stale",
        ),
        Action(
            name="Extend",
            parameters=(
                ActionParameter("consumer", "1..NumConsumers"),
                ActionParameter("newTimeout", "1..VisibilityTimeout"),
            ),
            preconditions=(
                "consumerState[consumer].holding /= NULL",
                r"consumerState[consumer].holding \in DOMAIN handles",
                "handles[consumerState[consumer].holding] = consumerState[consumer].handle",
                r"consumerState[consumer].holding \in DOMAIN invisible",
            ),
            updates={
                "invisible": "[invisible EXCEPT ![consumerState[consumer].holding].expiresAt = now + newTimeout]",
            },
            description="Extend visibility timeout for a message",
        ),
        Action(
            name="ExtendFail",
            parameters=(ActionParameter("consumer", "1..NumConsumers"),),
            preconditions=(
                "consumerState[consumer].holding /= NULL",
                r"consumerState[consumer].holding \notin DOMAIN handles \/ handles[consumerState[consumer].holding] /= consumerState[consumer].handle \/ consumerState[consumer].holding \notin DOMAIN invisible",
            ),
            updates={
                "consumerState": "[consumerState EXCEPT ![consumer] = [holding |-> NULL, handle |-> 0]]",
            },
            description="Extend fails if handle is stale or message not in invisible",
        ),
        Action(
            name="ReapOne",
            parameters=(),
            preconditions=(
                r"\E msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now",
            ),
            updates={
                r"pending": r"Append(pending, CHOOSE msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now)",
                r"invisible": r"RemoveKey(invisible, CHOOSE msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now)",
                r"handles": r"RemoveKey(handles, CHOOSE msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now)",
                r"consumerState": r"[c \in DOMAIN consumerState |-> IF consumerState[c].holding = (CHOOSE msgId \in DOMAIN invisible: invisible[msgId].expiresAt < now) THEN [holding |-> NULL, handle |-> 0] ELSE consumerState[c]]",
            },
            description="Move one expired message back to pending",
        ),
        Action(
            name="Tick",
            parameters=(),
            preconditions=(),
            updates={"now": "now + 1"},
            description="Advance abstract time",
        ),
    ],
    "invariants": [
        Invariant(
            id="INV-1",
            name="MessageStateExclusive",
            predicate=r"""
\A msgId \in 1..nextMsgId-1:
    LET inPending == InPending(msgId)
        inInvisible == msgId \in DOMAIN invisible
        inDeleted == msgId \in deleted
    IN (inPending /\ ~inInvisible /\ ~inDeleted) \/
       (~inPending /\ inInvisible /\ ~inDeleted) \/
       (~inPending /\ ~inInvisible /\ inDeleted)
""".strip(),
            description="A message must be in exactly one state: pending, invisible, or deleted",
        ),
        Invariant(
            id="INV-2-3",
            name="HandleValidity",
            predicate=r"""
\A c \in 1..NumConsumers:
    LET state == consumerState[c]
    IN state.holding /= NULL =>
        (state.holding \in DOMAIN handles =>
            handles[state.holding] = state.handle)
""".strip(),
            description="Consumers holding a message have a valid handle for it",
        ),
        Invariant(
            id="INV-4",
            name="DeliveryCountMonotonic",
            predicate=r"""
\A msgId \in DOMAIN deliveryHistory:
    LET history == deliveryHistory[msgId]
    IN \A i \in 1..Len(history)-1:
        history[i].count < history[i+1].count
""".strip(),
            description="Delivery counts are strictly increasing",
        ),
        Invariant(
            id="INV-4b",
            name="DeliveryCountPersistence",
            predicate=r"""
\A msgId \in DOMAIN deliveryCounts:
    \A i \in 1..Len(deliveryHistory[msgId]):
        deliveryHistory[msgId][i].count = i
""".strip(),
            description="Delivery counts persist across requeue",
        ),
        Invariant(
            id="INV-5",
            name="NoMessageLoss",
            predicate=r"""
\A msgId \in DOMAIN data:
    LET inPending == InPending(msgId)
        inInvisible == msgId \in DOMAIN invisible
    IN inPending \/ inInvisible
""".strip(),
            description="Every message with data is either pending or invisible",
        ),
        Invariant(
            id="INV-7",
            name="HandleUniqueness",
            predicate=r"""
\A msgId \in DOMAIN deliveryHistory:
    LET history == deliveryHistory[msgId]
    IN \A i, j \in 1..Len(history):
        i /= j => history[i].handle /= history[j].handle
""".strip(),
            description="Each delivery of a message gets a unique handle",
        ),
        Invariant(
            id="INV-8",
            name="PendingNoDuplicates",
            predicate=r"""
\A i, j \in 1..Len(pending):
    i /= j => pending[i] /= pending[j]
""".strip(),
            description="The pending queue contains no duplicate message IDs",
        ),
        Invariant(
            id="INV-9",
            name="DataIntegrity",
            predicate=r"""
\A msgId \in 1..nextMsgId-1:
    (InPending(msgId) \/ msgId \in DOMAIN invisible) => msgId \in DOMAIN data
""".strip(),
            description="Every message in pending or invisible has associated data",
        ),
    ],
    "constraint": "now <= 2",
}

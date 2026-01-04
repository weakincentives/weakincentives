#!/usr/bin/env python3
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

"""Example: Embedding TLA+ specs with Python implementation.

This demonstrates how to use the @formal_spec decorator to co-locate
TLA+ formal specifications with Python code.

Run fast extraction (no model checking):
    make verify-formal-fast

Run full verification with TLC model checking:
    make verify-formal
"""

from __future__ import annotations

from weakincentives.dbc import ensure, invariant, require
from weakincentives.formal import Action, Invariant, StateVar, formal_spec

# =============================================================================
# Example 1: Simple Counter with Embedded Spec
# =============================================================================


@formal_spec(
    module="Counter",
    state_vars=[
        StateVar("count", "Nat", "Current count value"),
    ],
    actions=[
        Action(
            name="Increment",
            parameters=(),
            preconditions=(),
            updates={"count": "count + 1"},
            description="Increment counter by 1",
        ),
        Action(
            name="Decrement",
            parameters=(),
            preconditions=("count > 0",),
            updates={"count": "count - 1"},
            description="Decrement counter by 1 (only if positive)",
        ),
        Action(
            name="Reset",
            parameters=(),
            preconditions=(),
            updates={"count": "0"},
            description="Reset counter to zero",
        ),
    ],
    invariants=[
        Invariant(
            id="INV-1",
            name="NonNegative",
            predicate="count >= 0",
            description="Counter is always non-negative",
        ),
    ],
    constants={"MaxCount": 100},
)
@invariant(lambda self: self.count >= 0)
class Counter:
    """Simple counter with non-negative invariant.

    The @formal_spec decorator embeds a TLA+ specification that
    can be extracted and model-checked.
    """

    def __init__(self) -> None:
        self.count = 0

    def increment(self) -> None:
        """Increment the counter."""
        self.count += 1

    @require(lambda self: self.count > 0)
    def decrement(self) -> None:
        """Decrement the counter (only if positive)."""
        self.count -= 1

    def reset(self) -> None:
        """Reset counter to zero."""
        self.count = 0


# =============================================================================
# Example 2: Mailbox Receive Operation (Simplified)
# =============================================================================


@formal_spec(
    module="SimpleMailbox",
    extends=("Integers", "Sequences", "FiniteSets"),
    constants={
        "MaxMessages": 3,
        "VisibilityTimeout": 30,
    },
    state_vars=[
        StateVar(
            "pending",
            "Seq(MessageId)",
            "Sequence of message IDs in pending queue",
        ),
        StateVar(
            "invisible",
            "Function",
            "Function from message ID to invisibility metadata",
        ),
        StateVar(
            "deliveryCounts",
            "Function",
            "Function from message ID to delivery count",
        ),
    ],
    helpers={
        "NULL": "0",
        "InPending(msgId)": r"\E i \in 1..Len(pending): pending[i] = msgId",
    },
    actions=[
        Action(
            name="Send",
            parameters=("msgId",),
            preconditions=(
                "msgId \\notin DOMAIN deliveryCounts",
                "Len(pending) < MaxMessages",
            ),
            updates={
                "pending": "Append(pending, msgId)",
                "deliveryCounts": "deliveryCounts @@ (msgId :> 0)",
            },
            description="Send a new message to the queue",
        ),
        Action(
            name="Receive",
            parameters=(),
            preconditions=("Len(pending) > 0",),
            updates={
                "pending": "Tail(pending)",
                "invisible": (
                    "invisible @@ (Head(pending) :> "
                    "[expiresAt |-> VisibilityTimeout, handle |-> 1])"
                ),
                "deliveryCounts": "[deliveryCounts EXCEPT ![Head(pending)] = @ + 1]",
            },
            description="Receive message from queue (move to invisible)",
        ),
        Action(
            name="Acknowledge",
            parameters=("msgId",),
            preconditions=(
                "msgId \\in DOMAIN invisible",
                "msgId \\in DOMAIN deliveryCounts",
            ),
            updates={
                "invisible": "[m \\in (DOMAIN invisible) \\\\ {msgId} |-> invisible[m]]",
                "deliveryCounts": "[m \\in (DOMAIN deliveryCounts) \\\\ {msgId} |-> deliveryCounts[m]]",
            },
            description="Acknowledge message (remove from queue)",
        ),
    ],
    invariants=[
        Invariant(
            id="INV-1",
            name="MessageStateExclusive",
            predicate=(
                r"\A msgId \in DOMAIN deliveryCounts:"
                "\n        \\/ InPending(msgId)"
                "\n        \\/ msgId \\in DOMAIN invisible"
            ),
            description="Message is either pending or invisible",
        ),
        Invariant(
            id="INV-2",
            name="DeliveryCountMonotonic",
            predicate=r"\A msgId \in DOMAIN deliveryCounts: deliveryCounts[msgId] >= 0",
            description="Delivery counts are non-negative",
        ),
        Invariant(
            id="INV-3",
            name="InvisibleImpliesDelivered",
            predicate=(
                r"\A msgId \in DOMAIN invisible:"
                "\n        /\\ msgId \\in DOMAIN deliveryCounts"
                "\n        /\\ deliveryCounts[msgId] > 0"
            ),
            description="Messages in invisible have been delivered at least once",
        ),
    ],
)
class SimpleMailbox:
    """Simplified mailbox demonstrating TLA+ spec embedding.

    This is a minimal example showing how the @formal_spec decorator
    captures the state machine semantics of the mailbox.
    """

    def __init__(self) -> None:
        self.pending: list[str] = []
        self.invisible: dict[str, dict[str, int]] = {}
        self.delivery_counts: dict[str, int] = {}

    @require(lambda self, msg_id: msg_id not in self.delivery_counts)
    @ensure(lambda self, msg_id: msg_id in self.pending)
    def send(self, msg_id: str) -> None:
        """Send a message to the queue."""
        self.pending.append(msg_id)
        self.delivery_counts[msg_id] = 0

    @require(lambda self: len(self.pending) > 0)
    @ensure(lambda self, result: result is not None)
    def receive(self) -> str | None:
        """Receive a message from the queue."""
        if not self.pending:
            return None

        msg_id = self.pending.pop(0)
        self.invisible[msg_id] = {"expiresAt": 30, "handle": 1}
        self.delivery_counts[msg_id] += 1
        return msg_id

    @require(lambda self, msg_id: msg_id in self.invisible)
    @ensure(lambda self, msg_id: msg_id not in self.invisible)
    def acknowledge(self, msg_id: str) -> None:
        """Acknowledge a message (remove from queue)."""
        del self.invisible[msg_id]
        del self.delivery_counts[msg_id]


# =============================================================================
# Usage Demo
# =============================================================================


def demo_counter() -> None:
    """Demonstrate counter with invariant checking."""
    from weakincentives.dbc import enable_dbc

    # Enable DbC to see runtime invariant checking
    enable_dbc()

    counter = Counter()
    print(f"Initial count: {counter.count}")

    counter.increment()
    print(f"After increment: {counter.count}")

    counter.decrement()
    print(f"After decrement: {counter.count}")

    try:
        counter.decrement()  # Will fail precondition
    except AssertionError as e:
        print(f"Expected error: {e}")


def demo_mailbox() -> None:
    """Demonstrate mailbox with state checking."""
    mailbox = SimpleMailbox()

    mailbox.send("msg-1")
    mailbox.send("msg-2")
    print(f"Pending: {mailbox.pending}")

    msg_id = mailbox.receive()
    print(f"Received: {msg_id}")
    print(f"Invisible: {mailbox.invisible}")
    print(f"Delivery counts: {mailbox.delivery_counts}")

    if msg_id:
        mailbox.acknowledge(msg_id)
        print(f"After ack - Invisible: {mailbox.invisible}")


def show_extracted_specs() -> None:
    """Show what TLA+ specs would be extracted."""
    print("=" * 77)
    print("Counter TLA+ Spec:")
    print("=" * 77)
    spec = Counter.__formal_spec__
    print(spec.to_tla())
    print()

    print("=" * 77)
    print("SimpleMailbox TLA+ Spec:")
    print("=" * 77)
    spec = SimpleMailbox.__formal_spec__
    print(spec.to_tla())
    print()

    print("=" * 77)
    print("TLC Config for SimpleMailbox:")
    print("=" * 77)
    print(spec.to_tla_config())


if __name__ == "__main__":
    print("Running examples...")
    print()

    demo_counter()
    print()

    demo_mailbox()
    print()

    show_extracted_specs()

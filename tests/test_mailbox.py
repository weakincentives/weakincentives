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

"""Tests for ``weakincentives.mailbox``."""

from __future__ import annotations

import threading

import pytest

from weakincentives.clock import FakeClock
from weakincentives.mailbox import (
    DeadLetterQueue,
    InMemoryMailbox,
    Mailbox,
    MailboxError,
    MailboxWorker,
    Message,
)


def test_implements_mailbox_protocol() -> None:
    assert isinstance(InMemoryMailbox(), Mailbox)


def test_put_then_poll_returns_message() -> None:
    mb = InMemoryMailbox()
    mb.put("hello")
    lease = mb.poll(lease_seconds=10)
    assert lease is not None
    assert lease.message.body == "hello"
    assert lease.message.delivery_count == 1
    mb.ack(lease.id)
    assert len(mb) == 0


def test_poll_empty_returns_none() -> None:
    assert InMemoryMailbox().poll(lease_seconds=1) is None


def test_nack_returns_message_to_queue() -> None:
    mb = InMemoryMailbox(max_deliveries=3)
    mb.put("retry me")
    lease = mb.poll(lease_seconds=10)
    assert lease is not None
    mb.nack(lease.id)
    again = mb.poll(lease_seconds=10)
    assert again is not None
    assert again.message.delivery_count == 2


def test_ack_unknown_raises() -> None:
    mb = InMemoryMailbox()
    with pytest.raises(MailboxError, match="unknown lease"):
        mb.ack("nope")


def test_nack_unknown_raises() -> None:
    mb = InMemoryMailbox()
    with pytest.raises(MailboxError, match="unknown lease"):
        mb.nack("nope")


def test_max_deliveries_enforced_and_dlq_collects_poison() -> None:
    dlq = DeadLetterQueue()
    mb = InMemoryMailbox(max_deliveries=2, dead_letter_queue=dlq)
    mb.put("dies eventually")
    for _ in range(2):
        lease = mb.poll(lease_seconds=10)
        assert lease is not None
        mb.nack(lease.id)
    assert len(mb) == 0
    assert len(dlq) == 1
    drained = dlq.drain()
    assert drained[0].body == "dies eventually"
    assert len(dlq) == 0


def test_drop_when_no_dlq_configured() -> None:
    mb = InMemoryMailbox(max_deliveries=1)
    mb.put("doomed")
    lease = mb.poll(lease_seconds=10)
    assert lease is not None
    mb.nack(lease.id)
    assert len(mb) == 0


def test_lease_expiry_returns_message() -> None:
    clock = FakeClock()
    mb = InMemoryMailbox(max_deliveries=3, clock=clock)
    mb.put("slow")
    lease = mb.poll(lease_seconds=5)
    assert lease is not None
    clock.advance(10)
    requeued = mb.poll(lease_seconds=5)
    assert requeued is not None
    assert requeued.message.delivery_count == 2


def test_reclaim_expired_skips_still_alive_leases() -> None:
    """Cover the loop branch where a lease has not yet expired."""
    clock = FakeClock()
    mb = InMemoryMailbox(max_deliveries=3, clock=clock)
    mb.put("first")
    mb.put("second")
    fresh = mb.poll(lease_seconds=100)
    assert fresh is not None
    stale = mb.poll(lease_seconds=5)
    assert stale is not None
    clock.advance(10)
    # Trigger reclaim — the fresh lease must survive while the stale one
    # is requeued. Putting a third message keeps the queue non-empty.
    mb.put("third")
    next_lease = mb.poll(lease_seconds=5)
    assert next_lease is not None
    # Acking the still-alive lease must work (i.e., it was not evicted).
    mb.ack(fresh.id)


def test_rejects_invalid_max_deliveries() -> None:
    with pytest.raises(MailboxError, match="max_deliveries"):
        InMemoryMailbox(max_deliveries=0)


def test_message_dataclass_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    message = Message(id="x", body="hi", enqueued_at=0.0)
    with pytest.raises(FrozenInstanceError):
        setattr(message, "body", "no")  # noqa: B010


def test_worker_step_acks_on_success() -> None:
    mb = InMemoryMailbox()
    mb.put("hi")
    seen: list[str] = []

    def handler(message: Message) -> None:
        seen.append(str(message.body))

    worker = MailboxWorker(mailbox=mb, handler=handler)
    assert worker.step() is True
    assert seen == ["hi"]
    assert len(mb) == 0


def test_worker_step_returns_false_when_empty() -> None:
    worker = MailboxWorker(mailbox=InMemoryMailbox(), handler=lambda _m: None)
    assert worker.step() is False


def test_worker_step_nacks_on_handler_error() -> None:
    mb = InMemoryMailbox()
    mb.put("retry")

    def handler(_message: Message) -> None:
        raise RuntimeError("boom")

    worker = MailboxWorker(mailbox=mb, handler=handler)
    assert worker.step() is True
    assert len(worker.failures) == 1
    assert len(mb) == 1


def test_worker_run_until_processes_then_idles() -> None:
    mb = InMemoryMailbox()
    mb.put("a")
    mb.put("b")
    seen: list[str] = []

    def handler(message: Message) -> None:
        seen.append(str(message.body))

    stop = threading.Event()

    def runner() -> None:
        worker = MailboxWorker(mailbox=mb, handler=handler, poll_idle_seconds=0.001)
        worker.run_until(stop)

    thread = threading.Thread(target=runner)
    thread.start()
    deadline_secs = 1.0
    while sorted(seen) != ["a", "b"] and deadline_secs > 0:
        deadline_secs -= 0.05
        threading.Event().wait(0.05)
    stop.set()
    thread.join(timeout=1.0)
    assert sorted(seen) == ["a", "b"]

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

"""Tests for AgentLoop mailbox close and receipt handle expiry behavior."""

from __future__ import annotations

import threading

from weakincentives.runtime.agent_loop import (
    AgentLoopRequest,
    AgentLoopResult,
)
from weakincentives.runtime.mailbox import (
    FakeMailbox,
    InMemoryMailbox,
)

from .conftest import (
    MockAdapter,
    SampleLoop,
    SampleOutput,
    SampleRequest,
)


def test_loop_handles_expired_receipt_handle_on_ack() -> None:
    """AgentLoop continues when receipt handle expires during processing."""
    results: FakeMailbox[AgentLoopResult[SampleOutput], None] = FakeMailbox(
        name="results"
    )
    requests: FakeMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = FakeMailbox(name="requests")

    adapter = MockAdapter()
    loop = SampleLoop(adapter=adapter, requests=requests)

    request = AgentLoopRequest(request=SampleRequest(message="hello"))
    requests.send(request, reply_to=results)

    # Receive the message to get the handle, then expire it
    msgs = requests.receive(max_messages=1)
    assert len(msgs) == 1
    msg = msgs[0]

    # Expire the handle to simulate slow processing
    requests.expire_handle(msg.receipt_handle)

    # Create a result to send
    result: AgentLoopResult[SampleOutput] = AgentLoopResult(
        request_id=request.request_id,
        output=SampleOutput(result="success"),
    )

    # Call _reply_and_ack directly - should handle expired handle gracefully
    loop._reply_and_ack(msg, result)

    # Should not raise - the expired handle is handled gracefully
    # Response should still be sent
    assert results.approximate_count() == 1


def test_loop_handles_expired_receipt_handle_on_nack() -> None:
    """AgentLoop continues when receipt handle expires during nack after send failure."""
    from weakincentives.runtime.mailbox import MailboxConnectionError

    results: FakeMailbox[AgentLoopResult[SampleOutput], None] = FakeMailbox(
        name="results"
    )
    requests: FakeMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = FakeMailbox(name="requests")

    adapter = MockAdapter()
    loop = SampleLoop(adapter=adapter, requests=requests)

    request = AgentLoopRequest(request=SampleRequest(message="hello"))
    requests.send(request, reply_to=results)

    # Receive the message to get the handle
    msgs = requests.receive(max_messages=1)
    assert len(msgs) == 1
    msg = msgs[0]

    # Expire the handle AND make send fail
    # This simulates: processing took too long, handle expired,
    # AND the response queue is also having issues
    requests.expire_handle(msg.receipt_handle)
    results.set_connection_error(MailboxConnectionError("connection lost"))

    # Create a result to send
    result: AgentLoopResult[SampleOutput] = AgentLoopResult(
        request_id=request.request_id,
        output=SampleOutput(result="success"),
    )

    # Call _reply_and_ack directly - should handle both failures gracefully
    loop._reply_and_ack(msg, result)

    # Should not raise - both failures are handled gracefully


def test_loop_exits_when_mailbox_closed() -> None:
    """AgentLoop.run() exits when requests mailbox is closed."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    adapter = MockAdapter()
    loop = SampleLoop(adapter=adapter, requests=requests)

    exited = []

    def run_loop() -> None:
        # Would run forever with max_iterations=None
        loop.run(max_iterations=None, wait_time_seconds=1)
        exited.append(True)

    thread = threading.Thread(target=run_loop)
    thread.start()

    # Close the mailbox - should cause loop to exit
    requests.close()
    results.close()

    # Thread should exit quickly
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(exited) == 1

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

from __future__ import annotations

from dataclasses import dataclass
from threading import Barrier, Thread

from weakincentives.events import InProcessEventBus, ToolInvoked
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.session import Session, ToolData


@dataclass(slots=True)
class _FakeParams:
    identifier: str


@dataclass(slots=True)
class _FakePayload:
    identifier: str


def test_session_collects_concurrent_tool_events() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    worker_count = 4
    per_worker = 16
    barrier = Barrier(worker_count)

    def _publish(worker_index: int) -> None:
        barrier.wait()
        for item_index in range(per_worker):
            identifier = f"{worker_index}-{item_index}"
            payload = _FakePayload(identifier=identifier)
            params = _FakeParams(identifier=identifier)
            result = ToolResult[object](message="ok", value=payload)
            event = ToolInvoked(
                prompt_name="test.prompt",
                adapter="test-adapter",
                name="dummy_tool",
                params=params,
                result=result,
                call_id=None,
            )
            bus.publish(event)

    threads = [Thread(target=_publish, args=(index,)) for index in range(worker_count)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    payloads = session.select_all(_FakePayload)
    assert len(payloads) == worker_count * per_worker
    assert {payload.identifier for payload in payloads} == {
        f"{worker}-{item}"
        for worker in range(worker_count)
        for item in range(per_worker)
    }

    tool_events = session.select_all(ToolData)
    assert len(tool_events) == worker_count * per_worker
    for event in tool_events:
        assert event.value is None or isinstance(event.value, _FakePayload)

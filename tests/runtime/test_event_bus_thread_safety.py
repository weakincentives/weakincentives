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

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Barrier, Lock, Thread

from weakincentives.runtime.events import InProcessEventBus, PublishResult


@dataclass(slots=True)
class _DummyEvent:
    event_id: int


def _make_handler(
    handler_id: int, call_log: list[tuple[int, int]], lock: Lock
) -> Callable[[object], None]:
    def handler(event: object) -> None:
        assert isinstance(event, _DummyEvent)
        with lock:
            call_log.append((handler_id, event.event_id))

    return handler


def _assert_all_handlers_called_once(
    call_log: list[tuple[int, int]], handler_count: int, event_ids: Iterable[int]
) -> None:
    counts = Counter(call_log)
    for event_id in event_ids:
        for handler_id in range(handler_count):
            assert counts[handler_id, event_id] == 1


def _assert_base_handlers_delivered(
    call_log: list[tuple[int, int]], publisher_count: int, base_handler_count: int
) -> None:
    for event_id in range(publisher_count):
        for handler_id in range(base_handler_count):
            assert call_log.count((handler_id, event_id)) == 1


def test_concurrent_publishes_invoke_all_handlers_once() -> None:
    handler_count = 8
    publisher_count = 12
    bus = InProcessEventBus()
    call_log: list[tuple[int, int]] = []
    log_lock = Lock()

    for handler_id in range(handler_count):
        bus.subscribe(_DummyEvent, _make_handler(handler_id, call_log, log_lock))

    start_barrier = Barrier(publisher_count)
    results: list[PublishResult] = []
    errors: list[BaseException] = []

    def publish(index: int) -> None:
        try:
            start_barrier.wait()
            result = bus.publish(_DummyEvent(event_id=index))
            results.append(result)
        except BaseException as error:  # pragma: no cover - defensive
            errors.append(error)

    threads = [
        Thread(target=publish, args=(index,)) for index in range(publisher_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(results) == publisher_count
    assert all(result.handlers_invoked for result in results)
    assert all(len(result.handlers_invoked) == handler_count for result in results)

    _assert_all_handlers_called_once(call_log, handler_count, range(publisher_count))


def test_subscriptions_and_publishes_can_race_without_errors() -> None:
    base_handler_count = 4
    new_handler_count = 6
    publisher_count = 5
    bus = InProcessEventBus()
    call_log: list[tuple[int, int]] = []
    log_lock = Lock()

    for handler_id in range(base_handler_count):
        bus.subscribe(_DummyEvent, _make_handler(handler_id, call_log, log_lock))

    start_barrier = Barrier(new_handler_count + publisher_count)
    results: list[PublishResult] = []
    errors: list[BaseException] = []

    def subscribe(offset: int) -> None:
        handler_id = base_handler_count + offset
        try:
            start_barrier.wait()
            bus.subscribe(_DummyEvent, _make_handler(handler_id, call_log, log_lock))
        except BaseException as error:  # pragma: no cover - defensive
            errors.append(error)

    def publish(event_id: int) -> None:
        try:
            start_barrier.wait()
            result = bus.publish(_DummyEvent(event_id=event_id))
            results.append(result)
        except BaseException as error:  # pragma: no cover - defensive
            errors.append(error)

    subscriber_threads = [
        Thread(target=subscribe, args=(index,)) for index in range(new_handler_count)
    ]
    publisher_threads = [
        Thread(target=publish, args=(index,)) for index in range(publisher_count)
    ]

    for thread in subscriber_threads + publisher_threads:
        thread.start()
    for thread in subscriber_threads + publisher_threads:
        thread.join()

    assert not errors
    assert len(results) == publisher_count
    assert all(len(result.handlers_invoked) >= base_handler_count for result in results)

    _assert_base_handlers_delivered(call_log, publisher_count, base_handler_count)

    final_event_id = publisher_count + 1
    final_result = bus.publish(_DummyEvent(event_id=final_event_id))

    assert len(final_result.handlers_invoked) == base_handler_count + new_handler_count

    _assert_all_handlers_called_once(
        call_log, base_handler_count + new_handler_count, [final_event_id]
    )


def test_concurrent_unsubscribes_and_publishes_can_race_without_errors() -> None:
    handler_count = 10
    publisher_count = 5
    unsubscribe_count = 5
    bus = InProcessEventBus()
    call_log: list[tuple[int, int]] = []
    log_lock = Lock()
    handlers: list[tuple[int, object]] = []

    for handler_id in range(handler_count):
        handler = _make_handler(handler_id, call_log, log_lock)
        handlers.append((handler_id, handler))
        bus.subscribe(_DummyEvent, handler)

    start_barrier = Barrier(unsubscribe_count + publisher_count)
    results: list[PublishResult] = []
    unsubscribe_results: list[bool] = []
    errors: list[BaseException] = []

    def unsubscribe(handler_id: int, handler: object) -> None:
        try:
            start_barrier.wait()
            result = bus.unsubscribe(_DummyEvent, handler)
            unsubscribe_results.append(result)
        except BaseException as error:  # pragma: no cover - defensive
            errors.append(error)

    def publish(event_id: int) -> None:
        try:
            start_barrier.wait()
            result = bus.publish(_DummyEvent(event_id=event_id))
            results.append(result)
        except BaseException as error:  # pragma: no cover - defensive
            errors.append(error)

    unsubscribe_threads = [
        Thread(target=unsubscribe, args=(handler_id, handler))
        for handler_id, handler in handlers[:unsubscribe_count]
    ]
    publisher_threads = [
        Thread(target=publish, args=(index,)) for index in range(publisher_count)
    ]

    for thread in unsubscribe_threads + publisher_threads:
        thread.start()
    for thread in unsubscribe_threads + publisher_threads:
        thread.join()

    assert not errors
    assert len(results) == publisher_count
    assert len(unsubscribe_results) == unsubscribe_count
    assert all(result is True for result in unsubscribe_results)

    remaining_handler_count = handler_count - unsubscribe_count
    final_event_id = publisher_count + 1
    final_result = bus.publish(_DummyEvent(event_id=final_event_id))

    assert len(final_result.handlers_invoked) == remaining_handler_count

    counts = Counter(call_log)
    for handler_id in range(unsubscribe_count, handler_count):
        assert counts[handler_id, final_event_id] == 1

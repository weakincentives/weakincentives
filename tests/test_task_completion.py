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

"""Tests for ``weakincentives.task_completion``."""

from __future__ import annotations

from datetime import datetime

from weakincentives.core import Session, ToolInvoked
from weakincentives.task_completion import (
    AllOfChecker,
    AnyOfChecker,
    CompletionChecker,
    CompletionStatus,
    ToolSucceededChecker,
    callable_checker,
)
from weakincentives.transcript import Transcript, TranscriptEntry


def _entry(name: str, success: bool) -> TranscriptEntry:
    return TranscriptEntry(
        timestamp=datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
        kind=ToolInvoked.__name__,
        payload={"name": name, "success": success, "message": ""},
    )


def test_tool_succeeded_passes_when_event_present() -> None:
    transcript = Transcript()
    transcript.append(_entry("apply", True))
    status = ToolSucceededChecker(tool_name="apply").check(Session(), transcript)
    assert status.is_complete is True


def test_tool_succeeded_ignores_failures() -> None:
    transcript = Transcript()
    transcript.append(_entry("apply", False))
    status = ToolSucceededChecker(tool_name="apply").check(Session(), transcript)
    assert status.is_complete is False


def test_tool_succeeded_ignores_other_events() -> None:
    transcript = Transcript()
    transcript.append(
        TranscriptEntry(
            timestamp=datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
            kind="OtherEvent",
            payload={"name": "apply"},
        )
    )
    status = ToolSucceededChecker(tool_name="apply").check(Session(), transcript)
    assert status.is_complete is False


def test_all_of_checker_passes_only_when_every_child_passes() -> None:
    transcript = Transcript()
    transcript.append(_entry("a", True))
    checker = AllOfChecker(
        checkers=(
            ToolSucceededChecker(tool_name="a"),
            ToolSucceededChecker(tool_name="b"),
        )
    )
    assert checker.check(Session(), transcript).is_complete is False
    transcript.append(_entry("b", True))
    assert checker.check(Session(), transcript).is_complete is True


def test_any_of_checker_passes_when_one_child_passes() -> None:
    transcript = Transcript()
    checker = AnyOfChecker(
        checkers=(
            ToolSucceededChecker(tool_name="a"),
            ToolSucceededChecker(tool_name="b"),
        )
    )
    assert checker.check(Session(), transcript).is_complete is False
    transcript.append(_entry("b", True))
    assert checker.check(Session(), transcript).is_complete is True


def test_callable_checker_returns_status_directly() -> None:
    checker = callable_checker(
        lambda _s, _t: CompletionStatus(is_complete=True, detail="custom")
    )
    status = checker.check(Session(), Transcript())
    assert status.is_complete is True
    assert status.detail == "custom"


def test_callable_checker_coerces_bool() -> None:
    truthy = callable_checker(lambda _s, _t: True)
    assert truthy.check(Session(), Transcript()).is_complete is True
    falsy = callable_checker(lambda _s, _t: False)
    assert falsy.check(Session(), Transcript()).is_complete is False


def test_checkers_satisfy_protocol() -> None:
    candidates: list[CompletionChecker] = [
        ToolSucceededChecker(tool_name="a"),
        AllOfChecker(checkers=()),
        AnyOfChecker(checkers=()),
        callable_checker(lambda _s, _t: True),
    ]
    for checker in candidates:
        assert isinstance(checker, CompletionChecker)

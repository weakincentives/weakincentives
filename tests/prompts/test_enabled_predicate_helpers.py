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
from typing import TYPE_CHECKING

from weakincentives.prompt._enabled_predicate import (
    callable_accepts_session_kwarg,
    callable_requires_positional_argument,
    normalize_enabled_predicate,
)
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session

if TYPE_CHECKING:
    from weakincentives.runtime.session._protocols import SessionProtocol


@dataclass
class ToggleParams:
    include: bool


def test_normalize_enabled_predicate_accepts_zero_arg_callable() -> None:
    predicate = normalize_enabled_predicate(lambda: False, params_type=None)

    assert predicate is not None
    assert predicate(None, None) is False
    assert callable_requires_positional_argument(lambda: False) is False


def test_normalize_enabled_predicate_passes_positional_value() -> None:
    recorded: list[object | None] = []

    def enabled(value: object | None) -> bool:
        recorded.append(value)
        return value is None

    predicate = normalize_enabled_predicate(enabled, params_type=None)

    assert predicate is not None
    assert predicate(None, None) is True
    assert recorded == [None]


def test_normalize_enabled_predicate_handles_parameterized_callable() -> None:
    def enabled(params: ToggleParams) -> bool:
        return params.include

    predicate = normalize_enabled_predicate(enabled, params_type=ToggleParams)

    assert predicate is not None
    assert predicate(ToggleParams(include=True), None) is True
    assert predicate(ToggleParams(include=False), None) is False


def test_callable_accepts_session_kwarg_detects_keyword_only() -> None:
    def with_session(*, session: SessionProtocol) -> bool:
        return True

    def without_session() -> bool:
        return True

    def positional_session(session: SessionProtocol) -> bool:
        return True

    assert callable_accepts_session_kwarg(with_session) is True
    assert callable_accepts_session_kwarg(without_session) is False
    # Only keyword-only parameters named session are accepted, not positional
    assert callable_accepts_session_kwarg(positional_session) is False


def test_normalize_enabled_predicate_with_session_kwarg() -> None:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)

    received_session: list[SessionProtocol | None] = []

    def enabled(*, session: SessionProtocol | None) -> bool:
        received_session.append(session)
        return session is not None

    predicate = normalize_enabled_predicate(enabled, params_type=None)

    assert predicate is not None
    assert predicate(None, session) is True
    assert received_session == [session]
    assert predicate(None, None) is False


def test_normalize_enabled_predicate_with_params_and_session() -> None:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)

    received: list[tuple[ToggleParams, SessionProtocol | None]] = []

    def enabled(params: ToggleParams, *, session: SessionProtocol | None) -> bool:
        received.append((params, session))
        return params.include and session is not None

    predicate = normalize_enabled_predicate(enabled, params_type=ToggleParams)

    assert predicate is not None
    params = ToggleParams(include=True)
    assert predicate(params, session) is True
    assert received == [(params, session)]

    params_false = ToggleParams(include=False)
    assert predicate(params_false, session) is False

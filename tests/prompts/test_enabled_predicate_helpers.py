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
from unittest.mock import MagicMock

from weakincentives.prompt._enabled_predicate import (
    _accepts_session_kwarg,
    callable_requires_positional_argument,
    normalize_enabled_predicate,
)

if TYPE_CHECKING:
    from weakincentives.runtime.session.protocols import SessionProtocol


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


def test_accepts_session_kwarg_detects_session_parameter() -> None:
    def no_session() -> bool:
        return True

    def with_params(params: ToggleParams) -> bool:
        return params.include

    def with_session_kwarg(*, session: SessionProtocol | None) -> bool:
        return session is not None

    def with_params_and_session(
        params: ToggleParams, *, session: SessionProtocol | None
    ) -> bool:
        return params.include and session is not None

    assert _accepts_session_kwarg(no_session) is False
    assert _accepts_session_kwarg(with_params) is False
    assert _accepts_session_kwarg(with_session_kwarg) is True
    assert _accepts_session_kwarg(with_params_and_session) is True


def test_normalize_enabled_predicate_with_session_only() -> None:
    """Test enabled predicate that only takes session kwarg."""
    recorded_sessions: list[object | None] = []

    def enabled(*, session: SessionProtocol | None) -> bool:
        recorded_sessions.append(session)
        return session is not None

    predicate = normalize_enabled_predicate(enabled, params_type=None)
    mock_session = MagicMock()

    assert predicate is not None
    assert predicate(None, None) is False
    assert predicate(None, mock_session) is True
    assert recorded_sessions == [None, mock_session]


def test_normalize_enabled_predicate_with_params_and_session() -> None:
    """Test enabled predicate with params and session."""
    recorded: list[tuple[object | None, object | None]] = []

    def enabled(params: ToggleParams, *, session: SessionProtocol | None) -> bool:
        recorded.append((params, session))
        return params.include and session is not None

    predicate = normalize_enabled_predicate(enabled, params_type=ToggleParams)
    mock_session = MagicMock()

    assert predicate is not None
    # Both conditions must be true
    assert predicate(ToggleParams(include=True), mock_session) is True
    # Missing session
    assert predicate(ToggleParams(include=True), None) is False
    # params.include is False
    assert predicate(ToggleParams(include=False), mock_session) is False

    assert recorded[0] == (ToggleParams(include=True), mock_session)
    assert recorded[1] == (ToggleParams(include=True), None)
    assert recorded[2] == (ToggleParams(include=False), mock_session)

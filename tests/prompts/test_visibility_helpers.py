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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from weakincentives.prompt._visibility import (
    SectionVisibility,
    _accepts_session_kwarg,
    _visibility_requires_positional_argument,
    normalize_visibility_selector,
)

if TYPE_CHECKING:
    from weakincentives.runtime.session.protocols import SessionProtocol


@dataclass
class _VisibilityParams:
    flag: bool = False


def test_visibility_selector_rejects_invalid_return_type() -> None:
    selector = normalize_visibility_selector(
        lambda params: cast(SectionVisibility, "not-visibility"), _VisibilityParams
    )

    with pytest.raises(TypeError):
        selector(_VisibilityParams(), None)


def test_visibility_requires_positional_argument_branches() -> None:
    assert (
        _visibility_requires_positional_argument(lambda: SectionVisibility.FULL)
        is False
    )
    assert (
        _visibility_requires_positional_argument(lambda params: SectionVisibility.FULL)
        is True
    )
    assert (
        _visibility_requires_positional_argument(
            cast(Callable[..., SectionVisibility], object())
        )
        is True
    )


def test_normalize_visibility_selector_static_value() -> None:
    """Test visibility selector with a static SectionVisibility value."""
    selector = normalize_visibility_selector(SectionVisibility.SUMMARY, None)

    assert selector(None, None) == SectionVisibility.SUMMARY
    assert selector(_VisibilityParams(), MagicMock()) == SectionVisibility.SUMMARY


def test_normalize_visibility_selector_zero_arg_callable() -> None:
    """Test visibility selector with zero-argument callable."""
    selector = normalize_visibility_selector(lambda: SectionVisibility.FULL, None)

    assert selector(None, None) == SectionVisibility.FULL


def test_normalize_visibility_selector_with_params() -> None:
    """Test visibility selector that uses params."""

    def selector_func(params: _VisibilityParams) -> SectionVisibility:
        return SectionVisibility.SUMMARY if params.flag else SectionVisibility.FULL

    selector = normalize_visibility_selector(selector_func, _VisibilityParams)

    assert selector(_VisibilityParams(flag=True), None) == SectionVisibility.SUMMARY
    assert selector(_VisibilityParams(flag=False), None) == SectionVisibility.FULL


def test_accepts_session_kwarg_detects_session_parameter() -> None:
    """Test detection of session keyword argument."""

    def no_session() -> SectionVisibility:
        return SectionVisibility.FULL

    def with_params(params: _VisibilityParams) -> SectionVisibility:
        return SectionVisibility.FULL

    def with_session_kwarg(*, session: SessionProtocol | None) -> SectionVisibility:
        return SectionVisibility.FULL if session else SectionVisibility.SUMMARY

    def with_params_and_session(
        params: _VisibilityParams, *, session: SessionProtocol | None
    ) -> SectionVisibility:
        return SectionVisibility.FULL

    assert _accepts_session_kwarg(no_session) is False
    assert _accepts_session_kwarg(with_params) is False
    assert _accepts_session_kwarg(with_session_kwarg) is True
    assert _accepts_session_kwarg(with_params_and_session) is True


def test_accepts_session_kwarg_handles_uninspectable_callable() -> None:
    """Test that uninspectable callables return False for session kwarg check."""
    # Cast an object to the expected callable type to trigger signature error
    uninspectable = cast(Callable[..., SectionVisibility], object())
    assert _accepts_session_kwarg(uninspectable) is False


def test_normalize_visibility_selector_with_session_only() -> None:
    """Test visibility selector that only takes session kwarg."""
    recorded_sessions: list[object | None] = []

    def selector_func(*, session: SessionProtocol | None) -> SectionVisibility:
        recorded_sessions.append(session)
        return SectionVisibility.FULL if session else SectionVisibility.SUMMARY

    selector = normalize_visibility_selector(selector_func, None)
    mock_session = MagicMock()

    assert selector(None, None) == SectionVisibility.SUMMARY
    assert selector(None, mock_session) == SectionVisibility.FULL
    assert recorded_sessions == [None, mock_session]


def test_normalize_visibility_selector_with_params_and_session() -> None:
    """Test visibility selector with params and session."""
    recorded: list[tuple[object | None, object | None]] = []

    def selector_func(
        params: _VisibilityParams, *, session: SessionProtocol | None
    ) -> SectionVisibility:
        recorded.append((params, session))
        if params.flag and session:
            return SectionVisibility.FULL
        return SectionVisibility.SUMMARY

    selector = normalize_visibility_selector(selector_func, _VisibilityParams)
    mock_session = MagicMock()

    # Both conditions met
    assert (
        selector(_VisibilityParams(flag=True), mock_session) == SectionVisibility.FULL
    )
    # Missing session
    assert selector(_VisibilityParams(flag=True), None) == SectionVisibility.SUMMARY
    # flag is False
    assert (
        selector(_VisibilityParams(flag=False), mock_session)
        == SectionVisibility.SUMMARY
    )

    assert recorded[0] == (_VisibilityParams(flag=True), mock_session)
    assert recorded[1] == (_VisibilityParams(flag=True), None)
    assert recorded[2] == (_VisibilityParams(flag=False), mock_session)

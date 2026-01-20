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

"""Section visibility control for :mod:`weakincentives.prompt`."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, cast

from ._enabled_predicate import callable_accepts_session_kwarg
from ._types import SupportsDataclass

if TYPE_CHECKING:
    from ..runtime.session._protocols import SessionProtocol


class SectionVisibility(Enum):
    """Controls how a section is rendered in a prompt.

    When a section has both a full template and a summary template,
    the visibility determines which one is used during rendering.
    """

    FULL = "full"
    """Render the full section content."""

    SUMMARY = "summary"
    """Render only the summary content."""


VisibilitySelector = (
    Callable[[SupportsDataclass], SectionVisibility]
    | Callable[[], SectionVisibility]
    | SectionVisibility
)

# Normalized callable signature that accepts params and session keyword argument
NormalizedVisibilitySelector = Callable[
    [SupportsDataclass | None, "SessionProtocol | None"], SectionVisibility
]


def _coerce_section_visibility(value: object) -> SectionVisibility:
    if isinstance(value, SectionVisibility):
        return value
    raise TypeError("Visibility selector must return SectionVisibility.")


def _normalize_zero_arg_visibility(
    visibility: Callable[..., SectionVisibility], accepts_session: bool
) -> NormalizedVisibilitySelector:
    """Normalize a zero-argument visibility callable."""
    if accepts_session:

        def _without_params_with_session(
            _: SupportsDataclass | None,
            session: SessionProtocol | None,
        ) -> SectionVisibility:
            return _coerce_section_visibility(visibility(session=session))

        return _without_params_with_session

    zero_arg_selector = cast(Callable[[], SectionVisibility], visibility)

    def _without_params(
        _: SupportsDataclass | None,
        session: SessionProtocol | None,
    ) -> SectionVisibility:
        del session
        return _coerce_section_visibility(zero_arg_selector())

    return _without_params


def _normalize_params_visibility(
    visibility: Callable[..., SectionVisibility], accepts_session: bool
) -> NormalizedVisibilitySelector:
    """Normalize a visibility callable that takes params."""
    if accepts_session:

        def _with_params_and_session(
            value: SupportsDataclass | None,
            session: SessionProtocol | None,
        ) -> SectionVisibility:
            return _coerce_section_visibility(
                visibility(cast(SupportsDataclass, value), session=session)
            )

        return _with_params_and_session

    selector = cast(Callable[[SupportsDataclass], SectionVisibility], visibility)

    def _with_params(
        value: SupportsDataclass | None,
        session: SessionProtocol | None,
    ) -> SectionVisibility:
        del session
        return _coerce_section_visibility(selector(cast(SupportsDataclass, value)))

    return _with_params


def normalize_visibility_selector(
    visibility: VisibilitySelector,
    params_type: type[SupportsDataclass] | None,
) -> NormalizedVisibilitySelector:
    """Normalize static or callable visibility into a shared interface.

    The returned callable always accepts params and session arguments.
    The session argument must be keyword-only in the original callable.
    If the original callable does not accept a session keyword argument,
    the session is not passed to it.

    Supported signatures:
        - () -> SectionVisibility
        - (*, session) -> SectionVisibility
        - (params) -> SectionVisibility
        - (params, *, session) -> SectionVisibility
        - SectionVisibility (constant)
    """
    if callable(visibility):
        accepts_session = callable_accepts_session_kwarg(visibility)
        requires_positional = _visibility_requires_positional_argument(visibility)

        if params_type is None and not requires_positional:
            return _normalize_zero_arg_visibility(visibility, accepts_session)
        return _normalize_params_visibility(visibility, accepts_session)

    constant_visibility = visibility

    def _constant(
        _: SupportsDataclass | None,
        session: SessionProtocol | None,
    ) -> SectionVisibility:
        del session
        return constant_visibility  # ty: ignore[invalid-return-type]  # ty narrowing issue

    return _constant


def _visibility_requires_positional_argument(
    callback: Callable[..., SectionVisibility],
) -> bool:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if (
            parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            and parameter.default is inspect.Signature.empty
        ):
            return True
    return False


__all__ = [
    "NormalizedVisibilitySelector",
    "SectionVisibility",
    "VisibilitySelector",
    "normalize_visibility_selector",
]

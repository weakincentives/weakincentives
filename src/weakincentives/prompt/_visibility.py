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

from ._types import SupportsDataclass

if TYPE_CHECKING:
    from ..runtime.session.protocols import SessionProtocol


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


def _coerce_section_visibility(value: object) -> SectionVisibility:
    if isinstance(value, SectionVisibility):
        return value
    raise TypeError("Visibility selector must return SectionVisibility.")


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


def _accepts_session_kwarg(callback: Callable[..., SectionVisibility]) -> bool:
    """Check if callback accepts a 'session' keyword argument."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.name == "session" and param.kind in {
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            return True
    return False


def normalize_visibility_selector(
    visibility: VisibilitySelector,
    params_type: type[SupportsDataclass] | None,
) -> Callable[[SupportsDataclass | None, SessionProtocol | None], SectionVisibility]:
    """Normalize static or callable visibility into a shared interface."""

    if callable(visibility):
        wants_session = _accepts_session_kwarg(visibility)
        needs_positional = _visibility_requires_positional_argument(visibility)

        if params_type is None and not needs_positional:
            if wants_session:
                # Cast to Any callable since we know it accepts session as kwarg
                zero_arg_with_session = cast(
                    Callable[..., SectionVisibility], visibility
                )

                def _without_params_with_session(
                    _: SupportsDataclass | None, session: SessionProtocol | None
                ) -> SectionVisibility:
                    return _coerce_section_visibility(
                        zero_arg_with_session(session=session)
                    )

                return _without_params_with_session

            zero_arg_selector = cast(Callable[[], SectionVisibility], visibility)

            def _without_params(
                _: SupportsDataclass | None, __: SessionProtocol | None
            ) -> SectionVisibility:
                return _coerce_section_visibility(zero_arg_selector())

            return _without_params

        if wants_session:
            # Cast to Any callable since we know it accepts params and session as kwarg
            selector_with_session = cast(
                Callable[..., SectionVisibility],
                visibility,
            )

            def _with_params_and_session(
                value: SupportsDataclass | None, session: SessionProtocol | None
            ) -> SectionVisibility:
                return _coerce_section_visibility(
                    selector_with_session(
                        cast(SupportsDataclass, value), session=session
                    )
                )

            return _with_params_and_session

        selector = cast(Callable[[SupportsDataclass], SectionVisibility], visibility)

        def _with_params(
            value: SupportsDataclass | None, _: SessionProtocol | None
        ) -> SectionVisibility:
            return _coerce_section_visibility(selector(cast(SupportsDataclass, value)))

        return _with_params

    constant_visibility = visibility

    def _constant(
        _: SupportsDataclass | None, __: SessionProtocol | None
    ) -> SectionVisibility:
        return constant_visibility

    return _constant


__all__ = [
    "SectionVisibility",
    "VisibilitySelector",
    "normalize_visibility_selector",
]

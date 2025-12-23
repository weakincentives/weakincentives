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

"""Section guards: normalized enabled and visibility predicates."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ..types.dataclass import SupportsDataclass
from ._visibility import SectionVisibility

if TYPE_CHECKING:
    from ..runtime.session.protocols import SessionProtocol


EnabledPredicate = Callable[[SupportsDataclass], bool] | Callable[[], bool]

NormalizedEnabledPredicate = Callable[
    [SupportsDataclass | None, "SessionProtocol | None"], bool
]

VisibilitySelector = (
    Callable[[SupportsDataclass], SectionVisibility]
    | Callable[[], SectionVisibility]
    | SectionVisibility
)

NormalizedVisibilitySelector = Callable[
    [SupportsDataclass | None, "SessionProtocol | None"], SectionVisibility
]


def callable_requires_positional_argument(callback: Callable[..., object]) -> bool:
    """Check if the callable requires at least one positional argument.

    Returns True if the callable has at least one required positional-only
    or positional-or-keyword parameter without a default value.
    """
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


def callable_accepts_session_kwarg(callback: Callable[..., object]) -> bool:
    """Check if the callable accepts a 'session' keyword-only argument."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    param = signature.parameters.get("session")
    if param is None:
        return False
    return param.kind == inspect.Parameter.KEYWORD_ONLY


def _normalize_zero_arg_enabled(
    enabled: Callable[..., bool], accepts_session: bool
) -> NormalizedEnabledPredicate:
    """Normalize a zero-argument enabled predicate."""
    if accepts_session:

        def _without_params_with_session(
            _: SupportsDataclass | None,
            session: SessionProtocol | None,
        ) -> bool:
            return bool(enabled(session=session))

        return _without_params_with_session

    zero_arg = cast(Callable[[], bool], enabled)

    def _without_params(
        _: SupportsDataclass | None,
        session: SessionProtocol | None,
    ) -> bool:
        del session
        return bool(zero_arg())

    return _without_params


def _normalize_params_enabled(
    enabled: Callable[..., bool], accepts_session: bool
) -> NormalizedEnabledPredicate:
    """Normalize an enabled predicate that takes params."""
    if accepts_session:

        def _with_params_and_session(
            value: SupportsDataclass | None,
            session: SessionProtocol | None,
        ) -> bool:
            return bool(enabled(cast(SupportsDataclass, value), session=session))

        return _with_params_and_session

    coerced = cast(Callable[[SupportsDataclass], bool], enabled)

    def _with_params(
        value: SupportsDataclass | None,
        session: SessionProtocol | None,
    ) -> bool:
        del session
        return bool(coerced(cast(SupportsDataclass, value)))

    return _with_params


def normalize_enabled_predicate(
    enabled: EnabledPredicate | None,
    params_type: type[SupportsDataclass] | None,
) -> NormalizedEnabledPredicate | None:
    """Normalize enabled predicate to a callable accepting (params, session).

    The returned callable always accepts params and session arguments.
    The session argument must be keyword-only in the original callable.
    If the original callable does not accept a session keyword argument,
    the session is not passed to it.

    Supported signatures:
        - () -> bool
        - (*, session) -> bool
        - (params) -> bool
        - (params, *, session) -> bool
    """
    if enabled is None:
        return None

    accepts_session = callable_accepts_session_kwarg(enabled)
    requires_positional = callable_requires_positional_argument(enabled)

    if params_type is None and not requires_positional:
        return _normalize_zero_arg_enabled(enabled, accepts_session)
    return _normalize_params_enabled(enabled, accepts_session)


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
        requires_positional = callable_requires_positional_argument(visibility)

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


@dataclass(slots=True, frozen=True)
class SectionGuards[SectionParamsT: SupportsDataclass]:
    """Normalized enabled and visibility predicates for a section.

    This helper consolidates the storage and invocation of section guards,
    providing a single typed interface for checking whether a section is
    enabled and what visibility to use.

    Attributes:
        enabled_predicate: The normalized enabled predicate, or None if always enabled.
        visibility_selector: The normalized visibility selector.
    """

    enabled_predicate: NormalizedEnabledPredicate | None
    visibility_selector: NormalizedVisibilitySelector

    @classmethod
    def create(
        cls,
        *,
        enabled: EnabledPredicate | None,
        visibility: VisibilitySelector,
        params_type: type[SupportsDataclass] | None,
    ) -> SectionGuards[SectionParamsT]:
        """Create guards from raw enabled predicate and visibility selector.

        Args:
            enabled: Optional enabled predicate (various signatures supported).
            visibility: Visibility selector (callable or constant).
            params_type: The section's params type for signature normalization.

        Returns:
            A SectionGuards instance with normalized predicates.
        """
        normalized_enabled = normalize_enabled_predicate(enabled, params_type)
        normalized_visibility = normalize_visibility_selector(visibility, params_type)
        return cls(
            enabled_predicate=normalized_enabled,
            visibility_selector=normalized_visibility,
        )

    def is_enabled(
        self,
        params: SupportsDataclass | None,
        session: SessionProtocol | None = None,
    ) -> bool:
        """Check whether the section is enabled for the given params/session.

        Returns True when no enabled predicate is set, otherwise delegates
        to the normalized predicate.
        """
        if self.enabled_predicate is None:
            return True
        return bool(self.enabled_predicate(params, session))

    def get_visibility(
        self,
        params: SupportsDataclass | None,
        session: SessionProtocol | None = None,
    ) -> SectionVisibility:
        """Compute the visibility for the given params/session."""
        return self.visibility_selector(params, session)


__all__ = [
    "EnabledPredicate",
    "NormalizedEnabledPredicate",
    "NormalizedVisibilitySelector",
    "SectionGuards",
    "VisibilitySelector",
    "callable_accepts_session_kwarg",
    "callable_requires_positional_argument",
    "normalize_enabled_predicate",
    "normalize_visibility_selector",
]

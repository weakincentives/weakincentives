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

# pyright: reportImportCycles=false
# This module is foundational and defines types used by many other modules.
# Deferred imports are used at runtime to avoid actual circular dependencies.

"""Section visibility control for :mod:`weakincentives.prompt`.

This module provides the unified visibility resolution system for prompt sections.
Visibility is determined by combining section-level predicates with session-based
overrides through the :class:`VisibilityResolver` class.

The resolution priority is:
1. Explicit override parameter (if provided)
2. Session state override (via VisibilityOverrides slice)
3. Section's visibility predicate (static, callable, or param-based)
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from ._enabled_predicate import callable_accepts_session_kwarg
from ._types import SupportsDataclass

if TYPE_CHECKING:
    from ..runtime.session.protocols import SessionProtocol

# Local type alias to avoid import cycle with errors.py
# (errors.py imports SectionVisibility from this module)
SectionPath = tuple[str, ...]


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


@dataclass(slots=True, frozen=True)
class RenderContext:
    """Context for visibility resolution during prompt rendering.

    This dataclass encapsulates all information needed to resolve section
    visibility, providing a clean interface for the VisibilityResolver.

    Attributes:
        params: Section parameters used by visibility predicates.
        session: Optional session for querying VisibilityOverrides state.
        path: Section path for looking up session-based overrides.
        override: Explicit visibility override (highest priority).
    """

    params: SupportsDataclass | None = None
    session: SessionProtocol | None = None
    path: SectionPath | None = None
    override: SectionVisibility | None = None


@runtime_checkable
class VisibilityPredicateProtocol(Protocol):
    """Protocol for section visibility predicates.

    Sections implementing this protocol provide a normalized visibility
    selector that can be called with params and session.
    """

    def __call__(
        self,
        params: SupportsDataclass | None,
        session: SessionProtocol | None,
    ) -> SectionVisibility: ...


@runtime_checkable
class SectionWithVisibility(Protocol):
    """Protocol for sections that support visibility resolution.

    This protocol defines the minimal interface required by VisibilityResolver
    to resolve visibility for a section.
    """

    @property
    def key(self) -> str:
        """The section's unique key."""
        ...

    @property
    def summary(self) -> str | None:
        """The section's summary template, if any."""
        ...

    def compute_predicate_visibility(
        self,
        params: SupportsDataclass | None,
        session: SessionProtocol | None,
    ) -> SectionVisibility:
        """Compute visibility using the section's predicate.

        This method evaluates the section's visibility selector/predicate
        without checking session overrides or explicit overrides.

        Args:
            params: Section parameters for the predicate.
            session: Optional session for predicates that inspect state.

        Returns:
            The visibility determined by the section's predicate.
        """
        ...


class VisibilityResolver:
    """Single source of truth for section visibility resolution.

    This class centralizes visibility logic that was previously split between
    section-level predicates and session-based overrides. It provides a
    consistent interface for determining section visibility during rendering.

    Resolution priority:
        1. Explicit override from RenderContext (if provided)
        2. Session state override from VisibilityOverrides slice
        3. Section's visibility predicate (static, callable, or param-based)

    Usage::

        from weakincentives.prompt import VisibilityResolver, RenderContext

        resolver = VisibilityResolver()
        context = RenderContext(
            params=section_params,
            session=session,
            path=("parent", "child"),
        )
        visibility = resolver.resolve(section, context)

    """

    def resolve(
        self,
        section: SectionWithVisibility,
        context: RenderContext,
    ) -> SectionVisibility:
        """Resolve the effective visibility for a section.

        Args:
            section: The section to resolve visibility for.
            context: The rendering context with params, session, and path.

        Returns:
            The effective visibility to use for rendering.

        Raises:
            PromptValidationError: If SUMMARY visibility is requested but no
                summary template is defined for the section.
        """
        from .errors import PromptValidationError

        visibility = self._resolve_visibility(section, context)

        # Validate SUMMARY requires a summary template
        if visibility == SectionVisibility.SUMMARY and section.summary is None:
            msg = (
                f"SUMMARY visibility requested for section '{section.key}' "
                "but no summary template is defined."
            )
            raise PromptValidationError(msg, section_path=context.path)

        return visibility

    def _resolve_visibility(
        self,
        section: SectionWithVisibility,
        context: RenderContext,
    ) -> SectionVisibility:
        """Internal visibility resolution without validation.

        Priority:
            1. Explicit override from context
            2. Session state override
            3. Section predicate
        """
        # Priority 1: Explicit override
        if context.override is not None:
            return context.override

        # Priority 2: Session state override
        session_override = self._get_session_override(context)
        if session_override is not None:
            return session_override

        # Priority 3: Section predicate
        return section.compute_predicate_visibility(context.params, context.session)

    @staticmethod
    def _get_session_override(
        context: RenderContext,
    ) -> SectionVisibility | None:
        """Query session state for visibility override.

        Returns None if session is None, path is None, or no override is set.
        """
        if context.session is None or context.path is None:
            return None

        from ..runtime.session.visibility_overrides import (
            get_session_visibility_override,
        )

        return get_session_visibility_override(context.session, context.path)


# Module-level singleton for convenience
_default_resolver: VisibilityResolver | None = None


def get_visibility_resolver() -> VisibilityResolver:
    """Get the default visibility resolver instance.

    Returns a module-level singleton VisibilityResolver for convenience.
    Users can also instantiate their own VisibilityResolver if needed.

    Returns:
        The default VisibilityResolver instance.
    """
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = VisibilityResolver()
    return _default_resolver


__all__ = [
    "NormalizedVisibilitySelector",
    "RenderContext",
    "SectionVisibility",
    "SectionWithVisibility",
    "VisibilityPredicateProtocol",
    "VisibilityResolver",
    "VisibilitySelector",
    "get_visibility_resolver",
    "normalize_visibility_selector",
]

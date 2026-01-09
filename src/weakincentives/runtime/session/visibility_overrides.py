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

"""Session-managed visibility overrides for prompt sections.

This module provides the VisibilityOverrides dataclass for storing
section visibility overrides in Session state. The visibility system
automatically checks session state before falling back to the
user-provided visibility selector or constant.
"""

from __future__ import annotations

from dataclasses import field, replace
from typing import TYPE_CHECKING

from ...dataclasses import FrozenDataclass
from ...prompt.errors import SectionPath
from ...prompt.section import SectionVisibility
from .slices import Replace
from .state_slice import reducer

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .protocols import SessionProtocol


@FrozenDataclass()
class SetVisibilityOverride:
    """Event to set a visibility override for a section path."""

    path: SectionPath = field(
        metadata={"description": "Section path as tuple of keys."}
    )
    visibility: SectionVisibility = field(
        metadata={"description": "Visibility to set for the section."}
    )


@FrozenDataclass()
class ClearVisibilityOverride:
    """Event to clear a visibility override for a section path."""

    path: SectionPath = field(
        metadata={"description": "Section path to clear override for."}
    )


@FrozenDataclass()
class ClearAllVisibilityOverrides:
    """Event to clear all visibility overrides."""


@FrozenDataclass()
class VisibilityOverrides:
    """Session state slice for section visibility overrides.

    Store this in session state to control section visibility dynamically
    based on session state rather than passing visibility_overrides to render.

    Usage::

        from weakincentives.runtime.session import VisibilityOverrides, SetVisibilityOverride
        from weakincentives.prompt import SectionVisibility

        # Set initial overrides
        overrides = VisibilityOverrides(
            overrides={
                ("instructions",): SectionVisibility.SUMMARY,
                ("tools",): SectionVisibility.FULL,
            }
        )
        session[VisibilityOverrides].seed(overrides)

        # Update overrides via event
        session.dispatch(
            SetVisibilityOverride(path=("instructions",), visibility=SectionVisibility.FULL)
        )

        # Query using natural syntax
        session[VisibilityOverrides].latest()

    """

    overrides: Mapping[SectionPath, SectionVisibility] = field(
        default_factory=lambda: dict[SectionPath, SectionVisibility]()
    )

    def get(self, path: SectionPath) -> SectionVisibility | None:
        """Return the override for a section path, or None if not set."""
        return self.overrides.get(path)

    def with_override(
        self, path: SectionPath, visibility: SectionVisibility
    ) -> VisibilityOverrides:
        """Return a new VisibilityOverrides with the given override added."""
        new_overrides = dict(self.overrides)
        new_overrides[path] = visibility
        return VisibilityOverrides(overrides=new_overrides)

    def without_override(self, path: SectionPath) -> VisibilityOverrides:
        """Return a new VisibilityOverrides with the given override removed."""
        new_overrides = dict(self.overrides)
        _ = new_overrides.pop(path, None)
        return VisibilityOverrides(overrides=new_overrides)

    @reducer(on=SetVisibilityOverride)
    def handle_set(self, event: SetVisibilityOverride) -> Replace[VisibilityOverrides]:
        """Handle SetVisibilityOverride event."""
        return Replace((self.with_override(event.path, event.visibility),))

    @reducer(on=ClearVisibilityOverride)
    def handle_clear(
        self, event: ClearVisibilityOverride
    ) -> Replace[VisibilityOverrides]:
        """Handle ClearVisibilityOverride event."""
        return Replace((self.without_override(event.path),))

    @reducer(on=ClearAllVisibilityOverrides)
    def handle_clear_all(
        self, event: ClearAllVisibilityOverrides
    ) -> Replace[VisibilityOverrides]:
        """Handle ClearAllVisibilityOverrides event."""
        del event
        return Replace((replace(self, overrides={}),))


def register_visibility_reducers(session: SessionProtocol) -> None:
    """Register the standard visibility override reducers with a session.

    Note: Sessions automatically register visibility reducers on creation,
    so calling this function manually is typically unnecessary. It is safe
    to call multiple times as it will not add duplicate registrations.

    Usage::

        session = Session(bus=bus)

        # Reducers are already registered - just dispatch events
        session.dispatch(
            SetVisibilityOverride(path=("section",), visibility=SectionVisibility.SUMMARY)
        )

    """
    session.install(VisibilityOverrides, initial=VisibilityOverrides)


def get_session_visibility_override(
    session: SessionProtocol | None,
    path: SectionPath,
) -> SectionVisibility | None:
    """Get a visibility override from session state.

    Args:
        session: The session to query, or None.
        path: The section path to look up.

    Returns:
        The visibility override if set, or None.
    """
    if session is None:
        return None
    overrides = session[VisibilityOverrides].latest()
    if overrides is None:
        return None
    return overrides.get(path)


__all__ = [
    "ClearAllVisibilityOverrides",
    "ClearVisibilityOverride",
    "SetVisibilityOverride",
    "VisibilityOverrides",
    "get_session_visibility_override",
    "register_visibility_reducers",
]

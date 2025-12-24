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

"""Tests for session-managed visibility overrides."""

from __future__ import annotations

from weakincentives.prompt import SectionVisibility
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import (
    ClearAllVisibilityOverrides,
    ClearVisibilityOverride,
    Session,
    SetVisibilityOverride,
    VisibilityOverrides,
    get_session_visibility_override,
)


def test_visibility_overrides_get() -> None:
    """VisibilityOverrides.get() returns override or None."""
    overrides = VisibilityOverrides(
        overrides={
            ("section_a",): SectionVisibility.SUMMARY,
            ("parent", "child"): SectionVisibility.FULL,
        }
    )

    assert overrides.get(("section_a",)) == SectionVisibility.SUMMARY
    assert overrides.get(("parent", "child")) == SectionVisibility.FULL
    assert overrides.get(("missing",)) is None


def test_visibility_overrides_with_override() -> None:
    """with_override() returns new instance with added override."""
    original = VisibilityOverrides()
    updated = original.with_override(("section",), SectionVisibility.SUMMARY)

    assert original.get(("section",)) is None
    assert updated.get(("section",)) == SectionVisibility.SUMMARY


def test_visibility_overrides_without_override() -> None:
    """without_override() returns new instance with removed override."""
    original = VisibilityOverrides(overrides={("section",): SectionVisibility.SUMMARY})
    updated = original.without_override(("section",))

    assert original.get(("section",)) == SectionVisibility.SUMMARY
    assert updated.get(("section",)) is None


def test_session_auto_registers_visibility_reducers() -> None:
    """Session automatically registers visibility reducers on creation."""
    bus = InProcessDispatcher()
    session = Session(bus=bus)

    # Initially no overrides
    assert session[VisibilityOverrides].latest() is None

    # Set an override - should work without explicit registration
    session.dispatch(
        SetVisibilityOverride(path=("section",), visibility=SectionVisibility.SUMMARY)
    )
    overrides = session[VisibilityOverrides].latest()
    assert overrides is not None
    assert overrides.get(("section",)) == SectionVisibility.SUMMARY

    # Set another override
    session.dispatch(
        SetVisibilityOverride(path=("other",), visibility=SectionVisibility.FULL)
    )
    overrides = session[VisibilityOverrides].latest()
    assert overrides is not None
    assert overrides.get(("section",)) == SectionVisibility.SUMMARY
    assert overrides.get(("other",)) == SectionVisibility.FULL


def test_clear_visibility_override_event() -> None:
    """ClearVisibilityOverride removes a single override."""
    bus = InProcessDispatcher()
    session = Session(bus=bus)

    # Set some overrides
    session.dispatch(
        SetVisibilityOverride(path=("a",), visibility=SectionVisibility.SUMMARY)
    )
    session.dispatch(
        SetVisibilityOverride(path=("b",), visibility=SectionVisibility.FULL)
    )

    # Clear one
    session.dispatch(ClearVisibilityOverride(path=("a",)))

    overrides = session[VisibilityOverrides].latest()
    assert overrides is not None
    assert overrides.get(("a",)) is None
    assert overrides.get(("b",)) == SectionVisibility.FULL


def test_clear_all_visibility_overrides_event() -> None:
    """ClearAllVisibilityOverrides removes all overrides."""
    bus = InProcessDispatcher()
    session = Session(bus=bus)

    # Set some overrides
    session.dispatch(
        SetVisibilityOverride(path=("a",), visibility=SectionVisibility.SUMMARY)
    )
    session.dispatch(
        SetVisibilityOverride(path=("b",), visibility=SectionVisibility.FULL)
    )

    # Clear all
    session.dispatch(ClearAllVisibilityOverrides())

    overrides = session[VisibilityOverrides].latest()
    assert overrides is not None
    assert overrides.get(("a",)) is None
    assert overrides.get(("b",)) is None


def test_get_session_visibility_override_returns_none_for_none_session() -> None:
    """get_session_visibility_override returns None when session is None."""
    assert get_session_visibility_override(None, ("section",)) is None


def test_get_session_visibility_override_returns_none_for_empty_session() -> None:
    """get_session_visibility_override returns None when no overrides set."""
    bus = InProcessDispatcher()
    session = Session(bus=bus)

    assert get_session_visibility_override(session, ("section",)) is None


def test_get_session_visibility_override_returns_override_from_session() -> None:
    """get_session_visibility_override returns override from session state."""
    bus = InProcessDispatcher()
    session = Session(bus=bus)

    session.dispatch(
        SetVisibilityOverride(path=("section",), visibility=SectionVisibility.SUMMARY)
    )

    assert (
        get_session_visibility_override(session, ("section",))
        == SectionVisibility.SUMMARY
    )
    assert get_session_visibility_override(session, ("other",)) is None


def test_cloned_session_preserves_visibility_reducers() -> None:
    """Session.clone preserves visibility reducers without duplicating them."""
    bus = InProcessDispatcher()
    session = Session(bus=bus)

    # Set an override on original session
    session.dispatch(
        SetVisibilityOverride(path=("original",), visibility=SectionVisibility.SUMMARY)
    )

    # Clone the session
    cloned = session.clone(bus=bus)

    # Cloned session should have reducers for visibility events
    assert cloned.state_manager.has_reducer(SetVisibilityOverride)
    assert cloned.state_manager.has_reducer(ClearVisibilityOverride)
    assert cloned.state_manager.has_reducer(ClearAllVisibilityOverrides)

    # Cloned session should work with visibility events
    cloned.dispatch(
        SetVisibilityOverride(path=("cloned",), visibility=SectionVisibility.FULL)
    )

    # Both sessions should have their overrides
    original_overrides = session[VisibilityOverrides].latest()
    cloned_overrides = cloned[VisibilityOverrides].latest()

    assert original_overrides is not None
    assert original_overrides.get(("original",)) == SectionVisibility.SUMMARY

    assert cloned_overrides is not None
    # Cloned session inherits state from original
    assert cloned_overrides.get(("original",)) == SectionVisibility.SUMMARY
    assert cloned_overrides.get(("cloned",)) == SectionVisibility.FULL


def test_builtin_reducer_registration_is_idempotent() -> None:
    """Calling _register_builtin_reducers multiple times is safe."""
    bus = InProcessDispatcher()
    session = Session(bus=bus)

    # Reducers are already registered from __init__
    # Calling again should be a no-op (guard prevents re-registration)
    session._register_builtin_reducers()

    # Should still work correctly
    session.dispatch(
        SetVisibilityOverride(path=("test",), visibility=SectionVisibility.FULL)
    )
    overrides = session[VisibilityOverrides].latest()
    assert overrides is not None
    assert overrides.get(("test",)) == SectionVisibility.FULL


def test_register_visibility_reducers_is_idempotent() -> None:
    """Calling register_visibility_reducers on a session adds reducers (idempotent)."""
    from weakincentives.runtime.session import register_visibility_reducers

    bus = InProcessDispatcher()
    session = Session(bus=bus)

    # Session already has reducers from __init__, but calling again is safe
    # (though it adds duplicate reducers - the guard is in _register_builtin_reducers)
    register_visibility_reducers(session)

    # Visibility events should work
    session.dispatch(
        SetVisibilityOverride(path=("test",), visibility=SectionVisibility.SUMMARY)
    )
    overrides = session[VisibilityOverrides].latest()
    assert overrides is not None
    assert overrides.get(("test",)) == SectionVisibility.SUMMARY

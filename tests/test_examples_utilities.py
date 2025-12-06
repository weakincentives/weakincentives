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

"""Tests for examples package utilities."""

from __future__ import annotations

import pytest

from examples import build_logged_session, render_plan_snapshot, resolve_override_tag
from tests.tools.helpers import find_tool, invoke_tool
from weakincentives.runtime import Session
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.tools import PlanningStrategy, PlanningToolsSection
from weakincentives.tools.planning import SetupPlan, UpdateStep

# Tests for build_logged_session


def test_build_logged_session_creates_session_with_event_bus() -> None:
    session = build_logged_session()

    assert session.event_bus is not None


def test_build_logged_session_applies_provided_tags() -> None:
    session = build_logged_session(tags={"app": "test-app", "env": "test"})

    # Tags are accessible on the session
    assert session.tags is not None


def test_build_logged_session_accepts_parent_session() -> None:
    parent = Session()
    child = build_logged_session(parent=parent)

    assert child.parent is parent


def test_build_logged_session_empty_tags_creates_valid_session() -> None:
    session = build_logged_session(tags={})

    assert session is not None


# Tests for resolve_override_tag


def test_resolve_override_tag_returns_explicit_tag_when_provided() -> None:
    result = resolve_override_tag("explicit-tag")

    assert result == "explicit-tag"


def test_resolve_override_tag_strips_whitespace_from_explicit_tag() -> None:
    result = resolve_override_tag("  padded-tag  ")

    assert result == "padded-tag"


def test_resolve_override_tag_returns_env_var_when_tag_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_OVERRIDE_TAG", "env-tag")

    result = resolve_override_tag(None, env_var="TEST_OVERRIDE_TAG")

    assert result == "env-tag"


def test_resolve_override_tag_returns_env_var_when_tag_whitespace_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_OVERRIDE_TAG", "env-tag")

    result = resolve_override_tag("   ", env_var="TEST_OVERRIDE_TAG")

    assert result == "env-tag"


def test_resolve_override_tag_returns_default_when_no_tag_or_env_var() -> None:
    result = resolve_override_tag(None)

    assert result == "latest"


def test_resolve_override_tag_returns_custom_default_when_specified() -> None:
    result = resolve_override_tag(None, default="custom-default")

    assert result == "custom-default"


def test_resolve_override_tag_ignores_env_var_when_explicit_tag_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_OVERRIDE_TAG", "env-tag")

    result = resolve_override_tag("explicit-tag", env_var="TEST_OVERRIDE_TAG")

    assert result == "explicit-tag"


def test_resolve_override_tag_returns_default_when_env_var_not_set() -> None:
    result = resolve_override_tag(None, env_var="NONEXISTENT_VAR")

    assert result == "latest"


def test_resolve_override_tag_strips_env_var_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_OVERRIDE_TAG", "  env-tag-padded  ")

    result = resolve_override_tag(None, env_var="TEST_OVERRIDE_TAG")

    assert result == "env-tag-padded"


# Tests for render_plan_snapshot


def test_render_plan_snapshot_returns_no_plan_message_when_empty() -> None:
    session = Session()

    result = render_plan_snapshot(session)

    assert result == "No active plan."


def test_render_plan_snapshot_renders_plan_with_objective_and_status() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session, strategy=PlanningStrategy.REACT)
    setup_tool = find_tool(section, "planning_setup_plan")
    invoke_tool(bus, setup_tool, SetupPlan(objective="Test the app"), session=session)

    result = render_plan_snapshot(session)

    assert "Objective: Test the app (status: active)" in result


def test_render_plan_snapshot_renders_plan_steps() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session, strategy=PlanningStrategy.REACT)
    setup_tool = find_tool(section, "planning_setup_plan")
    update_tool = find_tool(section, "planning_update_step")
    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(
            objective="Multi-step plan",
            initial_steps=("First step", "Second step", "Third step"),
        ),
        session=session,
    )
    invoke_tool(
        bus,
        update_tool,
        UpdateStep(step_id=1, status="done"),
        session=session,
    )
    invoke_tool(
        bus,
        update_tool,
        UpdateStep(step_id=2, status="in_progress"),
        session=session,
    )

    result = render_plan_snapshot(session)

    assert "- 1 [done] First step" in result
    assert "- 2 [in_progress] Second step" in result
    assert "- 3 [pending] Third step" in result


def test_render_plan_snapshot_renders_completed_plan() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session, strategy=PlanningStrategy.REACT)
    setup_tool = find_tool(section, "planning_setup_plan")
    update_tool = find_tool(section, "planning_update_step")
    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="Finished work", initial_steps=("Complete this",)),
        session=session,
    )
    # Mark step as done to auto-complete the plan
    invoke_tool(
        bus,
        update_tool,
        UpdateStep(step_id=1, status="done"),
        session=session,
    )

    result = render_plan_snapshot(session)

    assert "Objective: Finished work (status: completed)" in result
    assert "- 1 [done] Complete this" in result


def test_render_plan_snapshot_renders_plan_without_steps() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    section = PlanningToolsSection(session=session, strategy=PlanningStrategy.REACT)
    setup_tool = find_tool(section, "planning_setup_plan")
    invoke_tool(
        bus,
        setup_tool,
        SetupPlan(objective="Empty plan"),
        session=session,
    )

    result = render_plan_snapshot(session)

    assert "Objective: Empty plan (status: active)" in result
    # No step lines
    lines = result.split("\n")
    assert len(lines) == 1  # Only the objective line

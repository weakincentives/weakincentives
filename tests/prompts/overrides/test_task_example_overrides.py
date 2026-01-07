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

"""Tests for task example override loading and serialization."""

from __future__ import annotations

import pytest

from weakincentives.prompt.overrides import (
    HexDigest,
    PromptOverridesError,
    TaskExampleOverride,
    TaskStepOverride,
)
from weakincentives.prompt.overrides.validation import (
    load_task_example_overrides,
    serialize_task_example_overrides,
)

VALID_HASH = HexDigest("a" * 64)


def test_load_task_example_overrides_empty() -> None:
    assert load_task_example_overrides(None) == ()
    assert load_task_example_overrides([]) == ()


def test_load_task_example_overrides_not_a_list() -> None:
    with pytest.raises(PromptOverridesError, match="must be a list"):
        load_task_example_overrides({"not": "a list"})


def test_load_task_example_overrides_entry_not_object() -> None:
    with pytest.raises(PromptOverridesError, match="entry must be an object"):
        load_task_example_overrides(["not an object"])


def test_load_task_example_overrides_missing_path() -> None:
    with pytest.raises(PromptOverridesError, match="path must be a list"):
        load_task_example_overrides([{"index": 0, "action": "modify"}])


def test_load_task_example_overrides_missing_index() -> None:
    with pytest.raises(PromptOverridesError, match="index must be an integer"):
        load_task_example_overrides([{"path": ["a"], "action": "modify"}])


def test_load_task_example_overrides_invalid_action() -> None:
    with pytest.raises(PromptOverridesError, match="action must be"):
        load_task_example_overrides([{"path": ["a"], "index": 0, "action": "invalid"}])


def test_load_task_example_overrides_invalid_expected_hash() -> None:
    with pytest.raises(PromptOverridesError, match="expected_hash must be a string"):
        load_task_example_overrides(
            [{"path": ["a"], "index": 0, "action": "modify", "expected_hash": 123}]
        )


def test_load_task_example_overrides_invalid_objective() -> None:
    with pytest.raises(PromptOverridesError, match="objective must be a string"):
        load_task_example_overrides(
            [{"path": ["a"], "index": 0, "action": "modify", "objective": 123}]
        )


def test_load_task_example_overrides_invalid_outcome() -> None:
    with pytest.raises(PromptOverridesError, match="outcome must be a string"):
        load_task_example_overrides(
            [{"path": ["a"], "index": 0, "action": "modify", "outcome": 123}]
        )


def test_load_task_example_overrides_invalid_steps_to_remove() -> None:
    with pytest.raises(PromptOverridesError, match="steps_to_remove must be a list"):
        load_task_example_overrides(
            [
                {
                    "path": ["a"],
                    "index": 0,
                    "action": "modify",
                    "steps_to_remove": "not a list",
                }
            ]
        )


def test_load_task_example_overrides_valid() -> None:
    payload = [
        {
            "path": ["section", "example"],
            "index": 0,
            "expected_hash": str(VALID_HASH),
            "action": "modify",
            "objective": "Updated objective",
            "outcome": "Updated outcome",
            "steps_to_remove": [1, 2],
        }
    ]
    overrides = load_task_example_overrides(payload)
    assert len(overrides) == 1
    override = overrides[0]
    assert override.path == ("section", "example")
    assert override.index == 0
    assert override.expected_hash == VALID_HASH
    assert override.action == "modify"
    assert override.objective == "Updated objective"
    assert override.outcome == "Updated outcome"
    assert override.steps_to_remove == (1, 2)


def test_load_task_example_overrides_with_step_overrides() -> None:
    payload = [
        {
            "path": ["section"],
            "index": 0,
            "action": "modify",
            "step_overrides": [
                {
                    "index": 0,
                    "tool_name": "new_tool",
                    "description": "New description",
                    "input_json": '{"key": "value"}',
                    "output_json": '{"result": true}',
                }
            ],
        }
    ]
    overrides = load_task_example_overrides(payload)
    assert len(overrides) == 1
    assert len(overrides[0].step_overrides) == 1
    step = overrides[0].step_overrides[0]
    assert step.index == 0
    assert step.tool_name == "new_tool"
    assert step.description == "New description"
    assert step.input_json == '{"key": "value"}'
    assert step.output_json == '{"result": true}'


def test_load_step_overrides_not_a_list() -> None:
    with pytest.raises(PromptOverridesError, match="Step overrides must be a list"):
        load_task_example_overrides(
            [
                {
                    "path": ["a"],
                    "index": 0,
                    "action": "modify",
                    "step_overrides": "not a list",
                }
            ]
        )


def test_load_step_overrides_entry_not_object() -> None:
    with pytest.raises(
        PromptOverridesError, match="Step override entry must be an object"
    ):
        load_task_example_overrides(
            [
                {
                    "path": ["a"],
                    "index": 0,
                    "action": "modify",
                    "step_overrides": ["not an object"],
                }
            ]
        )


def test_load_step_overrides_missing_index() -> None:
    with pytest.raises(
        PromptOverridesError, match="Step override index must be an integer"
    ):
        load_task_example_overrides(
            [
                {
                    "path": ["a"],
                    "index": 0,
                    "action": "modify",
                    "step_overrides": [{"tool_name": "x"}],
                }
            ]
        )


def test_load_step_overrides_invalid_fields() -> None:
    with pytest.raises(PromptOverridesError, match="tool_name must be a string"):
        load_task_example_overrides(
            [
                {
                    "path": ["a"],
                    "index": 0,
                    "action": "modify",
                    "step_overrides": [{"index": 0, "tool_name": 123}],
                }
            ]
        )


def test_serialize_task_example_overrides_empty() -> None:
    assert serialize_task_example_overrides(()) == []


def test_serialize_task_example_overrides_round_trip() -> None:
    original = (
        TaskExampleOverride(
            path=("section", "example"),
            index=0,
            expected_hash=VALID_HASH,
            action="modify",
            objective="Test objective",
            outcome="Test outcome",
            step_overrides=(
                TaskStepOverride(
                    index=0,
                    tool_name="my_tool",
                    description="Step desc",
                    input_json='{"a": 1}',
                    output_json='{"b": 2}',
                ),
            ),
            steps_to_remove=(1, 2),
            steps_to_append=(TaskStepOverride(index=0, description="New step"),),
        ),
    )
    serialized = serialize_task_example_overrides(original)
    restored = load_task_example_overrides(serialized)
    assert restored == original


def test_serialize_task_example_overrides_minimal() -> None:
    override = TaskExampleOverride(
        path=("section",),
        index=0,
        expected_hash=None,
        action="append",
    )
    serialized = serialize_task_example_overrides((override,))
    assert serialized == [
        {
            "path": ["section"],
            "index": 0,
            "expected_hash": None,
            "action": "append",
        }
    ]


def test_serialize_task_example_overrides_step_with_only_index() -> None:
    """Test serialization of step override with only index field (all others None)."""
    override = TaskExampleOverride(
        path=("section",),
        index=0,
        expected_hash=None,
        action="modify",
        step_overrides=(TaskStepOverride(index=0),),  # Only index, all others None
    )
    serialized = serialize_task_example_overrides((override,))
    assert len(serialized) == 1
    assert serialized[0]["step_overrides"] == [{"index": 0}]

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

from collections.abc import Mapping
from typing import Literal, cast

from ...types import JSONValue
from .versioning import (
    HexDigest,
    PromptOverridesError,
    TaskExampleOverride,
    TaskStepOverride,
    ensure_hex_digest,
)


def _parse_task_example_path(item_map: Mapping[str, JSONValue]) -> tuple[str, ...]:
    """Parse and validate the path field from a task example override."""
    path_raw = item_map.get("path")
    if not isinstance(path_raw, list):
        raise PromptOverridesError("Task example override path must be a list.")
    return tuple(str(p) for p in path_raw)


def _parse_task_example_index(item_map: Mapping[str, JSONValue]) -> int:
    """Parse and validate the index field from a task example override."""
    index = item_map.get("index")
    if not isinstance(index, int):
        raise PromptOverridesError("Task example override index must be an integer.")
    return index


def _parse_task_example_hash(
    item_map: Mapping[str, JSONValue],
) -> HexDigest | None:
    """Parse and validate the expected_hash field from a task example override."""
    expected_hash_raw = item_map.get("expected_hash")
    if expected_hash_raw is None:
        return None
    if not isinstance(expected_hash_raw, str):
        raise PromptOverridesError("Task example expected_hash must be a string.")
    return ensure_hex_digest(expected_hash_raw, field_name="Task example expected_hash")


def _parse_task_example_action(
    item_map: Mapping[str, JSONValue],
) -> Literal["modify", "remove", "append"]:
    """Parse and validate the action field from a task example override."""
    action = item_map.get("action")
    if action not in {"modify", "remove", "append"}:
        raise PromptOverridesError(
            f"Task example override action must be 'modify', 'remove', or 'append', got {action!r}."
        )
    return action  # type: ignore[return-value]


def _parse_steps_to_remove(item_map: Mapping[str, JSONValue]) -> tuple[int, ...]:
    """Parse and validate the steps_to_remove field."""
    steps_to_remove_raw = item_map.get("steps_to_remove")
    if steps_to_remove_raw is None:
        return ()
    if not isinstance(steps_to_remove_raw, list):
        raise PromptOverridesError("steps_to_remove must be a list.")
    return tuple(i for i in steps_to_remove_raw if isinstance(i, int))


def _parse_task_example_entry(
    item_map: Mapping[str, JSONValue],
) -> TaskExampleOverride:
    """Parse a single task example override entry."""
    path = _parse_task_example_path(item_map)
    index = _parse_task_example_index(item_map)
    expected_hash = _parse_task_example_hash(item_map)
    action = _parse_task_example_action(item_map)

    objective = item_map.get("objective")
    if objective is not None and not isinstance(objective, str):
        raise PromptOverridesError("Task example objective must be a string.")
    outcome = item_map.get("outcome")
    if outcome is not None and not isinstance(outcome, str):
        raise PromptOverridesError("Task example outcome must be a string.")

    return TaskExampleOverride(
        path=path,
        index=index,
        expected_hash=expected_hash,
        action=action,
        objective=objective,
        outcome=outcome,
        step_overrides=_load_step_overrides(item_map.get("step_overrides")),
        steps_to_remove=_parse_steps_to_remove(item_map),
        steps_to_append=_load_step_overrides(item_map.get("steps_to_append")),
    )


def load_task_example_overrides(
    payload: JSONValue | None,
) -> tuple[TaskExampleOverride, ...]:
    """Load task example overrides from JSON payload."""
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise PromptOverridesError("Task example overrides must be a list.")
    overrides: list[TaskExampleOverride] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise PromptOverridesError("Task example override entry must be an object.")
        item_map = cast(Mapping[str, JSONValue], item)
        overrides.append(_parse_task_example_entry(item_map))
    return tuple(overrides)


def _parse_optional_string(
    item_map: Mapping[str, JSONValue],
    field: str,
    error_msg: str,
) -> str | None:
    """Parse an optional string field from a mapping."""
    value = item_map.get(field)
    if value is not None and not isinstance(value, str):
        raise PromptOverridesError(error_msg)
    return value


def _parse_step_override_entry(item_map: Mapping[str, JSONValue]) -> TaskStepOverride:
    """Parse a single step override entry."""
    index = item_map.get("index")
    if not isinstance(index, int):
        raise PromptOverridesError("Step override index must be an integer.")
    return TaskStepOverride(
        index=index,
        tool_name=_parse_optional_string(
            item_map, "tool_name", "Step override tool_name must be a string."
        ),
        description=_parse_optional_string(
            item_map, "description", "Step override description must be a string."
        ),
        input_json=_parse_optional_string(
            item_map, "input_json", "Step override input_json must be a string."
        ),
        output_json=_parse_optional_string(
            item_map, "output_json", "Step override output_json must be a string."
        ),
    )


def _load_step_overrides(
    payload: JSONValue | None,
) -> tuple[TaskStepOverride, ...]:
    """Load step overrides from JSON payload."""
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise PromptOverridesError("Step overrides must be a list.")
    steps: list[TaskStepOverride] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise PromptOverridesError("Step override entry must be an object.")
        item_map = cast(Mapping[str, JSONValue], item)
        steps.append(_parse_step_override_entry(item_map))
    return tuple(steps)


def serialize_task_example_overrides(
    overrides: tuple[TaskExampleOverride, ...],
) -> list[dict[str, JSONValue]]:
    """Serialize task example overrides to JSON-compatible format."""
    serialized: list[dict[str, JSONValue]] = []
    for override in overrides:
        entry: dict[str, JSONValue] = {
            "path": list(override.path),
            "index": override.index,
            "expected_hash": str(override.expected_hash)
            if override.expected_hash
            else None,
            "action": override.action,
        }
        if override.objective is not None:
            entry["objective"] = override.objective
        if override.outcome is not None:
            entry["outcome"] = override.outcome
        if override.step_overrides:
            entry["step_overrides"] = _serialize_step_overrides(override.step_overrides)
        if override.steps_to_remove:
            entry["steps_to_remove"] = list(override.steps_to_remove)
        if override.steps_to_append:
            entry["steps_to_append"] = _serialize_step_overrides(
                override.steps_to_append
            )
        serialized.append(entry)
    return serialized


def _serialize_step_overrides(
    steps: tuple[TaskStepOverride, ...],
) -> list[dict[str, JSONValue]]:
    """Serialize step overrides to JSON-compatible format."""
    serialized: list[dict[str, JSONValue]] = []
    for step in steps:
        entry: dict[str, JSONValue] = {"index": step.index}
        if step.tool_name is not None:
            entry["tool_name"] = step.tool_name
        if step.description is not None:
            entry["description"] = step.description
        if step.input_json is not None:
            entry["input_json"] = step.input_json
        if step.output_json is not None:
            entry["output_json"] = step.output_json
        serialized.append(entry)
    return serialized

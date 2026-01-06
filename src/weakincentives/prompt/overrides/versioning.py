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

import json
import re
from dataclasses import field
from hashlib import sha256
from typing import Literal, Protocol, Self, cast, overload

from ...dataclasses import FrozenDataclass
from ...errors import WinkError
from ...serde.schema import schema
from ...types import JSONValue
from ...types.dataclass import SupportsDataclass, SupportsDataclassOrNone


def _section_override_mapping_factory() -> dict[tuple[str, ...], SectionOverride]:
    return {}


def _tool_override_mapping_factory() -> dict[str, ToolOverride]:
    return {}


def _param_description_mapping_factory() -> dict[str, str]:
    return {}


def _tool_example_overrides_factory() -> tuple[ToolExampleOverride, ...]:
    return ()


def _task_example_overrides_factory() -> tuple[TaskExampleOverride, ...]:
    return ()


def _task_example_descriptors_factory() -> list[TaskExampleDescriptor]:
    return []


class ToolContractProtocol(Protocol):
    name: str
    description: str
    params_type: type[SupportsDataclass] | type[None]
    result_type: type[SupportsDataclassOrNone]
    result_container: Literal["object", "array"]
    accepts_overrides: bool


class SectionLike(Protocol):
    def original_body_template(self) -> str | None: ...

    def tools(self) -> tuple[ToolContractProtocol, ...]: ...

    accepts_overrides: bool


class SectionNodeLike(Protocol):
    path: tuple[str, ...]
    number: str
    section: SectionLike


class PromptLike(Protocol):
    ns: str
    key: str

    @property
    def sections(self) -> tuple[SectionNodeLike, ...]: ...


_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class HexDigest(str):
    """A validated lowercase hexadecimal SHA-256 digest."""

    __slots__ = ()

    def __new__(cls, value: object) -> Self:
        if not isinstance(value, str):
            msg = "HexDigest value must be a string."
            raise TypeError(msg)
        if not _HEX_DIGEST_RE.fullmatch(value):
            msg = f"Invalid hex digest value: {value!r}"
            raise ValueError(msg)
        return str.__new__(cls, value)


@overload
def ensure_hex_digest(value: HexDigest, *, field_name: str) -> HexDigest: ...


@overload
def ensure_hex_digest(value: str, *, field_name: str) -> HexDigest: ...


def ensure_hex_digest(value: object, *, field_name: str) -> HexDigest:
    """Normalize an object to a :class:`HexDigest` with helpful errors."""

    if isinstance(value, HexDigest):
        return value
    if isinstance(value, str):
        try:
            return HexDigest(value)
        except ValueError as error:
            msg = f"{field_name} must be a 64 character lowercase hex digest."
            raise PromptOverridesError(msg) from error
    msg = f"{field_name} must be a string."
    raise PromptOverridesError(msg)


@FrozenDataclass()
class SectionDescriptor:
    """Hash metadata for a single section within a prompt."""

    path: tuple[str, ...]
    content_hash: HexDigest
    number: str


@FrozenDataclass()
class ToolDescriptor:
    """Stable metadata describing a tool exposed by a prompt."""

    path: tuple[str, ...]
    name: str
    contract_hash: HexDigest
    example_hashes: tuple[HexDigest, ...] = ()


@FrozenDataclass()
class TaskExampleDescriptor:
    """Metadata describing a task example within a section."""

    path: tuple[str, ...]
    index: int
    content_hash: HexDigest


@FrozenDataclass()
class PromptDescriptor:
    """Stable metadata describing a prompt and its hash-aware sections."""

    ns: str
    key: str
    sections: list[SectionDescriptor]
    tools: list[ToolDescriptor]
    task_examples: list[TaskExampleDescriptor] = field(
        default_factory=_task_example_descriptors_factory
    )

    @classmethod
    def from_prompt(cls, prompt: PromptLike) -> PromptDescriptor:
        sections: list[SectionDescriptor] = []
        tools: list[ToolDescriptor] = []
        task_examples: list[TaskExampleDescriptor] = []
        for node in prompt.sections:
            if getattr(node.section, "accepts_overrides", True):
                template = node.section.original_body_template()
                if template is not None:
                    content_hash = hash_text(template)
                    sections.append(
                        SectionDescriptor(node.path, content_hash, node.number)
                    )
            tool_descriptors = [
                ToolDescriptor(
                    path=node.path,
                    name=tool.name,
                    contract_hash=_tool_contract_hash(tool),
                    example_hashes=_tool_example_hashes(tool),
                )
                for tool in node.section.tools()
                if tool.accepts_overrides
            ]
            tools.extend(tool_descriptors)
            # Collect task example descriptors
            task_example_descs = _task_example_descriptors(node)
            task_examples.extend(task_example_descs)
        return cls(prompt.ns, prompt.key, sections, tools, task_examples)


def descriptor_for_prompt(prompt: PromptLike) -> PromptDescriptor:
    """Return a cached prompt descriptor when available."""

    descriptor = getattr(prompt, "descriptor", None)
    if isinstance(descriptor, PromptDescriptor):
        return descriptor
    return PromptDescriptor.from_prompt(prompt)


@FrozenDataclass()
class SectionOverride:
    """Override payload for a prompt section validated by hash."""

    path: tuple[str, ...]
    expected_hash: HexDigest
    body: str


@FrozenDataclass()
class ToolExampleOverride:
    """Override for a single tool example."""

    index: int  # Original index, or -1 for append
    expected_hash: HexDigest | None  # None for new examples
    action: Literal["modify", "remove", "append"]
    description: str | None = None
    input_json: str | None = None  # JSON-serialized params
    output_json: str | None = None  # JSON-serialized result


@FrozenDataclass()
class ToolOverride:
    """Description overrides validated against a tool contract hash."""

    name: str
    expected_contract_hash: HexDigest
    description: str | None = None
    param_descriptions: dict[str, str] = field(
        default_factory=_param_description_mapping_factory
    )
    example_overrides: tuple[ToolExampleOverride, ...] = field(
        default_factory=_tool_example_overrides_factory
    )


@FrozenDataclass()
class TaskStepOverride:
    """Override for a single step within a task example."""

    index: int
    tool_name: str | None = None  # None = keep original
    description: str | None = None
    input_json: str | None = None
    output_json: str | None = None


@FrozenDataclass()
class TaskExampleOverride:
    """Override for a task example within TaskExamplesSection."""

    path: tuple[str, ...]
    index: int  # Original index, or -1 for append
    expected_hash: HexDigest | None  # None for new examples
    action: Literal["modify", "remove", "append"]
    objective: str | None = None
    outcome: str | None = None  # String or JSON-serialized dataclass
    step_overrides: tuple[TaskStepOverride, ...] = ()
    steps_to_remove: tuple[int, ...] = ()  # Indices to remove
    steps_to_append: tuple[TaskStepOverride, ...] = ()  # New steps


@FrozenDataclass()
class PromptOverride:
    """Runtime replacements for prompt sections validated by an overrides store."""

    ns: str
    prompt_key: str
    tag: str
    sections: dict[tuple[str, ...], SectionOverride] = field(
        default_factory=_section_override_mapping_factory
    )
    tool_overrides: dict[str, ToolOverride] = field(
        default_factory=_tool_override_mapping_factory
    )
    task_example_overrides: tuple[TaskExampleOverride, ...] = field(
        default_factory=_task_example_overrides_factory
    )


class PromptOverridesStore(Protocol):
    """Structural interface satisfied by prompt overrides stores."""

    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None: ...

    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride: ...

    def delete(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> None: ...

    def store(
        self,
        prompt: PromptLike,
        override: SectionOverride | ToolOverride | TaskExampleOverride,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        """Store a single override, dispatching by type."""
        ...

    def seed(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride: ...


class PromptOverridesError(WinkError):
    """Raised when prompt overrides fail validation or persistence."""


__all__ = [
    "HexDigest",
    "PromptDescriptor",
    "PromptOverride",
    "PromptOverridesError",
    "PromptOverridesStore",
    "SectionDescriptor",
    "SectionOverride",
    "TaskExampleDescriptor",
    "TaskExampleOverride",
    "TaskStepOverride",
    "ToolDescriptor",
    "ToolExampleOverride",
    "ToolOverride",
    "descriptor_for_prompt",
    "ensure_hex_digest",
    "hash_json",
    "hash_text",
]


def _tool_contract_hash(tool: ToolContractProtocol) -> HexDigest:
    description_hash = hash_text(tool.description)
    params_schema_hash = hash_json(_params_schema(tool.params_type))
    container = cast(
        Literal["object", "array"], getattr(tool, "result_container", "object")
    )
    result_schema_hash = hash_json(_result_schema(tool.result_type, container))
    return hash_text(
        "::".join((description_hash, params_schema_hash, result_schema_hash))
    )


def hash_text(value: str) -> HexDigest:
    return HexDigest(sha256(value.encode("utf-8")).hexdigest())


def hash_json(value: object) -> HexDigest:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hash_text(canonical)


def _params_schema(
    params_type: type[SupportsDataclass] | type[None],
) -> dict[str, JSONValue]:
    if params_type is type(None):
        return {
            "title": "NoneType",
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    return schema(params_type, extra="forbid")


def _result_schema(
    result_type: type[SupportsDataclassOrNone],
    container: Literal["object", "array"],
) -> dict[str, JSONValue]:
    if result_type is type(None):
        return {"title": "NoneType", "type": "null"}
    if container == "array":
        item_schema = schema(result_type, extra="ignore")
        return {
            "title": f"{result_type.__name__}List",
            "type": "array",
            "items": item_schema,
        }
    return schema(result_type, extra="ignore")


def _tool_example_hashes(tool: ToolContractProtocol) -> tuple[HexDigest, ...]:
    """Compute content hashes for all examples of a tool."""
    examples = getattr(tool, "examples", ())
    hashes: list[HexDigest] = []
    for example in examples:
        # Use None for missing values to distinguish from empty strings
        description = getattr(example, "description", None)
        example_data = {
            "description": description,
            "input": _serialize_example_value(getattr(example, "input", None)),
            "output": _serialize_example_value(getattr(example, "output", None)),
        }
        hashes.append(hash_json(example_data))
    return tuple(hashes)


def _serialize_example_value(value: object) -> JSONValue:
    """Serialize an example input/output value to JSON-compatible form."""
    if value is None:
        return None
    # Try to use serde.dump if it's a dataclass
    from ...serde import dump

    try:
        return dump(value, exclude_none=True)
    except Exception:
        # Fallback for non-dataclass values - log for debugging
        from ...runtime.logging import get_logger

        logger = get_logger(__name__, context={"component": "prompt_overrides"})
        logger.debug(
            "Falling back to str() for example value serialization.",
            event="example_value_fallback",
            context={"value_type": type(value).__name__},
        )
        return str(value)


def _task_example_descriptors(node: SectionNodeLike) -> list[TaskExampleDescriptor]:
    """Extract task example descriptors from a section node."""
    section = node.section
    # Check if this is a TaskExamplesSection by looking for children that are TaskExample
    children = getattr(section, "children", None)
    if children is None:
        return []

    descriptors: list[TaskExampleDescriptor] = []
    for idx, child in enumerate(children):
        # Check if child is a TaskExample by looking for objective, steps, outcome
        objective = getattr(child, "objective", None)
        steps = getattr(child, "steps", None)
        outcome = getattr(child, "outcome", None)
        if objective is None or steps is None or outcome is None:
            continue

        # Compute content hash for the task example
        step_data: list[dict[str, str]] = []
        for step in steps:
            tool_name: str = getattr(step, "tool_name", "")
            example = getattr(step, "example", None)
            description: str = getattr(example, "description", "") if example else ""
            step_data.append({"tool": tool_name, "description": description})

        example_data: dict[str, JSONValue] = {
            "objective": objective,
            "steps": step_data,
            "outcome": _serialize_example_value(outcome)
            if not isinstance(outcome, str)
            else outcome,
        }
        content_hash = hash_json(example_data)

        child_key = getattr(child, "key", f"example-{idx}")
        example_path = (*node.path, child_key)
        descriptors.append(
            TaskExampleDescriptor(
                path=example_path, index=idx, content_hash=content_hash
            )
        )

    return descriptors

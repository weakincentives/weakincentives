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
from pathlib import Path
from typing import cast, override

from ...runtime.logging import StructuredLogger, get_logger
from ...types import JSONValue
from ._fs import OverrideFilesystem
from .validation import (
    FORMAT_VERSION,
    filter_override_for_descriptor,
    load_sections,
    load_task_example_overrides,
    load_tools,
    seed_sections,
    seed_tools,
    serialize_sections,
    serialize_task_example_overrides,
    serialize_tools,
    validate_header,
    validate_sections_for_write,
    validate_tools_for_write,
)
from .versioning import (
    HexDigest,
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    SectionOverride,
    TaskExampleOverride,
    ToolOverride,
    descriptor_for_prompt,
)

_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "prompt_overrides"}
)
_DEFAULT_RELATIVE_PATH = Path(".weakincentives") / "prompts" / "overrides"


class LocalPromptOverridesStore(PromptOverridesStore):
    """Persist prompt overrides to disk within the project workspace."""

    def __init__(
        self,
        *,
        root_path: str | Path | None = None,
        overrides_relative_path: str | Path = _DEFAULT_RELATIVE_PATH,
    ) -> None:
        super().__init__()
        explicit_root = Path(root_path).resolve() if root_path else None
        overrides_relative = Path(overrides_relative_path)
        self._filesystem = OverrideFilesystem(
            explicit_root=explicit_root,
            overrides_relative_path=overrides_relative,
        )

    @override
    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None:
        normalized_tag = self._filesystem.validate_identifier(tag, "tag")
        file_path = self._filesystem.override_file_path(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=normalized_tag,
        )
        with self._filesystem.locked_override_path(file_path):
            if not file_path.exists():
                _LOGGER.debug(
                    "Override file not found.",
                    event="prompt_override_missing",
                    context={
                        "ns": descriptor.ns,
                        "prompt_key": descriptor.key,
                        "tag": normalized_tag,
                    },
                )
                return None

            payload: dict[str, JSONValue]
            try:
                with file_path.open("r", encoding="utf-8") as handle:
                    payload = cast(dict[str, JSONValue], json.load(handle))
            except json.JSONDecodeError as error:
                raise PromptOverridesError(
                    f"Failed to parse prompt override JSON: {file_path}"
                ) from error

        validate_header(payload, descriptor, normalized_tag, file_path)

        sections_payload = payload.get("sections")
        sections = load_sections(sections_payload, descriptor)
        tools_payload = payload.get("tools")
        tools = load_tools(tools_payload, descriptor)
        task_example_overrides = load_task_example_overrides(
            payload.get("task_example_overrides")
        )

        raw_override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=sections,
            tool_overrides=tools,
            task_example_overrides=task_example_overrides,
        )

        filtered_sections, filtered_tools = filter_override_for_descriptor(
            descriptor, raw_override
        )

        if not filtered_sections and not filtered_tools and not task_example_overrides:
            _LOGGER.debug(
                "No applicable overrides remain after validation.",
                event="prompt_override_empty",
                context={
                    "ns": descriptor.ns,
                    "prompt_key": descriptor.key,
                    "tag": normalized_tag,
                },
            )
            return None

        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=filtered_sections,
            tool_overrides=filtered_tools,
            task_example_overrides=task_example_overrides,
        )
        _LOGGER.info(
            "Resolved prompt override.",
            event="prompt_override_resolved",
            context={
                "ns": descriptor.ns,
                "prompt_key": descriptor.key,
                "tag": normalized_tag,
                "section_count": len(filtered_sections),
                "tool_count": len(filtered_tools),
                "task_example_count": len(task_example_overrides),
            },
        )
        return override

    @override
    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride:
        if override.ns != descriptor.ns or override.prompt_key != descriptor.key:
            raise PromptOverridesError(
                "Override metadata does not match descriptor.",
            )
        normalized_tag = self._filesystem.validate_identifier(override.tag, "tag")

        file_path = self._filesystem.override_file_path(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=normalized_tag,
        )

        validated_sections = validate_sections_for_write(
            override.sections,
            descriptor,
        )
        validated_tools = validate_tools_for_write(
            override.tool_overrides,
            descriptor,
        )

        payload = {
            "version": FORMAT_VERSION,
            "ns": descriptor.ns,
            "prompt_key": descriptor.key,
            "tag": normalized_tag,
            "sections": serialize_sections(validated_sections),
            "tools": serialize_tools(validated_tools),
            "task_example_overrides": serialize_task_example_overrides(
                override.task_example_overrides
            ),
        }

        with self._filesystem.locked_override_path(file_path):
            self._filesystem.atomic_write(file_path, payload)

        persisted = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=normalized_tag,
            sections=validated_sections,
            tool_overrides=validated_tools,
            task_example_overrides=override.task_example_overrides,
        )
        _LOGGER.info(
            "Persisted prompt override.",
            event="prompt_override_persisted",
            context={
                "ns": descriptor.ns,
                "prompt_key": descriptor.key,
                "tag": normalized_tag,
                "section_count": len(validated_sections),
                "tool_count": len(validated_tools),
                "task_example_count": len(override.task_example_overrides),
            },
        )
        return persisted

    @override
    def delete(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> None:
        normalized_tag = self._filesystem.validate_identifier(tag, "tag")
        file_path = self._filesystem.override_file_path(
            ns=ns,
            prompt_key=prompt_key,
            tag=normalized_tag,
        )
        with self._filesystem.locked_override_path(file_path):
            try:
                file_path.unlink()
            except FileNotFoundError:
                _LOGGER.debug(
                    "No override file to delete.",
                    event="prompt_override_delete_missing",
                    context={
                        "ns": ns,
                        "prompt_key": prompt_key,
                        "tag": normalized_tag,
                    },
                )

    @override
    def store(
        self,
        descriptor: PromptDescriptor,
        override: SectionOverride | ToolOverride | TaskExampleOverride,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        """Store a single override, dispatching by type.

        Holds a lock for the entire read-modify-write sequence to prevent
        TOCTOU race conditions.
        """
        _require_descriptor(descriptor)
        normalized_tag = self._filesystem.validate_identifier(tag, "tag")
        file_path = self._filesystem.override_file_path(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=normalized_tag,
        )

        # Hold lock for entire read-modify-write sequence
        with self._filesystem.locked_override_path(file_path):
            existing_override = self._resolve_unlocked(
                descriptor, normalized_tag, file_path
            )

            sections = dict(existing_override.sections) if existing_override else {}
            tools = dict(existing_override.tool_overrides) if existing_override else {}
            task_examples = (
                list(existing_override.task_example_overrides)
                if existing_override
                else []
            )

            if isinstance(override, SectionOverride):
                expected_hash = _lookup_section_hash(descriptor, override.path)
                if override.expected_hash != expected_hash:
                    msg = (
                        f"Hash mismatch for section {override.path!r}: expected "
                        f"{expected_hash}, got {override.expected_hash}."
                    )
                    raise PromptOverridesError(msg)
                sections[override.path] = override
            elif isinstance(override, ToolOverride):
                expected_hash = _lookup_tool_hash(descriptor, override.name)
                if override.expected_contract_hash != expected_hash:
                    msg = (
                        f"Hash mismatch for tool {override.name!r}: expected "
                        f"{expected_hash}, got {override.expected_contract_hash}."
                    )
                    raise PromptOverridesError(msg)
                tools[override.name] = override
            else:
                # TaskExampleOverride - type narrowed by pyright after above checks
                found = False
                for i, existing in enumerate(task_examples):
                    if (
                        existing.path == override.path
                        and existing.index == override.index
                    ):
                        task_examples[i] = override
                        found = True
                        break
                if not found:
                    task_examples.append(override)

            prompt_override = PromptOverride(
                ns=descriptor.ns,
                prompt_key=descriptor.key,
                tag=normalized_tag,
                sections=sections,
                tool_overrides=tools,
                task_example_overrides=tuple(task_examples),
            )
            return self._upsert_unlocked(descriptor, prompt_override, file_path)

    def _resolve_unlocked(
        self,
        descriptor: PromptDescriptor,
        tag: str,
        file_path: Path,
    ) -> PromptOverride | None:
        """Resolve override without acquiring lock (caller must hold lock)."""
        if not file_path.exists():
            return None

        payload: dict[str, JSONValue]
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                payload = cast(dict[str, JSONValue], json.load(handle))
        except json.JSONDecodeError as error:
            raise PromptOverridesError(
                f"Failed to parse prompt override JSON: {file_path}"
            ) from error

        validate_header(payload, descriptor, tag, file_path)

        sections = load_sections(payload.get("sections"), descriptor)
        tools = load_tools(payload.get("tools"), descriptor)
        task_example_overrides = load_task_example_overrides(
            payload.get("task_example_overrides")
        )

        raw_override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=sections,
            tool_overrides=tools,
            task_example_overrides=task_example_overrides,
        )

        filtered_sections, filtered_tools = filter_override_for_descriptor(
            descriptor, raw_override
        )

        if not filtered_sections and not filtered_tools and not task_example_overrides:
            return None

        return PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=filtered_sections,
            tool_overrides=filtered_tools,
            task_example_overrides=task_example_overrides,
        )

    def _upsert_unlocked(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
        file_path: Path,
    ) -> PromptOverride:
        """Upsert override without acquiring lock (caller must hold lock)."""
        # Uses self._filesystem.atomic_write - must be instance method
        validated_sections = validate_sections_for_write(
            override.sections,
            descriptor,
        )
        validated_tools = validate_tools_for_write(
            override.tool_overrides,
            descriptor,
        )

        payload = {
            "version": FORMAT_VERSION,
            "ns": descriptor.ns,
            "prompt_key": descriptor.key,
            "tag": override.tag,
            "sections": serialize_sections(validated_sections),
            "tools": serialize_tools(validated_tools),
            "task_example_overrides": serialize_task_example_overrides(
                override.task_example_overrides
            ),
        }

        self._filesystem.atomic_write(file_path, payload)

        return PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=override.tag,
            sections=validated_sections,
            tool_overrides=validated_tools,
            task_example_overrides=override.task_example_overrides,
        )

    @override
    def seed(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        descriptor = descriptor_for_prompt(prompt)
        normalized_tag = self._filesystem.validate_identifier(tag, "tag")
        file_path = self._filesystem.override_file_path(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=normalized_tag,
        )

        with self._filesystem.locked_override_path(file_path):
            if file_path.exists():
                existing = self.resolve(descriptor=descriptor, tag=normalized_tag)
                if existing is None:
                    raise PromptOverridesError(
                        "Override file exists but could not be resolved."
                    )
                return existing

            sections = seed_sections(prompt, descriptor)
            tools = seed_tools(prompt, descriptor)

            seed_override = PromptOverride(
                ns=descriptor.ns,
                prompt_key=descriptor.key,
                tag=normalized_tag,
                sections=sections,
                tool_overrides=tools,
                task_example_overrides=(),  # Task examples not seeded by default
            )
            return self.upsert(descriptor, seed_override)


def _require_descriptor(descriptor: object) -> None:
    """Fail fast if caller passes PromptLike instead of PromptDescriptor.

    Uses explicit isinstance() rather than @require decorator because DbC
    decorators are test-time only. This check must run in production to
    provide a clear TypeError with guidance on how to fix the call.
    """
    if not isinstance(descriptor, PromptDescriptor):
        msg = (
            "store() requires a PromptDescriptor, not a Prompt or PromptTemplate. "
            "Use PromptDescriptor.from_prompt(prompt) to create a descriptor."
        )
        raise TypeError(msg)


def _lookup_section_hash(
    descriptor: PromptDescriptor, path: tuple[str, ...]
) -> HexDigest:
    for candidate in descriptor.sections:
        if candidate.path == path:
            return candidate.content_hash
    raise PromptOverridesError(
        f"Section {path!r} not registered in prompt descriptor; cannot override."
    )


def _lookup_tool_hash(descriptor: PromptDescriptor, name: str) -> HexDigest:
    for candidate in descriptor.tools:
        if candidate.name == name:
            return candidate.contract_hash
    raise PromptOverridesError(
        f"Tool {name!r} not registered in prompt descriptor; cannot override."
    )


__all__ = ["LocalPromptOverridesStore"]

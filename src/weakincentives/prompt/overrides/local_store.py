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
from collections.abc import Mapping
from pathlib import Path
from typing import cast, override

from ...runtime.logging import StructuredLogger, get_logger
from ...types import JSONValue
from ._fs import OverrideFilesystem
from .validation import (
    FORMAT_VERSION,
    filter_override_for_descriptor,
    load_sections,
    load_tools,
    seed_sections,
    seed_tools,
    serialize_sections,
    serialize_tools,
    validate_header,
    validate_sections_for_write,
    validate_tools_for_write,
)
from .versioning import (
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
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
        sections = load_sections(
            cast(Mapping[str, JSONValue] | None, sections_payload), descriptor
        )
        tools_payload = payload.get("tools")
        tools = load_tools(cast(Mapping[str, JSONValue] | None, tools_payload), descriptor)

        raw_override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=sections,
            tool_overrides=tools,
        )

        filtered_sections, filtered_tools = filter_override_for_descriptor(
            descriptor, raw_override
        )

        if not filtered_sections and not filtered_tools:
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
        }

        with self._filesystem.locked_override_path(file_path):
            self._filesystem.atomic_write(file_path, payload)

        persisted = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=normalized_tag,
            sections=validated_sections,
            tool_overrides=validated_tools,
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
    def seed_if_necessary(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        descriptor = PromptDescriptor.from_prompt(prompt)
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
            )
            return self.upsert(descriptor, seed_override)


__all__ = ["LocalPromptOverridesStore"]

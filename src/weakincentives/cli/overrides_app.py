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

"""FastAPI app for editing prompt overrides from session snapshots."""

from __future__ import annotations

import json
import threading
import webbrowser
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote
from uuid import UUID

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt

from ..dataclasses import FrozenDataclass
from ..errors import WinkError
from ..prompt.overrides import (
    HexDigest,
    LocalPromptOverridesStore,
    PromptDescriptor,
    PromptOverride,
    SectionDescriptor,
    SectionOverride,
    ToolDescriptor,
    ToolOverride,
)
from ..prompt.overrides._fs import OverrideFilesystem
from ..runtime.logging import StructuredLogger, get_logger
from ..types import JSONValue
from .debug_app import (
    LoadedSnapshot,
    SnapshotLoader,
    SnapshotStore,
    load_snapshot,
)

logger: StructuredLogger = get_logger(__name__)

_markdown = MarkdownIt("commonmark", {"linkify": True})


# --- Error Types ---


class OverridesEditorError(WinkError, RuntimeError):
    """Base error for overrides editor operations."""


class PromptNotFoundError(OverridesEditorError):
    """Requested prompt not found in extracted prompts."""


class SectionNotFoundError(OverridesEditorError):
    """Requested section path not in prompt descriptor."""


class ToolNotFoundError(OverridesEditorError):
    """Requested tool not in prompt descriptor."""


class PromptNotSeededError(OverridesEditorError):
    """Prompt has no seed file; cannot create or edit overrides."""


class HashMismatchError(OverridesEditorError):
    """Override hash doesn't match current descriptor."""


# --- Data Types ---


@FrozenDataclass()
class ExtractedPrompt:
    """Prompt metadata extracted from a PromptRendered event."""

    ns: str
    key: str
    name: str | None
    descriptor: PromptDescriptor
    rendered_text: str
    created_at: datetime
    event_id: UUID


@FrozenDataclass()
class SectionState:
    """State for a single section override."""

    path: tuple[str, ...]
    number: str
    original_hash: HexDigest
    current_body: str | None
    is_overridden: bool
    is_stale: bool


@FrozenDataclass()
class ToolState:
    """State for a single tool override."""

    name: str
    path: tuple[str, ...]
    original_contract_hash: HexDigest
    current_description: str | None
    current_param_descriptions: dict[str, str]
    is_overridden: bool
    is_stale: bool


@FrozenDataclass()
class PromptOverrideState:
    """Combined state for editing a prompt's overrides."""

    prompt: ExtractedPrompt
    override: PromptOverride | None
    is_seeded: bool
    sections: list[SectionState]
    tools: list[ToolState]


# --- Prompt Extraction ---


def _parse_datetime(value: object) -> datetime:
    """Parse datetime from various formats."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Handle ISO format with timezone
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    msg = f"Cannot parse datetime from {type(value)}"
    raise TypeError(msg)


def _parse_uuid(value: object) -> UUID:
    """Parse UUID from string or UUID."""
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    msg = f"Cannot parse UUID from {type(value)}"
    raise TypeError(msg)


def _parse_descriptor(data: Mapping[str, Any]) -> PromptDescriptor:
    """Parse a PromptDescriptor from serialized data."""
    sections: list[SectionDescriptor] = []
    for section_data in data.get("sections", []):
        path = tuple(section_data["path"])
        content_hash = HexDigest(section_data["content_hash"])
        number = section_data["number"]
        sections.append(
            SectionDescriptor(path=path, content_hash=content_hash, number=number)
        )

    tools: list[ToolDescriptor] = []
    for tool_data in data.get("tools", []):
        path = tuple(tool_data["path"])
        name = tool_data["name"]
        contract_hash = HexDigest(tool_data["contract_hash"])
        tools.append(ToolDescriptor(path=path, name=name, contract_hash=contract_hash))

    return PromptDescriptor(
        ns=data["ns"],
        key=data["key"],
        sections=sections,
        tools=tools,
    )


def _extract_prompts_from_snapshot(
    snapshot: LoadedSnapshot,
) -> list[ExtractedPrompt]:
    """Extract prompts from a snapshot's PromptRendered events."""
    slice_type = "weakincentives.runtime.events:PromptRendered"
    slices = snapshot.slices
    if slice_type not in slices:
        return []

    slice_payload = slices[slice_type]
    prompts: list[ExtractedPrompt] = []

    for item in slice_payload.items:
        descriptor_data = item.get("descriptor")
        if descriptor_data is None:
            continue

        try:
            descriptor = _parse_descriptor(cast(Mapping[str, Any], descriptor_data))
            prompt = ExtractedPrompt(
                ns=str(item["prompt_ns"]),
                key=str(item["prompt_key"]),
                name=item.get("prompt_name"),  # type: ignore[arg-type]
                descriptor=descriptor,
                rendered_text=str(item["rendered_prompt"]),
                created_at=_parse_datetime(item["created_at"]),
                event_id=_parse_uuid(item["event_id"]),
            )
            prompts.append(prompt)
        except (KeyError, TypeError, ValueError) as error:
            logger.warning(
                "Failed to extract prompt from event",
                event="wink.overrides.extraction_error",
                context={"error": str(error)},
            )
            continue

    return prompts


def _deduplicate_prompts(
    prompts: list[ExtractedPrompt],
) -> dict[tuple[str, str], ExtractedPrompt]:
    """Deduplicate prompts by (ns, key), keeping most recent."""
    result: dict[tuple[str, str], ExtractedPrompt] = {}
    for prompt in prompts:
        key = (prompt.ns, prompt.key)
        existing = result.get(key)
        if existing is None or prompt.created_at > existing.created_at:
            if existing is not None and prompt.descriptor != existing.descriptor:
                logger.warning(
                    "Prompt descriptor hash drift detected",
                    event="wink.overrides.hash_drift",
                    context={"ns": prompt.ns, "key": prompt.key},
                )
            result[key] = prompt
    return result


# --- Overrides Store ---


class OverridesStore:
    """In-memory store managing override state."""

    def __init__(
        self,
        snapshot_path: Path,
        *,
        tag: str = "latest",
        store_root: Path | None = None,
        loader: SnapshotLoader,
        log: StructuredLogger,
    ) -> None:
        super().__init__()
        self._snapshot_path = snapshot_path.resolve()
        self._tag = tag
        self._store_root = store_root
        self._loader = loader
        self._logger = log
        self._snapshot_store = SnapshotStore(
            snapshot_path,
            loader=loader,
            logger=log,
        )
        self._prompts: dict[tuple[str, str], ExtractedPrompt] = {}
        self._local_store: LocalPromptOverridesStore | None = None
        self._filesystem: OverrideFilesystem | None = None
        self._extract_prompts()

    @property
    def prompts(self) -> tuple[ExtractedPrompt, ...]:
        return tuple(self._prompts.values())

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def snapshot_path(self) -> Path:
        return self._snapshot_path

    @property
    def store_root(self) -> Path | None:
        return self._store_root

    def _get_local_store(self) -> LocalPromptOverridesStore:
        """Get or create the local overrides store."""
        if self._local_store is None:
            self._local_store = LocalPromptOverridesStore(root_path=self._store_root)
        return self._local_store

    def _get_filesystem(self) -> OverrideFilesystem:
        """Get or create the override filesystem."""
        if self._filesystem is None:
            explicit_root = self._store_root.resolve() if self._store_root else None
            self._filesystem = OverrideFilesystem(
                explicit_root=explicit_root,
                overrides_relative_path=Path(".weakincentives")
                / "prompts"
                / "overrides",
            )
        return self._filesystem

    def _extract_prompts(self) -> None:
        """Extract prompts from all loaded snapshots."""
        all_prompts: list[ExtractedPrompt] = []
        for entry in self._snapshot_store.entries:
            all_prompts.extend(_extract_prompts_from_snapshot(entry))

        self._prompts = _deduplicate_prompts(all_prompts)

        if not self._prompts:
            self._logger.warning(
                "No PromptRendered events found in snapshot",
                event="wink.overrides.no_prompts",
                context={"snapshot_path": str(self._snapshot_path)},
            )
        else:
            self._logger.info(
                "Prompts extracted from snapshot",
                event="wink.overrides.prompts_extracted",
                context={"count": len(self._prompts)},
            )

    def _is_seeded(self, prompt: ExtractedPrompt) -> bool:
        """Check if an override file exists for the prompt."""
        fs = self._get_filesystem()
        try:
            file_path = fs.override_file_path(
                ns=prompt.ns,
                prompt_key=prompt.key,
                tag=self._tag,
            )
            return file_path.exists()
        except Exception:  # pragma: no cover
            return False

    def _read_raw_override_data(
        self, prompt: ExtractedPrompt
    ) -> tuple[dict[tuple[str, ...], dict[str, str]], dict[str, dict[str, Any]]] | None:
        """Read raw override file data without filtering.

        Returns (sections_data, tools_data) or None if no file exists.
        sections_data: {path: {"expected_hash": str, "body": str}}
        tools_data: {name: {"expected_contract_hash": str, "description": str, ...}}
        """
        fs = self._get_filesystem()
        file_path = fs.override_file_path(
            ns=prompt.ns,
            prompt_key=prompt.key,
            tag=self._tag,
        )
        if not file_path.exists():
            return None

        try:
            with file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:  # pragma: no cover
            return None

        sections_data: dict[tuple[str, ...], dict[str, str]] = {}
        raw_sections: dict[str, Any] = payload.get("sections", {})
        for path_key, section_payload in raw_sections.items():
            path = tuple(part for part in str(path_key).split("/") if part)
            if isinstance(section_payload, dict):
                section_dict = cast(dict[str, Any], section_payload)
                sections_data[path] = {
                    "expected_hash": str(section_dict.get("expected_hash", "")),
                    "body": str(section_dict.get("body", "")),
                }

        tools_data: dict[str, dict[str, Any]] = {}
        raw_tools: dict[str, Any] = payload.get("tools", {})
        for tool_name, tool_payload in raw_tools.items():
            if isinstance(tool_payload, dict):
                tool_dict = cast(dict[str, Any], tool_payload)
                tools_data[str(tool_name)] = {
                    "expected_contract_hash": str(
                        tool_dict.get("expected_contract_hash", "")
                    ),
                    "description": tool_dict.get("description"),
                    "param_descriptions": tool_dict.get("param_descriptions", {}),
                }

        return sections_data, tools_data

    @staticmethod
    def _build_section_state(
        section_desc: SectionDescriptor,
        raw_sections: dict[tuple[str, ...], dict[str, str]],
    ) -> SectionState:
        """Build a SectionState from descriptor and raw override data."""
        raw_section = raw_sections.get(section_desc.path)
        if raw_section:
            return SectionState(
                path=section_desc.path,
                number=section_desc.number,
                original_hash=section_desc.content_hash,
                current_body=raw_section["body"],
                is_overridden=True,
                is_stale=raw_section["expected_hash"] != section_desc.content_hash,
            )
        return SectionState(
            path=section_desc.path,
            number=section_desc.number,
            original_hash=section_desc.content_hash,
            current_body=None,
            is_overridden=False,
            is_stale=False,
        )

    @staticmethod
    def _build_tool_state(
        tool_desc: ToolDescriptor,
        raw_tools: dict[str, dict[str, Any]],
    ) -> ToolState:
        """Build a ToolState from descriptor and raw override data."""
        raw_tool = raw_tools.get(tool_desc.name)
        if raw_tool:
            param_descs = raw_tool.get("param_descriptions", {})
            param_dict: dict[str, str] = {}
            if isinstance(param_descs, dict):
                typed_params = cast(dict[str, Any], param_descs)
                param_dict = {str(k): str(v) for k, v in typed_params.items()}
            return ToolState(
                name=tool_desc.name,
                path=tool_desc.path,
                original_contract_hash=tool_desc.contract_hash,
                current_description=raw_tool.get("description"),
                current_param_descriptions=param_dict,
                is_overridden=True,
                is_stale=raw_tool["expected_contract_hash"] != tool_desc.contract_hash,
            )
        return ToolState(
            name=tool_desc.name,
            path=tool_desc.path,
            original_contract_hash=tool_desc.contract_hash,
            current_description=None,
            current_param_descriptions={},
            is_overridden=False,
            is_stale=False,
        )

    def get_prompt_state(self, ns: str, key: str) -> PromptOverrideState | None:
        """Get the full override state for a prompt."""
        prompt = self._prompts.get((ns, key))
        if prompt is None:
            return None

        override = self._get_local_store().resolve(
            descriptor=prompt.descriptor, tag=self._tag
        )

        raw_data = self._read_raw_override_data(prompt)
        raw_sections = raw_data[0] if raw_data else {}
        raw_tools = raw_data[1] if raw_data else {}

        return PromptOverrideState(
            prompt=prompt,
            override=override,
            is_seeded=self._is_seeded(prompt),
            sections=[
                self._build_section_state(s, raw_sections)
                for s in prompt.descriptor.sections
            ],
            tools=[
                self._build_tool_state(t, raw_tools) for t in prompt.descriptor.tools
            ],
        )

    def update_section(
        self, ns: str, key: str, path: tuple[str, ...], body: str
    ) -> SectionState:
        """Update a section override."""
        prompt = self._prompts.get((ns, key))
        if prompt is None:
            raise PromptNotFoundError(f"Prompt not found: {ns}:{key}")

        if not self._is_seeded(prompt):
            raise PromptNotSeededError(f"Prompt not seeded: {ns}:{key}")

        # Find the section descriptor
        section_desc: SectionDescriptor | None = None
        for desc in prompt.descriptor.sections:
            if desc.path == path:
                section_desc = desc
                break

        if section_desc is None:
            raise SectionNotFoundError(f"Section not found: {path}")

        store = self._get_local_store()
        existing = store.resolve(descriptor=prompt.descriptor, tag=self._tag)

        # Build updated override
        sections = dict(existing.sections) if existing else {}
        tools = dict(existing.tool_overrides) if existing else {}

        sections[path] = SectionOverride(
            expected_hash=section_desc.content_hash,
            body=body,
        )

        override = PromptOverride(
            ns=ns,
            prompt_key=key,
            tag=self._tag,
            sections=sections,
            tool_overrides=tools,
        )

        _ = store.upsert(prompt.descriptor, override)

        self._logger.info(
            "Section override updated",
            event="wink.overrides.section_updated",
            context={"ns": ns, "key": key, "path": list(path)},
        )

        return SectionState(
            path=path,
            number=section_desc.number,
            original_hash=section_desc.content_hash,
            current_body=body,
            is_overridden=True,
            is_stale=False,
        )

    def delete_section(self, ns: str, key: str, path: tuple[str, ...]) -> SectionState:
        """Remove a section override."""
        prompt = self._prompts.get((ns, key))
        if prompt is None:
            raise PromptNotFoundError(f"Prompt not found: {ns}:{key}")

        section_desc: SectionDescriptor | None = None
        for desc in prompt.descriptor.sections:
            if desc.path == path:
                section_desc = desc
                break

        if section_desc is None:
            raise SectionNotFoundError(f"Section not found: {path}")

        store = self._get_local_store()
        existing = store.resolve(descriptor=prompt.descriptor, tag=self._tag)

        if existing and path in existing.sections:
            sections = {k: v for k, v in existing.sections.items() if k != path}
            tools = dict(existing.tool_overrides)

            override = PromptOverride(
                ns=ns,
                prompt_key=key,
                tag=self._tag,
                sections=sections,
                tool_overrides=tools,
            )
            _ = store.upsert(prompt.descriptor, override)

        self._logger.info(
            "Section override deleted",
            event="wink.overrides.section_deleted",
            context={"ns": ns, "key": key, "path": list(path)},
        )

        return SectionState(
            path=path,
            number=section_desc.number,
            original_hash=section_desc.content_hash,
            current_body=None,
            is_overridden=False,
            is_stale=False,
        )

    def update_tool(
        self,
        ns: str,
        key: str,
        tool_name: str,
        description: str | None,
        param_descriptions: dict[str, str],
    ) -> ToolState:
        """Update a tool override."""
        prompt = self._prompts.get((ns, key))
        if prompt is None:
            raise PromptNotFoundError(f"Prompt not found: {ns}:{key}")

        if not self._is_seeded(prompt):
            raise PromptNotSeededError(f"Prompt not seeded: {ns}:{key}")

        tool_desc: ToolDescriptor | None = None
        for desc in prompt.descriptor.tools:
            if desc.name == tool_name:
                tool_desc = desc
                break

        if tool_desc is None:
            raise ToolNotFoundError(f"Tool not found: {tool_name}")

        store = self._get_local_store()
        existing = store.resolve(descriptor=prompt.descriptor, tag=self._tag)

        sections = dict(existing.sections) if existing else {}
        tools = dict(existing.tool_overrides) if existing else {}

        tools[tool_name] = ToolOverride(
            name=tool_name,
            expected_contract_hash=tool_desc.contract_hash,
            description=description,
            param_descriptions=param_descriptions,
        )

        override = PromptOverride(
            ns=ns,
            prompt_key=key,
            tag=self._tag,
            sections=sections,
            tool_overrides=tools,
        )

        _ = store.upsert(prompt.descriptor, override)

        self._logger.info(
            "Tool override updated",
            event="wink.overrides.tool_updated",
            context={"ns": ns, "key": key, "tool_name": tool_name},
        )

        return ToolState(
            name=tool_name,
            path=tool_desc.path,
            original_contract_hash=tool_desc.contract_hash,
            current_description=description,
            current_param_descriptions=param_descriptions,
            is_overridden=True,
            is_stale=False,
        )

    def delete_tool(self, ns: str, key: str, tool_name: str) -> ToolState:
        """Remove a tool override."""
        prompt = self._prompts.get((ns, key))
        if prompt is None:
            raise PromptNotFoundError(f"Prompt not found: {ns}:{key}")

        tool_desc: ToolDescriptor | None = None
        for desc in prompt.descriptor.tools:
            if desc.name == tool_name:
                tool_desc = desc
                break

        if tool_desc is None:
            raise ToolNotFoundError(f"Tool not found: {tool_name}")

        store = self._get_local_store()
        existing = store.resolve(descriptor=prompt.descriptor, tag=self._tag)

        if existing and tool_name in existing.tool_overrides:
            sections = dict(existing.sections)
            tools = {k: v for k, v in existing.tool_overrides.items() if k != tool_name}

            override = PromptOverride(
                ns=ns,
                prompt_key=key,
                tag=self._tag,
                sections=sections,
                tool_overrides=tools,
            )
            _ = store.upsert(prompt.descriptor, override)

        self._logger.info(
            "Tool override deleted",
            event="wink.overrides.tool_deleted",
            context={"ns": ns, "key": key, "tool_name": tool_name},
        )

        return ToolState(
            name=tool_name,
            path=tool_desc.path,
            original_contract_hash=tool_desc.contract_hash,
            current_description=None,
            current_param_descriptions={},
            is_overridden=False,
            is_stale=False,
        )

    def delete_prompt_overrides(self, ns: str, key: str) -> None:
        """Delete the entire override file for a prompt."""
        prompt = self._prompts.get((ns, key))
        if prompt is None:
            raise PromptNotFoundError(f"Prompt not found: {ns}:{key}")

        store = self._get_local_store()
        store.delete(ns=ns, prompt_key=key, tag=self._tag)

        self._logger.info(
            "Prompt overrides deleted",
            event="wink.overrides.prompt_deleted",
            context={"ns": ns, "key": key},
        )

    def reload(self) -> None:
        """Reload the snapshot file and refresh extracted prompts."""
        _ = self._snapshot_store.reload()
        self._extract_prompts()

        self._logger.info(
            "Snapshot reloaded",
            event="wink.overrides.reload",
            context={"snapshot_path": str(self._snapshot_path)},
        )


# --- API Handlers ---


class _OverridesAppHandlers:
    def __init__(
        self, *, store: OverridesStore, log: StructuredLogger, static_dir: Path
    ) -> None:
        super().__init__()
        self._store = store
        self._logger = log
        self._static_dir = static_dir

    def index(self) -> str:
        index_path = self._static_dir / "overrides" / "index.html"
        return index_path.read_text()

    def list_prompts(self) -> list[Mapping[str, JSONValue]]:
        result: list[Mapping[str, JSONValue]] = []
        for prompt in self._store.prompts:
            state = self._store.get_prompt_state(prompt.ns, prompt.key)
            if state is None:  # pragma: no cover
                continue

            has_overrides = any(s.is_overridden for s in state.sections) or any(
                t.is_overridden for t in state.tools
            )
            stale_count = sum(1 for s in state.sections if s.is_stale) + sum(
                1 for t in state.tools if t.is_stale
            )

            result.append(
                {
                    "ns": prompt.ns,
                    "key": prompt.key,
                    "name": prompt.name,
                    "section_count": len(state.sections),
                    "tool_count": len(state.tools),
                    "is_seeded": state.is_seeded,
                    "has_overrides": has_overrides,
                    "stale_count": stale_count,
                    "created_at": prompt.created_at.isoformat(),
                }
            )
        return result

    def get_prompt(self, encoded_ns: str, prompt_key: str) -> Mapping[str, JSONValue]:
        ns = unquote(encoded_ns)
        state = self._store.get_prompt_state(ns, prompt_key)
        if state is None:
            raise HTTPException(
                status_code=404, detail=f"Prompt not found: {ns}:{prompt_key}"
            )

        prompt = state.prompt
        rendered_html = _markdown.render(prompt.rendered_text)

        sections: list[Mapping[str, JSONValue]] = [
            {
                "path": list(s.path),
                "number": s.number,
                "original_hash": s.original_hash,
                "current_body": s.current_body,
                "is_overridden": s.is_overridden,
                "is_stale": s.is_stale,
            }
            for s in state.sections
        ]

        tools: list[Mapping[str, JSONValue]] = [
            {
                "name": t.name,
                "path": list(t.path),
                "original_contract_hash": t.original_contract_hash,
                "current_description": t.current_description,
                "current_param_descriptions": t.current_param_descriptions,
                "is_overridden": t.is_overridden,
                "is_stale": t.is_stale,
            }
            for t in state.tools
        ]

        return {
            "ns": prompt.ns,
            "key": prompt.key,
            "name": prompt.name,
            "rendered_prompt": {
                "text": prompt.rendered_text,
                "html": rendered_html,
            },
            "created_at": prompt.created_at.isoformat(),
            "tag": self._store.tag,
            "is_seeded": state.is_seeded,
            "sections": sections,
            "tools": tools,
        }

    def update_section(
        self,
        encoded_ns: str,
        prompt_key: str,
        encoded_path: str,
        payload: dict[str, Any],
    ) -> Mapping[str, JSONValue]:
        ns = unquote(encoded_ns)
        path = tuple(unquote(encoded_path).split("/"))
        body = payload.get("body")

        if not isinstance(body, str):
            raise HTTPException(status_code=400, detail="body is required")

        try:
            section = self._store.update_section(ns, prompt_key, path, body)
        except PromptNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except PromptNotSeededError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except SectionNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        return {
            "success": True,
            "section": {
                "path": list(section.path),
                "current_body": section.current_body,
                "is_overridden": section.is_overridden,
                "is_stale": section.is_stale,
            },
        }

    def delete_section(
        self, encoded_ns: str, prompt_key: str, encoded_path: str
    ) -> Mapping[str, JSONValue]:
        ns = unquote(encoded_ns)
        path = tuple(unquote(encoded_path).split("/"))

        try:
            section = self._store.delete_section(ns, prompt_key, path)
        except PromptNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except SectionNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        return {
            "success": True,
            "section": {
                "path": list(section.path),
                "current_body": section.current_body,
                "is_overridden": section.is_overridden,
                "is_stale": section.is_stale,
            },
        }

    def update_tool(
        self, encoded_ns: str, prompt_key: str, tool_name: str, payload: dict[str, Any]
    ) -> Mapping[str, JSONValue]:
        ns = unquote(encoded_ns)
        description = payload.get("description")
        raw_param_descriptions = payload.get("param_descriptions", {})

        if not isinstance(raw_param_descriptions, dict):
            raise HTTPException(
                status_code=400, detail="param_descriptions must be an object"
            )

        typed_raw: dict[object, object] = cast(
            dict[object, object], raw_param_descriptions
        )
        param_descriptions: dict[str, str] = {
            str(k): str(v) for k, v in typed_raw.items()
        }

        try:
            tool = self._store.update_tool(
                ns, prompt_key, tool_name, description, param_descriptions
            )
        except PromptNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except PromptNotSeededError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ToolNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        return {
            "success": True,
            "tool": {
                "name": tool.name,
                "current_description": tool.current_description,
                "current_param_descriptions": tool.current_param_descriptions,
                "is_overridden": tool.is_overridden,
                "is_stale": tool.is_stale,
            },
        }

    def delete_tool(
        self, encoded_ns: str, prompt_key: str, tool_name: str
    ) -> Mapping[str, JSONValue]:
        ns = unquote(encoded_ns)

        try:
            tool = self._store.delete_tool(ns, prompt_key, tool_name)
        except PromptNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ToolNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        return {
            "success": True,
            "tool": {
                "name": tool.name,
                "current_description": tool.current_description,
                "current_param_descriptions": tool.current_param_descriptions,
                "is_overridden": tool.is_overridden,
                "is_stale": tool.is_stale,
            },
        }

    def delete_prompt(
        self, encoded_ns: str, prompt_key: str
    ) -> Mapping[str, JSONValue]:
        ns = unquote(encoded_ns)

        try:
            self._store.delete_prompt_overrides(ns, prompt_key)
        except PromptNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        return {"success": True}

    def get_config(self) -> Mapping[str, JSONValue]:
        return {
            "tag": self._store.tag,
            "store_root": str(self._store.store_root)
            if self._store.store_root
            else None,
            "snapshot_path": str(self._store.snapshot_path),
        }

    def reload(self) -> Mapping[str, JSONValue]:
        self._store.reload()
        return {"success": True, "prompt_count": len(self._store.prompts)}


# --- App Builder ---


def build_overrides_app(store: OverridesStore, log: StructuredLogger) -> FastAPI:
    """Construct the FastAPI application for editing prompt overrides."""
    from importlib.resources import files

    static_dir = Path(str(files(__package__).joinpath("static")))
    handlers = _OverridesAppHandlers(store=store, log=log, static_dir=static_dir)

    app = FastAPI(title="wink overrides editor")
    app.state.overrides_store = store
    app.state.logger = log
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Routes
    _ = app.get("/", response_class=HTMLResponse)(handlers.index)
    _ = app.get("/api/prompts")(handlers.list_prompts)
    _ = app.get("/api/prompts/{encoded_ns}/{prompt_key}")(handlers.get_prompt)
    _ = app.put("/api/prompts/{encoded_ns}/{prompt_key}/sections/{encoded_path:path}")(
        handlers.update_section
    )
    _ = app.delete(
        "/api/prompts/{encoded_ns}/{prompt_key}/sections/{encoded_path:path}"
    )(handlers.delete_section)
    _ = app.put("/api/prompts/{encoded_ns}/{prompt_key}/tools/{tool_name}")(
        handlers.update_tool
    )
    _ = app.delete("/api/prompts/{encoded_ns}/{prompt_key}/tools/{tool_name}")(
        handlers.delete_tool
    )
    _ = app.delete("/api/prompts/{encoded_ns}/{prompt_key}")(handlers.delete_prompt)
    _ = app.get("/api/config")(handlers.get_config)
    _ = app.post("/api/reload")(handlers.reload)

    return app


# --- Server Runner ---


def run_overrides_server(
    app: FastAPI,
    *,
    host: str,
    port: int,
    open_browser: bool,
    log: StructuredLogger,
) -> int:
    """Run the uvicorn server for the overrides editor app."""
    url = f"http://{host}:{port}/"

    if open_browser:
        threading.Timer(0.2, _open_browser, args=(url, log)).start()

    log.info(
        "Starting wink overrides server",
        event="wink.overrides.start",
        context={
            "url": url,
            "snapshot_path": str(app.state.overrides_store.snapshot_path),
        },
    )

    try:
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_config=None,
        )
        server = uvicorn.Server(config)
        server.run()
    except Exception as error:  # pragma: no cover - exercised in wink tests
        log.exception(
            "Failed to start wink overrides server",
            event="wink.overrides.error",
            context={"url": url, "error": repr(error)},
        )
        return 3
    return 0


def _open_browser(url: str, log: StructuredLogger) -> None:
    try:
        _ = webbrowser.open(url)
    except Exception as error:  # pragma: no cover - best effort
        log.warning(
            "Unable to open browser",
            event="wink.overrides.browser",
            context={"url": url, "error": repr(error)},
        )


__all__ = [
    "ExtractedPrompt",
    "HashMismatchError",
    "OverridesEditorError",
    "OverridesStore",
    "PromptNotFoundError",
    "PromptNotSeededError",
    "PromptOverrideState",
    "SectionNotFoundError",
    "SectionState",
    "ToolNotFoundError",
    "ToolState",
    "build_overrides_app",
    "load_snapshot",
    "run_overrides_server",
]

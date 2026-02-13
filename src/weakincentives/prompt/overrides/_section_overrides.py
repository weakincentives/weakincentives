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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from ...runtime.logging import StructuredLogger, get_logger
from ...types import JSONValue
from .versioning import (
    HexDigest,
    PromptDescriptor,
    PromptLike,
    PromptOverridesError,
    SectionDescriptor,
    SectionOverride,
    ensure_hex_digest,
)

_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "prompt_overrides"}
)


def section_descriptor_index(
    descriptor: PromptDescriptor,
) -> dict[tuple[str, ...], SectionDescriptor]:
    return {section.path: section for section in descriptor.sections}


def format_section_path(path: tuple[str, ...]) -> str:
    return "/".join(path)


@dataclass(slots=True)
class SectionValidationConfig:
    strict: bool
    path_display: str
    body_error_message: str


def normalize_section_override(
    *,
    path: tuple[str, ...],
    descriptor_section: SectionDescriptor | None,
    expected_hash: JSONValue,
    body: JSONValue,
    summary: str | None,
    visibility: Literal["full", "summary"] | None,
    config: SectionValidationConfig,
) -> SectionOverride | None:
    if descriptor_section is None:
        if config.strict:
            raise PromptOverridesError(
                f"Unknown section path for override: {config.path_display}"
            )
        _LOGGER.debug(
            "Skipping unknown override section path.",
            event="prompt_override_unknown_section",
            context={"path": config.path_display},
        )
        return None
    expected_digest = ensure_hex_digest(
        cast(HexDigest | str, expected_hash),
        field_name="Section expected_hash",
    )
    if expected_digest != descriptor_section.content_hash:
        if config.strict:
            msg = (
                f"Hash mismatch for section {config.path_display}: expected "
                f"{descriptor_section.content_hash}, got {expected_digest}."
            )
            raise PromptOverridesError(msg)
        _LOGGER.debug(
            "Skipping stale section override.",
            event="prompt_override_stale_section",
            context={
                "path": config.path_display,
                "expected_hash": str(descriptor_section.content_hash),
                "found_hash": str(expected_digest),
            },
        )
        return None
    if not isinstance(body, str):
        raise PromptOverridesError(config.body_error_message)
    return SectionOverride(
        path=path,
        expected_hash=expected_digest,
        body=body,
        summary=summary,
        visibility=visibility,
    )


def _parse_section_summary(
    section_payload: Mapping[str, JSONValue],
) -> str | None:
    """Parse and validate the summary field from a section override payload."""
    summary = section_payload.get("summary")
    if summary is not None and not isinstance(summary, str):
        raise PromptOverridesError("Section summary must be a string.")
    return summary


def _parse_section_visibility(
    section_payload: Mapping[str, JSONValue],
) -> Literal["full", "summary"] | None:
    """Parse and validate the visibility field from a section override payload."""
    visibility = section_payload.get("visibility")
    if visibility is None:
        return None
    if visibility not in {"full", "summary"}:
        raise PromptOverridesError(
            f"Section visibility must be 'full' or 'summary', got {visibility!r}."
        )
    return visibility  # type: ignore[return-value]


def _load_section_override_entry(
    *,
    path_key_raw: object,
    section_payload_raw: JSONValue,
    descriptor_index: Mapping[tuple[str, ...], SectionDescriptor],
) -> tuple[tuple[str, ...], SectionOverride] | None:
    if not isinstance(path_key_raw, str):
        raise PromptOverridesError("Section keys must be strings.")
    path_key = path_key_raw
    path = tuple(part for part in path_key.split("/") if part)
    if not isinstance(section_payload_raw, Mapping):
        raise PromptOverridesError("Section payload must be an object.")
    section_payload = cast(Mapping[str, JSONValue], section_payload_raw)
    expected_hash = section_payload.get("expected_hash")
    body = section_payload.get("body")
    summary = _parse_section_summary(section_payload)
    visibility = _parse_section_visibility(section_payload)
    config = SectionValidationConfig(
        strict=False,
        path_display=path_key,
        body_error_message="Section body must be a string.",
    )
    section_override = normalize_section_override(
        path=path,
        descriptor_section=descriptor_index.get(path),
        expected_hash=expected_hash,
        body=body,
        summary=summary,
        visibility=visibility,
        config=config,
    )
    if section_override is None:
        return None
    return path, section_override


def load_sections(
    payload: JSONValue | None,
    descriptor: PromptDescriptor,
) -> dict[tuple[str, ...], SectionOverride]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise PromptOverridesError("Sections payload must be a mapping.")
    if not payload:
        return {}
    mapping_payload = cast(Mapping[object, JSONValue], payload)
    mapping_entries = cast(Iterable[tuple[object, JSONValue]], mapping_payload.items())
    descriptor_index = section_descriptor_index(descriptor)
    overrides: dict[tuple[str, ...], SectionOverride] = {}
    for path_key_raw, section_payload_raw in mapping_entries:
        normalized_section = _load_section_override_entry(
            path_key_raw=path_key_raw,
            section_payload_raw=section_payload_raw,
            descriptor_index=descriptor_index,
        )
        if normalized_section is not None:
            path, section_override = normalized_section
            overrides[path] = section_override
    return overrides


def validate_sections_for_write(
    sections: Mapping[tuple[str, ...], SectionOverride],
    descriptor: PromptDescriptor,
) -> dict[tuple[str, ...], SectionOverride]:
    descriptor_index = section_descriptor_index(descriptor)
    validated: dict[tuple[str, ...], SectionOverride] = {}
    for path, section_override in sections.items():
        path_display = "/".join(path)
        section_config = SectionValidationConfig(
            strict=True,
            path_display=path_display,
            body_error_message=(
                f"Section override body must be a string for {path_display}."
            ),
        )
        normalized_section = normalize_section_override(
            path=path,
            descriptor_section=descriptor_index.get(path),
            expected_hash=section_override.expected_hash,
            body=section_override.body,
            summary=section_override.summary,
            visibility=section_override.visibility,
            config=section_config,
        )
        validated[path] = cast(SectionOverride, normalized_section)
    return validated


def serialize_sections(
    sections: Mapping[tuple[str, ...], SectionOverride],
) -> dict[str, dict[str, object]]:
    serialized: dict[str, dict[str, object]] = {}
    for path, section_override in sections.items():
        key = "/".join(path)
        entry: dict[str, object] = {
            "path": list(path),
            "expected_hash": str(section_override.expected_hash),
            "body": section_override.body,
        }
        if section_override.summary is not None:
            entry["summary"] = section_override.summary
        if section_override.visibility is not None:
            entry["visibility"] = section_override.visibility
        serialized[key] = entry
    return serialized


def seed_sections(
    prompt: PromptLike,
    descriptor: PromptDescriptor,
) -> dict[tuple[str, ...], SectionOverride]:
    section_lookup = {node.path: node.section for node in prompt.sections}
    seeded: dict[tuple[str, ...], SectionOverride] = {}
    for section in descriptor.sections:
        section_obj = section_lookup.get(section.path)
        if section_obj is None:
            raise PromptOverridesError(
                f"Prompt missing section for descriptor path {'/'.join(section.path)}."
            )
        template = section_obj.original_body_template()
        if template is None:
            raise PromptOverridesError(
                "Cannot seed override for section without template."
            )
        seeded[section.path] = SectionOverride(
            path=section.path,
            expected_hash=section.content_hash,
            body=template,
        )
    return seeded

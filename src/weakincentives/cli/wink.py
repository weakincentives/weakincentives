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

"""Command line interface for managing prompt overrides."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess  # nosec B404
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..prompt.local_prompt_overrides_store import LocalPromptOverridesStore
from ..prompt.versioning import (
    PromptDescriptor,
    PromptOverride,
    PromptOverridesError,
    SectionDescriptor,
    SectionOverride,
    ToolDescriptor,
    ToolOverride,
)

_DESCRIPTORS_RELATIVE_PATH = Path(".weakincentives") / "prompts" / "descriptors"
_OVERRIDES_RELATIVE_PATH = Path(".weakincentives") / "prompts" / "overrides"
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class CliError(RuntimeError):
    """Raised when user-facing CLI validation fails."""


@dataclass(slots=True)
class CliOptions:
    root: Path | None
    output_format: str
    editor: str | None
    assume_yes: bool


@dataclass(slots=True)
class DescriptorRecord:
    descriptor: PromptDescriptor
    default_override_payload: dict[str, Any] | None


def main(argv: Sequence[str] | None = None) -> int:
    """Program entrypoint."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    options = CliOptions(
        root=args.root,
        output_format=args.output_format,
        editor=args.editor,
        assume_yes=args.yes,
    )

    command_handlers: dict[str, Callable[[CliOptions, argparse.Namespace], int]] = {
        "list": _command_list,
        "show": _command_show,
        "edit": _command_edit,
        "delete": _command_delete,
    }

    handler = command_handlers.get(args.command or "")
    if handler is None:  # pragma: no cover - argparse enforces valid commands
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    try:
        return handler(options, args)
    except CliError as error:
        print(str(error), file=sys.stderr)
        return 1
    except PromptOverridesError as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Aborted by user.", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wink")
    _ = parser.add_argument(
        "--root",
        type=Path,
        help="Override automatic project root discovery.",
    )
    _ = parser.add_argument(
        "--format",
        dest="output_format",
        choices=["table", "json"],
        default="table",
        help="Output format for supported commands.",
    )
    _ = parser.add_argument(
        "--editor",
        help="Editor command for interactive edits.",
    )
    _ = parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts on destructive actions.",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    list_parser = subparsers.add_parser("list", help="List available overrides.")
    _ = list_parser.add_argument("--ns", help="Filter by namespace.")
    _ = list_parser.add_argument("--prompt", help="Filter by prompt key.")
    _ = list_parser.add_argument("--tag", help="Filter by tag.")

    show_parser = subparsers.add_parser(
        "show", help="Show the contents of an override file."
    )
    _add_identifier_arguments(show_parser)

    edit_parser = subparsers.add_parser(
        "edit", help="Create or update an override using an editor."
    )
    _add_identifier_arguments(edit_parser)

    delete_parser = subparsers.add_parser("delete", help="Delete an override file.")
    _add_identifier_arguments(delete_parser)

    return parser


def _add_identifier_arguments(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--ns", required=True, help="Prompt namespace.")
    _ = parser.add_argument("--prompt", required=True, help="Prompt key.")
    _ = parser.add_argument(
        "--tag",
        default="latest",
        help="Override tag (defaults to 'latest').",
    )


def _command_list(options: CliOptions, args: argparse.Namespace) -> int:
    root_path = _resolve_root_path(options.root)
    overrides_dir = _overrides_dir(root_path)

    rows: list[dict[str, str]] = []
    if overrides_dir.exists():
        for file_path in sorted(overrides_dir.rglob("*.json")):
            relative = file_path.relative_to(overrides_dir)
            parts = relative.parts
            if len(parts) < 2:
                continue
            tag = Path(parts[-1]).stem
            prompt_key = parts[-2]
            ns_segments = parts[:-2]
            ns = "/".join(ns_segments)
            if args.ns and ns != args.ns:
                continue
            if args.prompt and prompt_key != args.prompt:
                continue
            if args.tag and tag != args.tag:
                continue
            rows.append(
                {
                    "ns": ns,
                    "prompt": prompt_key,
                    "tag": tag,
                    "path": str(file_path.relative_to(root_path)),
                }
            )

    if options.output_format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        _print_table(rows, columns=("ns", "prompt", "tag", "path"))
    return 0


def _command_show(options: CliOptions, args: argparse.Namespace) -> int:
    root_path = _resolve_root_path(options.root)
    override_path = _override_file_path(root_path, args.ns, args.prompt, args.tag)
    if not override_path.exists():
        raise CliError(f"No override found for {args.ns}/{args.prompt}:{args.tag}.")
    payload = _load_json_mapping(override_path)

    if options.output_format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{args.ns}/{args.prompt}:{args.tag}")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _command_edit(options: CliOptions, args: argparse.Namespace) -> int:
    root_path = _resolve_root_path(options.root)
    descriptor_record = _load_descriptor(root_path, args.ns, args.prompt)
    descriptor = descriptor_record.descriptor
    tag = args.tag
    override_path = _override_file_path(root_path, descriptor.ns, descriptor.key, tag)

    store = LocalPromptOverridesStore(root_path=root_path)

    if override_path.exists():
        payload_raw = _load_json_mapping(override_path)
        existing_override = _parse_override_payload(
            payload_raw,
            descriptor=descriptor,
        )
        payload = _override_to_payload(existing_override)
    else:
        default_payload = descriptor_record.default_override_payload
        if default_payload is None:
            raise CliError("No existing override or default template available.")
        seeded_override = _parse_override_payload(
            {
                "ns": descriptor.ns,
                "prompt_key": descriptor.key,
                "tag": tag,
                "sections": default_payload.get("sections", {}),
                "tools": default_payload.get("tools", {}),
            },
            descriptor=descriptor,
        )
        payload = _override_to_payload(seeded_override)

    editor_command = _resolve_editor_command(options.editor)

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        _ = handle.write("\n")

    try:
        _ = subprocess.run(  # nosec B603 B607
            [*editor_command, str(temp_path)],
            check=True,
        )
    except FileNotFoundError as error:
        temp_path.unlink(missing_ok=True)
        raise CliError(f"Failed to launch editor: {error}") from error
    except subprocess.CalledProcessError as error:
        temp_path.unlink(missing_ok=True)
        raise CliError(f"Editor exited with code {error.returncode}.") from error

    try:
        loaded_payload = _load_json_mapping(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    override = _parse_override_payload(
        loaded_payload,
        descriptor=descriptor,
    )

    stored = store.upsert(descriptor, override)
    stored_path = _override_file_path(
        root_path,
        descriptor.ns,
        descriptor.key,
        stored.tag,
    )
    stored_path_relative = stored_path.relative_to(root_path)
    print(stored_path_relative)
    return 0


def _command_delete(options: CliOptions, args: argparse.Namespace) -> int:
    root_path = _resolve_root_path(options.root)
    override_path = _override_file_path(root_path, args.ns, args.prompt, args.tag)
    store = LocalPromptOverridesStore(root_path=root_path)

    existed_before_delete = override_path.exists()
    if not options.assume_yes:
        confirmation = input(
            f"Delete override {args.ns}/{args.prompt}:{args.tag}? [y/N] "
        ).strip()
        if confirmation.lower() not in {"y", "yes"}:
            print("Aborted.", file=sys.stderr)
            return 1

    store.delete(ns=args.ns, prompt_key=args.prompt, tag=args.tag)
    if existed_before_delete:
        print(f"Deleted {override_path.relative_to(root_path)}")
    else:
        print("Nothing to delete.")
    return 0


def _resolve_editor_command(editor_option: str | None) -> list[str]:
    candidates = [
        editor_option,
        os.environ.get("WINK_EDITOR"),
        os.environ.get("VISUAL"),
        os.environ.get("EDITOR"),
    ]
    for candidate in candidates:
        if candidate and candidate.strip():
            return shlex.split(candidate)
    return ["vi"]


def _resolve_root_path(root_option: Path | None) -> Path:
    if root_option is not None:
        return root_option.resolve()

    try:
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        result = None
    if result and result.stdout.strip():
        return Path(result.stdout.strip()).resolve()

    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise CliError("Failed to locate repository root. Provide --root explicitly.")


def _overrides_dir(root: Path) -> Path:
    return root / _OVERRIDES_RELATIVE_PATH


def _descriptor_file_path(root: Path, ns: str, prompt_key: str) -> Path:
    descriptor_dir = root / _DESCRIPTORS_RELATIVE_PATH
    for segment in _split_namespace(ns):
        descriptor_dir /= segment
    prompt_component = _validate_identifier(prompt_key, "prompt key")
    return descriptor_dir / f"{prompt_component}.json"


def _override_file_path(root: Path, ns: str, prompt_key: str, tag: str) -> Path:
    overrides_dir = _overrides_dir(root)
    ns_segments = _split_namespace(ns)
    prompt_component = _validate_identifier(prompt_key, "prompt key")
    tag_component = _validate_identifier(tag, "tag")
    return overrides_dir.joinpath(
        *ns_segments, prompt_component, f"{tag_component}.json"
    )


def _validate_identifier(value: str, label: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise CliError(f"{label.capitalize()} must be a non-empty string.")
    if not _IDENTIFIER_PATTERN.fullmatch(stripped):
        raise CliError(
            f"{label.capitalize()} must match pattern ^[a-z0-9][a-z0-9._-]{{0,63}}$."
        )
    return stripped


def _split_namespace(ns: str) -> tuple[str, ...]:
    stripped = ns.strip()
    if not stripped:
        raise CliError("Namespace must be a non-empty string.")
    segments = [segment.strip() for segment in stripped.split("/") if segment.strip()]
    if not segments:
        raise CliError("Namespace must contain at least one segment.")
    return tuple(
        _validate_identifier(segment, "namespace segment") for segment in segments
    )


def _load_json_mapping(file_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CliError(f"Failed to parse JSON from {file_path}: {error}") from error
    if not isinstance(data, dict):
        raise CliError(f"Expected JSON object in {file_path}.")
    data_dict = cast(dict[object, Any], data)
    return _dict_with_string_keys(data_dict, f"JSON object in {file_path}")


def _dict_with_string_keys(data: Mapping[Any, Any], label: str) -> dict[str, Any]:
    typed: dict[str, Any] = {}
    for key_obj, value in data.items():
        if not isinstance(key_obj, str):
            raise CliError(f"{label} keys must be strings.")
        typed[key_obj] = value
    return typed


def _load_descriptor(root: Path, ns: str, prompt_key: str) -> DescriptorRecord:
    descriptor_path = _descriptor_file_path(root, ns, prompt_key)
    if not descriptor_path.is_file():
        raise CliError(f"Descriptor not found for {ns}/{prompt_key}: {descriptor_path}")
    payload = _load_json_mapping(descriptor_path)
    descriptor_data = payload.get("descriptor")
    if not isinstance(descriptor_data, dict):
        raise CliError("Descriptor payload missing 'descriptor' object.")
    descriptor_payload = _dict_with_string_keys(
        cast(Mapping[Any, Any], descriptor_data), "descriptor"
    )
    descriptor = _parse_descriptor(descriptor_payload)
    if descriptor.ns != ns or descriptor.key != prompt_key:
        raise CliError("Descriptor metadata does not match requested prompt.")
    default_payload_raw = payload.get("default_override")
    if default_payload_raw is None:
        default_payload: dict[str, Any] | None = None
    elif isinstance(default_payload_raw, dict):
        default_payload = _dict_with_string_keys(
            cast(Mapping[Any, Any], default_payload_raw), "default_override"
        )
    else:
        raise CliError("default_override must be a mapping when provided.")
    return DescriptorRecord(
        descriptor=descriptor, default_override_payload=default_payload
    )


def _parse_descriptor(payload: Mapping[str, Any]) -> PromptDescriptor:
    ns_obj = payload.get("ns")
    key_obj = payload.get("key")
    if not isinstance(ns_obj, str) or not ns_obj:
        raise CliError("Descriptor ns must be a non-empty string.")
    if not isinstance(key_obj, str) or not key_obj:
        raise CliError("Descriptor key must be a non-empty string.")

    raw_sections = payload.get("sections", [])
    if not isinstance(raw_sections, list):
        raise CliError("Descriptor sections must be a list.")
    raw_sections_list = cast(list[object], raw_sections)
    sections: list[SectionDescriptor] = []
    for entry_obj in raw_sections_list:
        if not isinstance(entry_obj, dict):
            raise CliError("Section descriptor entries must be objects.")
        entry = _dict_with_string_keys(
            cast(Mapping[Any, Any], entry_obj), "section descriptor"
        )
        path_value_obj = entry.get("path")
        content_hash_obj = entry.get("content_hash")
        if not isinstance(path_value_obj, list):
            raise CliError("Section descriptor path must be a list of strings.")
        path_components: list[str] = []
        path_list = cast(list[object], path_value_obj)
        for part_obj in path_list:
            if not isinstance(part_obj, str) or not part_obj:
                raise CliError(
                    "Section descriptor path must contain non-empty strings."
                )
            path_components.append(part_obj)
        if not isinstance(content_hash_obj, str) or not content_hash_obj:
            raise CliError("Section descriptor content_hash must be a string.")
        sections.append(SectionDescriptor(tuple(path_components), content_hash_obj))

    raw_tools = payload.get("tools", [])
    if not isinstance(raw_tools, list):
        raise CliError("Descriptor tools must be a list.")
    raw_tools_list = cast(list[object], raw_tools)
    tools: list[ToolDescriptor] = []
    for entry_obj in raw_tools_list:
        if not isinstance(entry_obj, dict):
            raise CliError("Tool descriptor entries must be objects.")
        entry = _dict_with_string_keys(
            cast(Mapping[Any, Any], entry_obj), "tool descriptor"
        )
        path_value_obj = entry.get("path")
        name_obj = entry.get("name")
        contract_hash_obj = entry.get("contract_hash")
        if not isinstance(path_value_obj, list):
            raise CliError("Tool descriptor path must be a list of strings.")
        path_components: list[str] = []
        path_list = cast(list[object], path_value_obj)
        for part_obj in path_list:
            if not isinstance(part_obj, str) or not part_obj:
                raise CliError("Tool descriptor path must contain non-empty strings.")
            path_components.append(part_obj)
        if not isinstance(name_obj, str) or not name_obj:
            raise CliError("Tool descriptor name must be a string.")
        if not isinstance(contract_hash_obj, str) or not contract_hash_obj:
            raise CliError("Tool descriptor contract_hash must be a string.")
        tools.append(
            ToolDescriptor(tuple(path_components), name_obj, contract_hash_obj)
        )

    return PromptDescriptor(ns=ns_obj, key=key_obj, sections=sections, tools=tools)


def _parse_override_payload(
    payload: Mapping[str, Any],
    *,
    descriptor: PromptDescriptor,
) -> PromptOverride:
    ns_obj = payload.get("ns")
    prompt_key_obj = payload.get("prompt_key")
    if ns_obj != descriptor.ns or prompt_key_obj != descriptor.key:
        raise CliError("Override metadata does not match descriptor.")

    tag_obj = payload.get("tag", "latest")
    if not isinstance(tag_obj, str):
        raise CliError("Override tag must be a string.")
    tag_value = _validate_identifier(tag_obj, "tag")

    sections_field = payload.get("sections")
    if sections_field is None:
        sections_field = {}
    if not isinstance(sections_field, dict):
        raise CliError("Sections must be an object mapping paths to overrides.")
    sections_source = cast(dict[object, Any], sections_field)
    sections: dict[tuple[str, ...], SectionOverride] = {}
    for path_key_obj, section_obj in sections_source.items():
        if not isinstance(path_key_obj, str):
            raise CliError("Section keys must be strings.")
        if not isinstance(section_obj, dict):
            raise CliError("Section overrides must be objects.")
        section_mapping = _dict_with_string_keys(
            cast(Mapping[Any, Any], section_obj), "section override"
        )
        expected_hash_obj = section_mapping.get("expected_hash")
        body_obj = section_mapping.get("body")
        if not isinstance(expected_hash_obj, str):
            raise CliError(
                f"Section expected_hash must be a string for {path_key_obj}."
            )
        if not isinstance(body_obj, str):
            raise CliError(f"Section body must be a string for {path_key_obj}.")
        path = tuple(part for part in path_key_obj.split("/") if part)
        sections[path] = SectionOverride(
            expected_hash=expected_hash_obj,
            body=body_obj,
        )

    tools_field = payload.get("tools")
    if tools_field is None:
        tools_field = {}
    if not isinstance(tools_field, dict):
        raise CliError("Tools must be an object mapping names to overrides.")
    tools_source = cast(dict[object, Any], tools_field)
    tools: dict[str, ToolOverride] = {}
    for tool_name_obj, tool_obj in tools_source.items():
        if not isinstance(tool_name_obj, str):
            raise CliError("Tool names must be strings.")
        if not isinstance(tool_obj, dict):
            raise CliError("Tool overrides must be objects.")
        tool_mapping = _dict_with_string_keys(
            cast(Mapping[Any, Any], tool_obj), "tool override"
        )
        expected_hash_obj = tool_mapping.get("expected_contract_hash")
        description_obj = tool_mapping.get("description")
        param_payload_obj = tool_mapping.get("param_descriptions")
        if not isinstance(expected_hash_obj, str):
            raise CliError(
                f"Tool expected_contract_hash must be a string for {tool_name_obj}."
            )
        if description_obj is not None and not isinstance(description_obj, str):
            raise CliError(f"Tool description must be a string for {tool_name_obj}.")
        if param_payload_obj is None:
            param_payload_obj = {}
        if not isinstance(param_payload_obj, dict):
            raise CliError(
                f"Tool param_descriptions must be a mapping for {tool_name_obj}."
            )
        param_payload_mapping = cast(dict[object, Any], param_payload_obj)
        param_mapping: dict[str, str] = {}
        for param_key_obj, param_value_obj in param_payload_mapping.items():
            if not isinstance(param_key_obj, str) or not isinstance(
                param_value_obj, str
            ):
                raise CliError("Tool param_descriptions must map strings to strings.")
            param_mapping[param_key_obj] = param_value_obj
        param_descriptions = dict(param_mapping)
        tools[tool_name_obj] = ToolOverride(
            name=tool_name_obj,
            expected_contract_hash=expected_hash_obj,
            description=description_obj,
            param_descriptions=param_descriptions,
        )

    return PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag=tag_value,
        sections=sections,
        tool_overrides=tools,
    )


def _override_to_payload(override: PromptOverride) -> dict[str, Any]:
    return {
        "version": 1,
        "ns": override.ns,
        "prompt_key": override.prompt_key,
        "tag": override.tag,
        "sections": _serialize_sections(override.sections),
        "tools": _serialize_tools(override.tool_overrides),
    }


def _serialize_sections(
    sections: Mapping[tuple[str, ...], SectionOverride],
) -> dict[str, dict[str, str]]:
    serialized: dict[str, dict[str, str]] = {}
    for path, section_override in sections.items():
        key = "/".join(path)
        serialized[key] = {
            "expected_hash": section_override.expected_hash,
            "body": section_override.body,
        }
    return serialized


def _serialize_tools(tools: Mapping[str, ToolOverride]) -> dict[str, dict[str, Any]]:
    serialized: dict[str, dict[str, Any]] = {}
    for name, tool_override in tools.items():
        serialized[name] = {
            "expected_contract_hash": tool_override.expected_contract_hash,
            "description": tool_override.description,
            "param_descriptions": dict(tool_override.param_descriptions),
        }
    return serialized


def _print_table(rows: Sequence[Mapping[str, str]], columns: tuple[str, ...]) -> None:
    if not rows:
        header = "  ".join(column for column in columns)
        print(header)
        return
    widths = [len(column) for column in columns]
    for row in rows:
        for index, column in enumerate(columns):
            widths[index] = max(widths[index], len(row.get(column, "")))
    header = "  ".join(column for column in columns)
    print(header)
    for row in rows:
        line = "  ".join(
            row.get(column, "").ljust(widths[index])
            for index, column in enumerate(columns)
        )
        print(line)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    sys.exit(main())

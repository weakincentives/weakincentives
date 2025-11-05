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

import argparse
import json
import os
import shlex
import subprocess  # nosec B404
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any, Literal, cast

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    tomllib = cast(Any, None)

from ..prompt import (
    Prompt,
    PromptOverride,
    PromptOverridesError,
    SectionOverride,
    ToolOverride,
)
from ..prompt.local_prompt_overrides_store import LocalPromptOverridesStore
from ..prompt.registry import get_prompt
from ..prompt.versioning import PromptDescriptor, PromptLike

Format = Literal["table", "json"]


@dataclass(slots=True)
class CLISettings:
    """Resolved global CLI configuration."""

    root: Path
    output_format: Format
    editor_command: list[str] | None
    assume_yes: bool
    quiet: bool
    confirm_deletes: bool


class CLIError(Exception):
    """Raised when wink encounters a recoverable CLI error."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    store = LocalPromptOverridesStore(root_path=args.root)
    try:
        root = store._resolve_root()  # pyright: ignore[reportPrivateUsage]
    except PromptOverridesError as error:
        print(str(error), file=sys.stderr)
        return 1

    config = _load_wink_config(root)
    settings = _build_settings(args, config, root)

    command = args.command
    try:
        if command == "list":
            return _handle_list(store, settings, args)
        if command == "show":
            return _handle_show(store, settings, args)
        if command == "edit":
            return _handle_edit(store, settings, args)
        if command == "delete":
            return _handle_delete(store, settings, args)
        if command == "diff":
            return _handle_diff(store, settings, args)
    except CLIError as error:
        print(str(error), file=sys.stderr)
        return 1
    except PromptOverridesError as error:
        print(str(error), file=sys.stderr)
        return 1

    try:
        parser.error(f"Unknown command: {command}")
    except SystemExit:
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wink")
    _ = parser.add_argument("--root", type=Path, default=None)
    _ = parser.add_argument(
        "--format",
        choices=("table", "json"),
        default=None,
        dest="output_format",
    )
    _ = parser.add_argument("--editor", dest="editor_command", default=None)
    _ = parser.add_argument("--yes", action="store_true", dest="assume_yes")
    _ = parser.add_argument("-q", "--quiet", action="store_true", dest="quiet")

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    _add_identifier_arguments(list_parser, optional=True)

    show_parser = subparsers.add_parser("show")
    _add_identifier_arguments(show_parser)
    _ = show_parser.add_argument(
        "--summary-only",
        action="store_true",
        dest="summary_only",
    )

    edit_parser = subparsers.add_parser("edit")
    _add_identifier_arguments(edit_parser)
    _ = edit_parser.add_argument(
        "--create-only",
        action="store_true",
        dest="create_only",
    )

    delete_parser = subparsers.add_parser("delete")
    _add_identifier_arguments(delete_parser)

    diff_parser = subparsers.add_parser("diff")
    _add_identifier_arguments(diff_parser)

    return parser


def _add_identifier_arguments(
    parser: argparse.ArgumentParser, *, optional: bool = False
) -> None:
    required = not optional
    _ = parser.add_argument("--ns", required=required)
    _ = parser.add_argument("--prompt", dest="prompt_key", required=required)
    _ = parser.add_argument("--tag", default="latest")


def _build_settings(
    args: argparse.Namespace, config: Mapping[str, Any], root: Path
) -> CLISettings:
    output_format = args.output_format or config.get("default_format", "table")
    if output_format not in {"table", "json"}:
        raise CLIError(f"Invalid format configured: {output_format}")

    editor_value = args.editor_command or config.get("editor")
    editor_command = _resolve_editor_command(editor_value)

    confirm_deletes = config.get("confirm_deletes", True)
    return CLISettings(
        root=root,
        output_format=output_format,
        editor_command=editor_command,
        assume_yes=args.assume_yes,
        quiet=args.quiet,
        confirm_deletes=bool(confirm_deletes),
    )


def _load_wink_config(root: Path) -> dict[str, Any]:
    config_path = root / ".weakincentives" / "wink.toml"
    if not config_path.exists() or tomllib is None:
        return {}
    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}

    ui = data.get("ui", {})
    paths = data.get("paths", {})

    config: dict[str, Any] = {}
    default_format = ui.get("default_format")
    if default_format in {"table", "json"}:
        config["default_format"] = default_format
    confirm_deletes = ui.get("confirm_deletes")
    if isinstance(confirm_deletes, bool):
        config["confirm_deletes"] = confirm_deletes
    editor = paths.get("editor")
    if isinstance(editor, str) and editor.strip():
        config["editor"] = editor
    return config


def _resolve_editor_command(value: str | None) -> list[str] | None:
    if value and value.strip():
        return shlex.split(value.strip())

    env = os.environ
    for variable in ("WINK_EDITOR", "VISUAL", "EDITOR"):
        env_value = env.get(variable)
        if env_value:
            return shlex.split(env_value)
    return None


def _handle_list(
    store: LocalPromptOverridesStore,
    settings: CLISettings,
    args: argparse.Namespace,
) -> int:
    overrides_dir = store._overrides_dir()  # pyright: ignore[reportPrivateUsage]
    if not overrides_dir.exists():
        if settings.output_format == "json":
            print("[]")
        elif not settings.quiet:
            print("No overrides found.")
        return 0

    records = list(_collect_override_metadata(overrides_dir))
    filtered = [
        record
        for record in records
        if _matches_filters(record, args.ns, args.prompt_key, args.tag)
    ]

    if settings.output_format == "json":
        payload = [record.to_json(settings.root) for record in filtered]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if not filtered:
        if not settings.quiet:
            print("No overrides found.")
        return 0

    headers = ["ns", "prompt", "tag", "sections", "tools", "path"]
    rows = [record.to_row(settings.root) for record in filtered]
    print(_format_table(headers, rows))
    return 0


def _matches_filters(
    record: OverrideRecord,
    ns: str | None,
    prompt_key: str | None,
    tag: str | None,
) -> bool:
    if ns and record.ns != ns:
        return False
    if prompt_key and record.prompt_key != prompt_key:
        return False
    return not (tag and record.tag != tag)


@dataclass(slots=True)
class OverrideRecord:
    ns: str
    prompt_key: str
    tag: str
    sections: dict[tuple[str, ...], dict[str, str]]
    tools: dict[str, dict[str, Any]]
    path: Path

    def to_json(self, root: Path) -> dict[str, Any]:
        rel_path = str(self.path.relative_to(root))
        sections = [
            {
                "path": list(path),
                "expected_hash": payload.get("expected_hash"),
            }
            for path, payload in self.sections.items()
        ]
        tools = [
            {
                "name": name,
                "expected_contract_hash": payload.get("expected_contract_hash"),
            }
            for name, payload in self.tools.items()
        ]
        return {
            "ns": self.ns,
            "prompt_key": self.prompt_key,
            "tag": self.tag,
            "path": rel_path,
            "sections": sections,
            "tools": tools,
        }

    def to_row(self, root: Path) -> list[str]:
        section_count = str(len(self.sections))
        tool_count = str(len(self.tools))
        rel_path = str(self.path.relative_to(root))
        return [self.ns, self.prompt_key, self.tag, section_count, tool_count, rel_path]


def _collect_override_metadata(directory: Path) -> Iterable[OverrideRecord]:
    for path in sorted(directory.rglob("*.json")):
        payload = _load_override_payload(path)
        ns = payload.get("ns")
        prompt_key = payload.get("prompt_key")
        tag = payload.get("tag")
        if (
            not isinstance(ns, str)
            or not isinstance(prompt_key, str)
            or not isinstance(tag, str)
        ):
            continue
        sections_payload = payload.get("sections")
        tools_payload = payload.get("tools")
        sections = _normalize_sections_payload(sections_payload)
        tools = _normalize_tools_payload(tools_payload)
        yield OverrideRecord(ns, prompt_key, tag, sections, tools, path)


def _load_override_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_sections_payload(
    payload: Mapping[object, object] | None,
) -> dict[tuple[str, ...], dict[str, str]]:
    if not isinstance(payload, Mapping):
        return {}
    sections: dict[tuple[str, ...], dict[str, str]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            continue
        path = tuple(part for part in key.split("/") if part)
        mapping_value = cast(Mapping[str, object], value)
        sections[path] = {
            "expected_hash": str(mapping_value.get("expected_hash", "")),
            "body": str(mapping_value.get("body", "")),
        }
    return sections


def _normalize_tools_payload(
    payload: Mapping[object, object] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return {}
    tools: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            continue
        mapping_value = cast(Mapping[str, object], value)
        tools[key] = {
            "expected_contract_hash": str(
                mapping_value.get("expected_contract_hash", "")
            ),
            "description": mapping_value.get("description"),
        }
    return tools


def _handle_show(
    store: LocalPromptOverridesStore,
    settings: CLISettings,
    args: argparse.Namespace,
) -> int:
    descriptor, override = _resolve_override_for_prompt(
        store, args.ns, args.prompt_key, args.tag
    )
    if override is None:
        raise CLIError("Prompt override not found.")

    if settings.output_format == "json":
        json_path = store._override_file_path(  # pyright: ignore[reportPrivateUsage]
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=args.tag,
        )
        print(
            json.dumps(
                _override_to_json(override, json_path, settings.root),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    header = f"{override.ns}/{override.prompt_key}:{override.tag}"
    path = store._override_file_path(  # pyright: ignore[reportPrivateUsage]
        ns=descriptor.ns, prompt_key=descriptor.key, tag=args.tag
    )
    print(f"{header}\n{path}")

    if override.sections:
        section_rows: list[list[str]] = []
        for path_key, section in override.sections.items():
            first_line = (
                section.body.strip().splitlines()[0] if section.body.strip() else ""
            )
            section_rows.append(
                [
                    "/".join(path_key),
                    section.expected_hash,
                    first_line,
                ]
            )
        print("\nSections:")
        print(_format_table(["path", "hash", "first line"], section_rows))

    if override.tool_overrides:
        tool_rows: list[list[str]] = []
        for name, tool in override.tool_overrides.items():
            description = tool.description or ""
            first_line = description.strip().splitlines()[0] if description else ""
            tool_rows.append([name, tool.expected_contract_hash, first_line])
        print("\nTools:")
        print(_format_table(["name", "hash", "description"], tool_rows))

    if not args.summary_only:
        print("\n---\n")
        for path_key, section in override.sections.items():
            print(f"[{'/'.join(path_key)}]")
            print(section.body)
            print()
        for name, tool in override.tool_overrides.items():
            print(f"[tool:{name}]")
            if tool.description:
                print(tool.description)
            if tool.param_descriptions:
                print(json.dumps(tool.param_descriptions, indent=2, sort_keys=True))
            print()
    return 0


def _override_to_json(
    override: PromptOverride, path: Path, root: Path
) -> dict[str, Any]:
    return {
        "ns": override.ns,
        "prompt_key": override.prompt_key,
        "tag": override.tag,
        "path": str(path.relative_to(root)),
        "sections": [
            {
                "path": list(path_key),
                "expected_hash": section.expected_hash,
                "body": section.body,
            }
            for path_key, section in override.sections.items()
        ],
        "tools": [
            {
                "name": name,
                "expected_contract_hash": tool.expected_contract_hash,
                "description": tool.description,
                "param_descriptions": dict(tool.param_descriptions),
            }
            for name, tool in override.tool_overrides.items()
        ],
    }


def _handle_edit(
    store: LocalPromptOverridesStore,
    settings: CLISettings,
    args: argparse.Namespace,
) -> int:
    descriptor, override = _resolve_override_for_prompt(
        store, args.ns, args.prompt_key, args.tag
    )
    prompt = _require_prompt(args.ns, args.prompt_key)

    if override is None:
        override = store.seed_if_necessary(prompt, tag=args.tag)

    if args.create_only:
        if not settings.quiet:
            print("Override seeded.")
        return 0

    if settings.editor_command is None:
        raise CLIError("No editor configured. Set $EDITOR or pass --editor.")

    temp_path = _write_override_tempfile(override)
    try:
        original = temp_path.read_text(encoding="utf-8")
        editor_command = [*settings.editor_command, str(temp_path)]
        try:
            completed = subprocess.run(editor_command)  # nosec B603
        except FileNotFoundError as error:
            raise CLIError(f"Failed to launch editor: {error}") from error

        if completed.returncode != 0:
            raise CLIError("Editor aborted. No changes were saved.")

        edited = temp_path.read_text(encoding="utf-8")
        if edited == original:
            if not settings.quiet:
                print("No changes.")
            return 0

        payload = json.loads(edited)
        updated_override = _prompt_override_from_payload(payload)
        stored = store.upsert(descriptor, updated_override)
        if not settings.quiet:
            file_path = store._override_file_path(  # pyright: ignore[reportPrivateUsage]
                ns=descriptor.ns, prompt_key=descriptor.key, tag=stored.tag
            )
            print(f"Saved override to {file_path}")
        return 0
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _write_override_tempfile(override: PromptOverride) -> Path:
    prefix = f"wink-{override.ns}-{override.prompt_key}-{override.tag}-"
    with tempfile.NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", prefix=prefix, suffix=".json"
    ) as handle:
        json.dump(_override_to_json_blob(override), handle, indent=2, sort_keys=True)
        _ = handle.write("\n")
        return Path(handle.name)


def _override_to_json_blob(override: PromptOverride) -> dict[str, Any]:
    return {
        "ns": override.ns,
        "prompt_key": override.prompt_key,
        "tag": override.tag,
        "sections": {
            "/".join(path): {
                "expected_hash": section.expected_hash,
                "body": section.body,
            }
            for path, section in override.sections.items()
        },
        "tools": {
            name: {
                "expected_contract_hash": tool.expected_contract_hash,
                "description": tool.description,
                "param_descriptions": dict(tool.param_descriptions),
            }
            for name, tool in override.tool_overrides.items()
        },
    }


def _prompt_override_from_payload(payload: Mapping[str, Any]) -> PromptOverride:
    ns = payload.get("ns")
    prompt_key = payload.get("prompt_key")
    tag = payload.get("tag")
    if (
        not isinstance(ns, str)
        or not isinstance(prompt_key, str)
        or not isinstance(tag, str)
    ):
        raise CLIError("Override JSON must include ns, prompt_key, and tag.")

    sections_payload_raw = payload.get("sections")
    if sections_payload_raw is None:
        sections_payload_raw = {}
    if not isinstance(sections_payload_raw, Mapping):
        raise CLIError("Sections must be a JSON object.")
    sections_payload = cast(Mapping[object, object], sections_payload_raw)
    sections: dict[tuple[str, ...], SectionOverride] = {}
    for path_key, section_payload in sections_payload.items():
        if not isinstance(path_key, str) or not isinstance(section_payload, Mapping):
            raise CLIError("Section entries must map path strings to objects.")
        section_mapping = cast(Mapping[str, object], section_payload)
        expected_hash = section_mapping.get("expected_hash")
        body = section_mapping.get("body")
        if not isinstance(expected_hash, str) or not isinstance(body, str):
            raise CLIError("Section overrides require expected_hash and body strings.")
        sections[tuple(part for part in path_key.split("/") if part)] = SectionOverride(
            expected_hash=expected_hash,
            body=body,
        )

    tools_payload_raw = payload.get("tools")
    if tools_payload_raw is None:
        tools_payload_raw = {}
    if not isinstance(tools_payload_raw, Mapping):
        raise CLIError("Tools must be a JSON object.")
    tools_payload = cast(Mapping[object, object], tools_payload_raw)
    tools: dict[str, ToolOverride] = {}
    for name, tool_payload in tools_payload.items():
        if not isinstance(name, str) or not isinstance(tool_payload, Mapping):
            raise CLIError("Tool overrides must map tool names to objects.")
        tool_mapping = cast(Mapping[str, object], tool_payload)
        expected_hash = tool_mapping.get("expected_contract_hash")
        description = tool_mapping.get("description")
        param_payload = tool_mapping.get("param_descriptions")
        if not isinstance(expected_hash, str):
            raise CLIError("Tool overrides require expected_contract_hash.")
        if description is not None and not isinstance(description, str):
            raise CLIError("Tool description must be a string when provided.")
        if param_payload is None:
            param_descriptions: dict[str, str] = {}
        else:
            if not isinstance(param_payload, Mapping):
                raise CLIError(
                    "Tool param_descriptions must be a mapping when provided."
                )
            param_mapping = cast(Mapping[object, object], param_payload)
            param_descriptions = {}
            for key, value in param_mapping.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise CLIError(
                        "Tool param_descriptions must map strings to strings."
                    )
                param_descriptions[key] = value
        tools[name] = ToolOverride(
            name=name,
            expected_contract_hash=expected_hash,
            description=description,
            param_descriptions=param_descriptions,
        )

    return PromptOverride(
        ns=ns,
        prompt_key=prompt_key,
        tag=tag,
        sections=sections,
        tool_overrides=tools,
    )


def _handle_delete(
    store: LocalPromptOverridesStore,
    settings: CLISettings,
    args: argparse.Namespace,
) -> int:
    path = store._override_file_path(  # pyright: ignore[reportPrivateUsage]
        ns=args.ns, prompt_key=args.prompt_key, tag=args.tag
    )
    exists = path.exists()
    if not exists and not settings.quiet:
        print("Override not found; nothing to delete.")

    should_confirm = settings.confirm_deletes and not settings.assume_yes and exists
    if should_confirm:
        response = input(f"Delete {path}? [y/N] ")
        if response.lower() not in {"y", "yes"}:
            if not settings.quiet:
                print("Aborted.")
            return 0

    store.delete(ns=args.ns, prompt_key=args.prompt_key, tag=args.tag)
    if not settings.quiet:
        print(f"Deleted {path}")
    return 0


def _handle_diff(
    store: LocalPromptOverridesStore,
    settings: CLISettings,
    args: argparse.Namespace,
) -> int:
    descriptor, override = _resolve_override_for_prompt(
        store, args.ns, args.prompt_key, args.tag
    )
    if override is None:
        return 1

    prompt = _require_prompt(args.ns, args.prompt_key)
    baseline = _seed_preview(store, prompt, descriptor, args.tag)

    baseline_blob = json.dumps(
        _override_to_json_blob(baseline), indent=2, sort_keys=True
    )
    override_blob = json.dumps(
        _override_to_json_blob(override), indent=2, sort_keys=True
    )
    if baseline_blob == override_blob:
        return 1

    if settings.editor_command is not None:
        baseline_path = _write_temp_blob(baseline_blob, suffix="-baseline.json")
        override_path = _write_temp_blob(override_blob, suffix="-override.json")
        try:
            command = [*settings.editor_command, str(baseline_path), str(override_path)]
            result = subprocess.run(command)  # nosec B603
            return result.returncode
        finally:
            baseline_path.unlink(missing_ok=True)
            override_path.unlink(missing_ok=True)

    diff = unified_diff(
        baseline_blob.splitlines(),
        override_blob.splitlines(),
        fromfile="baseline",
        tofile="override",
        lineterm="",
    )
    for line in diff:
        print(line)
    return 0


def _write_temp_blob(content: str, *, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", prefix="wink-diff-", suffix=suffix
    ) as handle:
        _ = handle.write(content)
        _ = handle.write("\n")
        return Path(handle.name)


def _seed_preview(
    store: LocalPromptOverridesStore,
    prompt: PromptLike,
    descriptor: PromptDescriptor,
    tag: str,
) -> PromptOverride:
    sections = store._seed_sections(  # pyright: ignore[reportPrivateUsage]
        prompt, descriptor
    )
    tools = store._seed_tools(prompt, descriptor)  # pyright: ignore[reportPrivateUsage]
    return PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag=tag,
        sections=sections,
        tool_overrides=tools,
    )


def _resolve_override_for_prompt(
    store: LocalPromptOverridesStore,
    ns: str,
    prompt_key: str,
    tag: str,
) -> tuple[PromptDescriptor, PromptOverride | None]:
    prompt = get_prompt(ns, prompt_key)
    if prompt is None:
        raise CLIError("Prompt not found. Verify namespace and prompt key.")
    descriptor = PromptDescriptor.from_prompt(prompt)
    override = store.resolve(descriptor, tag=tag)
    return descriptor, override


def _require_prompt(ns: str, prompt_key: str) -> PromptLike:
    prompt = get_prompt(ns, prompt_key)
    if prompt is None or not isinstance(prompt, Prompt):
        raise CLIError("Prompt not found. Verify namespace and prompt key.")
    return prompt


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    columns = len(headers)
    widths = [len(header) for header in headers]
    normalized_rows = [tuple(str(cell) for cell in row) for row in rows]
    for row in normalized_rows:
        for index in range(columns):
            widths[index] = max(widths[index], len(row[index]))
    header_line = " ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    divider = " ".join("-" * widths[index] for index in range(columns))
    body_lines = [
        " ".join(row[index].ljust(widths[index]) for index in range(columns))
        for row in normalized_rows
    ]
    return "\n".join([header_line, divider, *body_lines])


__all__ = ["CLIError", "CLISettings", "main"]

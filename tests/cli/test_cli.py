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
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

import weakincentives.cli as wink_cli
from weakincentives.cli import CLIError, main
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    Section,
    Tool,
    clear_registry,
    register_prompt,
)
from weakincentives.prompt.local_prompt_overrides_store import LocalPromptOverridesStore
from weakincentives.prompt.versioning import PromptDescriptor, PromptOverridesError


@pytest.fixture(autouse=True)
def _clear_registry() -> Iterator[None]:
    clear_registry()
    yield
    clear_registry()


@dataclass
class _GreetingParams:
    subject: str


@dataclass
class _ToolParams:
    subject: str


@dataclass
class _ToolResultPayload:
    message: str


def _register_prompt(*, with_tool: bool = False) -> Prompt:
    sections: list[Section[_GreetingParams]] = [
        MarkdownSection[_GreetingParams](
            title="Greeting",
            template="Hello ${subject}.",
            key="greeting",
        )
    ]
    if with_tool:
        tool = Tool[_ToolParams, _ToolResultPayload](
            name="greeter",
            description="Return a friendly greeting.",
            handler=None,
        )
        sections.append(
            MarkdownSection[_GreetingParams](
                title="Tool",
                template="Body",
                key="tool",
                tools=[tool],
            )
        )
    prompt = Prompt(
        ns="demo",
        key="greeting" if not with_tool else "greeting-with-tool",
        sections=sections,
    )
    register_prompt(prompt)
    return prompt


def _override_path(
    root: Path, *, prompt_key: str = "greeting", tag: str = "latest"
) -> Path:
    return (
        root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "demo"
        / prompt_key
        / f"{tag}.json"
    )


def _write_editor_script(
    tmp_path: Path,
    *,
    body_text: str | None = None,
    exit_code: int = 0,
    mutate: bool = True,
    invalid_json: bool = False,
) -> Path:
    script_path = tmp_path / "editor.py"
    replacement = body_text or "Hello ${subject}!"
    lines = [
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        "path = Path(sys.argv[-1])",
        'text = path.read_text(encoding="utf-8")',
        f"if {mutate!r}:",
        f"    if {invalid_json!r}:",
        '        path.write_text("{", encoding="utf-8")',
        "    else:",
        "        payload = json.loads(text)",
        f'        payload["sections"]["greeting"]["body"] = {replacement!r}',
        '        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")',
        f"sys.exit({exit_code})",
    ]
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def _run_cli(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    exit_code = main(args)
    captured = capsys.readouterr()
    return exit_code, captured.out, captured.err


def test_list_outputs_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "--format",
            "json",
            "list",
        ],
        capsys,
    )

    assert code == 0
    payload = json.loads(out)
    assert payload[0]["ns"] == "demo"


def test_list_no_overrides_prints_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _register_prompt()
    _, out, _ = _run_cli(["--root", str(tmp_path), "list"], capsys)
    assert "No overrides" in out


def test_list_filters(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "list",
            "--ns",
            "other",
        ],
        capsys,
    )
    assert code == 0
    assert "No overrides" in out


def test_list_json_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "--format",
            "json",
            "list",
        ],
        capsys,
    )
    assert code == 0
    assert out.strip() == "[]"


def test_list_table_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)

    code, out, _ = _run_cli(["--root", str(tmp_path), "list"], capsys)
    assert code == 0
    assert "ns" in out and "demo" in out


def test_list_prompt_filter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "list",
            "--prompt",
            "other",
            "--tag",
            "missing",
        ],
        capsys,
    )
    assert code == 0
    assert "No overrides" in out


def test_matches_filters_branches() -> None:
    record = wink_cli.OverrideRecord(
        ns="demo",
        prompt_key="greeting",
        tag="latest",
        sections={},
        tools={},
        path=Path("dummy"),
    )
    assert not wink_cli._matches_filters(record, "other", None, None)
    assert not wink_cli._matches_filters(record, None, "other", None)
    assert not wink_cli._matches_filters(record, None, None, "other")
    assert wink_cli._matches_filters(record, None, None, None)


def test_show_emits_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "--format",
            "json",
            "show",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )

    assert code == 0
    payload = json.loads(out)
    assert payload["sections"][0]["body"].startswith("Hello")


def test_show_table_includes_sections_and_tools(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = _register_prompt(with_tool=True)
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)
    path = _override_path(tmp_path, prompt_key="greeting-with-tool")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tools"]["greeter"]["param_descriptions"] = {"subject": "Name"}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "show",
            "--ns",
            "demo",
            "--prompt",
            "greeting-with-tool",
            "--tag",
            "latest",
            "--summary-only",
        ],
        capsys,
    )

    assert code == 0
    assert "Sections:" in out
    assert "Tools:" in out


def test_show_prints_tool_details(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = _register_prompt(with_tool=True)
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)
    path = _override_path(tmp_path, prompt_key="greeting-with-tool")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tools"]["greeter"]["param_descriptions"] = {"subject": "Name"}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "show",
            "--ns",
            "demo",
            "--prompt",
            "greeting-with-tool",
            "--tag",
            "latest",
        ],
        capsys,
    )

    assert code == 0
    assert "[tool:greeter]" in out


def test_show_prints_full_body(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "show",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )

    assert code == 0
    assert "[greeting]" in out
    assert "Hello ${subject}." in out


def test_show_missing_override_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _register_prompt()
    code, _, err = _run_cli(
        [
            "--root",
            str(tmp_path),
            "show",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "missing",
        ],
        capsys,
    )

    assert code == 1
    assert "Prompt override not found" in err


def test_edit_create_only_seeds_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "edit",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "experiment",
            "--create-only",
        ],
        capsys,
    )

    assert code == 0
    assert "Override seeded" in out
    assert _override_path(tmp_path, tag="experiment").exists()


def test_edit_requires_editor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)

    code, _, err = _run_cli(
        [
            "--root",
            str(tmp_path),
            "edit",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )

    assert code == 1
    assert "No editor configured" in err


def test_edit_no_changes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)
    script = _write_editor_script(tmp_path, mutate=False)

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "--editor",
            f"{sys.executable} {script}",
            "edit",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )

    assert code == 0
    assert "No changes" in out


def test_edit_applies_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)
    script = _write_editor_script(tmp_path, body_text="Cheers ${subject}!")

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "--editor",
            f"{sys.executable} {script}",
            "edit",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )

    assert code == 0
    assert "Saved override" in out
    payload = json.loads(_override_path(tmp_path).read_text(encoding="utf-8"))
    assert payload["sections"]["greeting"]["body"] == "Cheers ${subject}!"


def test_edit_handles_missing_editor_binary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)

    code, _, err = _run_cli(
        [
            "--root",
            str(tmp_path),
            "--editor",
            "nonexistent-editor-binary",  # guaranteed to fail resolution
            "edit",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )

    assert code == 1
    assert "Failed to launch editor" in err


def test_edit_aborts_on_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)
    script = _write_editor_script(tmp_path, exit_code=2)

    code, _, err = _run_cli(
        [
            "--root",
            str(tmp_path),
            "--editor",
            f"{sys.executable} {script}",
            "edit",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )

    assert code == 1
    assert "Editor aborted" in err


def test_prompt_override_payload_validation() -> None:
    with pytest.raises(CLIError):
        wink_cli._prompt_override_from_payload({})
    with pytest.raises(CLIError):
        wink_cli._prompt_override_from_payload(
            {"ns": "demo", "prompt_key": "greeting", "tag": "latest", "sections": []}
        )
    with pytest.raises(CLIError):
        wink_cli._prompt_override_from_payload(
            {
                "ns": "demo",
                "prompt_key": "greeting",
                "tag": "latest",
                "sections": {"greeting": {"expected_hash": "hash"}},
            }
        )
    with pytest.raises(CLIError):
        wink_cli._prompt_override_from_payload(
            {
                "ns": "demo",
                "prompt_key": "greeting",
                "tag": "latest",
                "sections": {"greeting": {"expected_hash": "hash", "body": "value"}},
                "tools": {"greeter": {"expected_contract_hash": 1}},
            }
        )
    with pytest.raises(CLIError):
        wink_cli._prompt_override_from_payload(
            {
                "ns": "demo",
                "prompt_key": "greeting",
                "tag": "latest",
                "sections": {1: {"expected_hash": "hash", "body": "value"}},
            }
        )
    with pytest.raises(CLIError):
        wink_cli._prompt_override_from_payload(
            {
                "ns": "demo",
                "prompt_key": "greeting",
                "tag": "latest",
                "sections": {"greeting": []},
            }
        )
    with pytest.raises(CLIError):
        wink_cli._prompt_override_from_payload(
            {
                "ns": "demo",
                "prompt_key": "greeting",
                "tag": "latest",
                "sections": {"greeting": {"expected_hash": "hash", "body": "value"}},
                "tools": [],
            }
        )
    with pytest.raises(CLIError):
        wink_cli._prompt_override_from_payload(
            {
                "ns": "demo",
                "prompt_key": "greeting",
                "tag": "latest",
                "sections": {"greeting": {"expected_hash": "hash", "body": "value"}},
                "tools": {1: {"expected_contract_hash": "hash"}},
            }
        )
    with pytest.raises(CLIError):
        wink_cli._prompt_override_from_payload(
            {
                "ns": "demo",
                "prompt_key": "greeting",
                "tag": "latest",
                "sections": {"greeting": {"expected_hash": "hash", "body": "value"}},
                "tools": {
                    "greeter": {"expected_contract_hash": "hash", "description": 1}
                },
            }
        )
    with pytest.raises(CLIError):
        wink_cli._prompt_override_from_payload(
            {
                "ns": "demo",
                "prompt_key": "greeting",
                "tag": "latest",
                "sections": {"greeting": {"expected_hash": "hash", "body": "value"}},
                "tools": {
                    "greeter": {
                        "expected_contract_hash": "hash",
                        "param_descriptions": [],
                    }
                },
            }
        )
    with pytest.raises(CLIError):
        wink_cli._prompt_override_from_payload(
            {
                "ns": "demo",
                "prompt_key": "greeting",
                "tag": "latest",
                "sections": {"greeting": {"expected_hash": "hash", "body": "value"}},
                "tools": {
                    "greeter": {
                        "expected_contract_hash": "hash",
                        "param_descriptions": {"x": 1},
                    }
                },
            }
        )


def test_prompt_override_payload_success() -> None:
    payload = {
        "ns": "demo",
        "prompt_key": "greeting",
        "tag": "latest",
        "sections": {"greeting": {"expected_hash": "hash", "body": "value"}},
        "tools": {
            "greeter": {
                "expected_contract_hash": "hash",
                "description": "desc",
                "param_descriptions": {"key": "value"},
            }
        },
    }
    override = wink_cli._prompt_override_from_payload(payload)
    assert override.sections["greeting",].body == "value"
    assert override.tool_overrides["greeter"].param_descriptions == {"key": "value"}


def test_prompt_override_payload_defaults() -> None:
    payload = {
        "ns": "demo",
        "prompt_key": "greeting",
        "tag": "latest",
        "tools": {
            "greeter": {
                "expected_contract_hash": "hash",
                "description": "desc",
            }
        },
    }
    override = wink_cli._prompt_override_from_payload(payload)
    assert override.sections == {}
    assert override.tool_overrides["greeter"].param_descriptions == {}


def test_prompt_override_payload_empty_collections() -> None:
    payload = {
        "ns": "demo",
        "prompt_key": "greeting",
        "tag": "latest",
    }
    override = wink_cli._prompt_override_from_payload(payload)
    assert override.sections == {}
    assert override.tool_overrides == {}


def test_normalize_payload_filters_invalid_entries() -> None:
    sections = wink_cli._normalize_sections_payload(
        cast(
            Mapping[object, object],
            {
                "valid": {"expected_hash": "hash", "body": "body"},
                1: {"expected_hash": "hash", "body": "ignored"},
                "bad": [],
            },
        )
    )
    assert list(sections) == [("valid",)]

    tools = wink_cli._normalize_tools_payload(
        cast(
            Mapping[object, object],
            {
                "greeter": {
                    "expected_contract_hash": "hash",
                    "description": "d",
                },
                1: {"expected_contract_hash": "hash"},
                "other": [],
            },
        )
    )
    assert set(tools) == {"greeter"}


def test_normalize_payload_handles_non_mapping() -> None:
    assert wink_cli._normalize_sections_payload(None) == {}
    assert wink_cli._normalize_tools_payload(None) == {}


def test_delete_removes_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt, tag="experiment")

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "--yes",
            "delete",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "experiment",
        ],
        capsys,
    )

    assert code == 0
    assert "Deleted" in out
    assert not _override_path(tmp_path, tag="experiment").exists()


def test_delete_handles_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _register_prompt()
    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "delete",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "missing",
        ],
        capsys,
    )
    assert code == 0
    assert "nothing to delete" in out


def test_delete_confirmation_abort(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)

    monkeypatch.setattr("builtins.input", lambda _: "n")
    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "delete",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )
    assert code == 0
    assert "Aborted" in out


def test_diff_prints_unified_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    override = store.seed_if_necessary(prompt)
    override.sections["greeting",].body = "Hello ${subject}!"
    store.upsert(PromptDescriptor.from_prompt(prompt), override)

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "diff",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )
    assert code == 0
    assert "Hello ${subject}!" in out


def test_diff_missing_override_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _register_prompt()
    code, _, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "diff",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "missing",
        ],
        capsys,
    )
    assert code == 1


def test_diff_no_changes_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)

    code, _, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "diff",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )
    assert code == 1


def test_diff_uses_editor_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    override = store.seed_if_necessary(prompt)
    override.sections["greeting",].body = "Updated"
    store.upsert(PromptDescriptor.from_prompt(prompt), override)
    script = _write_editor_script(tmp_path, mutate=False)

    code, _, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "--editor",
            f"{sys.executable} {script}",
            "diff",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )
    assert code == 0


def test_unknown_prompt_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = _run_cli(
        [
            "--root",
            str(tmp_path),
            "show",
            "--ns",
            "demo",
            "--prompt",
            "missing",
            "--tag",
            "latest",
        ],
        capsys,
    )
    assert code == 1
    assert "Prompt not found" in err


def test_require_prompt_error() -> None:
    with pytest.raises(CLIError):
        wink_cli._require_prompt("demo", "missing")


def test_build_settings_invalid_format(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_format="invalid",
        editor_command=None,
        assume_yes=False,
        quiet=False,
    )
    with pytest.raises(CLIError):
        wink_cli._build_settings(args, {}, tmp_path)


def test_load_wink_config_invalid_toml(tmp_path: Path) -> None:
    config_dir = tmp_path / ".weakincentives"
    config_dir.mkdir(parents=True)
    (config_dir / "wink.toml").write_text("=invalid", encoding="utf-8")
    assert wink_cli._load_wink_config(tmp_path) == {}


def test_config_overrides_defaults(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)
    config_dir = tmp_path / ".weakincentives"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "wink.toml").write_text(
        """
[ui]
default_format = "json"
confirm_deletes = false

[paths]
editor = "{cmd}"
        """.format(cmd=f"{sys.executable} {_write_editor_script(tmp_path)}"),
        encoding="utf-8",
    )

    code, out, _ = _run_cli(["--root", str(tmp_path), "list"], capsys)
    assert code == 0
    assert out.strip().startswith("[")

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "delete",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )
    assert code == 0
    assert "Deleted" in out


def test_editor_resolves_from_environment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)
    script = _write_editor_script(tmp_path)
    monkeypatch.setenv("WINK_EDITOR", f"{sys.executable} {script}")

    code, out, _ = _run_cli(
        [
            "--root",
            str(tmp_path),
            "edit",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )
    assert code == 0
    assert "Saved override" in out


def test_parser_unknown_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, str] = {}

    class _FakeParser:
        def parse_args(self, args: Sequence[str] | None = None) -> argparse.Namespace:
            return argparse.Namespace(
                root=tmp_path,
                output_format="table",
                editor_command=None,
                assume_yes=False,
                quiet=False,
                command="mystery",
            )

        def error(self, message: str) -> None:
            captured["message"] = message
            raise SystemExit(2)

    monkeypatch.setattr(wink_cli, "_build_parser", lambda: _FakeParser())
    monkeypatch.setattr(
        LocalPromptOverridesStore, "_resolve_root", lambda self: tmp_path
    )

    result = main([])
    assert result == 1
    assert "Unknown command" in captured["message"]


def test_list_handles_malformed_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / ".weakincentives" / "prompts" / "overrides"
    directory.mkdir(parents=True)
    malformed = directory / "bad.json"
    malformed.write_text("{}", encoding="utf-8")

    code, out, _ = _run_cli(["--root", str(tmp_path), "list"], capsys)
    assert code == 0
    assert "No overrides" in out


def test_main_handles_root_resolution_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(self: LocalPromptOverridesStore) -> Path:  # type: ignore[override]
        raise PromptOverridesError("boom")

    monkeypatch.setattr(LocalPromptOverridesStore, "_resolve_root", _boom)
    code, _, err = _run_cli(["--root", str(tmp_path), "list"], capsys)
    assert code == 1
    assert "boom" in err


def test_main_handles_prompt_overrides_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = _register_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)
    store.seed_if_necessary(prompt)

    def _fail(
        self: LocalPromptOverridesStore,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> wink_cli.PromptOverride | None:
        raise PromptOverridesError("fail")

    monkeypatch.setattr(LocalPromptOverridesStore, "resolve", _fail)
    code, _, err = _run_cli(
        [
            "--root",
            str(tmp_path),
            "show",
            "--ns",
            "demo",
            "--prompt",
            "greeting",
            "--tag",
            "latest",
        ],
        capsys,
    )
    assert code == 1
    assert "fail" in err

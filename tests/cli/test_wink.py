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
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest

from weakincentives.cli import wink

_DESCRIPTOR_CONTENT = {
    "descriptor": {
        "ns": "demo",
        "key": "welcome",
        "sections": [
            {"path": ["system"], "content_hash": "hash-system"},
            {"path": ["user"], "content_hash": "hash-user"},
        ],
        "tools": [
            {
                "path": ["system"],
                "name": "search",
                "contract_hash": "hash-search",
            }
        ],
    },
    "default_override": {
        "tag": "latest",
        "sections": {
            "system": {
                "expected_hash": "hash-system",
                "body": "System body",
            },
            "user": {
                "expected_hash": "hash-user",
                "body": "User body",
            },
        },
        "tools": {
            "search": {
                "expected_contract_hash": "hash-search",
                "description": "Search for information.",
                "param_descriptions": {"query": "Query string"},
            }
        },
    },
}

_OVERRIDE_CONTENT = {
    "version": 1,
    "ns": "demo",
    "prompt_key": "welcome",
    "tag": "latest",
    "sections": _DESCRIPTOR_CONTENT["default_override"]["sections"],
    "tools": _DESCRIPTOR_CONTENT["default_override"]["tools"],
}


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    descriptor_path = tmp_path / ".weakincentives" / "prompts" / "descriptors" / "demo"
    descriptor_path.mkdir(parents=True)
    (descriptor_path / "welcome.json").write_text(
        json.dumps(_DESCRIPTOR_CONTENT, indent=2)
    )
    return tmp_path


def test_list_empty_workspace(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = wink.main(["--root", str(workspace), "list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "ns  prompt  tag  path"
    assert captured.err == ""


def test_list_with_override(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    override_dir = (
        workspace / ".weakincentives" / "prompts" / "overrides" / "demo" / "welcome"
    )
    override_dir.mkdir(parents=True)
    (override_dir / "latest.json").write_text(json.dumps(_OVERRIDE_CONTENT, indent=2))

    exit_code = wink.main(["--root", str(workspace), "list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    output_lines = captured.out.strip().splitlines()
    assert output_lines[0].startswith("ns  prompt  tag  path")
    assert any("demo" in line and "welcome" in line for line in output_lines[1:])
    assert captured.err == ""


def test_show_json(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    override_dir = (
        workspace / ".weakincentives" / "prompts" / "overrides" / "demo" / "welcome"
    )
    override_dir.mkdir(parents=True)
    (override_dir / "latest.json").write_text(json.dumps(_OVERRIDE_CONTENT, indent=2))

    exit_code = wink.main(
        [
            "--root",
            str(workspace),
            "--format",
            "json",
            "show",
            "--ns",
            "demo",
            "--prompt",
            "welcome",
            "--tag",
            "latest",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["ns"] == "demo"
    assert payload["prompt_key"] == "welcome"
    assert captured.err == ""


def test_edit_creates_override_from_default(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script_path = workspace / "editor.py"
    script_path.write_text(
        """
import json
import sys
from pathlib import Path

override_path = Path(sys.argv[1])
content = json.loads(override_path.read_text())
content["sections"]["system"]["body"] = "Edited body"
override_path.write_text(json.dumps(content))
""".strip()
    )

    exit_code = wink.main(
        [
            "--root",
            str(workspace),
            "--editor",
            f"{sys.executable} {script_path}",
            "edit",
            "--ns",
            "demo",
            "--prompt",
            "welcome",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    override_path = (
        workspace
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "demo"
        / "welcome"
        / "latest.json"
    )
    assert override_path.is_file()
    stored_payload = json.loads(override_path.read_text())
    assert stored_payload["sections"]["system"]["body"] == "Edited body"
    assert (
        captured.out.strip()
        == ".weakincentives/prompts/overrides/demo/welcome/latest.json"
    )
    assert captured.err == ""


def test_delete_override(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    override_dir = (
        workspace / ".weakincentives" / "prompts" / "overrides" / "demo" / "welcome"
    )
    override_dir.mkdir(parents=True)
    (override_dir / "latest.json").write_text(json.dumps(_OVERRIDE_CONTENT, indent=2))

    exit_code = wink.main(
        [
            "--root",
            str(workspace),
            "--yes",
            "delete",
            "--ns",
            "demo",
            "--prompt",
            "welcome",
            "--tag",
            "latest",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Deleted" in captured.out
    assert not (override_dir / "latest.json").exists()
    assert captured.err == ""


def test_show_missing_override_errors(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = wink.main(
        [
            "--root",
            str(workspace),
            "show",
            "--ns",
            "demo",
            "--prompt",
            "welcome",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No override found" in captured.err


def test_list_filters_and_json_output(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    overrides_root = workspace / ".weakincentives" / "prompts" / "overrides"
    (overrides_root / "demo" / "welcome").mkdir(parents=True)
    (overrides_root / "demo" / "other").mkdir(parents=True)
    (overrides_root / "demo" / "welcome" / "latest.json").write_text(
        json.dumps(_OVERRIDE_CONTENT, indent=2)
    )
    (overrides_root / "demo" / "welcome" / "beta.json").write_text(
        json.dumps({**_OVERRIDE_CONTENT, "tag": "beta"}, indent=2)
    )
    (overrides_root / "demo" / "other" / "latest.json").write_text(
        json.dumps({**_OVERRIDE_CONTENT, "prompt_key": "other"}, indent=2)
    )
    # Orphan file triggers len(parts) < 2 guard.
    (overrides_root / "orphan.json").write_text("{}")

    exit_code = wink.main(
        [
            "--root",
            str(workspace),
            "--format",
            "json",
            "list",
            "--ns",
            "demo",
            "--prompt",
            "welcome",
            "--tag",
            "latest",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    rows = json.loads(captured.out)
    assert rows == [
        {
            "ns": "demo",
            "prompt": "welcome",
            "tag": "latest",
            "path": ".weakincentives/prompts/overrides/demo/welcome/latest.json",
        }
    ]


def test_list_filters_ignore_non_matching(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    overrides_root = workspace / ".weakincentives" / "prompts" / "overrides"
    (overrides_root / "demo" / "welcome").mkdir(parents=True)
    (overrides_root / "demo" / "welcome" / "latest.json").write_text(
        json.dumps(_OVERRIDE_CONTENT, indent=2)
    )

    exit_code = wink.main(
        [
            "--root",
            str(workspace),
            "--format",
            "json",
            "list",
            "--ns",
            "other",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == []


def test_show_table_output(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    override_dir = (
        workspace / ".weakincentives" / "prompts" / "overrides" / "demo" / "welcome"
    )
    override_dir.mkdir(parents=True)
    (override_dir / "latest.json").write_text(json.dumps(_OVERRIDE_CONTENT, indent=2))

    exit_code = wink.main(
        [
            "--root",
            str(workspace),
            "show",
            "--ns",
            "demo",
            "--prompt",
            "welcome",
            "--tag",
            "latest",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.splitlines()[0] == "demo/welcome:latest"


def test_edit_existing_override_uses_current_payload(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    descriptor_dir = workspace / ".weakincentives" / "prompts" / "descriptors" / "demo"
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    (descriptor_dir / "welcome.json").write_text(
        json.dumps(_DESCRIPTOR_CONTENT, indent=2)
    )
    override_dir = (
        workspace / ".weakincentives" / "prompts" / "overrides" / "demo" / "welcome"
    )
    override_dir.mkdir(parents=True)
    (override_dir / "latest.json").write_text(json.dumps(_OVERRIDE_CONTENT, indent=2))

    script_path = workspace / "editor_existing.py"
    script_path.write_text(
        """
import json
import sys
from pathlib import Path

override_path = Path(sys.argv[1])
payload = json.loads(override_path.read_text())
payload["tools"]["search"]["description"] = "Updated"
override_path.write_text(json.dumps(payload))
""".strip()
    )

    exit_code = wink.main(
        [
            "--root",
            str(workspace),
            "--editor",
            f"{sys.executable} {script_path}",
            "edit",
            "--ns",
            "demo",
            "--prompt",
            "welcome",
            "--tag",
            "latest",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    stored_payload = json.loads((override_dir / "latest.json").read_text())
    assert stored_payload["tools"]["search"]["description"] == "Updated"
    assert (
        captured.out.strip()
        == ".weakincentives/prompts/overrides/demo/welcome/latest.json"
    )


def test_edit_without_default_template_errors(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    descriptor_dir = workspace / ".weakincentives" / "prompts" / "descriptors" / "demo"
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    descriptor_without_default = {
        "descriptor": {
            **_DESCRIPTOR_CONTENT["descriptor"],
            "key": "custom",
        },
    }
    (descriptor_dir / "custom.json").write_text(
        json.dumps(descriptor_without_default, indent=2)
    )

    exit_code = wink.main(
        [
            "--root",
            str(workspace),
            "--editor",
            sys.executable,
            "edit",
            "--ns",
            "demo",
            "--prompt",
            "custom",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "default template" in captured.err


def test_edit_editor_not_found(
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor_dir = workspace / ".weakincentives" / "prompts" / "descriptors" / "demo"
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    (descriptor_dir / "welcome.json").write_text(
        json.dumps(_DESCRIPTOR_CONTENT, indent=2)
    )

    def fake_run(*args: object, **kwargs: object) -> NoReturn:
        raise FileNotFoundError("missing editor")

    monkeypatch.setattr(wink.subprocess, "run", fake_run)

    exit_code = wink.main(
        [
            "--root",
            str(workspace),
            "--editor",
            "nonexistent-editor",
            "edit",
            "--ns",
            "demo",
            "--prompt",
            "welcome",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Failed to launch editor" in captured.err


def test_edit_editor_failure_exit_code(
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor_dir = workspace / ".weakincentives" / "prompts" / "descriptors" / "demo"
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    (descriptor_dir / "welcome.json").write_text(
        json.dumps(_DESCRIPTOR_CONTENT, indent=2)
    )

    def fake_run(*args: object, **kwargs: object) -> NoReturn:
        raise subprocess.CalledProcessError(returncode=3, cmd=["editor"])

    monkeypatch.setattr(wink.subprocess, "run", fake_run)

    exit_code = wink.main(
        [
            "--root",
            str(workspace),
            "--editor",
            "failing-editor",
            "edit",
            "--ns",
            "demo",
            "--prompt",
            "welcome",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Editor exited with code" in captured.err


def test_delete_requires_confirmation(
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_dir = (
        workspace / ".weakincentives" / "prompts" / "overrides" / "demo" / "welcome"
    )
    override_dir.mkdir(parents=True)
    (override_dir / "latest.json").write_text(json.dumps(_OVERRIDE_CONTENT, indent=2))
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")

    exit_code = wink.main(
        [
            "--root",
            str(workspace),
            "delete",
            "--ns",
            "demo",
            "--prompt",
            "welcome",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Aborted." in captured.err


def test_delete_when_override_missing(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = wink.main(
        [
            "--root",
            str(workspace),
            "--yes",
            "delete",
            "--ns",
            "demo",
            "--prompt",
            "welcome",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "Nothing to delete."


def test_main_handles_prompt_override_error(
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> NoReturn:
        raise wink.PromptOverridesError("boom")

    monkeypatch.setattr(wink, "_command_list", boom)
    exit_code = wink.main(["--root", str(workspace), "list"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.strip() == "boom"


def test_main_handles_keyboard_interrupt(
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupter(*_args: object, **_kwargs: object) -> NoReturn:
        raise KeyboardInterrupt

    monkeypatch.setattr(wink, "_command_list", interrupter)
    exit_code = wink.main(["--root", str(workspace), "list"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Aborted by user." in captured.err


def test_resolve_root_path_uses_git(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_result = SimpleNamespace(stdout="/tmp/repo\n")

    def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return fake_result

    monkeypatch.setattr(wink.subprocess, "run", fake_run)
    resolved = wink._resolve_root_path(None)
    assert resolved == Path("/tmp/repo").resolve()


def test_resolve_root_path_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> NoReturn:
        raise FileNotFoundError

    monkeypatch.setattr(wink.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(wink.CliError):
        wink._resolve_root_path(None)


def test_resolve_root_path_discovers_git_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> NoReturn:
        raise FileNotFoundError

    monkeypatch.setattr(wink.subprocess, "run", fake_run)
    monkeypatch.chdir(Path.cwd())
    resolved = wink._resolve_root_path(None)
    assert (resolved / ".git").is_dir()


def test_resolve_editor_command_defaults_to_vi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WINK_EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    assert wink._resolve_editor_command(None) == ["vi"]


def test_validate_identifier_errors() -> None:
    with pytest.raises(wink.CliError):
        wink._validate_identifier("   ", "prompt key")
    with pytest.raises(wink.CliError):
        wink._validate_identifier("Invalid?", "prompt key")


def test_split_namespace_errors() -> None:
    with pytest.raises(wink.CliError):
        wink._split_namespace(" ")
    with pytest.raises(wink.CliError):
        wink._split_namespace("///")


def test_load_json_mapping_errors(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{not json}")
    with pytest.raises(wink.CliError):
        wink._load_json_mapping(invalid_path)

    array_path = tmp_path / "array.json"
    array_path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(wink.CliError):
        wink._load_json_mapping(array_path)


def test_dict_with_string_keys_error() -> None:
    with pytest.raises(wink.CliError):
        wink._dict_with_string_keys({1: "value"}, "payload")


def test_load_descriptor_errors(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(wink.CliError):
        wink._load_descriptor(workspace, "demo", "missing")

    descriptor_dir = workspace / ".weakincentives" / "prompts" / "descriptors" / "demo"
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    broken_path = descriptor_dir / "broken.json"
    broken_path.write_text(json.dumps({"descriptor": None}, indent=2))
    with pytest.raises(wink.CliError):
        wink._load_descriptor(workspace, "demo", "broken")

    mismatch_path = descriptor_dir / "mismatch.json"
    mismatch_payload = {
        "descriptor": {**_DESCRIPTOR_CONTENT["descriptor"], "ns": "other"},
        "default_override": _DESCRIPTOR_CONTENT["default_override"],
    }
    mismatch_path.write_text(json.dumps(mismatch_payload, indent=2))
    with pytest.raises(wink.CliError):
        wink._load_descriptor(workspace, "demo", "mismatch")

    bad_default_path = descriptor_dir / "bad_default.json"
    bad_payload = {
        "descriptor": {
            **_DESCRIPTOR_CONTENT["descriptor"],
            "key": "bad_default",
        },
        "default_override": ["not", "a", "mapping"],
    }
    bad_default_path.write_text(json.dumps(bad_payload, indent=2))
    with pytest.raises(wink.CliError):
        wink._load_descriptor(workspace, "demo", "bad_default")


def test_parse_descriptor_validation_errors() -> None:
    base = {
        "ns": "demo",
        "key": "welcome",
        "sections": [
            {"path": ["system"], "content_hash": "hash"},
        ],
        "tools": [
            {"path": ["system"], "name": "tool", "contract_hash": "hash"},
        ],
    }

    cases = [
        {"ns": ""},
        {"key": ""},
        {"sections": "not a list"},
        {"sections": ["wrong"]},
        {"sections": [{"path": "not list", "content_hash": "hash"}]},
        {
            "sections": [
                {"path": [""], "content_hash": "hash"},
            ]
        },
        {
            "sections": [
                {"path": ["system"], "content_hash": None},
            ]
        },
        {"tools": "not a list"},
        {"tools": ["wrong"]},
        {"tools": [{"path": "not list", "name": "tool", "contract_hash": "hash"}]},
        {
            "tools": [
                {"path": [""], "name": "tool", "contract_hash": "hash"},
            ]
        },
        {
            "tools": [
                {"path": ["system"], "name": 123, "contract_hash": "hash"},
            ]
        },
        {
            "tools": [
                {"path": ["system"], "name": "tool", "contract_hash": None},
            ]
        },
    ]

    for patch_data in cases:
        payload = json.loads(json.dumps(base))  # deep copy
        for key, value in patch_data.items():
            payload[key] = value
        with pytest.raises(wink.CliError):
            wink._parse_descriptor(payload)


def test_parse_override_payload_validation_errors() -> None:
    descriptor = wink.PromptDescriptor(
        ns="demo",
        key="welcome",
        sections=[wink.SectionDescriptor(("system",), "hash")],
        tools=[wink.ToolDescriptor(("system",), "tool", "hash")],
    )
    base_payload: dict[str, object] = {
        "ns": "demo",
        "prompt_key": "welcome",
        "tag": "latest",
        "sections": {"system": {"expected_hash": "hash", "body": "Body"}},
        "tools": {
            "tool": {
                "expected_contract_hash": "hash",
                "description": "desc",
                "param_descriptions": {"arg": "desc"},
            }
        },
    }

    scenarios = [
        {"ns": "other"},
        {"tag": 123},
        {"sections": []},
        {"sections": {1: {}}},
        {"sections": {"system": []}},
        {"sections": {"system": {"expected_hash": 1}}},
        {"sections": {"system": {"expected_hash": "hash", "body": 5}}},
        {"tools": []},
        {"tools": {1: {}}},
        {"tools": {"tool": []}},
        {"tools": {"tool": {"expected_contract_hash": 1}}},
        {
            "tools": {
                "tool": {
                    "expected_contract_hash": "hash",
                    "description": 5,
                }
            }
        },
        {
            "tools": {
                "tool": {
                    "expected_contract_hash": "hash",
                    "param_descriptions": [],
                }
            }
        },
        {
            "tools": {
                "tool": {
                    "expected_contract_hash": "hash",
                    "param_descriptions": {1: "desc"},
                }
            }
        },
    ]

    for patch_data in scenarios:
        payload = json.loads(json.dumps(base_payload))
        for key, value in patch_data.items():
            payload[key] = value
        with pytest.raises(wink.CliError):
            wink._parse_override_payload(payload, descriptor=descriptor)


def test_parse_override_payload_allows_missing_optional_fields() -> None:
    descriptor = wink.PromptDescriptor(
        ns="demo",
        key="welcome",
        sections=[wink.SectionDescriptor(("system",), "hash")],
        tools=[wink.ToolDescriptor(("system",), "tool", "hash")],
    )
    payload = {
        "ns": "demo",
        "prompt_key": "welcome",
        "tag": "latest",
    }
    result = wink._parse_override_payload(payload, descriptor=descriptor)
    assert result.sections == {}
    assert result.tool_overrides == {}

    payload_with_tool = {
        "ns": "demo",
        "prompt_key": "welcome",
        "tag": "latest",
        "tools": {"tool": {"expected_contract_hash": "hash"}},
    }
    tool_result = wink._parse_override_payload(payload_with_tool, descriptor=descriptor)
    assert tool_result.tool_overrides["tool"].param_descriptions == {}

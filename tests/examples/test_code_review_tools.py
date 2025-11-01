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

from types import SimpleNamespace

import pytest

from weakincentives.examples import code_review_tools


def _subprocess_result(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_git_log_handler_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_args: dict[str, list[str]] = {}

    def fake_run(args: list[str]) -> SimpleNamespace:
        captured_args["args"] = list(args)
        return _subprocess_result(stdout="abc123 Added feature\n")

    monkeypatch.setattr(code_review_tools, "_run_git_command", fake_run)

    params = code_review_tools.GitLogParams(
        revision_range="main..HEAD",
        path="src",
        max_count=2,
        skip=3,
        author="Jane Developer",
        since="2024-01-01",
        until="2024-02-01",
        grep="feature",
        additional_args=("--decorate", "--stat"),
    )
    result = code_review_tools.git_log_handler(params)

    assert captured_args["args"] == [
        "git",
        "log",
        "--oneline",
        "-n2",
        "--skip",
        "3",
        "--author",
        "Jane Developer",
        "--since",
        "2024-01-01",
        "--until",
        "2024-02-01",
        "--grep",
        "feature",
        "--decorate",
        "--stat",
        "main..HEAD",
        "--",
        "src",
    ]
    assert result.value.entries == ["abc123 Added feature"]
    assert "Returned 1 git log entry" in result.message


def test_git_log_handler_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str]) -> SimpleNamespace:
        return _subprocess_result(stderr="fatal: bad revision", returncode=128)

    monkeypatch.setattr(code_review_tools, "_run_git_command", fake_run)

    result = code_review_tools.git_log_handler(
        code_review_tools.GitLogParams(revision_range="bad")
    )

    assert result.value.entries == []
    assert result.message == "git log failed: fatal: bad revision"


def test_branch_list_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_args: dict[str, list[str]] = {}

    def fake_run(args: list[str]) -> SimpleNamespace:
        captured_args["args"] = list(args)
        return _subprocess_result(stdout="* main\n  feature/one\n")

    monkeypatch.setattr(code_review_tools, "_run_git_command", fake_run)

    params = code_review_tools.BranchListParams(
        include_remote=True, pattern="feature*", contains="abc123"
    )
    result = code_review_tools.branch_list_handler(params)

    assert captured_args["args"] == [
        "git",
        "branch",
        "--list",
        "--all",
        "--contains",
        "abc123",
        "feature*",
    ]
    assert result.value.branches == ["main", "feature/one"]
    assert "Returned 2 branch entries" in result.message


def test_tag_list_handler_no_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str]) -> SimpleNamespace:
        return _subprocess_result()

    monkeypatch.setattr(code_review_tools, "_run_git_command", fake_run)

    result = code_review_tools.tag_list_handler(
        code_review_tools.TagListParams(sort="-version:refname")
    )

    assert result.value.tags == []
    assert result.message == "No tags matched the query."


def test_build_tools_returns_expected_handlers() -> None:
    tools = code_review_tools.build_tools()
    names = [tool.name for tool in tools]

    assert names == [
        "show_git_log",
        "show_current_time",
        "show_git_branches",
        "show_git_tags",
    ]
    assert tools[0].handler is code_review_tools.git_log_handler
    assert tools[1].handler is code_review_tools.current_time_handler
    assert tools[2].handler is code_review_tools.branch_list_handler
    assert tools[3].handler is code_review_tools.tag_list_handler

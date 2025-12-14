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

"""Tests for Claude Agent SDK typed tool results."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from weakincentives.adapters.claude_agent_sdk import (
    SdkBashResult,
    SdkEditResult,
    SdkFileRead,
    SdkGlobResult,
    SdkGrepResult,
    SdkWriteResult,
)
from weakincentives.adapters.claude_agent_sdk._hooks import (
    HookContext,
    create_post_tool_use_hook,
)
from weakincentives.adapters.claude_agent_sdk._sdk_results import parse_sdk_tool_result
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session


@pytest.fixture
def session() -> Session:
    bus = InProcessEventBus()
    return Session(bus=bus)


@pytest.fixture
def hook_context(session: Session) -> HookContext:
    return HookContext(
        session=session,
        adapter_name="claude_agent_sdk",
        prompt_name="test_prompt",
    )


class TestSdkFileRead:
    def test_parse_with_dict_response(self) -> None:
        result = parse_sdk_tool_result(
            "Read",
            {"file_path": "/home/user/test.py"},
            {"stdout": "def hello():\n    print('hello')"},
        )

        assert isinstance(result, SdkFileRead)
        assert result.path == "/home/user/test.py"
        assert result.content == "def hello():\n    print('hello')"
        assert result.line_count == 2
        assert result.offset == 0
        assert result.limit is None
        assert result.truncated is False

    def test_parse_with_string_response(self) -> None:
        result = parse_sdk_tool_result(
            "Read",
            {"file_path": "/test.txt"},
            "line1\nline2\nline3",
        )

        assert isinstance(result, SdkFileRead)
        assert result.path == "/test.txt"
        assert result.content == "line1\nline2\nline3"
        assert result.line_count == 3

    def test_parse_with_offset_and_limit(self) -> None:
        result = parse_sdk_tool_result(
            "Read",
            {"file_path": "/test.txt", "offset": 10, "limit": 50},
            "content",
        )

        assert isinstance(result, SdkFileRead)
        assert result.offset == 10
        assert result.limit == 50

    def test_parse_returns_none_without_path(self) -> None:
        result = parse_sdk_tool_result(
            "Read",
            {},
            "content",
        )

        assert result is None

    def test_parse_with_non_str_non_dict_response(self) -> None:
        result = parse_sdk_tool_result(
            "Read",
            {"file_path": "/test.txt"},
            123,  # type: ignore[arg-type]
        )

        assert isinstance(result, SdkFileRead)
        assert result.path == "/test.txt"
        assert result.content == "123"

    def test_render(self) -> None:
        file_read = SdkFileRead(
            path="/test.py",
            content="print('hello')",
            line_count=1,
            offset=0,
            limit=None,
            truncated=False,
        )

        assert file_read.render() == "Read /test.py: 1 lines"

    def test_render_truncated(self) -> None:
        file_read = SdkFileRead(
            path="/big.py",
            content="...",
            line_count=5000,
            offset=0,
            limit=None,
            truncated=True,
        )

        assert "(truncated)" in file_read.render()


class TestSdkBashResult:
    def test_parse_with_dict_response(self) -> None:
        result = parse_sdk_tool_result(
            "Bash",
            {"command": "ls -la"},
            {"stdout": "file1\nfile2", "stderr": "", "exitCode": 0},
        )

        assert isinstance(result, SdkBashResult)
        assert result.command == "ls -la"
        assert result.stdout == "file1\nfile2"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.interrupted is False

    def test_parse_with_exit_code_key(self) -> None:
        result = parse_sdk_tool_result(
            "Bash",
            {"command": "false"},
            {"stdout": "", "stderr": "error", "exit_code": 1},
        )

        assert isinstance(result, SdkBashResult)
        assert result.exit_code == 1
        assert result.stderr == "error"

    def test_parse_with_string_response(self) -> None:
        result = parse_sdk_tool_result(
            "Bash",
            {"command": "echo hello"},
            "hello",
        )

        assert isinstance(result, SdkBashResult)
        assert result.stdout == "hello"
        assert result.stderr == ""
        assert result.exit_code == 0

    def test_parse_with_interrupted(self) -> None:
        result = parse_sdk_tool_result(
            "Bash",
            {"command": "sleep 1000"},
            {"stdout": "", "stderr": "", "interrupted": True},
        )

        assert isinstance(result, SdkBashResult)
        assert result.interrupted is True

    def test_parse_returns_none_without_command(self) -> None:
        result = parse_sdk_tool_result(
            "Bash",
            {},
            {"stdout": "output"},
        )

        assert result is None

    def test_parse_with_none_exit_code(self) -> None:
        result = parse_sdk_tool_result(
            "Bash",
            {"command": "test"},
            {"stdout": "ok", "stderr": "", "exitCode": None},
        )

        assert isinstance(result, SdkBashResult)
        assert result.exit_code == 0

    def test_parse_returns_none_for_non_str_non_dict_response(self) -> None:
        result = parse_sdk_tool_result(
            "Bash",
            {"command": "echo"},
            123,  # type: ignore[arg-type]
        )

        assert result is None

    def test_render_success(self) -> None:
        bash_result = SdkBashResult(
            command="ls",
            stdout="file1",
            stderr="",
            exit_code=0,
            interrupted=False,
        )

        rendered = bash_result.render()
        assert "$ ls [OK]" in rendered
        assert "file1" in rendered

    def test_render_failure(self) -> None:
        bash_result = SdkBashResult(
            command="bad_command",
            stdout="",
            stderr="not found",
            exit_code=127,
            interrupted=False,
        )

        rendered = bash_result.render()
        assert "[exit 127]" in rendered

    def test_render_long_output_truncated(self) -> None:
        long_output = "x" * 200
        bash_result = SdkBashResult(
            command="cat bigfile",
            stdout=long_output,
            stderr="",
            exit_code=0,
            interrupted=False,
        )

        rendered = bash_result.render()
        assert "..." in rendered
        assert len(rendered) < len(long_output) + 50  # Account for header


class TestSdkGlobResult:
    def test_parse_with_string_response(self) -> None:
        result = parse_sdk_tool_result(
            "Glob",
            {"pattern": "**/*.py", "path": "/src"},
            "file1.py\nfile2.py\nfile3.py",
        )

        assert isinstance(result, SdkGlobResult)
        assert result.pattern == "**/*.py"
        assert result.path == "/src"
        assert result.matches == ("file1.py", "file2.py", "file3.py")
        assert result.match_count == 3

    def test_parse_with_dict_response(self) -> None:
        result = parse_sdk_tool_result(
            "Glob",
            {"pattern": "*.txt", "path": "."},
            {"stdout": "a.txt\nb.txt"},
        )

        assert isinstance(result, SdkGlobResult)
        assert result.matches == ("a.txt", "b.txt")

    def test_parse_with_empty_response(self) -> None:
        result = parse_sdk_tool_result(
            "Glob",
            {"pattern": "*.nonexistent"},
            "",
        )

        assert isinstance(result, SdkGlobResult)
        assert result.matches == ()
        assert result.match_count == 0

    def test_parse_returns_none_without_pattern(self) -> None:
        result = parse_sdk_tool_result(
            "Glob",
            {"path": "/src"},
            "file1.py",
        )

        assert result is None

    def test_render(self) -> None:
        glob_result = SdkGlobResult(
            pattern="*.py",
            path="/src",
            matches=("a.py", "b.py"),
            match_count=2,
        )

        rendered = glob_result.render()
        assert "Glob *.py" in rendered
        assert "a.py" in rendered


class TestSdkGrepResult:
    def test_parse_with_string_response(self) -> None:
        result = parse_sdk_tool_result(
            "Grep",
            {"pattern": "TODO", "path": "/src"},
            "file1.py:10: # TODO fix\nfile2.py:5: # TODO clean",
        )

        assert isinstance(result, SdkGrepResult)
        assert result.pattern == "TODO"
        assert result.path == "/src"
        assert len(result.matches) == 2
        assert result.match_count == 2

    def test_parse_with_dict_response(self) -> None:
        result = parse_sdk_tool_result(
            "Grep",
            {"pattern": "error", "path": "."},
            {"stdout": "log.txt:1: error occurred"},
        )

        assert isinstance(result, SdkGrepResult)
        assert result.matches == ("log.txt:1: error occurred",)

    def test_parse_with_empty_response(self) -> None:
        result = parse_sdk_tool_result(
            "Grep",
            {"pattern": "notfound"},
            "",
        )

        assert isinstance(result, SdkGrepResult)
        assert result.matches == ()
        assert result.match_count == 0

    def test_parse_returns_none_without_pattern(self) -> None:
        result = parse_sdk_tool_result(
            "Grep",
            {"path": "/src"},
            "match",
        )

        assert result is None

    def test_render(self) -> None:
        grep_result = SdkGrepResult(
            pattern="error",
            path="/logs",
            matches=("line1", "line2"),
            match_count=2,
            files_searched=0,
        )

        rendered = grep_result.render()
        assert "Grep error" in rendered


class TestSdkWriteResult:
    def test_parse(self) -> None:
        result = parse_sdk_tool_result(
            "Write",
            {"file_path": "/test.txt", "content": "hello world"},
            "File written",
        )

        assert isinstance(result, SdkWriteResult)
        assert result.path == "/test.txt"
        assert result.bytes_written == len(b"hello world")
        assert result.created is True

    def test_parse_returns_none_without_path(self) -> None:
        result = parse_sdk_tool_result(
            "Write",
            {"content": "hello"},
            "ok",
        )

        assert result is None

    def test_render(self) -> None:
        write_result = SdkWriteResult(
            path="/test.txt",
            bytes_written=100,
            created=True,
        )

        assert "Created /test.txt" in write_result.render()


class TestSdkEditResult:
    def test_parse(self) -> None:
        result = parse_sdk_tool_result(
            "Edit",
            {
                "file_path": "/test.py",
                "old_string": "foo",
                "new_string": "bar",
            },
            "Edit successful",
        )

        assert isinstance(result, SdkEditResult)
        assert result.path == "/test.py"
        assert result.old_string == "foo"
        assert result.new_string == "bar"
        assert result.replacements == 1

    def test_parse_returns_none_without_path(self) -> None:
        result = parse_sdk_tool_result(
            "Edit",
            {"old_string": "a", "new_string": "b"},
            "ok",
        )

        assert result is None

    def test_parse_returns_none_without_old_string(self) -> None:
        result = parse_sdk_tool_result(
            "Edit",
            {"file_path": "/test.py", "new_string": "b"},
            "ok",
        )

        assert result is None

    def test_render(self) -> None:
        edit_result = SdkEditResult(
            path="/test.py",
            old_string="old",
            new_string="new",
            replacements=3,
        )

        assert "Edited /test.py: 3 replacement(s)" in edit_result.render()


class TestUnknownTool:
    def test_unknown_tool_returns_none(self) -> None:
        result = parse_sdk_tool_result(
            "UnknownTool",
            {"some": "params"},
            "output",
        )

        assert result is None


class TestParseErrorHandling:
    def test_handles_malformed_input_gracefully(self) -> None:
        # This shouldn't raise, just return None
        result = parse_sdk_tool_result(
            "Read",
            None,  # type: ignore[arg-type]
            "content",
        )

        assert result is None


class TestHookDispatchesTypedResults:
    def test_dispatches_file_read_to_session(self, session: Session) -> None:
        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data: dict[str, Any] = {
            "tool_name": "Read",
            "tool_input": {"file_path": "/test.py"},
            "tool_response": {"stdout": "print('hello')"},
        }

        asyncio.run(hook(input_data, "call-read", context))

        results = session.query(SdkFileRead).all()
        assert len(results) == 1
        assert results[0].path == "/test.py"
        assert results[0].content == "print('hello')"

    def test_dispatches_bash_result_to_session(self, session: Session) -> None:
        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data: dict[str, Any] = {
            "tool_name": "Bash",
            "tool_input": {"command": "echo hello"},
            "tool_response": {"stdout": "hello", "stderr": "", "exitCode": 0},
        }

        asyncio.run(hook(input_data, "call-bash", context))

        results = session.query(SdkBashResult).all()
        assert len(results) == 1
        assert results[0].command == "echo hello"
        assert results[0].stdout == "hello"
        assert results[0].exit_code == 0

    def test_dispatches_glob_result_to_session(self, session: Session) -> None:
        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data: dict[str, Any] = {
            "tool_name": "Glob",
            "tool_input": {"pattern": "*.py", "path": "/src"},
            "tool_response": {"stdout": "a.py\nb.py"},
        }

        asyncio.run(hook(input_data, "call-glob", context))

        results = session.query(SdkGlobResult).all()
        assert len(results) == 1
        assert results[0].pattern == "*.py"
        assert results[0].matches == ("a.py", "b.py")

    def test_dispatches_grep_result_to_session(self, session: Session) -> None:
        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data: dict[str, Any] = {
            "tool_name": "Grep",
            "tool_input": {"pattern": "TODO", "path": "/src"},
            "tool_response": {"stdout": "file.py:1: # TODO"},
        }

        asyncio.run(hook(input_data, "call-grep", context))

        results = session.query(SdkGrepResult).all()
        assert len(results) == 1
        assert results[0].pattern == "TODO"

    def test_dispatches_write_result_to_session(self, session: Session) -> None:
        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data: dict[str, Any] = {
            "tool_name": "Write",
            "tool_input": {"file_path": "/new.txt", "content": "hello"},
            "tool_response": "File written",
        }

        asyncio.run(hook(input_data, "call-write", context))

        results = session.query(SdkWriteResult).all()
        assert len(results) == 1
        assert results[0].path == "/new.txt"

    def test_dispatches_edit_result_to_session(self, session: Session) -> None:
        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)
        input_data: dict[str, Any] = {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": "/test.py",
                "old_string": "foo",
                "new_string": "bar",
            },
            "tool_response": "Edit successful",
        }

        asyncio.run(hook(input_data, "call-edit", context))

        results = session.query(SdkEditResult).all()
        assert len(results) == 1
        assert results[0].path == "/test.py"
        assert results[0].old_string == "foo"
        assert results[0].new_string == "bar"

    def test_multiple_tools_accumulate_in_session(self, session: Session) -> None:
        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)

        # Execute multiple tools
        asyncio.run(
            hook(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/file1.py"},
                    "tool_response": {"stdout": "content1"},
                },
                "call-1",
                context,
            )
        )
        asyncio.run(
            hook(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/file2.py"},
                    "tool_response": {"stdout": "content2"},
                },
                "call-2",
                context,
            )
        )
        asyncio.run(
            hook(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "ls"},
                    "tool_response": {"stdout": "files", "stderr": "", "exitCode": 0},
                },
                "call-3",
                context,
            )
        )

        # Query all file reads
        file_reads = session.query(SdkFileRead).all()
        assert len(file_reads) == 2

        # Query all bash results
        bash_results = session.query(SdkBashResult).all()
        assert len(bash_results) == 1

    def test_query_with_predicate(self, session: Session) -> None:
        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)

        # Execute some bash commands with different exit codes
        asyncio.run(
            hook(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "ls"},
                    "tool_response": {"stdout": "ok", "stderr": "", "exitCode": 0},
                },
                "call-1",
                context,
            )
        )
        asyncio.run(
            hook(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "bad_cmd"},
                    "tool_response": {
                        "stdout": "",
                        "stderr": "not found",
                        "exitCode": 1,
                    },
                },
                "call-2",
                context,
            )
        )
        asyncio.run(
            hook(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "echo hello"},
                    "tool_response": {"stdout": "hello", "stderr": "", "exitCode": 0},
                },
                "call-3",
                context,
            )
        )

        # Query failed commands only
        failed = session.query(SdkBashResult).where(lambda r: r.exit_code != 0)
        assert len(failed) == 1
        assert failed[0].command == "bad_cmd"

        # Query successful commands
        successful = session.query(SdkBashResult).where(lambda r: r.exit_code == 0)
        assert len(successful) == 2

    def test_does_not_dispatch_for_unknown_tools(self, session: Session) -> None:
        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)

        asyncio.run(
            hook(
                {
                    "tool_name": "UnknownTool",
                    "tool_input": {"param": "value"},
                    "tool_response": {"stdout": "output"},
                },
                "call-unknown",
                context,
            )
        )

        # No typed results should be in any slice
        assert session.query(SdkFileRead).all() == ()
        assert session.query(SdkBashResult).all() == ()


class TestSdkFileReadQueryPatterns:
    def test_query_by_path_pattern(self, session: Session) -> None:
        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)

        # Read several files
        for path in ["/src/main.py", "/src/utils.py", "/tests/test_main.py"]:
            asyncio.run(
                hook(
                    {
                        "tool_name": "Read",
                        "tool_input": {"file_path": path},
                        "tool_response": {"stdout": f"content of {path}"},
                    },
                    f"call-{path}",
                    context,
                )
            )

        # Query only test files
        test_files = session.query(SdkFileRead).where(lambda r: "/tests/" in r.path)
        assert len(test_files) == 1
        assert test_files[0].path == "/tests/test_main.py"

        # Query src files
        src_files = session.query(SdkFileRead).where(lambda r: "/src/" in r.path)
        assert len(src_files) == 2

    def test_latest_file_read(self, session: Session) -> None:
        context = HookContext(
            session=session,
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        hook = create_post_tool_use_hook(context)

        asyncio.run(
            hook(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/first.py"},
                    "tool_response": {"stdout": "first"},
                },
                "call-1",
                context,
            )
        )
        asyncio.run(
            hook(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/second.py"},
                    "tool_response": {"stdout": "second"},
                },
                "call-2",
                context,
            )
        )

        latest = session.query(SdkFileRead).latest()
        assert latest is not None
        assert latest.path == "/second.py"

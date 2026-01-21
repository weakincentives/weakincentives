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

"""Integration tests for transactional tool calling with state rollback.

These tests verify that the transaction system correctly rolls back both
filesystem and session state when tools fail, ensuring atomic tool execution
semantics.

The tests use ClaudeAgentWorkspaceSection which provides a HostFilesystem backed
by a temporary directory. The HostFilesystem supports snapshotting via git commits
for transactional rollback.

Debug bundles are captured to verify the final filesystem and session state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pytest

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
)
from weakincentives.dataclasses import FrozenDataclass
from weakincentives.debug import BundleConfig, BundleWriter, DebugBundle
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.runtime.session import Replace, Session, SliceOp, reducer

pytest.importorskip("claude_agent_sdk")


def _is_bedrock_mode() -> bool:
    """Check if running in Bedrock mode based on environment."""
    return os.getenv("CLAUDE_CODE_USE_BEDROCK") == "1" and "AWS_REGION" in os.environ


def _has_credentials() -> bool:
    """Check if Bedrock or Anthropic API credentials are available."""
    return _is_bedrock_mode() or "ANTHROPIC_API_KEY" in os.environ


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _has_credentials(),
        reason="Neither CLAUDE_CODE_USE_BEDROCK+AWS_REGION nor ANTHROPIC_API_KEY set.",
    ),
    pytest.mark.timeout(120),  # Allow time for multi-turn tool execution
]

_MODEL_ENV_VAR: Final[str] = "CLAUDE_AGENT_SDK_TEST_MODEL"
_PROMPT_NS: Final[str] = "integration/transactional-tools"


def _get_model() -> str:
    """Return the model name used for integration tests.

    Uses Opus 4.5 for better instruction following with custom tools.
    """
    if _MODEL_ENV_VAR in os.environ:
        return os.environ[_MODEL_ENV_VAR]
    # Use Opus for better tool following - Sonnet sometimes uses native tools
    if _is_bedrock_mode():
        return "us.anthropic.claude-opus-4-5-20251101-v1:0"
    return "claude-opus-4-5-20251101"


def _make_config(cwd: Path, **kwargs: object) -> ClaudeAgentSDKClientConfig:
    """Build a ClaudeAgentSDKClientConfig with explicit cwd."""
    config_kwargs: dict[str, object] = {
        "permission_mode": "bypassPermissions",
        "cwd": str(cwd),
    }
    config_kwargs.update(kwargs)
    return ClaudeAgentSDKClientConfig(**config_kwargs)


def _make_adapter(cwd: Path, **kwargs: object) -> ClaudeAgentSDKAdapter[object]:
    """Create a Claude Agent SDK adapter for transaction tests."""
    model = kwargs.pop("model", None) or _get_model()
    client_config = kwargs.pop("client_config", None) or _make_config(cwd)
    return ClaudeAgentSDKAdapter(
        model=model,  # type: ignore[arg-type]
        client_config=client_config,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


# =============================================================================
# Session State Slice for Tracking Tool Operations
# =============================================================================


@FrozenDataclass()
class ToolOperation:
    """Records a tool operation in session state.

    This is dispatched INSIDE the tool handler to track operations.
    When a tool fails, the snapshot restore should remove this record.
    """

    operation_id: str
    tool_name: str
    filename: str


@FrozenDataclass()
class RecordToolOperation:
    """Event to record a tool operation."""

    operation: ToolOperation


@FrozenDataclass()
class ToolOperationLog:
    """Session state slice that tracks tool operations.

    Operations are appended when tools dispatch RecordToolOperation.
    When a tool fails, the snapshot restore removes the operation.
    """

    operations: tuple[ToolOperation, ...] = field(default=())

    @reducer(on=RecordToolOperation)
    def on_record(self, event: RecordToolOperation) -> SliceOp[ToolOperationLog]:
        """Append a new operation record to the accumulated log."""
        new_operations = (*self.operations, event.operation)
        return Replace((ToolOperationLog(operations=new_operations),))

    def has_operation(self, operation_id: str) -> bool:
        """Check if an operation with the given ID exists."""
        return any(op.operation_id == operation_id for op in self.operations)


# =============================================================================
# Transactional Tools
# =============================================================================


@dataclass(slots=True)
class WriteAndFailParams:
    """Parameters for the write-then-fail tool."""

    filename: str
    content: str
    operation_id: str = ""  # Optional - generated if not provided


@dataclass(slots=True)
class WriteAndFailResult:
    """Result from write-and-fail (never returned due to failure)."""

    filename: str

    def render(self) -> str:
        return f"Wrote and then failed: {self.filename}"


def _build_write_and_fail_tool() -> Tool[WriteAndFailParams, WriteAndFailResult]:
    """Build a tool that writes a file then immediately fails.

    This tool:
    1. Writes a file to the filesystem
    2. Dispatches a session event to record the operation
    3. Returns a failure result (success=False)

    Due to transactional semantics, BOTH the file write AND the session
    event should be rolled back when the tool fails.
    """

    def handler(
        params: WriteAndFailParams, *, context: ToolContext
    ) -> ToolResult[WriteAndFailResult]:
        if context.filesystem is None:
            return ToolResult(
                message="No filesystem available",
                value=None,
                success=False,
            )

        # Step 1: Write the file
        context.filesystem.write(params.filename, params.content, mode="create")

        # Step 2: Record operation in session state (INSIDE transaction boundary)
        op_id = params.operation_id or f"fail_{params.filename}"
        operation = ToolOperation(
            operation_id=op_id,
            tool_name="write_and_fail",
            filename=params.filename,
        )
        context.session.dispatch(RecordToolOperation(operation=operation))

        # Step 3: Always fail after writing
        return ToolResult(
            message=f"Failed after writing {params.filename} - should be rolled back",
            value=None,
            success=False,
        )

    return Tool[WriteAndFailParams, WriteAndFailResult](
        name="write_and_fail",
        description=(
            "Write a file to the filesystem and record in session, then fail. "
            "Both the file and session record will be rolled back."
        ),
        handler=handler,
    )


@dataclass(slots=True)
class WriteAndSucceedParams:
    """Parameters for the write-and-succeed tool."""

    filename: str
    content: str
    operation_id: str = ""  # Optional - generated if not provided


@dataclass(slots=True)
class WriteAndSucceedResult:
    """Result from write-and-succeed."""

    filename: str
    bytes_written: int

    def render(self) -> str:
        return f"Successfully wrote {self.bytes_written} bytes to {self.filename}"


def _build_write_and_succeed_tool() -> Tool[
    WriteAndSucceedParams, WriteAndSucceedResult
]:
    """Build a tool that writes a file and succeeds.

    This tool:
    1. Writes a file to the filesystem
    2. Dispatches a session event to record the operation
    3. Returns success

    Both changes should persist since the tool succeeds.
    """

    def handler(
        params: WriteAndSucceedParams, *, context: ToolContext
    ) -> ToolResult[WriteAndSucceedResult]:
        if context.filesystem is None:
            return ToolResult(
                message="No filesystem available",
                value=None,
                success=False,
            )

        # Step 1: Write the file
        context.filesystem.write(params.filename, params.content, mode="create")
        bytes_written = len(params.content.encode("utf-8"))

        # Step 2: Record operation in session state
        op_id = params.operation_id or f"success_{params.filename}"
        operation = ToolOperation(
            operation_id=op_id,
            tool_name="write_and_succeed",
            filename=params.filename,
        )
        context.session.dispatch(RecordToolOperation(operation=operation))

        return ToolResult.ok(
            WriteAndSucceedResult(
                filename=params.filename, bytes_written=bytes_written
            ),
            message=f"Successfully wrote {params.filename}",
        )

    return Tool[WriteAndSucceedParams, WriteAndSucceedResult](
        name="write_and_succeed",
        description="Write a file and record in session. Returns success.",
        handler=handler,
    )


# =============================================================================
# List Files Tool (for verification without allowing native tools)
# =============================================================================


@dataclass(slots=True)
class ListFilesParams:
    """Parameters for list_files tool."""

    pass


@dataclass(slots=True)
class ListFilesResult:
    """Result from list_files tool."""

    files: tuple[str, ...]

    def render(self) -> str:
        if not self.files:
            return "No files in workspace"
        return "Files in workspace:\n" + "\n".join(f"  - {f}" for f in self.files)


def _build_list_files_tool() -> Tool[ListFilesParams, ListFilesResult]:
    """Build a tool that lists files in the workspace."""

    def handler(
        params: ListFilesParams, *, context: ToolContext
    ) -> ToolResult[ListFilesResult]:
        del params  # unused
        if context.filesystem is None:
            return ToolResult(
                message="No filesystem available",
                value=None,
                success=False,
            )

        try:
            entries = context.filesystem.list(".")
            files = tuple(e.name for e in entries if e.is_file)
            return ToolResult.ok(
                ListFilesResult(files=files),
                message=f"Found {len(files)} files",
            )
        except Exception as e:
            return ToolResult(
                message=f"Error listing files: {e}",
                value=None,
                success=False,
            )

    return Tool[ListFilesParams, ListFilesResult](
        name="list_files",
        description="List all files in the workspace. Use to verify which files exist.",
        handler=handler,
    )


# =============================================================================
# Prompt Builders
# =============================================================================


@dataclass(slots=True)
class TransactionTestParams:
    """Parameters for the transaction test prompt."""

    pass


@dataclass(slots=True)
class TransactionTestResult:
    """Result of the transaction test."""

    success_file_exists: bool
    rollback_file_exists: bool
    success_message: str
    rollback_message: str


def _build_transaction_test_prompt(
    write_succeed_tool: Tool[WriteAndSucceedParams, WriteAndSucceedResult],
    write_fail_tool: Tool[WriteAndFailParams, WriteAndFailResult],
    list_files_tool: Tool[ListFilesParams, ListFilesResult],
    workspace_section: ClaudeAgentWorkspaceSection,
) -> PromptTemplate[TransactionTestResult]:
    """Build a prompt that tests transactional rollback behavior."""
    task_section = MarkdownSection[TransactionTestParams](
        title="Transaction Test Task",
        template="""Perform these operations in order using the provided tools:

1. Call write_and_succeed with:
   - filename: "success.txt"
   - content: "persisted"
   - operation_id: "op_success"

2. Call write_and_fail with:
   - filename: "rollback.txt"
   - content: "should be rolled back"
   - operation_id: "op_failure"
   (This tool will intentionally fail - the file should be rolled back)

3. Call list_files to see which files exist in the workspace.

Return a structured result:
- success_file_exists: true if success.txt exists
- rollback_file_exists: true if rollback.txt exists (should be false!)
- success_message: what you observed about success.txt
- rollback_message: what you observed about rollback.txt""",
        tools=(write_succeed_tool, write_fail_tool, list_files_tool),
        key="task",
    )

    return PromptTemplate[TransactionTestResult](
        ns=_PROMPT_NS,
        key="transaction-test",
        name="transactional_tools_test",
        sections=[workspace_section, task_section],
    )


# =============================================================================
# Debug Bundle Verification Helpers
# =============================================================================


def _parse_session_snapshot(session_json: str) -> dict:
    """Parse session snapshot JSON into a dictionary."""
    return json.loads(session_json)


def _find_operation_ids_in_session(session_data: dict) -> set[str]:
    """Extract ToolOperation IDs from session snapshot data."""
    operation_ids: set[str] = set()

    for slice_entry in session_data.get("slices", []):
        # Look for ToolOperationLog slices
        if "ToolOperationLog" in slice_entry.get("slice_type", ""):
            for item in slice_entry.get("items", []):
                # Each item has an "operations" field with tuple of operations
                operations = item.get("operations", [])
                for op in operations:
                    if "operation_id" in op:
                        operation_ids.add(op["operation_id"])

    return operation_ids


def _verify_filesystem_in_bundle(
    bundle: DebugBundle,
    *,
    expected_files: tuple[str, ...],
    unexpected_files: tuple[str, ...],
) -> None:
    """Verify filesystem state in debug bundle."""
    bundle_files = bundle.list_files()
    filesystem_files = [f for f in bundle_files if f.startswith("filesystem/")]

    print("\nDebug Bundle Filesystem Contents:")
    for f in filesystem_files:
        print(f"  {f}")

    for expected in expected_files:
        assert any(expected in f for f in filesystem_files), (
            f"Expected {expected} in debug bundle. Files: {filesystem_files}"
        )

    for unexpected in unexpected_files:
        assert not any(unexpected in f for f in filesystem_files), (
            f"Expected {unexpected} NOT in debug bundle. Files: {filesystem_files}"
        )


def _verify_session_operations_in_bundle(
    bundle: DebugBundle,
    *,
    expected_ops: tuple[str, ...],
    unexpected_ops: tuple[str, ...],
) -> None:
    """Verify session state operation IDs in debug bundle."""
    session_after = bundle.session_after
    assert session_after is not None, "Expected session_after in debug bundle"

    session_data = _parse_session_snapshot(session_after)
    operation_ids = _find_operation_ids_in_session(session_data)

    print("\nDebug Bundle Session State (operation IDs):")
    for op_id in sorted(operation_ids):
        print(f"  - {op_id}")

    for expected in expected_ops:
        assert expected in operation_ids, (
            f"Expected {expected} in debug bundle session. Found: {operation_ids}"
        )

    for unexpected in unexpected_ops:
        assert unexpected not in operation_ids, (
            f"Expected {unexpected} NOT in debug bundle session. Found: {operation_ids}"
        )


def _verify_session_operations_direct(
    session: Session,
    *,
    expected_ops: tuple[str, ...],
    unexpected_ops: tuple[str, ...],
) -> None:
    """Verify session state operation IDs directly."""
    operation_log = session[ToolOperationLog].latest()

    print("\nSession State (direct):")
    if operation_log:
        for op in operation_log.operations:
            print(f"  - {op.operation_id}: {op.tool_name} -> {op.filename}")
    else:
        print("  (no operations)")

    assert operation_log is not None, "Expected ToolOperationLog in session"

    for expected in expected_ops:
        assert operation_log.has_operation(expected), (
            f"Expected {expected} in session state"
        )

    for unexpected in unexpected_ops:
        assert not operation_log.has_operation(unexpected), (
            f"Expected {unexpected} NOT in session state"
        )


# =============================================================================
# Integration Tests
# =============================================================================


def test_transactional_tool_rollback_on_failure(tmp_path: Path) -> None:
    """Test that failed tools roll back both filesystem and session state.

    This test verifies the core transactional semantics:
    1. A successful tool call persists both file and session changes
    2. A failed tool call rolls back both file and session changes

    The debug bundle is inspected to verify:
    - Filesystem: success.txt exists, rollback.txt does not
    - Session: op_success is recorded, op_failure is NOT recorded

    Note: We programmatically call our tools via the MCP bridge to ensure
    they run, then use the model to observe and report the results.
    """
    from weakincentives.adapters.claude_agent_sdk._bridge import create_bridged_tools

    session = Session()
    # Provide initial factory so reducers have state to work with on first dispatch
    session.install(ToolOperationLog, initial=ToolOperationLog)

    workspace = ClaudeAgentWorkspaceSection(session=session)
    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()

    try:
        write_succeed_tool = _build_write_and_succeed_tool()
        write_fail_tool = _build_write_and_fail_tool()
        list_files_tool = _build_list_files_tool()

        prompt_template = _build_transaction_test_prompt(
            write_succeed_tool,
            write_fail_tool,
            list_files_tool,
            workspace,
        )
        prompt = Prompt(prompt_template).bind(TransactionTestParams())

        # Manually execute tools using the bridge mechanism to test transactions
        # This ensures we test the actual transactional behavior
        with prompt.resources:
            adapter = _make_adapter(workspace.temp_dir, allowed_tools=())

            # Create bridged tools to test the bridge mechanism
            bridged = create_bridged_tools(
                (write_succeed_tool, write_fail_tool),
                session=session,
                adapter=adapter,
                prompt=prompt,
                rendered_prompt=None,
                deadline=None,
                budget_tracker=None,
            )
            write_succeed_bridged = bridged[0]
            write_fail_bridged = bridged[1]

            # Execute successful tool - should persist both file and session
            result1 = write_succeed_bridged(
                {
                    "filename": "success.txt",
                    "content": "persisted",
                    "operation_id": "op_success",
                }
            )
            assert not result1.get("isError", False), f"write_succeed failed: {result1}"

            # Execute failing tool - should rollback both file and session
            result2 = write_fail_bridged(
                {
                    "filename": "rollback.txt",
                    "content": "should rollback",
                    "operation_id": "op_failure",
                }
            )
            assert result2.get("isError", True), "write_fail should return error"

        # Capture debug bundle after programmatic tool execution
        bundle_config = BundleConfig(target=bundle_dir)
        with BundleWriter(target=bundle_dir, config=bundle_config, trigger="test") as w:
            w.set_prompt_info(ns=prompt.ns, key=prompt.key, adapter="claude_agent_sdk")
            w.write_session_after(session)
            w.write_filesystem(workspace.filesystem)

        bundle = DebugBundle.load(w.path)

        # Verify filesystem directly
        print("\nFilesystem State:")
        print(f"  success.txt exists: {workspace.filesystem.exists('success.txt')}")
        print(f"  rollback.txt exists: {workspace.filesystem.exists('rollback.txt')}")

        assert workspace.filesystem.exists("success.txt"), "success.txt should exist"
        assert not workspace.filesystem.exists("rollback.txt"), (
            "rollback.txt should NOT exist"
        )

        # Verify filesystem in debug bundle
        _verify_filesystem_in_bundle(
            bundle,
            expected_files=("success.txt",),
            unexpected_files=("rollback.txt",),
        )

        # Verify file content in bundle
        content = bundle.read_file("filesystem/success.txt")
        assert content.decode("utf-8") == "persisted"

        # Verify session state directly
        _verify_session_operations_direct(
            session,
            expected_ops=("op_success",),
            unexpected_ops=("op_failure",),
        )

        # Verify session state in debug bundle
        _verify_session_operations_in_bundle(
            bundle,
            expected_ops=("op_success",),
            unexpected_ops=("op_failure",),
        )

    finally:
        workspace.cleanup()


@dataclass(slots=True)
class SequentialOpsParams:
    """Parameters for sequential operations test."""

    pass


@dataclass(slots=True)
class SequentialOpsResult:
    """Result of sequential operations test."""

    first_file_exists: bool
    second_file_exists: bool
    third_file_exists: bool
    summary: str


def _build_sequential_ops_prompt(
    write_succeed_tool: Tool[WriteAndSucceedParams, WriteAndSucceedResult],
    write_fail_tool: Tool[WriteAndFailParams, WriteAndFailResult],
    list_files_tool: Tool[ListFilesParams, ListFilesResult],
    workspace_section: ClaudeAgentWorkspaceSection,
) -> PromptTemplate[SequentialOpsResult]:
    """Build a prompt for sequential operations with mixed success/failure."""
    task_section = MarkdownSection[SequentialOpsParams](
        title="Sequential Operations Task",
        template="""Perform these operations in sequence using the provided tools:

1. Call write_and_succeed: filename="first.txt", content="first", operation_id="op1"
2. Call write_and_fail: filename="second.txt", content="second", operation_id="op2"
3. Call write_and_succeed: filename="third.txt", content="third", operation_id="op3"

Then call list_files to see which files exist.

Expected behavior:
- first.txt SHOULD exist (op1 succeeded)
- second.txt SHOULD NOT exist (op2 failed, rolled back)
- third.txt SHOULD exist (op3 succeeded, independent of op2)

Return:
- first_file_exists: true/false
- second_file_exists: true/false
- third_file_exists: true/false
- summary: describe what you observed""",
        tools=(write_succeed_tool, write_fail_tool, list_files_tool),
        key="task",
    )

    return PromptTemplate[SequentialOpsResult](
        ns=_PROMPT_NS,
        key="sequential-ops-test",
        name="sequential_operations_test",
        sections=[workspace_section, task_section],
    )


def test_transactional_tool_sequential_operations(tmp_path: Path) -> None:
    """Test that transaction rollback is isolated to the failing operation.

    This test verifies that:
    1. Operations before a failure are committed and persist
    2. The failing operation is rolled back (both filesystem and session)
    3. Operations after a failure succeed independently

    The debug bundle is inspected to verify:
    - Filesystem: first.txt, third.txt exist; second.txt does not
    - Session: op1, op3 are recorded; op2 is NOT recorded

    Note: We programmatically call our tools via the MCP bridge to ensure
    they run, then verify the results in the debug bundle.
    """
    from weakincentives.adapters.claude_agent_sdk._bridge import create_bridged_tools

    session = Session()
    # Provide initial factory so reducers have state to work with on first dispatch
    session.install(ToolOperationLog, initial=ToolOperationLog)
    workspace = ClaudeAgentWorkspaceSection(session=session)
    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()

    try:
        write_succeed_tool = _build_write_and_succeed_tool()
        write_fail_tool = _build_write_and_fail_tool()
        list_files_tool = _build_list_files_tool()

        prompt_template = _build_sequential_ops_prompt(
            write_succeed_tool,
            write_fail_tool,
            list_files_tool,
            workspace,
        )
        prompt = Prompt(prompt_template).bind(SequentialOpsParams())

        # Manually execute tools using the bridge mechanism to test transactions
        with prompt.resources:
            adapter = _make_adapter(workspace.temp_dir, allowed_tools=())

            # Create bridged tools to test the bridge mechanism
            bridged = create_bridged_tools(
                (write_succeed_tool, write_fail_tool),
                session=session,
                adapter=adapter,
                prompt=prompt,
                rendered_prompt=None,
                deadline=None,
                budget_tracker=None,
            )
            write_succeed_bridged = bridged[0]
            write_fail_bridged = bridged[1]

            # Operation 1: Success - should persist
            result1 = write_succeed_bridged(
                {"filename": "first.txt", "content": "first", "operation_id": "op1"}
            )
            assert not result1.get("isError", False), f"op1 failed: {result1}"

            # Operation 2: Failure - should rollback
            result2 = write_fail_bridged(
                {"filename": "second.txt", "content": "second", "operation_id": "op2"}
            )
            assert result2.get("isError", True), "op2 should return error"

            # Operation 3: Success - should persist (independent of op2)
            result3 = write_succeed_bridged(
                {"filename": "third.txt", "content": "third", "operation_id": "op3"}
            )
            assert not result3.get("isError", False), f"op3 failed: {result3}"

        # Capture debug bundle after programmatic tool execution
        bundle_config = BundleConfig(target=bundle_dir)
        with BundleWriter(target=bundle_dir, config=bundle_config, trigger="test") as w:
            w.set_prompt_info(ns=prompt.ns, key=prompt.key, adapter="claude_agent_sdk")
            w.write_session_after(session)
            w.write_filesystem(workspace.filesystem)

        bundle = DebugBundle.load(w.path)

        # Verify filesystem directly
        print("\nFilesystem State:")
        print(f"  first.txt exists: {workspace.filesystem.exists('first.txt')}")
        print(f"  second.txt exists: {workspace.filesystem.exists('second.txt')}")
        print(f"  third.txt exists: {workspace.filesystem.exists('third.txt')}")

        assert workspace.filesystem.exists("first.txt"), "first.txt should exist"
        assert not workspace.filesystem.exists("second.txt"), (
            "second.txt should NOT exist"
        )
        assert workspace.filesystem.exists("third.txt"), "third.txt should exist"

        # Verify filesystem in debug bundle
        _verify_filesystem_in_bundle(
            bundle,
            expected_files=("first.txt", "third.txt"),
            unexpected_files=("second.txt",),
        )

        # Verify file contents in bundle
        assert bundle.read_file("filesystem/first.txt").decode() == "first"
        assert bundle.read_file("filesystem/third.txt").decode() == "third"

        # Verify session state directly
        _verify_session_operations_direct(
            session,
            expected_ops=("op1", "op3"),
            unexpected_ops=("op2",),
        )

        # Verify session state in debug bundle
        _verify_session_operations_in_bundle(
            bundle,
            expected_ops=("op1", "op3"),
            unexpected_ops=("op2",),
        )

    finally:
        workspace.cleanup()

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

"""Integration tests for transactional tool rollback with the Codex App Server adapter.

These tests verify that the shared transaction system correctly rolls back both
filesystem and session state when tools fail, using the Codex adapter and
WorkspaceSection for workspace management.

Tools are called programmatically via ``create_bridged_tools`` with
``adapter_name="codex_app_server"`` to ensure we test the actual transactional
behavior without requiring LLM interaction.

Debug bundles are captured to verify the final filesystem and session state.
"""

# pyright: reportArgumentType=false

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pytest

from weakincentives.adapters.codex_app_server import (
    CodexAppServerAdapter,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
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
    WorkspaceSection,
)
from weakincentives.runtime.session import Replace, Session, SliceOp, reducer


def _has_codex() -> bool:
    return shutil.which("codex") is not None


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _has_codex(), reason="codex CLI not found on PATH"),
    pytest.mark.timeout(120),
]

_MODEL_ENV_VAR: Final[str] = "CODEX_APP_SERVER_TEST_MODEL"
_PROMPT_NS: Final[str] = "integration/codex-transactional-tools"


def _get_model() -> str:
    """Return the model name for integration tests."""
    return os.environ.get(_MODEL_ENV_VAR, "gpt-5.3-codex")


def _make_adapter(cwd: Path, **kwargs: object) -> CodexAppServerAdapter:
    """Create a Codex adapter for transaction tests."""
    return CodexAppServerAdapter(
        model_config=CodexAppServerModelConfig(model=_get_model()),
        client_config=CodexAppServerClientConfig(
            cwd=str(cwd),
            approval_policy="never",
        ),
    )


# =============================================================================
# Session State Slice for Tracking Tool Operations
# =============================================================================


@FrozenDataclass()
class ToolOperation:
    """Records a tool operation in session state."""

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
    operation_id: str = ""


@dataclass(slots=True)
class WriteAndFailResult:
    """Result from write-and-fail (never returned due to failure)."""

    filename: str

    def render(self) -> str:
        return f"Wrote and then failed: {self.filename}"


def _build_write_and_fail_tool() -> Tool[WriteAndFailParams, WriteAndFailResult]:
    """Build a tool that writes a file then immediately fails."""

    def handler(
        params: WriteAndFailParams, *, context: ToolContext
    ) -> ToolResult[WriteAndFailResult]:
        if context.filesystem is None:
            return ToolResult(
                message="No filesystem available",
                value=None,
                success=False,
            )

        context.filesystem.write(params.filename, params.content, mode="create")

        op_id = params.operation_id or f"fail_{params.filename}"
        operation = ToolOperation(
            operation_id=op_id,
            tool_name="write_and_fail",
            filename=params.filename,
        )
        context.session.dispatch(RecordToolOperation(operation=operation))

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
    operation_id: str = ""


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
    """Build a tool that writes a file and succeeds."""

    def handler(
        params: WriteAndSucceedParams, *, context: ToolContext
    ) -> ToolResult[WriteAndSucceedResult]:
        if context.filesystem is None:
            return ToolResult(
                message="No filesystem available",
                value=None,
                success=False,
            )

        context.filesystem.write(params.filename, params.content, mode="create")
        bytes_written = len(params.content.encode("utf-8"))

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
# Prompt Builders
# =============================================================================


@dataclass(slots=True)
class TransactionTestParams:
    """Parameters for the transaction test prompt."""

    pass


def _build_transaction_test_prompt(
    write_succeed_tool: Tool[WriteAndSucceedParams, WriteAndSucceedResult],
    write_fail_tool: Tool[WriteAndFailParams, WriteAndFailResult],
    workspace_section: WorkspaceSection,
) -> PromptTemplate[object]:
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
   (This tool will intentionally fail - the file should be rolled back)""",
        tools=(write_succeed_tool, write_fail_tool),
        key="task",
    )

    return PromptTemplate(
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
        if "ToolOperationLog" in slice_entry.get("slice_type", ""):
            for item in slice_entry.get("items", []):
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


def test_codex_transactional_tool_rollback_on_failure(tmp_path: Path) -> None:
    """Test that failed tools roll back both filesystem and session state.

    This test verifies the core transactional semantics via the shared bridge:
    1. A successful tool call persists both file and session changes
    2. A failed tool call rolls back both file and session changes

    Tools are called programmatically via ``create_bridged_tools`` with
    ``adapter_name="codex_app_server"`` to isolate transaction behavior.
    """
    from weakincentives.adapters._shared._bridge import create_bridged_tools

    session = Session()
    session.install(ToolOperationLog, initial=ToolOperationLog)

    workspace = WorkspaceSection(session=session)
    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()

    try:
        write_succeed_tool = _build_write_and_succeed_tool()
        write_fail_tool = _build_write_and_fail_tool()

        prompt_template = _build_transaction_test_prompt(
            write_succeed_tool,
            write_fail_tool,
            workspace,
        )
        prompt = Prompt(prompt_template).bind(TransactionTestParams())

        with prompt.resources:
            adapter = _make_adapter(workspace.temp_dir)

            bridged = create_bridged_tools(
                (write_succeed_tool, write_fail_tool),
                session=session,
                adapter=adapter,
                prompt=prompt,
                rendered_prompt=None,
                deadline=None,
                budget_tracker=None,
                adapter_name="codex_app_server",
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
            w.set_prompt_info(ns=prompt.ns, key=prompt.key, adapter="codex_app_server")
            w.write_session_after(session)
            w.write_filesystem(workspace.filesystem)

        bundle = DebugBundle.load(w.path)

        # Verify filesystem directly
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


def test_codex_transactional_tool_sequential_operations(tmp_path: Path) -> None:
    """Test that transaction rollback is isolated to the failing operation.

    Three sequential operations (success, fail, success): the first and third
    persist, while the second is rolled back. Verified via direct state and
    debug bundle.
    """
    from weakincentives.adapters._shared._bridge import create_bridged_tools

    session = Session()
    session.install(ToolOperationLog, initial=ToolOperationLog)
    workspace = WorkspaceSection(session=session)
    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()

    try:
        write_succeed_tool = _build_write_and_succeed_tool()
        write_fail_tool = _build_write_and_fail_tool()

        prompt_template = _build_transaction_test_prompt(
            write_succeed_tool,
            write_fail_tool,
            workspace,
        )
        prompt = Prompt(prompt_template).bind(TransactionTestParams())

        with prompt.resources:
            adapter = _make_adapter(workspace.temp_dir)

            bridged = create_bridged_tools(
                (write_succeed_tool, write_fail_tool),
                session=session,
                adapter=adapter,
                prompt=prompt,
                rendered_prompt=None,
                deadline=None,
                budget_tracker=None,
                adapter_name="codex_app_server",
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

        # Capture debug bundle
        bundle_config = BundleConfig(target=bundle_dir)
        with BundleWriter(target=bundle_dir, config=bundle_config, trigger="test") as w:
            w.set_prompt_info(ns=prompt.ns, key=prompt.key, adapter="codex_app_server")
            w.write_session_after(session)
            w.write_filesystem(workspace.filesystem)

        bundle = DebugBundle.load(w.path)

        # Verify filesystem directly
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

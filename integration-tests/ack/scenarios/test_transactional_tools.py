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

"""Tier 3 ACK scenarios for transactional tool rollback semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from weakincentives.adapters._shared._bridge import create_bridged_tools
from weakincentives.dataclasses import FrozenDataclass
from weakincentives.debug import BundleConfig, BundleWriter, DebugBundle
from weakincentives.prompt import (
    Prompt,
    Tool,
    ToolContext,
    ToolResult,
    WorkspaceSection,
)
from weakincentives.runtime.session import Replace, Session, SliceOp, reducer

from ..adapters import AdapterFixture
from . import TransactionPromptParams, build_transactional_prompt, make_adapter_ns

pytestmark = pytest.mark.ack_capability("transactional_tools")


@FrozenDataclass()
class ToolOperation:
    """A session-recorded tool operation identifier."""

    operation_id: str
    tool_name: str
    filename: str


@FrozenDataclass()
class RecordToolOperation:
    """Event used to append operations to ``ToolOperationLog``."""

    operation: ToolOperation


@FrozenDataclass()
class ToolOperationLog:
    """Session slice for transactional operation tracking."""

    operations: tuple[ToolOperation, ...] = field(default=())

    @reducer(on=RecordToolOperation)
    def on_record(self, event: RecordToolOperation) -> SliceOp[ToolOperationLog]:
        return Replace(
            (ToolOperationLog(operations=(*self.operations, event.operation)),)
        )

    def has_operation(self, operation_id: str) -> bool:
        return any(op.operation_id == operation_id for op in self.operations)


@dataclass(slots=True)
class WriteAndFailParams:
    """Parameters for the write-and-fail transactional tool."""

    filename: str
    content: str
    operation_id: str


@dataclass(slots=True, frozen=True)
class WriteAndFailResult:
    """Never-returned payload type for write-and-fail."""

    filename: str


@dataclass(slots=True)
class WriteAndSucceedParams:
    """Parameters for the write-and-succeed transactional tool."""

    filename: str
    content: str
    operation_id: str


@dataclass(slots=True, frozen=True)
class WriteAndSucceedResult:
    """Payload for successful transactional writes."""

    filename: str
    bytes_written: int


def _build_write_and_fail_tool() -> Tool[WriteAndFailParams, WriteAndFailResult]:
    """Build a tool that writes then fails, triggering rollback."""

    def handler(
        params: WriteAndFailParams,
        *,
        context: ToolContext,
    ) -> ToolResult[WriteAndFailResult]:
        if context.filesystem is None:
            return ToolResult(
                message="No filesystem available", value=None, success=False
            )

        context.filesystem.write(params.filename, params.content, mode="create")
        context.session.dispatch(
            RecordToolOperation(
                operation=ToolOperation(
                    operation_id=params.operation_id,
                    tool_name="write_and_fail",
                    filename=params.filename,
                )
            )
        )
        return ToolResult(
            message=f"Failure after writing {params.filename}",
            value=None,
            success=False,
        )

    return Tool[WriteAndFailParams, WriteAndFailResult](
        name="write_and_fail",
        description="Write a file, record operation, then fail to trigger rollback.",
        handler=handler,
    )


def _build_write_and_succeed_tool() -> Tool[
    WriteAndSucceedParams, WriteAndSucceedResult
]:
    """Build a tool that writes and commits transaction changes."""

    def handler(
        params: WriteAndSucceedParams,
        *,
        context: ToolContext,
    ) -> ToolResult[WriteAndSucceedResult]:
        if context.filesystem is None:
            return ToolResult(
                message="No filesystem available", value=None, success=False
            )

        context.filesystem.write(params.filename, params.content, mode="create")
        context.session.dispatch(
            RecordToolOperation(
                operation=ToolOperation(
                    operation_id=params.operation_id,
                    tool_name="write_and_succeed",
                    filename=params.filename,
                )
            )
        )
        return ToolResult.ok(
            WriteAndSucceedResult(
                filename=params.filename,
                bytes_written=len(params.content.encode("utf-8")),
            ),
            message=f"Wrote {params.filename}",
        )

    return Tool[WriteAndSucceedParams, WriteAndSucceedResult](
        name="write_and_succeed",
        description="Write a file and persist transactional session/filesystem state.",
        handler=handler,
    )


def _operation_ids(session: Session) -> set[str]:
    log = session[ToolOperationLog].latest()
    if log is None:
        return set()
    return {operation.operation_id for operation in log.operations}


def _parse_operation_ids_from_bundle(bundle: DebugBundle) -> set[str]:
    snapshot = bundle.session_after
    assert snapshot is not None

    parsed = cast("dict[str, object]", json.loads(snapshot))
    operation_ids: set[str] = set()
    for operation in _iter_logged_operations(parsed):
        operation_id = operation.get("operation_id")
        if isinstance(operation_id, str):
            operation_ids.add(operation_id)
    return operation_ids


def _iter_logged_operations(parsed: dict[str, object]) -> list[dict[str, object]]:
    slices = parsed.get("slices")
    if not isinstance(slices, list):
        return []

    operation_entries: list[dict[str, object]] = []
    for entry in slices:
        if not isinstance(entry, dict):
            continue
        if "ToolOperationLog" not in str(entry.get("slice_type", "")):
            continue

        items = entry.get("items")
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            operations = item.get("operations")
            if not isinstance(operations, list):
                continue

            operation_entries.extend(
                cast("dict[str, object]", operation)
                for operation in operations
                if isinstance(operation, dict)
            )

    return operation_entries


def _run_tool_sequence(
    *,
    adapter_fixture: AdapterFixture,
    session: Session,
    workspace: WorkspaceSection,
    write_succeed_tool: Tool[WriteAndSucceedParams, WriteAndSucceedResult],
    write_fail_tool: Tool[WriteAndFailParams, WriteAndFailResult],
) -> None:
    prompt = Prompt(
        build_transactional_prompt(
            make_adapter_ns(adapter_fixture.adapter_name),
            write_succeed_tool,
            write_fail_tool,
            workspace,
        )
    ).bind(TransactionPromptParams())

    with prompt.resources:
        adapter = adapter_fixture.create_adapter(workspace.temp_dir)
        bridged = create_bridged_tools(
            (write_succeed_tool, write_fail_tool),
            session=session,
            adapter=adapter,
            prompt=prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
            adapter_name=adapter_fixture.adapter_name,
            prompt_name=prompt.name,
        )

        write_succeed = bridged[0]
        write_fail = bridged[1]

        result_1 = write_succeed(
            {
                "filename": "success.txt",
                "content": "persisted",
                "operation_id": "op_success",
            }
        )
        assert not result_1.get("isError", False), f"Unexpected failure: {result_1}"

        result_2 = write_fail(
            {
                "filename": "rollback.txt",
                "content": "should be rolled back",
                "operation_id": "op_failure",
            }
        )
        assert result_2.get("isError", True), "Expected write_and_fail to return error"


def _run_mixed_tool_sequence(
    *,
    adapter_fixture: AdapterFixture,
    session: Session,
    workspace: WorkspaceSection,
    write_succeed_tool: Tool[WriteAndSucceedParams, WriteAndSucceedResult],
    write_fail_tool: Tool[WriteAndFailParams, WriteAndFailResult],
) -> None:
    prompt = Prompt(
        build_transactional_prompt(
            make_adapter_ns(adapter_fixture.adapter_name),
            write_succeed_tool,
            write_fail_tool,
            workspace,
        )
    ).bind(TransactionPromptParams())

    with prompt.resources:
        adapter = adapter_fixture.create_adapter(workspace.temp_dir)
        bridged = create_bridged_tools(
            (write_succeed_tool, write_fail_tool),
            session=session,
            adapter=adapter,
            prompt=prompt,
            rendered_prompt=None,
            deadline=None,
            budget_tracker=None,
            adapter_name=adapter_fixture.adapter_name,
            prompt_name=prompt.name,
        )

        write_succeed = bridged[0]
        write_fail = bridged[1]

        first = write_succeed(
            {
                "filename": "first.txt",
                "content": "first",
                "operation_id": "op1",
            }
        )
        assert not first.get("isError", False), f"Unexpected failure: {first}"

        second = write_fail(
            {
                "filename": "second.txt",
                "content": "second",
                "operation_id": "op2",
            }
        )
        assert second.get("isError", True), "Expected second operation to fail"

        third = write_succeed(
            {
                "filename": "third.txt",
                "content": "third",
                "operation_id": "op3",
            }
        )
        assert not third.get("isError", False), f"Unexpected failure: {third}"


def test_tool_failure_rolls_back_filesystem(
    adapter_fixture: AdapterFixture,
    session: Session,
) -> None:
    """Failed tool calls roll back filesystem changes while successes persist."""
    session.install(ToolOperationLog, initial=ToolOperationLog)
    workspace = WorkspaceSection(session=session)

    try:
        write_succeed_tool = _build_write_and_succeed_tool()
        write_fail_tool = _build_write_and_fail_tool()

        _run_tool_sequence(
            adapter_fixture=adapter_fixture,
            session=session,
            workspace=workspace,
            write_succeed_tool=write_succeed_tool,
            write_fail_tool=write_fail_tool,
        )

        assert workspace.filesystem.exists("success.txt")
        assert not workspace.filesystem.exists("rollback.txt")
    finally:
        workspace.cleanup()


def test_tool_failure_rolls_back_session_state(
    adapter_fixture: AdapterFixture,
    session: Session,
) -> None:
    """Failed tool calls roll back session slice mutations."""
    session.install(ToolOperationLog, initial=ToolOperationLog)
    workspace = WorkspaceSection(session=session)

    try:
        write_succeed_tool = _build_write_and_succeed_tool()
        write_fail_tool = _build_write_and_fail_tool()

        _run_tool_sequence(
            adapter_fixture=adapter_fixture,
            session=session,
            workspace=workspace,
            write_succeed_tool=write_succeed_tool,
            write_fail_tool=write_fail_tool,
        )

        operation_ids = _operation_ids(session)
        assert "op_success" in operation_ids
        assert "op_failure" not in operation_ids
    finally:
        workspace.cleanup()


def test_sequential_operations_isolation(
    adapter_fixture: AdapterFixture,
    session: Session,
) -> None:
    """Rollback affects only failed operations in mixed success/failure sequences."""
    session.install(ToolOperationLog, initial=ToolOperationLog)
    workspace = WorkspaceSection(session=session)

    try:
        write_succeed_tool = _build_write_and_succeed_tool()
        write_fail_tool = _build_write_and_fail_tool()

        _run_mixed_tool_sequence(
            adapter_fixture=adapter_fixture,
            session=session,
            workspace=workspace,
            write_succeed_tool=write_succeed_tool,
            write_fail_tool=write_fail_tool,
        )

        assert workspace.filesystem.exists("first.txt")
        assert not workspace.filesystem.exists("second.txt")
        assert workspace.filesystem.exists("third.txt")

        operation_ids = _operation_ids(session)
        assert "op1" in operation_ids
        assert "op2" not in operation_ids
        assert "op3" in operation_ids
    finally:
        workspace.cleanup()


def test_rollback_verified_in_debug_bundle(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Debug bundles reflect transactional rollback in filesystem and session data."""
    session.install(ToolOperationLog, initial=ToolOperationLog)
    workspace = WorkspaceSection(session=session)
    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()

    try:
        write_succeed_tool = _build_write_and_succeed_tool()
        write_fail_tool = _build_write_and_fail_tool()

        _run_mixed_tool_sequence(
            adapter_fixture=adapter_fixture,
            session=session,
            workspace=workspace,
            write_succeed_tool=write_succeed_tool,
            write_fail_tool=write_fail_tool,
        )

        with BundleWriter(
            target=bundle_dir,
            config=BundleConfig(target=bundle_dir),
            trigger="ack_test",
        ) as writer:
            writer.set_prompt_info(
                ns=make_adapter_ns(adapter_fixture.adapter_name),
                key="ack-transactional",
                adapter=adapter_fixture.adapter_name,
            )
            writer.write_session_after(session)
            writer.write_filesystem(workspace.filesystem)

        bundle = DebugBundle.load(writer.path)

        bundle_files = bundle.list_files()
        filesystem_files = [
            name for name in bundle_files if name.startswith("filesystem/")
        ]
        assert any("first.txt" in name for name in filesystem_files)
        assert any("third.txt" in name for name in filesystem_files)
        assert not any("second.txt" in name for name in filesystem_files)

        operation_ids = _parse_operation_ids_from_bundle(bundle)
        assert "op1" in operation_ids
        assert "op2" not in operation_ids
        assert "op3" in operation_ids
    finally:
        workspace.cleanup()

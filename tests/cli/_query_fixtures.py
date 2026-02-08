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

"""Reusable fixtures for wink query tests."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from weakincentives.debug.bundle import BundleConfig, BundleWriter
from weakincentives.runtime.session import Session


@dataclass(slots=True, frozen=True)
class _AgentPlan:
    goal: str
    steps: int


@dataclass(slots=True, frozen=True)
class _TaskStatus:
    task_id: str
    completed: bool


def create_test_bundle(
    target_dir: Path,
    *,
    with_logs: bool = True,
    with_error: bool = False,
    with_config: bool = True,
    with_metrics: bool = True,
) -> Path:
    """Create a test debug bundle with various artifacts."""
    session = Session()
    session.dispatch(_AgentPlan(goal="Test goal", steps=3))
    session.dispatch(_TaskStatus(task_id="task-001", completed=True))

    with BundleWriter(target_dir, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({"task": "test"})
        writer.write_request_output({"status": "ok"})

        if with_config:
            writer.write_config({"adapter": {"model": "gpt-4"}, "max_tokens": 1000})

        if with_metrics:
            writer.write_metrics(
                {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_ms": 1500,
                }
            )

        if with_error:
            writer.write_error(
                {
                    "type": "ValueError",
                    "message": "Test error message",
                    "traceback": ["line 1", "line 2"],
                }
            )

        if with_logs:
            with writer.capture_logs():
                logger = logging.getLogger("test.logger")
                logger.setLevel(logging.DEBUG)
                logger.info(
                    "Test message",
                    extra={
                        "event": "test.event",
                        "context": {"key": "value"},
                    },
                )

    assert writer.path is not None
    return writer.path


def create_bundle_with_logs(target_dir: Path) -> Path:
    """Create bundle with custom log content."""
    session = Session()
    session.dispatch(_AgentPlan(goal="Test goal", steps=3))

    with BundleWriter(target_dir, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({"task": "test"})
        writer.write_request_output({"status": "ok"})

        # Manually trigger log capture to write to file
        with writer.capture_logs():
            logger = logging.getLogger("test.logger")
            logger.setLevel(logging.DEBUG)
            logger.info(
                "Test message",
                extra={
                    "event": "test.event",
                    "context": {"key": "value"},
                },
            )
            logger.error(
                "Error message",
                extra={
                    "event": "test.error",
                    "context": {"traceback": "stack trace"},
                },
            )
            # Tool call log
            logger.info(
                "Tool executed",
                extra={
                    "event": "tool.execution.complete",
                    "context": {
                        "tool_name": "read_file",
                        "params": {"path": "/test.txt"},
                        "duration_ms": 15.5,
                    },
                },
            )
            # New unified transcript format (transcript.entry)
            logger.debug(
                "transcript entry: user_message",
                extra={
                    "event": "transcript.entry",
                    "context": {
                        "component": "transcript",
                        "prompt_name": "test-prompt",
                        "adapter": "claude_agent_sdk",
                        "source": "main",
                        "entry_type": "user_message",
                        "sequence_number": 1,
                        "detail": {
                            "sdk_entry": {
                                "type": "user",
                                "message": {
                                    "role": "user",
                                    "content": "Hello",
                                },
                            },
                        },
                        "raw": json.dumps(
                            {
                                "type": "user",
                                "message": {
                                    "role": "user",
                                    "content": "Hello",
                                },
                            }
                        ),
                    },
                },
            )

    assert writer.path is not None
    return writer.path


__all__ = [
    "create_bundle_with_logs",
    "create_test_bundle",
]

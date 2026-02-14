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

"""Tests for AgentLoopConfig, AgentLoopRequest, and AgentLoopResult data types."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from weakincentives.budget import Budget
from weakincentives.deadlines import Deadline
from weakincentives.runtime.agent_loop import (
    AgentLoopConfig,
    AgentLoopRequest,
    AgentLoopResult,
)

from .conftest import CustomResource, SampleOutput, SampleRequest

# =============================================================================
# AgentLoopConfig Tests
# =============================================================================


def test_config_default_values() -> None:
    """AgentLoopConfig has sensible defaults."""
    config = AgentLoopConfig()
    assert config.budget is None


def test_config_custom_values() -> None:
    """AgentLoopConfig accepts custom values."""
    budget = Budget(max_total_tokens=1000)
    config = AgentLoopConfig(
        budget=budget,
    )
    assert config.budget is budget


def test_config_accepts_resources() -> None:
    """AgentLoopConfig accepts resources parameter."""
    resource = CustomResource(name="config-resource")
    resources: dict[type[object], object] = {CustomResource: resource}
    config = AgentLoopConfig(resources=resources)
    assert config.resources is resources


# =============================================================================
# AgentLoopRequest Tests
# =============================================================================


def test_request_default_values() -> None:
    """AgentLoopRequest has sensible defaults."""
    request = AgentLoopRequest(request=SampleRequest(message="hello"))
    assert request.request == SampleRequest(message="hello")
    assert request.budget is None
    assert request.deadline is None
    assert isinstance(request.request_id, UUID)
    assert request.created_at.tzinfo == UTC


def test_request_custom_values() -> None:
    """AgentLoopRequest accepts custom values."""
    budget = Budget(max_total_tokens=1000)
    deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
    request = AgentLoopRequest(
        request=SampleRequest(message="hello"),
        budget=budget,
        deadline=deadline,
    )
    assert request.budget is budget
    assert request.deadline is deadline


def test_request_accepts_resources() -> None:
    """AgentLoopRequest accepts resources parameter."""
    resource = CustomResource(name="request-resource")
    resources: dict[type[object], object] = {CustomResource: resource}
    request = AgentLoopRequest(
        request=SampleRequest(message="hello"),
        resources=resources,
    )
    assert request.resources is resources


# =============================================================================
# AgentLoopResult Tests
# =============================================================================


def test_result_success_case() -> None:
    """AgentLoopResult represents successful completion."""
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    session_id = UUID("87654321-4321-8765-4321-876543218765")
    output = SampleOutput(result="success")
    result: AgentLoopResult[SampleOutput] = AgentLoopResult(
        request_id=request_id,
        output=output,
        session_id=session_id,
    )
    assert result.request_id == request_id
    assert result.output == output
    assert result.error is None
    assert result.session_id == session_id
    assert result.success is True
    assert result.completed_at.tzinfo == UTC


def test_result_error_case() -> None:
    """AgentLoopResult represents failure."""
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    result: AgentLoopResult[SampleOutput] = AgentLoopResult(
        request_id=request_id,
        error="adapter failure",
    )
    assert result.request_id == request_id
    assert result.output is None
    assert result.error == "adapter failure"
    assert result.session_id is None
    assert result.success is False


def test_result_is_frozen() -> None:
    """AgentLoopResult is immutable."""
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    result: AgentLoopResult[SampleOutput] = AgentLoopResult(
        request_id=request_id,
        output=SampleOutput(result="success"),
    )
    with pytest.raises(AttributeError):
        result.output = None  # type: ignore[misc]

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

"""Tests for exception handler chain."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from weakincentives import DeadlineExceededError, ToolValidationError
from weakincentives.adapters._exception_handlers import (
    DEFAULT_EXCEPTION_CHAIN,
    DeadlineErrorHandler,
    ExceptionContext,
    ExceptionHandlerChain,
    TypeErrorHandler,
    UnexpectedErrorHandler,
    ValidationErrorHandler,
)
from weakincentives.adapters.core import (
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
)
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
from weakincentives.prompt.tool import ToolResult
from weakincentives.runtime.logging import StructuredLogger, get_logger
from weakincentives.runtime.session.snapshots import Snapshot
from weakincentives.runtime.transactions import CompositeSnapshot
from weakincentives.types.dataclass import SupportsToolResult


@dataclass(slots=True)
class DummyParams:
    """Dummy tool parameters for testing."""

    value: str


def _make_snapshot() -> CompositeSnapshot:
    """Create a minimal test snapshot."""
    return CompositeSnapshot(
        snapshot_id=uuid4(),
        created_at=datetime.now(UTC),
        session=Snapshot(
            created_at=datetime.now(UTC),
        ),
    )


def _make_logger() -> StructuredLogger:
    """Create a test logger."""
    return get_logger("test", context={"test": True})


def _make_context(
    error: Exception,
    *,
    tool_name: str = "test_tool",
    prompt_name: str = "test_prompt",
    deadline: Deadline | None = None,
    provider_payload: dict[str, Any] | None = None,
    tool_params: DummyParams | None = None,
    arguments_mapping: dict[str, Any] | None = None,
) -> ExceptionContext:
    """Create an exception context for testing."""
    return ExceptionContext(
        error=error,
        tool_name=tool_name,
        prompt_name=prompt_name,
        deadline=deadline,
        provider_payload=provider_payload or {},
        log=_make_logger(),
        snapshot=_make_snapshot(),
        tool_params=tool_params,
        arguments_mapping=arguments_mapping or {},
    )


class TestValidationErrorHandler:
    """Tests for ValidationErrorHandler."""

    def test_can_handle_returns_true_for_validation_error(self) -> None:
        handler = ValidationErrorHandler()
        error = ToolValidationError("Invalid parameter")
        assert handler.can_handle(error) is True

    def test_can_handle_returns_false_for_other_errors(self) -> None:
        handler = ValidationErrorHandler()
        assert handler.can_handle(ValueError("other")) is False
        assert handler.can_handle(TypeError("type")) is False
        assert handler.can_handle(RuntimeError("runtime")) is False

    def test_handle_returns_error_result(self) -> None:
        handler = ValidationErrorHandler()
        error = ToolValidationError("Invalid value for field 'count'")
        ctx = _make_context(error)

        result = handler.handle(ctx)

        assert result.success is False
        assert result.value is None
        assert "Tool validation failed" in result.message
        assert "Invalid value for field 'count'" in result.message

    def test_does_not_require_restore(self) -> None:
        handler = ValidationErrorHandler()
        assert not hasattr(handler, "requires_restore") or not handler.requires_restore


class TestDeadlineErrorHandler:
    """Tests for DeadlineErrorHandler."""

    def test_can_handle_returns_true_for_deadline_error(self) -> None:
        handler = DeadlineErrorHandler()
        error = DeadlineExceededError("Deadline exceeded")
        assert handler.can_handle(error) is True

    def test_can_handle_returns_false_for_other_errors(self) -> None:
        handler = DeadlineErrorHandler()
        assert handler.can_handle(ValueError("other")) is False
        assert handler.can_handle(ToolValidationError("validation")) is False
        assert handler.can_handle(TimeoutError("timeout")) is False

    def test_handle_returns_prompt_evaluation_error(self) -> None:
        handler = DeadlineErrorHandler()
        error = DeadlineExceededError("Time limit reached")
        ctx = _make_context(error, tool_name="slow_tool", prompt_name="my_prompt")

        result = handler.handle(ctx)

        assert isinstance(result, PromptEvaluationError)
        assert result.prompt_name == "my_prompt"
        assert result.phase == PROMPT_EVALUATION_PHASE_TOOL

    def test_handle_uses_error_message_when_present(self) -> None:
        handler = DeadlineErrorHandler()
        error = DeadlineExceededError("Custom deadline message")
        ctx = _make_context(error)

        result = handler.handle(ctx)

        assert isinstance(result, PromptEvaluationError)
        assert "Custom deadline message" in str(result)

    def test_handle_uses_fallback_message_when_empty(self) -> None:
        handler = DeadlineErrorHandler()
        error = DeadlineExceededError("")
        ctx = _make_context(error, tool_name="my_tool")

        result = handler.handle(ctx)

        assert isinstance(result, PromptEvaluationError)
        assert "my_tool" in str(result)
        assert "exceeded the deadline" in str(result)

    def test_does_not_require_restore(self) -> None:
        handler = DeadlineErrorHandler()
        # Deadline errors are re-raised, context manager handles restore
        assert not hasattr(handler, "requires_restore") or not handler.requires_restore


class TestTypeErrorHandler:
    """Tests for TypeErrorHandler."""

    def test_can_handle_returns_true_for_type_error(self) -> None:
        handler = TypeErrorHandler()
        error = TypeError("got unexpected keyword argument")
        assert handler.can_handle(error) is True

    def test_can_handle_returns_false_for_other_errors(self) -> None:
        handler = TypeErrorHandler()
        assert handler.can_handle(ValueError("other")) is False
        assert handler.can_handle(ToolValidationError("validation")) is False

    def test_handle_returns_error_result(self) -> None:
        handler = TypeErrorHandler()
        error = TypeError("missing required positional argument: 'context'")
        ctx = _make_context(error, tool_name="broken_tool")

        result = handler.handle(ctx)

        assert result.success is False
        assert result.value is None
        assert "broken_tool" in result.message
        assert "TypeError" in result.message
        assert "missing required positional argument" in result.message

    def test_requires_restore_is_true(self) -> None:
        handler = TypeErrorHandler()
        assert handler.requires_restore is True


class TestUnexpectedErrorHandler:
    """Tests for UnexpectedErrorHandler (fallback handler)."""

    def test_can_handle_returns_true_for_any_exception(self) -> None:
        handler = UnexpectedErrorHandler()
        assert handler.can_handle(ValueError("any")) is True
        assert handler.can_handle(RuntimeError("runtime")) is True
        assert handler.can_handle(Exception("generic")) is True
        assert handler.can_handle(KeyError("key")) is True

    def test_handle_returns_error_result(self) -> None:
        handler = UnexpectedErrorHandler()
        error = RuntimeError("Something unexpected happened")
        ctx = _make_context(error, tool_name="failing_tool")

        result = handler.handle(ctx)

        assert result.success is False
        assert result.value is None
        assert "failing_tool" in result.message
        assert "execution failed" in result.message
        assert "Something unexpected happened" in result.message

    def test_requires_restore_is_true(self) -> None:
        handler = UnexpectedErrorHandler()
        assert handler.requires_restore is True


class TestExceptionHandlerChain:
    """Tests for ExceptionHandlerChain."""

    def test_handlers_property_returns_tuple(self) -> None:
        handlers = [ValidationErrorHandler(), TypeErrorHandler()]
        chain = ExceptionHandlerChain(handlers)
        assert chain.handlers == tuple(handlers)

    def test_dispatches_to_first_matching_handler(self) -> None:
        chain = ExceptionHandlerChain(
            [
                ValidationErrorHandler(),
                TypeErrorHandler(),
                UnexpectedErrorHandler(),
            ]
        )
        error = ToolValidationError("Invalid")
        ctx = _make_context(error)
        restore_calls: list[str] = []

        result = chain.handle(ctx, restore_fn=lambda r: restore_calls.append(r))

        assert result.success is False
        assert "Tool validation failed" in result.message
        # Validation handler doesn't require restore
        assert restore_calls == []

    def test_calls_restore_when_handler_requires_it(self) -> None:
        chain = ExceptionHandlerChain(
            [
                ValidationErrorHandler(),
                TypeErrorHandler(),
                UnexpectedErrorHandler(),
            ]
        )
        error = TypeError("type mismatch")
        ctx = _make_context(error)
        restore_calls: list[str] = []

        result = chain.handle(ctx, restore_fn=lambda r: restore_calls.append(r))

        assert result.success is False
        assert "TypeError" in result.message
        assert restore_calls == ["exception"]

    def test_raises_exception_when_handler_returns_exception(self) -> None:
        chain = ExceptionHandlerChain(
            [
                ValidationErrorHandler(),
                DeadlineErrorHandler(),
                UnexpectedErrorHandler(),
            ]
        )
        error = DeadlineExceededError("Timed out")
        ctx = _make_context(error)

        with pytest.raises(PromptEvaluationError) as exc_info:
            chain.handle(ctx, restore_fn=lambda r: None)

        assert exc_info.value.phase == PROMPT_EVALUATION_PHASE_TOOL
        assert exc_info.value.__cause__ is error

    def test_fallback_handler_catches_unmatched_errors(self) -> None:
        chain = ExceptionHandlerChain(
            [
                ValidationErrorHandler(),
                DeadlineErrorHandler(),
                UnexpectedErrorHandler(),
            ]
        )
        error = KeyError("missing_key")
        ctx = _make_context(error, tool_name="lookup_tool")
        restore_calls: list[str] = []

        result = chain.handle(ctx, restore_fn=lambda r: restore_calls.append(r))

        assert result.success is False
        assert "lookup_tool" in result.message
        assert restore_calls == ["exception"]

    def test_handler_order_matters(self) -> None:
        # TypeErrorHandler should match before UnexpectedErrorHandler
        chain = ExceptionHandlerChain(
            [
                TypeErrorHandler(),
                UnexpectedErrorHandler(),
            ]
        )
        error = TypeError("specific type error")
        ctx = _make_context(error, tool_name="my_tool")

        result = chain.handle(ctx, restore_fn=lambda r: None)

        # TypeErrorHandler should handle it, not UnexpectedErrorHandler
        assert "TypeError" in result.message
        assert "execution failed" not in result.message


class TestDefaultExceptionChain:
    """Tests for the DEFAULT_EXCEPTION_CHAIN."""

    def test_contains_expected_handlers(self) -> None:
        handlers = DEFAULT_EXCEPTION_CHAIN.handlers
        handler_types = [type(h) for h in handlers]

        assert ValidationErrorHandler in handler_types
        assert DeadlineErrorHandler in handler_types
        assert TypeErrorHandler in handler_types
        assert UnexpectedErrorHandler in handler_types

    def test_validation_error_ordering(self) -> None:
        # ValidationErrorHandler should come before UnexpectedErrorHandler
        handlers = DEFAULT_EXCEPTION_CHAIN.handlers
        handler_types = [type(h) for h in handlers]

        val_idx = handler_types.index(ValidationErrorHandler)
        unexpected_idx = handler_types.index(UnexpectedErrorHandler)
        assert val_idx < unexpected_idx

    def test_handles_validation_error(self) -> None:
        error = ToolValidationError("Bad input")
        ctx = _make_context(error)

        result = DEFAULT_EXCEPTION_CHAIN.handle(ctx, restore_fn=lambda r: None)

        assert result.success is False
        assert "validation failed" in result.message

    def test_handles_deadline_error(self) -> None:
        error = DeadlineExceededError("Timeout")
        ctx = _make_context(error)

        with pytest.raises(PromptEvaluationError):
            DEFAULT_EXCEPTION_CHAIN.handle(ctx, restore_fn=lambda r: None)

    def test_handles_type_error(self) -> None:
        error = TypeError("Wrong type")
        ctx = _make_context(error)
        restore_calls: list[str] = []

        result = DEFAULT_EXCEPTION_CHAIN.handle(
            ctx, restore_fn=lambda r: restore_calls.append(r)
        )

        assert result.success is False
        assert "TypeError" in result.message
        assert restore_calls == ["exception"]

    def test_handles_unexpected_error(self) -> None:
        error = RuntimeError("Something broke")
        ctx = _make_context(error)
        restore_calls: list[str] = []

        result = DEFAULT_EXCEPTION_CHAIN.handle(
            ctx, restore_fn=lambda r: restore_calls.append(r)
        )

        assert result.success is False
        assert "execution failed" in result.message
        assert restore_calls == ["exception"]


class TestExceptionContext:
    """Tests for ExceptionContext dataclass."""

    def test_creation_with_all_fields(self) -> None:
        clock = FakeClock()
        clock.set_wall(datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC))
        deadline = Deadline(
            expires_at=datetime(2024, 6, 1, 13, 0, 0, tzinfo=UTC),
            clock=clock,
        )
        error = ValueError("test error")
        params = DummyParams(value="test")

        ctx = ExceptionContext(
            error=error,
            tool_name="my_tool",
            prompt_name="my_prompt",
            deadline=deadline,
            provider_payload={"model": "gpt-4"},
            log=_make_logger(),
            snapshot=_make_snapshot(),
            tool_params=params,
            arguments_mapping={"value": "test"},
        )

        assert ctx.error is error
        assert ctx.tool_name == "my_tool"
        assert ctx.prompt_name == "my_prompt"
        assert ctx.deadline is deadline
        assert ctx.provider_payload == {"model": "gpt-4"}
        assert ctx.tool_params is params
        assert ctx.arguments_mapping == {"value": "test"}

    def test_creation_with_none_optional_fields(self) -> None:
        ctx = ExceptionContext(
            error=ValueError("test"),
            tool_name="tool",
            prompt_name="prompt",
            deadline=None,
            provider_payload={},
            log=_make_logger(),
            snapshot=_make_snapshot(),
            tool_params=None,
            arguments_mapping={},
        )

        assert ctx.deadline is None
        assert ctx.tool_params is None

    def test_is_frozen(self) -> None:
        ctx = _make_context(ValueError("test"))
        with pytest.raises(AttributeError):
            ctx.tool_name = "new_name"  # type: ignore[misc]


class TestCustomHandler:
    """Tests for extending the chain with custom handlers."""

    def test_custom_handler_can_be_added(self) -> None:
        @dataclass(slots=True, frozen=True)
        class CustomErrorHandler:
            """Handler for a custom error type."""

            requires_restore: bool = True

            def can_handle(self, error: Exception) -> bool:
                return isinstance(error, KeyError)

            def handle(self, ctx: ExceptionContext) -> ToolResult[SupportsToolResult]:
                return ToolResult(
                    message=f"Custom error: {ctx.error}",
                    value=None,
                    success=False,
                )

        # Create chain with custom handler
        chain = ExceptionHandlerChain(
            [
                CustomErrorHandler(),
                UnexpectedErrorHandler(),
            ]
        )

        # Verify custom handler is checked first
        error = KeyError("missing")

        # CustomErrorHandler matches
        assert chain.handlers[0].can_handle(error) is True

    def test_chain_preserves_handler_sequence(self) -> None:
        handlers = [
            ValidationErrorHandler(),
            DeadlineErrorHandler(),
            TypeErrorHandler(),
            UnexpectedErrorHandler(),
        ]
        chain = ExceptionHandlerChain(handlers)

        # Verify order is preserved
        for i, handler in enumerate(chain.handlers):
            assert handler is handlers[i]

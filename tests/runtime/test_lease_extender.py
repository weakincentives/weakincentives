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

"""Tests for LeaseExtender and Heartbeat callback support."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from weakincentives.runtime.lease_extender import LeaseExtender, LeaseExtenderConfig
from weakincentives.runtime.mailbox import (
    InMemoryMailbox,
    Message,
    ReceiptHandleExpiredError,
)
from weakincentives.runtime.watchdog import Heartbeat

# =============================================================================
# Heartbeat Callback Tests
# =============================================================================


def test_heartbeat_callback_invoked_on_beat() -> None:
    """Verify callbacks are invoked when beat() is called."""
    calls: list[str] = []
    heartbeat = Heartbeat()
    heartbeat.add_callback(lambda: calls.append("callback1"))
    heartbeat.add_callback(lambda: calls.append("callback2"))

    heartbeat.beat()

    assert calls == ["callback1", "callback2"]


def test_heartbeat_callback_removed() -> None:
    """Verify callbacks can be removed."""
    calls: list[str] = []
    heartbeat = Heartbeat()
    callback = lambda: calls.append("callback")  # noqa: E731
    heartbeat.add_callback(callback)

    heartbeat.beat()
    assert calls == ["callback"]

    heartbeat.remove_callback(callback)
    heartbeat.beat()
    assert calls == ["callback"]  # Not called again


def test_heartbeat_remove_unregistered_callback_raises() -> None:
    """Verify removing an unregistered callback raises ValueError."""
    heartbeat = Heartbeat()

    with pytest.raises(ValueError):
        heartbeat.remove_callback(lambda: None)


def test_heartbeat_callback_exception_logged_not_propagated() -> None:
    """Verify callback exceptions are logged but don't stop other callbacks."""
    calls: list[str] = []
    heartbeat = Heartbeat()
    heartbeat.add_callback(lambda: calls.append("before"))
    heartbeat.add_callback(lambda: (_ for _ in ()).throw(RuntimeError("test error")))
    heartbeat.add_callback(lambda: calls.append("after"))

    # Should not raise, and both surrounding callbacks should be invoked
    heartbeat.beat()

    assert calls == ["before", "after"]


def test_heartbeat_elapsed_updated_on_beat() -> None:
    """Verify elapsed() is updated when beat() is called."""
    heartbeat = Heartbeat()

    time.sleep(0.1)
    elapsed_before = heartbeat.elapsed()

    heartbeat.beat()
    elapsed_after = heartbeat.elapsed()

    assert elapsed_before >= 0.1
    assert elapsed_after < 0.05  # Just beat, should be near zero


# =============================================================================
# Subscription Tests
# =============================================================================


def test_heartbeat_subscribe_returns_subscription() -> None:
    """Verify subscribe() returns a Subscription object."""
    from weakincentives.runtime.watchdog import Subscription

    calls: list[str] = []
    heartbeat = Heartbeat()
    sub = heartbeat.subscribe(lambda: calls.append("callback"))

    assert isinstance(sub, Subscription)
    assert sub.active

    heartbeat.beat()
    assert calls == ["callback"]


def test_subscription_cancel_removes_callback() -> None:
    """Verify Subscription.cancel() removes the callback."""
    calls: list[str] = []
    heartbeat = Heartbeat()
    sub = heartbeat.subscribe(lambda: calls.append("callback"))

    heartbeat.beat()
    assert calls == ["callback"]

    sub.cancel()
    assert not sub.active

    heartbeat.beat()
    assert calls == ["callback"]  # Not called again


def test_subscription_context_manager_cleanup() -> None:
    """Verify Subscription works as a context manager."""
    calls: list[str] = []
    heartbeat = Heartbeat()

    with heartbeat.subscribe(lambda: calls.append("callback")) as sub:
        assert sub.active
        heartbeat.beat()
        assert calls == ["callback"]

    assert not sub.active
    heartbeat.beat()
    assert calls == ["callback"]  # Not called again after exit


def test_subscription_cancel_idempotent() -> None:
    """Verify cancel() can be called multiple times safely."""
    heartbeat = Heartbeat()
    sub = heartbeat.subscribe(lambda: None)

    sub.cancel()
    assert not sub.active

    # Should not raise
    sub.cancel()
    sub.cancel()
    assert not sub.active


def test_subscription_cancel_handles_already_removed() -> None:
    """Verify cancel() is safe even if callback was already removed."""
    heartbeat = Heartbeat()
    callback = lambda: None  # noqa: E731
    sub = heartbeat.subscribe(callback)

    # Remove via deprecated method first
    heartbeat.remove_callback(callback)

    # cancel() should not raise
    sub.cancel()


def test_multiple_subscriptions() -> None:
    """Verify multiple subscriptions work independently."""
    calls: list[str] = []
    heartbeat = Heartbeat()

    sub1 = heartbeat.subscribe(lambda: calls.append("sub1"))
    sub2 = heartbeat.subscribe(lambda: calls.append("sub2"))

    heartbeat.beat()
    assert calls == ["sub1", "sub2"]

    sub1.cancel()
    calls.clear()

    heartbeat.beat()
    assert calls == ["sub2"]

    sub2.cancel()


# =============================================================================
# LeaseExtenderConfig Tests
# =============================================================================


def test_lease_extender_config_defaults() -> None:
    """Verify default configuration values."""
    config = LeaseExtenderConfig()

    assert config.interval == 60.0
    assert config.extension == 300
    assert config.enabled is True


def test_lease_extender_config_custom() -> None:
    """Verify custom configuration values."""
    config = LeaseExtenderConfig(interval=30.0, extension=600, enabled=False)

    assert config.interval == 30.0
    assert config.extension == 600
    assert config.enabled is False


# =============================================================================
# LeaseExtender Tests
# =============================================================================


def _create_test_message() -> tuple[Message[str, str], MagicMock]:
    """Create a test message with a mocked extend_visibility function."""
    extend_calls: list[int] = []
    extend_fn: MagicMock = MagicMock(
        side_effect=lambda timeout: extend_calls.append(timeout)
    )

    mailbox: InMemoryMailbox[str, str] = InMemoryMailbox(name="test")
    mailbox.send("test body")
    msgs = mailbox.receive(max_messages=1)
    original_msg = msgs[0]

    # Create a new message with the mocked extend function
    # Message is a dataclass, so we create a new one with the mock
    from dataclasses import replace as dc_replace

    msg = dc_replace(original_msg, _extend_fn=extend_fn)

    return msg, extend_fn


def test_lease_extender_extends_on_beat() -> None:
    """Verify extension happens when heartbeat fires."""
    msg, mock_extend = _create_test_message()
    heartbeat = Heartbeat()
    config = LeaseExtenderConfig(interval=0.0)  # No rate limit
    extender = LeaseExtender(config=config)

    with extender.attach(msg, heartbeat):
        heartbeat.beat()
        heartbeat.beat()
        heartbeat.beat()

    assert mock_extend.call_count == 3
    mock_extend.assert_called_with(300)  # Default extension


def test_lease_extender_rate_limits() -> None:
    """Verify extension is rate-limited by interval."""
    msg, mock_extend = _create_test_message()
    heartbeat = Heartbeat()
    config = LeaseExtenderConfig(interval=0.2)  # 200ms limit
    extender = LeaseExtender(config=config)

    with extender.attach(msg, heartbeat):
        heartbeat.beat()  # Extends
        heartbeat.beat()  # Skipped (interval not elapsed)
        heartbeat.beat()  # Skipped
        time.sleep(0.25)
        heartbeat.beat()  # Extends (interval elapsed)

    assert mock_extend.call_count == 2


def test_lease_extender_disabled() -> None:
    """Verify no extension when disabled."""
    msg, mock_extend = _create_test_message()
    heartbeat = Heartbeat()
    config = LeaseExtenderConfig(enabled=False)
    extender = LeaseExtender(config=config)

    with extender.attach(msg, heartbeat):
        heartbeat.beat()

    assert mock_extend.call_count == 0


def test_lease_extender_custom_extension() -> None:
    """Verify custom extension value is used."""
    msg, mock_extend = _create_test_message()
    heartbeat = Heartbeat()
    config = LeaseExtenderConfig(interval=0.0, extension=600)
    extender = LeaseExtender(config=config)

    with extender.attach(msg, heartbeat):
        heartbeat.beat()

    mock_extend.assert_called_once_with(600)


def test_lease_extender_detaches_on_exit() -> None:
    """Verify callback is removed when context exits."""
    msg, mock_extend = _create_test_message()
    heartbeat = Heartbeat()
    config = LeaseExtenderConfig(interval=0.0)
    extender = LeaseExtender(config=config)

    with extender.attach(msg, heartbeat):
        heartbeat.beat()
        assert mock_extend.call_count == 1

    # After exit, beats should not trigger extension
    heartbeat.beat()
    assert mock_extend.call_count == 1


def test_lease_extender_handles_receipt_expired_error() -> None:
    """Verify ReceiptHandleExpiredError is logged but doesn't raise."""
    msg, mock_extend = _create_test_message()
    mock_extend.side_effect = ReceiptHandleExpiredError("test")
    heartbeat = Heartbeat()
    config = LeaseExtenderConfig(interval=0.0)
    extender = LeaseExtender(config=config)

    # Should not raise
    with extender.attach(msg, heartbeat):
        heartbeat.beat()

    assert mock_extend.call_count == 1


def test_lease_extender_handles_generic_exception() -> None:
    """Verify generic exceptions are logged but don't raise."""
    msg, mock_extend = _create_test_message()
    mock_extend.side_effect = RuntimeError("network error")
    heartbeat = Heartbeat()
    config = LeaseExtenderConfig(interval=0.0)
    extender = LeaseExtender(config=config)

    # Should not raise
    with extender.attach(msg, heartbeat):
        heartbeat.beat()

    assert mock_extend.call_count == 1


def test_lease_extender_already_attached_raises() -> None:
    """Verify attaching when already attached raises RuntimeError."""
    msg, _ = _create_test_message()
    msg2, _ = _create_test_message()
    heartbeat = Heartbeat()
    config = LeaseExtenderConfig(interval=0.0)
    extender = LeaseExtender(config=config)

    with extender.attach(msg, heartbeat):
        with pytest.raises(RuntimeError, match="already attached"):
            with extender.attach(msg2, heartbeat):
                pass


def test_lease_extender_no_op_after_detach() -> None:
    """Verify _on_beat is a no-op after detach."""
    msg, mock_extend = _create_test_message()
    heartbeat = Heartbeat()
    config = LeaseExtenderConfig(interval=0.0)
    extender = LeaseExtender(config=config)

    with extender.attach(msg, heartbeat):
        heartbeat.beat()

    # Manually call _on_beat after detach (simulates race condition)
    extender._on_beat()

    # Should only have the one call from before detach
    assert mock_extend.call_count == 1


def test_lease_extender_with_multiple_callbacks() -> None:
    """Verify LeaseExtender coexists with other heartbeat callbacks."""
    msg, mock_extend = _create_test_message()
    heartbeat = Heartbeat()
    config = LeaseExtenderConfig(interval=0.0)
    extender = LeaseExtender(config=config)

    other_calls: list[str] = []
    heartbeat.add_callback(lambda: other_calls.append("metrics"))

    with extender.attach(msg, heartbeat):
        heartbeat.beat()

    # Both callbacks should be invoked
    assert other_calls == ["metrics"]
    assert mock_extend.call_count == 1


def test_lease_extender_default_config() -> None:
    """Verify LeaseExtender uses default config when None provided."""
    extender = LeaseExtender()
    assert extender.config.interval == 60.0
    assert extender.config.extension == 300
    assert extender.config.enabled is True


def test_lease_extender_detach_when_not_attached() -> None:
    """Verify _detach is safe when heartbeat is None (never attached)."""
    extender = LeaseExtender()

    # Should not raise - covers the heartbeat is None branch in _detach
    extender._detach()


# =============================================================================
# Integration Tests for Heartbeat Propagation
# =============================================================================


def test_tool_context_beat_with_heartbeat() -> None:
    """Verify ToolContext.beat() invokes heartbeat.beat() when set."""
    from weakincentives.prompt import Prompt, PromptTemplate, ToolContext
    from weakincentives.runtime.session import Session

    heartbeat = Heartbeat()
    beat_count = [0]
    heartbeat.add_callback(lambda: beat_count.__setitem__(0, beat_count[0] + 1))

    session = Session()
    template: PromptTemplate[None] = PromptTemplate(ns="test", key="test", name="test")
    prompt: Prompt[None] = Prompt(template)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=None,  # type: ignore[arg-type]
        adapter=None,  # type: ignore[arg-type]
        session=session,
        deadline=None,
        heartbeat=heartbeat,
    )

    context.beat()
    assert beat_count[0] == 1


def test_tool_execution_context_beat_with_heartbeat() -> None:
    """Verify ToolExecutionContext.beat() invokes heartbeat.beat() when set."""
    from weakincentives.adapters.tool_executor import ToolExecutionContext
    from weakincentives.prompt import Prompt, PromptTemplate
    from weakincentives.runtime.session import Session

    heartbeat = Heartbeat()
    beat_count = [0]
    heartbeat.add_callback(lambda: beat_count.__setitem__(0, beat_count[0] + 1))

    session = Session()
    template: PromptTemplate[None] = PromptTemplate(ns="test", key="test", name="test")
    prompt: Prompt[None] = Prompt(template)
    context = ToolExecutionContext(
        adapter_name="test",
        adapter=None,  # type: ignore[arg-type]
        prompt=prompt,
        rendered_prompt=None,
        tool_registry={},
        session=session,
        prompt_name="test",
        parse_arguments=lambda _tc, _h: None,  # type: ignore[arg-type,return-value]
        format_dispatch_failures=lambda _: "",
        deadline=None,
        heartbeat=heartbeat,
    )

    context.beat()
    assert beat_count[0] == 1


def test_hook_context_beat_with_heartbeat() -> None:
    """Verify HookContext.beat() invokes heartbeat.beat() when set."""
    from weakincentives.adapters.claude_agent_sdk._hooks import HookContext
    from weakincentives.prompt import Prompt, PromptTemplate
    from weakincentives.runtime.session import Session

    heartbeat = Heartbeat()
    beat_count = [0]
    heartbeat.add_callback(lambda: beat_count.__setitem__(0, beat_count[0] + 1))

    session = Session()
    template: PromptTemplate[None] = PromptTemplate(ns="test", key="test", name="test")
    prompt: Prompt[None] = Prompt(template)
    context = HookContext(
        prompt=prompt,
        session=session,
        adapter_name="test",
        prompt_name="test",
        heartbeat=heartbeat,
    )

    context.beat()
    assert beat_count[0] == 1


def test_inner_loop_beat_method_with_heartbeat() -> None:
    """Verify InnerLoop._beat() invokes heartbeat.beat() when configured."""
    from weakincentives.adapters.inner_loop import InnerLoop, InnerLoopConfig
    from weakincentives.prompt import Prompt, PromptTemplate
    from weakincentives.prompt.prompt import RenderedPrompt
    from weakincentives.runtime.session import Session

    heartbeat = Heartbeat()
    beat_count = [0]
    heartbeat.add_callback(lambda: beat_count.__setitem__(0, beat_count[0] + 1))

    session = Session()
    template = PromptTemplate[None](ns="test", key="test", name="test")
    prompt = Prompt(template)
    rendered = RenderedPrompt[None](text="test prompt")

    # Import InnerLoopInputs
    from weakincentives.adapters.inner_loop import InnerLoopInputs

    inputs = InnerLoopInputs[None](
        adapter_name="test",
        adapter=None,  # type: ignore[arg-type]
        prompt=prompt,
        prompt_name="test",
        rendered=rendered,
        render_inputs=(),
        initial_messages=[{"role": "system", "content": "test"}],
    )
    config = InnerLoopConfig(
        session=session,
        tool_choice="auto",
        response_format=None,
        require_structured_output_text=False,
        call_provider=lambda *args: None,  # type: ignore[arg-type,return-value]
        select_choice=lambda r: r,  # type: ignore[arg-type,return-value]
        serialize_tool_message_fn=lambda *args: {},  # type: ignore[arg-type,return-value]
        heartbeat=heartbeat,
    )

    loop = InnerLoop(inputs=inputs, config=config)
    # Directly call _beat to test the heartbeat invocation
    loop._beat()

    assert beat_count[0] == 1


def test_tool_context_beat_without_heartbeat() -> None:
    """Verify ToolContext.beat() is safe when heartbeat is None."""
    from weakincentives.prompt import Prompt, PromptTemplate, ToolContext
    from weakincentives.runtime.session import Session

    session = Session()
    template: PromptTemplate[None] = PromptTemplate(ns="test", key="test", name="test")
    prompt: Prompt[None] = Prompt(template)
    context = ToolContext(
        prompt=prompt,
        rendered_prompt=None,  # type: ignore[arg-type]
        adapter=None,  # type: ignore[arg-type]
        session=session,
        deadline=None,
        heartbeat=None,  # No heartbeat
    )

    # Should not raise even with no heartbeat
    context.beat()


__all__ = []

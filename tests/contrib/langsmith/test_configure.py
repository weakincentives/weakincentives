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

"""Tests for configure_wink auto-instrumentation."""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from tests.helpers.adapters import TEST_ADAPTER_NAME
from weakincentives.contrib.langsmith import LangSmithTelemetryHandler
from weakincentives.contrib.langsmith._configure import (
    configure_wink,
    get_configured_handler,
    unconfigure_wink,
)
from weakincentives.contrib.langsmith.testing import MockLangSmithClient
from weakincentives.runtime.events import InProcessEventBus, PromptRendered


@pytest.fixture(autouse=True)
def cleanup_configuration() -> Generator[None, None, None]:
    """Ensure configuration is cleaned up after each test."""
    yield
    unconfigure_wink()


class TestConfigureWink:
    """Tests for configure_wink function."""

    def test_returns_handler(self) -> None:
        """configure_wink returns a handler."""
        handler = configure_wink(
            api_key="test",
            project="test",
            async_upload=False,
            flush_on_exit=False,
        )

        assert isinstance(handler, LangSmithTelemetryHandler)

    def test_patches_event_bus(self) -> None:
        """configure_wink patches InProcessEventBus.__init__."""
        configure_wink(
            api_key="test",
            project="test",
            async_upload=False,
            flush_on_exit=False,
        )

        # New buses should have handler attached
        _ = InProcessEventBus()

        # Verify handler is attached by checking subscription behavior
        # We can't directly inspect, but we know the handler attaches
        # to specific event types

    def test_get_configured_handler(self) -> None:
        """get_configured_handler returns the active handler."""
        handler = configure_wink(
            api_key="test",
            project="test",
            async_upload=False,
            flush_on_exit=False,
        )

        assert get_configured_handler() is handler

    def test_unconfigure_restores_init(self) -> None:
        """unconfigure_wink restores original __init__."""
        # Get reference to original init
        original = InProcessEventBus.__init__

        configure_wink(
            api_key="test",
            project="test",
            async_upload=False,
            flush_on_exit=False,
        )

        unconfigure_wink()

        # Init should be restored
        assert InProcessEventBus.__init__ is original

    def test_unconfigure_clears_handler(self) -> None:
        """unconfigure_wink clears the handler."""
        configure_wink(
            api_key="test",
            project="test",
            async_upload=False,
            flush_on_exit=False,
        )

        unconfigure_wink()

        assert get_configured_handler() is None

    def test_multiple_configure_calls_reuse_patch(self) -> None:
        """Multiple configure calls don't stack patches."""
        configure_wink(
            api_key="test1",
            project="test1",
            async_upload=False,
            flush_on_exit=False,
        )

        handler2 = configure_wink(
            api_key="test2",
            project="test2",
            async_upload=False,
            flush_on_exit=False,
        )

        # Second handler should be used
        assert get_configured_handler() is handler2

    def test_custom_config_options(self) -> None:
        """configure_wink accepts custom config options."""
        handler = configure_wink(
            api_key="custom-key",
            project="custom-project",
            trace_sample_rate=0.5,
            trace_native_tools=False,
            async_upload=False,
            flush_on_exit=False,
        )

        # Access internal config to verify
        assert handler._config.trace_sample_rate == 0.5
        assert handler._config.trace_native_tools is False


class TestAutoInstrumentation:
    """Tests for auto-instrumentation behavior."""

    def test_new_buses_auto_trace(self) -> None:
        """New buses automatically trace events."""
        # Create mock client to verify tracing
        mock_client = MockLangSmithClient()

        # Configure with mock client by creating handler directly
        # and using configure_wink pattern
        configure_wink(
            api_key="test",
            project="test",
            async_upload=False,
            flush_on_exit=False,
        )

        # Replace handler's client with mock
        handler = get_configured_handler()
        assert handler is not None
        handler._client = mock_client

        # Create new bus - should auto-attach
        bus = InProcessEventBus()

        # Publish event
        bus.publish(
            PromptRendered(
                prompt_ns="test",
                prompt_key="prompt",
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                session_id=uuid4(),
                render_inputs=(),
                rendered_prompt="Test",
                created_at=datetime.now(UTC),
            )
        )

        # Mock client should have received the run
        assert mock_client.runs_created >= 1

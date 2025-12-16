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

"""Tests for LangSmithConfig."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from weakincentives.contrib.langsmith import LangSmithConfig


class TestLangSmithConfig:
    """Tests for LangSmithConfig dataclass."""

    def test_default_values(self) -> None:
        """Config has sensible defaults."""
        config = LangSmithConfig()

        assert config.api_key is None
        assert config.api_url == "https://api.smith.langchain.com"
        assert config.project is None
        assert config.tracing_enabled is True
        assert config.trace_sample_rate == 1.0
        assert config.async_upload is True
        assert config.upload_batch_size == 100
        assert config.upload_interval_seconds == 1.0
        assert config.max_queue_size == 10000
        assert config.flush_on_exit is True
        assert config.trace_native_tools is True
        assert config.hub_enabled is True
        assert config.cache_ttl_seconds == 300.0
        assert config.cache_versioned_indefinitely is True

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        config = LangSmithConfig(
            api_key="test-key",
            project="test-project",
            trace_sample_rate=0.5,
            async_upload=False,
        )

        assert config.api_key == "test-key"
        assert config.project == "test-project"
        assert config.trace_sample_rate == 0.5
        assert config.async_upload is False

    def test_frozen_dataclass(self) -> None:
        """Config is immutable."""
        config = LangSmithConfig()

        with pytest.raises(AttributeError):
            config.api_key = "new-key"

    def test_resolved_api_key_with_explicit_value(self) -> None:
        """resolved_api_key returns explicit value."""
        config = LangSmithConfig(api_key="explicit-key")
        assert config.resolved_api_key() == "explicit-key"

    def test_resolved_api_key_from_env(self) -> None:
        """resolved_api_key falls back to env var."""
        config = LangSmithConfig()
        with patch.dict(os.environ, {"LANGCHAIN_API_KEY": "env-key"}):
            assert config.resolved_api_key() == "env-key"

    def test_resolved_api_key_prefers_explicit(self) -> None:
        """resolved_api_key prefers explicit over env."""
        config = LangSmithConfig(api_key="explicit-key")
        with patch.dict(os.environ, {"LANGCHAIN_API_KEY": "env-key"}):
            assert config.resolved_api_key() == "explicit-key"

    def test_resolved_project_with_explicit_value(self) -> None:
        """resolved_project returns explicit value."""
        config = LangSmithConfig(project="explicit-project")
        assert config.resolved_project() == "explicit-project"

    def test_resolved_project_from_env(self) -> None:
        """resolved_project falls back to env var."""
        config = LangSmithConfig()
        with patch.dict(os.environ, {"LANGCHAIN_PROJECT": "env-project"}):
            assert config.resolved_project() == "env-project"

    def test_resolved_project_default(self) -> None:
        """resolved_project returns default when no value set."""
        config = LangSmithConfig()
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env vars
            os.environ.pop("LANGCHAIN_PROJECT", None)
            assert config.resolved_project() == "default"

    def test_resolved_api_url_from_env(self) -> None:
        """resolved_api_url uses env var when set."""
        config = LangSmithConfig()
        with patch.dict(os.environ, {"LANGCHAIN_ENDPOINT": "https://custom.api"}):
            assert config.resolved_api_url() == "https://custom.api"

    def test_resolved_api_url_default(self) -> None:
        """resolved_api_url returns config value when no env."""
        config = LangSmithConfig(api_url="https://explicit.api")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LANGCHAIN_ENDPOINT", None)
            assert config.resolved_api_url() == "https://explicit.api"

    def test_is_tracing_enabled_default(self) -> None:
        """is_tracing_enabled returns True by default."""
        config = LangSmithConfig()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
            assert config.is_tracing_enabled() is True

    def test_is_tracing_enabled_when_disabled(self) -> None:
        """is_tracing_enabled returns False when config disabled."""
        config = LangSmithConfig(tracing_enabled=False)
        assert config.is_tracing_enabled() is False

    def test_is_tracing_enabled_env_override(self) -> None:
        """is_tracing_enabled respects env var."""
        config = LangSmithConfig(tracing_enabled=True)
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false"}):
            assert config.is_tracing_enabled() is False

    def test_is_tracing_enabled_env_true(self) -> None:
        """is_tracing_enabled allows env var to be true."""
        config = LangSmithConfig(tracing_enabled=True)
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}):
            assert config.is_tracing_enabled() is True

    def test_update_method(self) -> None:
        """Config supports update method from FrozenDataclass."""
        from dataclasses import replace

        config = LangSmithConfig(project="original")
        updated = replace(config, project="updated")

        assert config.project == "original"
        assert updated.project == "updated"

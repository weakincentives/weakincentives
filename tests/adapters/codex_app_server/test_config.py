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

"""Tests for Codex App Server configuration dataclasses."""

from __future__ import annotations

from weakincentives.adapters.codex_app_server.config import (
    DEFAULT_MODEL,
    ApiKeyAuth,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
    ExternalTokenAuth,
)


class TestDefaultModel:
    def test_default_model_is_gpt52(self) -> None:
        assert DEFAULT_MODEL == "gpt-5.2"


class TestApiKeyAuth:
    def test_creation(self) -> None:
        auth = ApiKeyAuth(api_key="sk-test-123")
        assert auth.api_key == "sk-test-123"

    def test_frozen(self) -> None:
        auth = ApiKeyAuth(api_key="sk-test")
        try:
            auth.api_key = "other"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised


class TestExternalTokenAuth:
    def test_creation(self) -> None:
        auth = ExternalTokenAuth(id_token="id-tok", access_token="access-tok")
        assert auth.id_token == "id-tok"
        assert auth.access_token == "access-tok"


class TestCodexAppServerClientConfig:
    def test_defaults(self) -> None:
        cfg = CodexAppServerClientConfig()
        assert cfg.codex_bin == "codex"
        assert cfg.cwd is None
        assert cfg.env is None
        assert cfg.suppress_stderr is True
        assert cfg.startup_timeout_s == 10.0
        assert cfg.approval_policy == "never"
        assert cfg.sandbox_mode is None
        assert cfg.auth_mode is None
        assert cfg.reuse_thread is False
        assert cfg.mcp_servers is None
        assert cfg.ephemeral is False
        assert cfg.client_name == "wink"
        assert cfg.client_version == "0.1.0"

    def test_custom_values(self) -> None:
        auth = ApiKeyAuth(api_key="key")
        cfg = CodexAppServerClientConfig(
            codex_bin="/usr/local/bin/codex",
            cwd="/tmp/work",
            env={"OPENAI_API_KEY": "key"},
            suppress_stderr=False,
            startup_timeout_s=30.0,
            approval_policy="on-request",
            sandbox_mode="read-only",
            auth_mode=auth,
            reuse_thread=True,
            mcp_servers={"server1": {"command": "npx", "args": ["mcp"]}},
            ephemeral=True,
            client_name="my-agent",
            client_version="2.0.0",
        )
        assert cfg.codex_bin == "/usr/local/bin/codex"
        assert cfg.cwd == "/tmp/work"
        assert cfg.approval_policy == "on-request"
        assert cfg.sandbox_mode == "read-only"
        assert cfg.auth_mode is auth
        assert cfg.reuse_thread is True
        assert cfg.mcp_servers is not None
        assert "server1" in cfg.mcp_servers
        assert cfg.ephemeral is True
        assert cfg.client_name == "my-agent"


class TestCodexAppServerModelConfig:
    def test_defaults(self) -> None:
        cfg = CodexAppServerModelConfig()
        assert cfg.model == "gpt-5.2"
        assert cfg.effort is None
        assert cfg.summary is None
        assert cfg.personality is None

    def test_custom_values(self) -> None:
        cfg = CodexAppServerModelConfig(
            model="o3",
            effort="high",
            summary="concise",
            personality="friendly",
        )
        assert cfg.model == "o3"
        assert cfg.effort == "high"
        assert cfg.summary == "concise"
        assert cfg.personality == "friendly"

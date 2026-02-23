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

"""Tests for Claude Agent SDK EphemeralHome: creation, settings, and environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

from weakincentives.adapters.claude_agent_sdk._ephemeral_home import (
    EphemeralHome,
)
from weakincentives.adapters.claude_agent_sdk.isolation import (
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)


class TestEphemeralHome:
    def test_creates_temp_directory(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            assert Path(home.home_path).is_dir()
            # Temp dir location is platform-dependent (e.g., /tmp on Linux, /var/folders on macOS)
            assert "claude-agent-" in home.home_path

    def test_creates_claude_directory(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            assert home.claude_dir.is_dir()
            assert home.claude_dir == Path(home.home_path) / ".claude"

    def test_creates_settings_json(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            assert home.settings_path.is_file()
            settings = json.loads(home.settings_path.read_text())
            assert "sandbox" in settings

    def test_cleanup_removes_directory(self) -> None:
        config = IsolationConfig()
        home = EphemeralHome(config)
        home_path = Path(home.home_path)
        assert home_path.is_dir()
        home.cleanup()
        assert not home_path.exists()

    def test_cleanup_is_idempotent(self) -> None:
        config = IsolationConfig()
        home = EphemeralHome(config)
        home.cleanup()
        home.cleanup()  # Should not raise

    def test_context_manager_cleanup(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home_path = Path(home.home_path)
            assert home_path.is_dir()
        assert not home_path.exists()

    def test_get_setting_sources_returns_user(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            sources = home.get_setting_sources()
            # Returns ["user"] to load settings from ephemeral HOME
            assert sources == ["user"]


class TestEphemeralHomeSettingsGeneration:
    def test_default_sandbox_settings(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["enabled"] is True
            assert settings["sandbox"]["autoAllowBashIfSandboxed"] is True
            assert settings["sandbox"]["allowUnsandboxedCommands"] is False
            assert settings["sandbox"]["network"]["allowedDomains"] == []

    def test_network_policy_allowed_domains(self) -> None:
        config = IsolationConfig(
            network_policy=NetworkPolicy(
                allowed_domains=("api.anthropic.com", "api.github.com")
            )
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["network"]["allowedDomains"] == [
                "api.anthropic.com",
                "api.github.com",
            ]

    def test_sandbox_disabled(self) -> None:
        config = IsolationConfig(
            sandbox=SandboxConfig(enabled=False, bash_auto_allow=False)
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["enabled"] is False
            assert settings["sandbox"]["autoAllowBashIfSandboxed"] is False

    def test_sandbox_excluded_commands(self) -> None:
        config = IsolationConfig(
            sandbox=SandboxConfig(
                excluded_commands=("docker", "podman"),
                allow_unsandboxed_commands=True,
            )
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["excludedCommands"] == ["docker", "podman"]
            assert settings["sandbox"]["allowUnsandboxedCommands"] is True

    def test_sandbox_writable_paths(self) -> None:
        config = IsolationConfig(
            sandbox=SandboxConfig(writable_paths=("/tmp/output", "/var/log"))
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            # When sandbox is enabled, Claude Code's temp dir is auto-added
            expected = ["/tmp/output", "/var/log", f"/tmp/claude-{os.getuid()}"]
            assert settings["sandbox"]["writablePaths"] == expected

    def test_sandbox_writable_paths_already_includes_claude_temp(self) -> None:
        """Test that Claude temp dir is not duplicated if already in writable_paths."""
        claude_temp_dir = f"/tmp/claude-{os.getuid()}"
        config = IsolationConfig(
            sandbox=SandboxConfig(writable_paths=("/tmp/output", claude_temp_dir))
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            # Should not duplicate the Claude temp dir
            expected = ["/tmp/output", claude_temp_dir]
            assert settings["sandbox"]["writablePaths"] == expected

    def test_sandbox_readable_paths(self) -> None:
        config = IsolationConfig(
            sandbox=SandboxConfig(readable_paths=("/data/readonly",))
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["readablePaths"] == ["/data/readonly"]

    def test_enable_weaker_nested_sandbox(self) -> None:
        config = IsolationConfig(
            sandbox=SandboxConfig(enable_weaker_nested_sandbox=True)
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["enableWeakerNestedSandbox"] is True

    def test_enable_weaker_nested_sandbox_default_omitted(self) -> None:
        """When enable_weaker_nested_sandbox is False, it should be omitted."""
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert "enableWeakerNestedSandbox" not in settings["sandbox"]

    def test_ignore_file_violations(self) -> None:
        config = IsolationConfig(
            sandbox=SandboxConfig(ignore_file_violations=("/tmp/noisy", "/var/log"))
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["ignoreViolations"]["file"] == [
                "/tmp/noisy",
                "/var/log",
            ]

    def test_ignore_network_violations(self) -> None:
        config = IsolationConfig(
            sandbox=SandboxConfig(ignore_network_violations=("localhost", "127.0.0.1"))
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["ignoreViolations"]["network"] == [
                "localhost",
                "127.0.0.1",
            ]

    def test_ignore_violations_omitted_when_empty(self) -> None:
        """When no violations are configured, ignoreViolations is omitted."""
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert "ignoreViolations" not in settings["sandbox"]

    def test_network_allow_unix_sockets(self) -> None:
        config = IsolationConfig(
            network_policy=NetworkPolicy(
                allow_unix_sockets=("/tmp/ssh-agent.sock",),
            )
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["network"]["allowUnixSockets"] == [
                "/tmp/ssh-agent.sock",
            ]

    def test_network_allow_all_unix_sockets(self) -> None:
        config = IsolationConfig(
            network_policy=NetworkPolicy(allow_all_unix_sockets=True)
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["network"]["allowAllUnixSockets"] is True

    def test_network_allow_local_binding(self) -> None:
        config = IsolationConfig(network_policy=NetworkPolicy(allow_local_binding=True))
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["network"]["allowLocalBinding"] is True

    def test_network_proxy_ports(self) -> None:
        config = IsolationConfig(
            network_policy=NetworkPolicy(http_proxy_port=8080, socks_proxy_port=1080)
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["network"]["httpProxyPort"] == 8080
            assert settings["sandbox"]["network"]["socksProxyPort"] == 1080

    def test_network_optional_fields_omitted_when_default(self) -> None:
        """Network optional fields should be omitted when using defaults."""
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            network = settings["sandbox"]["network"]
            assert "allowUnixSockets" not in network
            assert "allowAllUnixSockets" not in network
            assert "allowLocalBinding" not in network
            assert "httpProxyPort" not in network
            assert "socksProxyPort" not in network

    def test_env_section_inherits_host_auth_when_no_api_key(self) -> None:
        """Settings should not disable Bedrock when inheriting host auth.

        When no explicit api_key is set, the ephemeral home inherits authentication
        from the host environment. This allows both Bedrock and Anthropic API.
        """
        # Test without Bedrock configured
        with mock.patch.dict(os.environ, {}, clear=True):
            config = IsolationConfig()
            with EphemeralHome(config) as home:
                settings = json.loads(home.settings_path.read_text())
                assert "env" in settings
                # Should NOT have Bedrock setting when not configured in host
                assert "CLAUDE_CODE_USE_BEDROCK" not in settings["env"]
                assert "CLAUDE_USE_BEDROCK" not in settings["env"]
                # Should still disable autoupdater
                assert settings["env"]["DISABLE_AUTOUPDATER"] == "1"

    def test_env_section_enables_bedrock_when_host_has_bedrock(
        self, tmp_path: Path
    ) -> None:
        """Settings should enable Bedrock when host has it configured."""
        # Bedrock requires AWS config to be present
        aws_dir = tmp_path / ".aws"
        aws_dir.mkdir()
        (aws_dir / "config").write_text("[default]\nregion = us-east-1")

        with mock.patch.dict(
            os.environ,
            {
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "AWS_REGION": "us-east-1",
                "HOME": str(tmp_path),
            },
            clear=True,
        ):
            config = IsolationConfig()
            with EphemeralHome(config) as home:
                settings = json.loads(home.settings_path.read_text())
                assert "env" in settings
                # Should enable Bedrock when host has it configured
                assert settings["env"]["CLAUDE_CODE_USE_BEDROCK"] == "1"
                assert settings["env"]["DISABLE_AUTOUPDATER"] == "1"

    def test_env_section_disables_bedrock_with_explicit_api_key(self) -> None:
        """Settings should disable Bedrock when explicit api_key is set.

        When an explicit api_key is provided, Bedrock is disabled to force
        Anthropic API usage with the provided key.
        """
        config = IsolationConfig(api_key="sk-ant-test")
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert "env" in settings
            assert settings["env"]["CLAUDE_CODE_USE_BEDROCK"] == "0"
            assert settings["env"]["CLAUDE_USE_BEDROCK"] == "0"
            assert settings["env"]["DISABLE_AUTOUPDATER"] == "1"


class TestEphemeralHomeEnv:
    def test_home_is_ephemeral_directory(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["HOME"] == home.home_path

    def test_api_key_from_config(self) -> None:
        config = IsolationConfig(api_key="sk-ant-test-key")
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["ANTHROPIC_API_KEY"] == "sk-ant-test-key"

    def test_api_key_from_environment(self) -> None:
        config = IsolationConfig()
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-from-env"}):
            with EphemeralHome(config) as home:
                env = home.get_env()
                assert env["ANTHROPIC_API_KEY"] == "sk-ant-from-env"

    def test_config_api_key_overrides_env(self) -> None:
        config = IsolationConfig(api_key="sk-ant-from-config")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-from-env"}):
            with EphemeralHome(config) as home:
                env = home.get_env()
                assert env["ANTHROPIC_API_KEY"] == "sk-ant-from-config"

    def test_no_api_key_inherits_host_auth(self) -> None:
        """Without explicit api_key, inherits authentication from host environment."""
        config = IsolationConfig()
        # Clear environment to simulate no auth available
        with mock.patch.dict(os.environ, {}, clear=True):
            with EphemeralHome(config) as home:
                env = home.get_env()
                # Should have HOME
                assert "HOME" in env
                # Should have DISABLE_AUTOUPDATER
                assert env["DISABLE_AUTOUPDATER"] == "1"
                # Should NOT have ANTHROPIC_API_KEY (no explicit key, none in env)
                assert "ANTHROPIC_API_KEY" not in env
                # Should NOT disable Bedrock (inheriting host auth)
                assert "CLAUDE_CODE_USE_BEDROCK" not in env
                assert "CLAUDE_USE_BEDROCK" not in env

    def test_custom_env_vars(self) -> None:
        config = IsolationConfig(env={"MY_CUSTOM_VAR": "custom_value"})
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["MY_CUSTOM_VAR"] == "custom_value"

    def test_custom_env_vars_override_generated(self) -> None:
        # Custom env should take precedence over generated values
        config = IsolationConfig(
            api_key="sk-ant-generated",
            env={"ANTHROPIC_API_KEY": "sk-ant-custom-override"},
        )
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["ANTHROPIC_API_KEY"] == "sk-ant-custom-override"

    def test_include_host_env_false_excludes_all(self) -> None:
        """With include_host_env=False, only passes through auth-related vars."""
        config = IsolationConfig(include_host_env=False)
        with mock.patch.dict(
            os.environ,
            {"PATH": "/usr/bin", "MY_VAR": "value", "ANTHROPIC_API_KEY": "key"},
            clear=True,
        ):
            with EphemeralHome(config) as home:
                env = home.get_env()
                # Should have HOME
                assert "HOME" in env
                # Should have DISABLE_AUTOUPDATER
                assert env["DISABLE_AUTOUPDATER"] == "1"
                # Should pass through ANTHROPIC_API_KEY (for inherit host auth mode)
                assert "ANTHROPIC_API_KEY" in env
                # Should NOT have general env vars
                assert "MY_VAR" not in env
                # Should NOT disable Bedrock (inheriting host auth)
                assert "CLAUDE_CODE_USE_BEDROCK" not in env
                # PATH is passed through for finding node/npx
                assert env["PATH"] == "/usr/bin"

    def test_include_host_env_true_copies_safe_vars(self) -> None:
        config = IsolationConfig(include_host_env=True)
        with mock.patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin",
                "MY_VAR": "value",
                "ANTHROPIC_API_KEY": "key",
                "HOME": "/home/user",
                "CLAUDE_CONFIG": "something",
                "AWS_ACCESS_KEY": "secret",
            },
            clear=True,
        ):
            with EphemeralHome(config) as home:
                env = home.get_env()
                # Should include safe vars
                assert env["PATH"] == "/usr/bin"
                assert env["MY_VAR"] == "value"
                # Should NOT include sensitive vars from host
                # (but HOME is overridden and ANTHROPIC_API_KEY is copied)
                assert env["HOME"] == home.home_path  # Overridden
                assert env["ANTHROPIC_API_KEY"] == "key"  # Explicitly copied
                # Should exclude other sensitive prefixes
                assert "CLAUDE_CONFIG" not in env
                assert "AWS_ACCESS_KEY" not in env

    def test_sensitive_prefixes_excluded(self) -> None:
        """Sensitive prefixes are excluded when include_host_env=True.

        Note: In inherit host auth mode, ANTHROPIC_API_KEY and some AWS vars
        are explicitly passed through for authentication.
        """
        config = IsolationConfig(include_host_env=True)
        sensitive_vars = {
            "HOME": "/home/user",
            "CLAUDE_CONFIG_DIR": "/claude",
            "CLAUDE_API_KEY": "key1",
            "ANTHROPIC_API_KEY": "key2",
            "ANTHROPIC_BASE_URL": "url",
            "AWS_SECRET_KEY": "secret",
            "AWS_ACCESS_KEY_ID": "id",
            "AWS_REGION": "us-west-2",
            "GOOGLE_APPLICATION_CREDENTIALS": "creds",
            "GOOGLE_API_KEY": "key",
            "AZURE_CLIENT_SECRET": "secret",
            "OPENAI_API_KEY": "key",
        }
        with mock.patch.dict(os.environ, sensitive_vars, clear=True):
            with EphemeralHome(config) as home:
                env = home.get_env()
                # HOME should be overridden to ephemeral
                assert env["HOME"] == home.home_path
                # ANTHROPIC_API_KEY is explicitly passed through
                assert "ANTHROPIC_API_KEY" in env
                # AWS auth vars are passed through for Bedrock
                assert "AWS_ACCESS_KEY_ID" in env
                assert "AWS_REGION" in env
                # Other sensitive vars should not be inherited
                excluded_vars = [
                    "CLAUDE_CONFIG_DIR",
                    "CLAUDE_API_KEY",
                    "ANTHROPIC_BASE_URL",
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    "GOOGLE_API_KEY",
                    "AZURE_CLIENT_SECRET",
                    "OPENAI_API_KEY",
                ]
                for key in excluded_vars:
                    assert key not in env, f"{key} should be excluded"


class TestEphemeralHomeCustomPrefix:
    def test_custom_prefix(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config, temp_dir_prefix="my-custom-") as home:
            assert "my-custom-" in home.home_path


class TestEphemeralHomeWorkspacePath:
    def test_workspace_path_stored(self) -> None:
        config = IsolationConfig()
        # The workspace_path is currently just stored for potential future use
        home = EphemeralHome(config, workspace_path="/my/workspace")
        try:
            assert home._workspace_path == "/my/workspace"
        finally:
            home.cleanup()

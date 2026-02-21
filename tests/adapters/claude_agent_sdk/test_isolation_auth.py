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

"""Tests for Claude Agent SDK isolation module (auth): AWS config, env, settings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from weakincentives.adapters.claude_agent_sdk._ephemeral_home import (
    EphemeralHome,
    _is_sensitive_key,
)
from weakincentives.adapters.claude_agent_sdk.isolation import (
    IsolationAuthError,
    IsolationConfig,
    _get_effective_env_value,
    _read_host_claude_settings,
)


class TestIsolationConfigAwsConfigPath:
    """Tests for IsolationConfig with aws_config_path."""

    def test_aws_config_path_default_none(self) -> None:
        config = IsolationConfig()
        assert config.aws_config_path is None

    def test_aws_config_path_custom(self) -> None:
        config = IsolationConfig(aws_config_path="/mnt/aws")
        assert config.aws_config_path == "/mnt/aws"


class TestEphemeralHomeInheritHostAuth:
    """Tests for EphemeralHome inheriting host authentication."""

    def test_inherit_host_auth_passes_through_aws_env_vars(
        self, tmp_path: Path
    ) -> None:
        """When no api_key is set, AWS env vars should be passed through."""
        # Bedrock requires AWS config directory
        aws_dir = tmp_path / ".aws"
        aws_dir.mkdir()
        (aws_dir / "config").write_text("[default]\nregion = us-west-2")

        config = IsolationConfig()
        with mock.patch.dict(
            os.environ,
            {
                "AWS_PROFILE": "my-profile",
                "AWS_REGION": "us-west-2",
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "PATH": "/usr/bin",
                "HOME": str(tmp_path),
            },
            clear=True,
        ):
            with EphemeralHome(config) as home:
                env = home.get_env()
                assert env["AWS_PROFILE"] == "my-profile"
                assert env["AWS_REGION"] == "us-west-2"
                assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"
                assert env["PATH"] == "/usr/bin"

    def test_inherit_host_auth_passes_through_anthropic_key(self) -> None:
        """When no api_key is set and ANTHROPIC_API_KEY exists, pass it through."""
        config = IsolationConfig()
        with mock.patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "sk-ant-test", "PATH": "/usr/bin", "HOME": "/home"},
            clear=True,
        ):
            with EphemeralHome(config) as home:
                env = home.get_env()
                assert env["ANTHROPIC_API_KEY"] == "sk-ant-test"

    def test_explicit_api_key_disables_bedrock(self) -> None:
        """When api_key is set, Bedrock should be disabled."""
        config = IsolationConfig(api_key="sk-ant-explicit")
        with mock.patch.dict(
            os.environ,
            {"CLAUDE_CODE_USE_BEDROCK": "1", "HOME": "/home"},
            clear=True,
        ):
            with EphemeralHome(config) as home:
                env = home.get_env()
                assert env["ANTHROPIC_API_KEY"] == "sk-ant-explicit"
                assert env["CLAUDE_CODE_USE_BEDROCK"] == "0"
                assert env["CLAUDE_USE_BEDROCK"] == "0"

    def test_explicit_api_key_does_not_pass_aws_vars(self) -> None:
        """When api_key is set, AWS env vars should not be passed through."""
        config = IsolationConfig(api_key="sk-ant-explicit")
        with mock.patch.dict(
            os.environ,
            {"AWS_PROFILE": "my-profile", "AWS_REGION": "us-west-2", "HOME": "/home"},
            clear=True,
        ):
            with EphemeralHome(config) as home:
                env = home.get_env()
                assert "AWS_PROFILE" not in env
                assert "AWS_REGION" not in env

    def test_explicit_api_key_always_passes_path(self) -> None:
        """PATH is always passed through, even with explicit API key.

        The SDK needs PATH to find node/npx for MCP tools.
        """
        config = IsolationConfig(api_key="sk-ant-explicit")
        with mock.patch.dict(
            os.environ,
            {"PATH": "/usr/bin:/usr/local/bin", "HOME": "/home"},
            clear=True,
        ):
            with EphemeralHome(config) as home:
                env = home.get_env()
                assert env["PATH"] == "/usr/bin:/usr/local/bin"

    def test_inherit_aws_vars_from_host_settings(self, tmp_path: Path) -> None:
        """When AWS vars not in shell env, inherit from host settings.json."""
        # Set up a mock home directory with settings.json
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {
            "env": {
                "AWS_REGION": "eu-west-1",
                "AWS_PROFILE": "host-profile",
                "CLAUDE_CODE_USE_BEDROCK": "1",
            }
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings))

        # Bedrock requires AWS config directory
        aws_dir = tmp_path / ".aws"
        aws_dir.mkdir()
        (aws_dir / "config").write_text("[default]\nregion = eu-west-1")

        config = IsolationConfig()
        with mock.patch.dict(
            os.environ,
            {"HOME": str(tmp_path), "PATH": "/usr/bin"},
            clear=True,
        ):
            with EphemeralHome(config) as home:
                env = home.get_env()
                # Should inherit from host settings.json
                assert env["AWS_REGION"] == "eu-west-1"
                assert env["AWS_PROFILE"] == "host-profile"
                assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"

    def test_shell_env_takes_priority_over_host_settings(self, tmp_path: Path) -> None:
        """Shell env vars should take priority over host settings.json."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {"env": {"AWS_REGION": "eu-west-1"}}
        (claude_dir / "settings.json").write_text(json.dumps(settings))

        config = IsolationConfig()
        with mock.patch.dict(
            os.environ,
            {"HOME": str(tmp_path), "PATH": "/usr/bin", "AWS_REGION": "us-east-1"},
            clear=True,
        ):
            with EphemeralHome(config) as home:
                env = home.get_env()
                # Shell env should win
                assert env["AWS_REGION"] == "us-east-1"

    def test_sensitive_keys_from_host_settings_are_inherited(
        self, tmp_path: Path
    ) -> None:
        """Sensitive AWS keys from host settings.json should be inherited."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        # Include sensitive keys in host settings
        settings = {
            "env": {
                "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
                "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "CLAUDE_CODE_USE_BEDROCK": "1",
            }
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings))

        config = IsolationConfig()
        with mock.patch.dict(
            os.environ,
            {"HOME": str(tmp_path), "PATH": "/usr/bin"},
            clear=True,
        ):
            with EphemeralHome(config) as home:
                env = home.get_env()
                # Sensitive keys should be inherited (values passed through)
                assert env["AWS_ACCESS_KEY_ID"] == "AKIAIOSFODNN7EXAMPLE"
                assert (
                    env["AWS_SECRET_ACCESS_KEY"]
                    == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
                )


class TestReadHostClaudeSettings:
    """Tests for _read_host_claude_settings function."""

    def test_returns_empty_when_no_settings(self, tmp_path: Path) -> None:
        """Returns empty dict when settings.json doesn't exist."""
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
            result = _read_host_claude_settings()
            assert result == {}

    def test_reads_valid_settings(self, tmp_path: Path) -> None:
        """Reads and returns valid settings.json."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {"env": {"FOO": "bar"}, "other": "value"}
        (claude_dir / "settings.json").write_text(json.dumps(settings))

        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
            result = _read_host_claude_settings()
            assert result == settings

    def test_returns_empty_on_invalid_json(self, tmp_path: Path) -> None:
        """Returns empty dict when settings.json has invalid JSON."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text("not valid json {{{")

        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
            result = _read_host_claude_settings()
            assert result == {}


class TestGetEffectiveEnvValue:
    """Tests for _get_effective_env_value function."""

    def test_returns_shell_env_first(self) -> None:
        """Shell environment takes priority."""
        with mock.patch.dict(os.environ, {"TEST_VAR": "from_shell"}, clear=True):
            result = _get_effective_env_value(
                "TEST_VAR", {"env": {"TEST_VAR": "from_host"}}
            )
            assert result == "from_shell"

    def test_falls_back_to_host_settings(self) -> None:
        """Falls back to host settings when not in shell env."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _get_effective_env_value(
                "TEST_VAR", {"env": {"TEST_VAR": "from_host"}}
            )
            assert result == "from_host"

    def test_returns_none_when_not_found(self) -> None:
        """Returns None when var not in shell or host settings."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _get_effective_env_value("TEST_VAR", {"env": {}})
            assert result is None

    def test_loads_host_settings_when_none(self, tmp_path: Path) -> None:
        """Loads host settings when not provided."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {"env": {"LAZY_VAR": "lazy_value"}}
        (claude_dir / "settings.json").write_text(json.dumps(settings))

        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
            result = _get_effective_env_value("LAZY_VAR")  # No host_settings passed
            assert result == "lazy_value"


class TestIsSensitiveKey:
    """Tests for _is_sensitive_key function."""

    def test_detects_secret_keys(self) -> None:
        assert _is_sensitive_key("AWS_SECRET_ACCESS_KEY") is True
        assert _is_sensitive_key("MY_SECRET") is True

    def test_detects_token_keys(self) -> None:
        assert _is_sensitive_key("AWS_SESSION_TOKEN") is True
        assert _is_sensitive_key("AUTH_TOKEN") is True

    def test_detects_access_key_id(self) -> None:
        assert _is_sensitive_key("AWS_ACCESS_KEY_ID") is True

    def test_non_sensitive_keys(self) -> None:
        assert _is_sensitive_key("AWS_REGION") is False
        assert _is_sensitive_key("AWS_PROFILE") is False
        assert _is_sensitive_key("PATH") is False


class TestEphemeralHomeAwsConfigCopy:
    """Tests for AWS config copying in EphemeralHome."""

    def test_copies_aws_config_with_explicit_path(self, tmp_path: Path) -> None:
        """When aws_config_path is set, copies from that path."""
        # Set up source AWS config
        aws_source = tmp_path / "aws_source"
        aws_source.mkdir()
        (aws_source / "config").write_text("[default]\nregion = us-west-2")
        (aws_source / "credentials").write_text("[default]\naws_access_key_id = test")

        config = IsolationConfig(aws_config_path=str(aws_source))
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
            with EphemeralHome(config) as home:
                ephemeral_aws = Path(home.home_path) / ".aws"
                assert ephemeral_aws.exists()
                assert (ephemeral_aws / "config").exists()
                assert (ephemeral_aws / "credentials").exists()

    def test_copies_aws_config_from_home(self, tmp_path: Path) -> None:
        """When no aws_config_path, copies from $HOME/.aws."""
        # Set up AWS config in fake home
        aws_dir = tmp_path / ".aws"
        aws_dir.mkdir()
        (aws_dir / "config").write_text("[default]\nregion = eu-central-1")

        config = IsolationConfig()
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
            with EphemeralHome(config) as home:
                ephemeral_aws = Path(home.home_path) / ".aws"
                assert ephemeral_aws.exists()
                assert (
                    ephemeral_aws / "config"
                ).read_text() == "[default]\nregion = eu-central-1"

    def test_handles_copy_failure_gracefully_without_bedrock(
        self, tmp_path: Path
    ) -> None:
        """When AWS config copy fails without Bedrock, should log warning and continue."""
        import shutil

        # Set up AWS config in fake home
        aws_dir = tmp_path / ".aws"
        aws_dir.mkdir()
        (aws_dir / "config").write_text("[default]\nregion = eu-central-1")

        config = IsolationConfig()
        # No Bedrock configured - failure should be graceful
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
            # Mock copytree to raise OSError
            with mock.patch.object(
                shutil, "copytree", side_effect=OSError("Permission denied")
            ):
                # Should not raise, just log warning
                with EphemeralHome(config) as home:
                    # Ephemeral home should still be created
                    assert Path(home.home_path).exists()

    def test_copy_failure_raises_when_bedrock_enabled(self, tmp_path: Path) -> None:
        """When Bedrock is configured and AWS config copy fails, should raise."""
        import shutil

        # Set up AWS config in fake home
        aws_dir = tmp_path / ".aws"
        aws_dir.mkdir()
        (aws_dir / "config").write_text("[default]\nregion = eu-central-1")

        config = IsolationConfig()
        # Bedrock configured - failure should raise
        with mock.patch.dict(
            os.environ,
            {
                "HOME": str(tmp_path),
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "AWS_REGION": "us-east-1",
            },
            clear=True,
        ):
            # Mock copytree to raise OSError
            with mock.patch.object(
                shutil, "copytree", side_effect=OSError("Permission denied")
            ):
                with pytest.raises(IsolationAuthError) as exc_info:
                    EphemeralHome(config)
                assert "Failed to copy AWS config for Bedrock" in str(exc_info.value)

    def test_no_home_raises_when_bedrock_enabled(self) -> None:
        """When Bedrock is configured but HOME is not set, should raise."""
        config = IsolationConfig()
        with mock.patch.dict(
            os.environ,
            {
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "AWS_REGION": "us-east-1",
            },
            clear=True,  # Removes HOME
        ):
            with pytest.raises(IsolationAuthError) as exc_info:
                EphemeralHome(config)
            assert "HOME not set but Bedrock is configured" in str(exc_info.value)

    def test_missing_aws_dir_raises_when_bedrock_enabled(self, tmp_path: Path) -> None:
        """When Bedrock is configured but ~/.aws doesn't exist, should raise."""
        # No AWS config directory
        config = IsolationConfig()
        with mock.patch.dict(
            os.environ,
            {
                "HOME": str(tmp_path),
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "AWS_REGION": "us-east-1",
            },
            clear=True,
        ):
            with pytest.raises(IsolationAuthError) as exc_info:
                EphemeralHome(config)
            assert "AWS config directory not found" in str(exc_info.value)
            assert "but Bedrock is configured" in str(exc_info.value)


class TestEphemeralHomeSettingsAwsAuthRefresh:
    """Tests for awsAuthRefresh handling in settings generation."""

    def test_copies_aws_auth_refresh_from_host(self, tmp_path: Path) -> None:
        """awsAuthRefresh should be copied from host settings."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {
            "env": {"CLAUDE_CODE_USE_BEDROCK": "1"},
            "awsAuthRefresh": {"command": "aws sso login", "timeout": 300},
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings))

        config = IsolationConfig()
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
            with EphemeralHome(config) as home:
                # Read the generated settings
                generated = json.loads(
                    (Path(home.home_path) / ".claude" / "settings.json").read_text()
                )
                assert "awsAuthRefresh" in generated
                assert generated["awsAuthRefresh"]["command"] == "aws sso login"

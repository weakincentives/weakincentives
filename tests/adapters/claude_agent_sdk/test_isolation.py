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

"""Tests for Claude Agent SDK isolation module."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from weakincentives.adapters.claude_agent_sdk.isolation import (
    BedrockConfig,
    EphemeralHome,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)


class TestNetworkPolicy:
    def test_defaults(self) -> None:
        policy = NetworkPolicy()
        assert policy.allowed_domains == ()

    def test_no_network_factory(self) -> None:
        policy = NetworkPolicy.no_network()
        assert policy.allowed_domains == ()

    def test_with_domains_factory(self) -> None:
        policy = NetworkPolicy.with_domains("api.github.com", "pypi.org")
        assert policy.allowed_domains == ("api.github.com", "pypi.org")


class TestSandboxConfig:
    def test_defaults(self) -> None:
        config = SandboxConfig()
        assert config.enabled is True
        assert config.writable_paths == ()
        assert config.readable_paths == ()
        assert config.excluded_commands == ()
        assert config.allow_unsandboxed_commands is False
        assert config.bash_auto_allow is True

    def test_with_paths(self) -> None:
        config = SandboxConfig(
            writable_paths=("/tmp/output",),
            readable_paths=("/data/readonly",),
        )
        assert config.writable_paths == ("/tmp/output",)
        assert config.readable_paths == ("/data/readonly",)

    def test_with_excluded_commands(self) -> None:
        config = SandboxConfig(
            excluded_commands=("docker", "podman"),
            allow_unsandboxed_commands=True,
        )
        assert config.excluded_commands == ("docker", "podman")
        assert config.allow_unsandboxed_commands is True

    def test_disabled_sandbox(self) -> None:
        config = SandboxConfig(enabled=False, bash_auto_allow=False)
        assert config.enabled is False
        assert config.bash_auto_allow is False


class TestIsolationConfig:
    def test_defaults(self) -> None:
        config = IsolationConfig()
        assert config.network_policy is None
        assert config.sandbox is None
        assert config.env is None
        assert config.api_key is None
        assert config.include_host_env is False

    def test_with_network_policy(self) -> None:
        policy = NetworkPolicy.no_network()
        config = IsolationConfig(network_policy=policy)
        assert config.network_policy is policy

    def test_with_sandbox(self) -> None:
        sandbox = SandboxConfig(enabled=True)
        config = IsolationConfig(sandbox=sandbox)
        assert config.sandbox is sandbox

    def test_with_env(self) -> None:
        config = IsolationConfig(env={"MY_VAR": "value"})
        assert config.env == {"MY_VAR": "value"}

    def test_with_api_key(self) -> None:
        config = IsolationConfig(api_key="sk-ant-test")
        assert config.api_key == "sk-ant-test"

    def test_with_include_host_env(self) -> None:
        config = IsolationConfig(include_host_env=True)
        assert config.include_host_env is True


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
            assert settings["sandbox"]["writablePaths"] == ["/tmp/output", "/var/log"]

    def test_sandbox_readable_paths(self) -> None:
        config = IsolationConfig(
            sandbox=SandboxConfig(readable_paths=("/data/readonly",))
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["readablePaths"] == ["/data/readonly"]


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
        config = IsolationConfig(include_host_env=False)
        with mock.patch.dict(
            os.environ,
            {"PATH": "/usr/bin", "MY_VAR": "value", "ANTHROPIC_API_KEY": "key"},
            clear=True,
        ):
            with EphemeralHome(config) as home:
                env = home.get_env()
                # Should only have HOME and ANTHROPIC_API_KEY (from environ)
                assert "PATH" not in env
                assert "MY_VAR" not in env
                assert "HOME" in env
                assert "ANTHROPIC_API_KEY" in env

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
        config = IsolationConfig(include_host_env=True)
        sensitive_vars = {
            "HOME": "/home/user",
            "CLAUDE_CONFIG_DIR": "/claude",
            "CLAUDE_API_KEY": "key1",
            "ANTHROPIC_API_KEY": "key2",
            "ANTHROPIC_BASE_URL": "url",
            "AWS_SECRET_KEY": "secret",
            "AWS_ACCESS_KEY_ID": "id",
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
                # ANTHROPIC_API_KEY is explicitly copied
                assert "ANTHROPIC_API_KEY" in env
                # Other sensitive vars should not be inherited
                for key in sensitive_vars:
                    if key == "HOME":
                        continue  # Overridden
                    if key == "ANTHROPIC_API_KEY":
                        continue  # Explicitly copied
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


class TestBedrockConfig:
    """Tests for BedrockConfig dataclass."""

    def test_region_required(self) -> None:
        config = BedrockConfig(region="us-east-1")
        assert config.region == "us-east-1"

    def test_static_credentials(self) -> None:
        config = BedrockConfig(
            region="us-east-1",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        assert config.access_key_id == "AKIAIOSFODNN7EXAMPLE"
        assert config.secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert config.session_token is None

    def test_static_credentials_with_session_token(self) -> None:
        config = BedrockConfig(
            region="us-east-1",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            session_token="FwoGZXIvYXdzE...",
        )
        assert config.session_token == "FwoGZXIvYXdzE..."

    def test_profile(self) -> None:
        config = BedrockConfig(
            region="us-west-2",
            profile="bedrock-prod",
        )
        assert config.profile == "bedrock-prod"

    def test_role_assumption(self) -> None:
        config = BedrockConfig(
            region="eu-west-1",
            role_arn="arn:aws:iam::123456789012:role/BedrockRole",
            role_session_name="test-session",
        )
        assert config.role_arn == "arn:aws:iam::123456789012:role/BedrockRole"
        assert config.role_session_name == "test-session"

    def test_role_with_external_id(self) -> None:
        config = BedrockConfig(
            region="us-east-1",
            role_arn="arn:aws:iam::123456789012:role/CrossAccountRole",
            external_id="my-external-id",
        )
        assert config.external_id == "my-external-id"

    def test_web_identity(self) -> None:
        config = BedrockConfig(
            region="us-east-1",
            role_arn="arn:aws:iam::123456789012:role/EKSRole",
            web_identity_token_file="/var/run/secrets/token",
        )
        assert config.web_identity_token_file == "/var/run/secrets/token"

    def test_endpoint_url(self) -> None:
        config = BedrockConfig(
            region="us-east-1",
            endpoint_url="http://localhost:4566",
        )
        assert config.endpoint_url == "http://localhost:4566"


class TestBedrockConfigValidation:
    """Tests for BedrockConfig validation rules."""

    def test_access_key_requires_secret(self) -> None:
        with pytest.raises(
            ValueError, match="Both access_key_id and secret_access_key"
        ):
            BedrockConfig(region="us-east-1", access_key_id="AKIAEXAMPLE")

    def test_secret_requires_access_key(self) -> None:
        with pytest.raises(
            ValueError, match="Both access_key_id and secret_access_key"
        ):
            BedrockConfig(region="us-east-1", secret_access_key="secret")

    def test_session_token_requires_static_credentials(self) -> None:
        with pytest.raises(ValueError, match="session_token requires"):
            BedrockConfig(region="us-east-1", session_token="token")

    def test_web_identity_requires_role_arn(self) -> None:
        with pytest.raises(
            ValueError, match="web_identity_token_file requires role_arn"
        ):
            BedrockConfig(
                region="us-east-1", web_identity_token_file="/var/run/secrets/token"
            )

    def test_external_id_requires_role_arn(self) -> None:
        with pytest.raises(ValueError, match="external_id requires role_arn"):
            BedrockConfig(region="us-east-1", external_id="my-external-id")

    def test_role_session_name_requires_role_arn(self) -> None:
        with pytest.raises(ValueError, match="role_session_name requires role_arn"):
            BedrockConfig(region="us-east-1", role_session_name="my-session")


class TestBedrockConfigFactoryMethods:
    """Tests for BedrockConfig factory methods."""

    def test_from_static_credentials(self) -> None:
        config = BedrockConfig.from_static_credentials(
            region="us-east-1",
            access_key_id="AKIAEXAMPLE",
            secret_access_key="secret",
            session_token="token",
            endpoint_url="http://localhost:4566",
        )
        assert config.region == "us-east-1"
        assert config.access_key_id == "AKIAEXAMPLE"
        assert config.secret_access_key == "secret"
        assert config.session_token == "token"
        assert config.endpoint_url == "http://localhost:4566"

    def test_from_profile(self) -> None:
        config = BedrockConfig.from_profile(
            region="us-west-2",
            profile="my-profile",
            endpoint_url="http://localhost:4566",
        )
        assert config.region == "us-west-2"
        assert config.profile == "my-profile"
        assert config.endpoint_url == "http://localhost:4566"

    def test_from_role(self) -> None:
        config = BedrockConfig.from_role(
            region="eu-west-1",
            role_arn="arn:aws:iam::123456789012:role/MyRole",
            role_session_name="my-session",
            external_id="ext-id",
            endpoint_url="http://localhost:4566",
        )
        assert config.region == "eu-west-1"
        assert config.role_arn == "arn:aws:iam::123456789012:role/MyRole"
        assert config.role_session_name == "my-session"
        assert config.external_id == "ext-id"

    def test_from_web_identity(self) -> None:
        config = BedrockConfig.from_web_identity(
            region="us-east-1",
            role_arn="arn:aws:iam::123456789012:role/OIDCRole",
            web_identity_token_file="/var/run/secrets/token",
            role_session_name="oidc-session",
        )
        assert config.region == "us-east-1"
        assert config.role_arn == "arn:aws:iam::123456789012:role/OIDCRole"
        assert config.web_identity_token_file == "/var/run/secrets/token"
        assert config.role_session_name == "oidc-session"

    def test_from_environment(self) -> None:
        config = BedrockConfig.from_environment(region="us-east-1")
        assert config.region == "us-east-1"
        assert config.access_key_id is None
        assert config.profile is None
        assert config.role_arn is None


class TestIsolationConfigWithBedrock:
    """Tests for IsolationConfig with BedrockConfig."""

    def test_bedrock_field(self) -> None:
        bedrock = BedrockConfig(region="us-east-1")
        config = IsolationConfig(bedrock=bedrock)
        assert config.bedrock is bedrock
        assert config.api_key is None

    def test_api_key_and_bedrock_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both api_key and bedrock"):
            IsolationConfig(
                api_key="sk-ant-test",
                bedrock=BedrockConfig(region="us-east-1"),
            )


class TestEphemeralHomeBedrockEnv:
    """Tests for EphemeralHome environment generation with Bedrock."""

    def test_bedrock_sets_region(self) -> None:
        config = IsolationConfig(bedrock=BedrockConfig(region="us-east-1"))
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["AWS_REGION"] == "us-east-1"
            assert env["AWS_DEFAULT_REGION"] == "us-east-1"

    def test_bedrock_static_credentials(self) -> None:
        config = IsolationConfig(
            bedrock=BedrockConfig(
                region="us-east-1",
                access_key_id="AKIAEXAMPLE",
                secret_access_key="secret",
            )
        )
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["AWS_ACCESS_KEY_ID"] == "AKIAEXAMPLE"
            assert env["AWS_SECRET_ACCESS_KEY"] == "secret"
            assert "AWS_SESSION_TOKEN" not in env

    def test_bedrock_static_credentials_with_session_token(self) -> None:
        config = IsolationConfig(
            bedrock=BedrockConfig(
                region="us-east-1",
                access_key_id="AKIAEXAMPLE",
                secret_access_key="secret",
                session_token="token123",
            )
        )
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["AWS_SESSION_TOKEN"] == "token123"

    def test_bedrock_profile(self) -> None:
        config = IsolationConfig(
            bedrock=BedrockConfig(region="us-west-2", profile="my-profile")
        )
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["AWS_PROFILE"] == "my-profile"
            assert "AWS_ACCESS_KEY_ID" not in env

    def test_bedrock_role_arn(self) -> None:
        config = IsolationConfig(
            bedrock=BedrockConfig(
                region="eu-west-1",
                role_arn="arn:aws:iam::123456789012:role/MyRole",
            )
        )
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["AWS_ROLE_ARN"] == "arn:aws:iam::123456789012:role/MyRole"

    def test_bedrock_role_session_name(self) -> None:
        config = IsolationConfig(
            bedrock=BedrockConfig(
                region="us-east-1",
                role_arn="arn:aws:iam::123456789012:role/MyRole",
                role_session_name="my-session",
            )
        )
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["AWS_ROLE_SESSION_NAME"] == "my-session"

    def test_bedrock_external_id(self) -> None:
        config = IsolationConfig(
            bedrock=BedrockConfig(
                region="us-east-1",
                role_arn="arn:aws:iam::123456789012:role/MyRole",
                external_id="ext-123",
            )
        )
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["AWS_EXTERNAL_ID"] == "ext-123"

    def test_bedrock_web_identity(self) -> None:
        config = IsolationConfig(
            bedrock=BedrockConfig(
                region="us-east-1",
                role_arn="arn:aws:iam::123456789012:role/OIDCRole",
                web_identity_token_file="/var/run/secrets/token",
            )
        )
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["AWS_WEB_IDENTITY_TOKEN_FILE"] == "/var/run/secrets/token"

    def test_bedrock_endpoint_url(self) -> None:
        config = IsolationConfig(
            bedrock=BedrockConfig(
                region="us-east-1",
                endpoint_url="http://localhost:4566",
            )
        )
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["AWS_ENDPOINT_URL_BEDROCK_RUNTIME"] == "http://localhost:4566"

    def test_bedrock_does_not_set_anthropic_api_key(self) -> None:
        config = IsolationConfig(bedrock=BedrockConfig(region="us-east-1"))
        # Clear the environment to ensure no fallback
        with mock.patch.dict(os.environ, {}, clear=True):
            with EphemeralHome(config) as home:
                env = home.get_env()
                assert "ANTHROPIC_API_KEY" not in env

    def test_bedrock_custom_env_overrides(self) -> None:
        config = IsolationConfig(
            bedrock=BedrockConfig(region="us-east-1"),
            env={"AWS_REGION": "us-west-2"},
        )
        with EphemeralHome(config) as home:
            env = home.get_env()
            # Custom env should override generated values
            assert env["AWS_REGION"] == "us-west-2"

    def test_bedrock_full_configuration(self) -> None:
        """Test a complete Bedrock configuration with multiple fields."""
        config = IsolationConfig(
            bedrock=BedrockConfig(
                region="us-east-1",
                access_key_id="AKIAEXAMPLE",
                secret_access_key="secret",
                session_token="token",
                endpoint_url="http://localhost:4566",
            ),
            network_policy=NetworkPolicy.no_network(),
            sandbox=SandboxConfig(enabled=True),
        )
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["AWS_REGION"] == "us-east-1"
            assert env["AWS_DEFAULT_REGION"] == "us-east-1"
            assert env["AWS_ACCESS_KEY_ID"] == "AKIAEXAMPLE"
            assert env["AWS_SECRET_ACCESS_KEY"] == "secret"
            assert env["AWS_SESSION_TOKEN"] == "token"
            assert env["AWS_ENDPOINT_URL_BEDROCK_RUNTIME"] == "http://localhost:4566"
            assert env["HOME"] == home.home_path
            # Should not have Anthropic API key
            assert "ANTHROPIC_API_KEY" not in env

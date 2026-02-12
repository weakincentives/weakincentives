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

from weakincentives.adapters.claude_agent_sdk._ephemeral_home import (
    AwsConfigResolution,
    _copy_skill,
    _is_sensitive_key,
)
from weakincentives.adapters.claude_agent_sdk.isolation import (
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_MODEL,
    AuthMode,
    BedrockConfig,
    EphemeralHome,
    IsolationAuthError,
    IsolationConfig,
    IsolationOptions,
    NetworkPolicy,
    SandboxConfig,
    _get_effective_env_value,
    _read_host_claude_settings,
    get_default_model,
    get_supported_bedrock_models,
    to_anthropic_model_name,
    to_bedrock_model_id,
)
from weakincentives.skills import (
    SkillMount,
    SkillMountError,
    SkillNotFoundError,
    SkillValidationError,
    resolve_skill_name,
    validate_skill,
    validate_skill_name,
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


class TestIsolationConfigFactoryMethods:
    """Tests for IsolationConfig factory methods with validation."""

    def test_inherit_host_auth_with_anthropic_key(self) -> None:
        """inherit_host_auth succeeds when ANTHROPIC_API_KEY is set."""
        with mock.patch.dict(
            os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True
        ):
            config = IsolationConfig.inherit_host_auth()
            assert config.api_key is None  # Inherits from env, not stored

    def test_inherit_host_auth_with_bedrock(self) -> None:
        """inherit_host_auth succeeds when Bedrock is configured."""
        with mock.patch.dict(
            os.environ,
            {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "us-east-1"},
            clear=True,
        ):
            config = IsolationConfig.inherit_host_auth()
            assert config.api_key is None

    def test_inherit_host_auth_fails_without_auth(self) -> None:
        """inherit_host_auth raises IsolationAuthError when no auth is configured."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(IsolationAuthError) as exc_info:
                IsolationConfig.inherit_host_auth()
            assert "No authentication configured" in str(exc_info.value)
            assert "ANTHROPIC_API_KEY" in str(exc_info.value)
            assert "CLAUDE_CODE_USE_BEDROCK" in str(exc_info.value)

    def test_inherit_host_auth_passes_options(self) -> None:
        """inherit_host_auth passes through optional parameters."""
        policy = NetworkPolicy.no_network()
        sandbox = SandboxConfig(enabled=True)
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}, clear=True):
            config = IsolationConfig.inherit_host_auth(
                network_policy=policy,
                sandbox=sandbox,
                include_host_env=True,
            )
            assert config.network_policy is policy
            assert config.sandbox is sandbox
            assert config.include_host_env is True

    def test_with_api_key_succeeds(self) -> None:
        """with_api_key creates config with explicit key."""
        config = IsolationConfig.with_api_key("sk-ant-test-key")
        assert config.api_key == "sk-ant-test-key"

    def test_with_api_key_fails_on_empty(self) -> None:
        """with_api_key raises IsolationAuthError for empty key."""
        with pytest.raises(IsolationAuthError) as exc_info:
            IsolationConfig.with_api_key("")
        assert "api_key is required" in str(exc_info.value)

    def test_with_api_key_passes_options(self) -> None:
        """with_api_key passes through optional parameters."""
        policy = NetworkPolicy.no_network()
        options = IsolationOptions(network_policy=policy, include_host_env=True)
        config = IsolationConfig.with_api_key(
            "sk-ant-key",
            options=options,
        )
        assert config.api_key == "sk-ant-key"
        assert config.network_policy is policy
        assert config.include_host_env is True

    def test_for_bedrock_succeeds_when_configured(self) -> None:
        """for_bedrock succeeds when Bedrock env vars are set."""
        with mock.patch.dict(
            os.environ,
            {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "us-west-2"},
            clear=True,
        ):
            config = IsolationConfig.for_bedrock()
            assert config.api_key is None
            assert config.aws_config_path is None

    def test_for_bedrock_with_aws_config_path(self) -> None:
        """for_bedrock accepts custom aws_config_path for Docker."""
        with mock.patch.dict(
            os.environ,
            {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "us-east-1"},
            clear=True,
        ):
            config = IsolationConfig.for_bedrock(aws_config_path="/mnt/aws")
            assert config.aws_config_path == "/mnt/aws"

    def test_for_bedrock_fails_without_bedrock_flag(self) -> None:
        """for_bedrock raises IsolationAuthError without CLAUDE_CODE_USE_BEDROCK."""
        with mock.patch.dict(os.environ, {"AWS_REGION": "us-east-1"}, clear=True):
            with pytest.raises(IsolationAuthError) as exc_info:
                IsolationConfig.for_bedrock()
            assert "Bedrock authentication not configured" in str(exc_info.value)

    def test_for_bedrock_fails_without_region(self) -> None:
        """for_bedrock raises IsolationAuthError without AWS_REGION."""
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_USE_BEDROCK": "1"}, clear=True):
            with pytest.raises(IsolationAuthError) as exc_info:
                IsolationConfig.for_bedrock()
            assert "AWS_REGION" in str(exc_info.value)

    def test_for_bedrock_passes_options(self) -> None:
        """for_bedrock passes through optional parameters."""
        sandbox = SandboxConfig(enabled=False)
        options = IsolationOptions(sandbox=sandbox, include_host_env=True)
        with mock.patch.dict(
            os.environ,
            {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "eu-west-1"},
            clear=True,
        ):
            config = IsolationConfig.for_bedrock(
                options=options,
            )
            assert config.sandbox is sandbox
            assert config.include_host_env is True


class TestModelIdFunctions:
    """Tests for model ID conversion and default model functions."""

    def test_default_model_constants(self) -> None:
        """Default model constants should be Opus 4.6."""
        assert "opus" in DEFAULT_MODEL.lower()
        assert "opus" in DEFAULT_BEDROCK_MODEL.lower()

    def test_get_default_model_anthropic(self) -> None:
        """get_default_model returns Anthropic format when not using Bedrock."""
        with mock.patch.dict(os.environ, {}, clear=True):
            model = get_default_model()
            assert model == DEFAULT_MODEL
            assert not model.startswith("us.")

    def test_get_default_model_bedrock(self) -> None:
        """get_default_model returns Bedrock format when Bedrock is configured."""
        with mock.patch.dict(
            os.environ,
            {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "us-east-1"},
            clear=True,
        ):
            model = get_default_model()
            assert model == DEFAULT_BEDROCK_MODEL
            assert model.startswith("us.")

    def test_to_bedrock_model_id_converts_known(self) -> None:
        """to_bedrock_model_id converts known Anthropic models."""
        assert to_bedrock_model_id("claude-opus-4-6").startswith("us.")
        assert "opus" in to_bedrock_model_id("claude-opus-4-6").lower()

    def test_to_bedrock_model_id_passes_through_bedrock(self) -> None:
        """to_bedrock_model_id passes through existing Bedrock IDs."""
        bedrock_id = "us.anthropic.claude-opus-4-6-v1"
        assert to_bedrock_model_id(bedrock_id) == bedrock_id

    def test_to_bedrock_model_id_passes_through_unknown(self) -> None:
        """to_bedrock_model_id passes through unknown models unchanged."""
        unknown = "some-unknown-model"
        assert to_bedrock_model_id(unknown) == unknown

    def test_to_anthropic_model_name_converts_known(self) -> None:
        """to_anthropic_model_name converts known Bedrock IDs."""
        result = to_anthropic_model_name("us.anthropic.claude-opus-4-6-v1")
        assert result == "claude-opus-4-6"

    def test_to_anthropic_model_name_passes_through_anthropic(self) -> None:
        """to_anthropic_model_name passes through existing Anthropic names."""
        anthropic_name = "claude-opus-4-6"
        assert to_anthropic_model_name(anthropic_name) == anthropic_name

    def test_to_anthropic_model_name_passes_through_unknown(self) -> None:
        """to_anthropic_model_name passes through unknown Bedrock IDs."""
        unknown = "us.anthropic.some-unknown-model-v1:0"
        assert to_anthropic_model_name(unknown) == unknown

    def test_get_supported_bedrock_models(self) -> None:
        """get_supported_bedrock_models returns the model mapping."""
        models = get_supported_bedrock_models()
        assert isinstance(models, dict)
        assert "claude-opus-4-6" in models
        assert models["claude-opus-4-6"].startswith("us.")


class TestAuthMode:
    """Tests for the AuthMode enum."""

    def test_enum_values(self) -> None:
        """AuthMode has expected values."""
        assert AuthMode.INHERIT_HOST.value == "inherit_host"
        assert AuthMode.EXPLICIT_API_KEY.value == "explicit_api_key"
        assert AuthMode.ANTHROPIC_API.value == "anthropic_api"
        assert AuthMode.BEDROCK.value == "bedrock"

    def test_enum_members(self) -> None:
        """AuthMode has all expected members."""
        members = list(AuthMode)
        assert len(members) == 4
        assert AuthMode.INHERIT_HOST in members
        assert AuthMode.EXPLICIT_API_KEY in members
        assert AuthMode.ANTHROPIC_API in members
        assert AuthMode.BEDROCK in members


class TestBedrockConfig:
    """Tests for the BedrockConfig dataclass."""

    def test_from_environment_with_bedrock(self) -> None:
        """BedrockConfig.from_environment detects Bedrock configuration."""
        with mock.patch.dict(
            os.environ,
            {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "us-west-2"},
            clear=True,
        ):
            config = BedrockConfig.from_environment()
            assert config is not None
            assert config.region == "us-west-2"
            assert config.source == "shell"

    def test_from_environment_with_profile(self) -> None:
        """BedrockConfig.from_environment includes AWS profile."""
        with mock.patch.dict(
            os.environ,
            {
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "AWS_REGION": "us-east-1",
                "AWS_PROFILE": "my-profile",
            },
            clear=True,
        ):
            config = BedrockConfig.from_environment()
            assert config is not None
            assert config.profile == "my-profile"

    def test_from_environment_without_bedrock(self) -> None:
        """BedrockConfig.from_environment returns None without Bedrock config."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = BedrockConfig.from_environment()
            assert config is None

    def test_from_environment_missing_region(self) -> None:
        """BedrockConfig.from_environment returns None without AWS_REGION."""
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_USE_BEDROCK": "1"}, clear=True):
            config = BedrockConfig.from_environment()
            assert config is None

    def test_from_environment_from_host_settings(self, tmp_path: Path) -> None:
        """BedrockConfig.from_environment reads from host settings.json."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings = {
            "env": {
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "AWS_REGION": "eu-west-1",
            }
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings))

        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True):
            config = BedrockConfig.from_environment()
            assert config is not None
            assert config.region == "eu-west-1"
            assert config.source == "host_settings"


class TestAwsConfigResolution:
    """Tests for the AwsConfigResolution named tuple."""

    def test_with_path(self) -> None:
        """AwsConfigResolution with a valid path."""
        resolution = AwsConfigResolution(path=Path("/home/user/.aws"), skip_reason=None)
        assert resolution.path == Path("/home/user/.aws")
        assert resolution.skip_reason is None

    def test_without_path(self) -> None:
        """AwsConfigResolution without a path."""
        resolution = AwsConfigResolution(path=None, skip_reason="HOME_not_set")
        assert resolution.path is None
        assert resolution.skip_reason == "HOME_not_set"


class TestIsolationConfigForAnthropicApi:
    """Tests for IsolationConfig.for_anthropic_api() factory method."""

    def test_for_anthropic_api_succeeds_with_key(self) -> None:
        """for_anthropic_api succeeds when ANTHROPIC_API_KEY is set."""
        with mock.patch.dict(
            os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True
        ):
            config = IsolationConfig.for_anthropic_api()
            assert config.api_key is None  # Inherits from env, not explicit

    def test_for_anthropic_api_fails_without_key(self) -> None:
        """for_anthropic_api raises error when ANTHROPIC_API_KEY is not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(IsolationAuthError) as exc_info:
                IsolationConfig.for_anthropic_api()
            assert "ANTHROPIC_API_KEY environment variable is not set" in str(
                exc_info.value
            )

    def test_for_anthropic_api_passes_options(self) -> None:
        """for_anthropic_api passes through optional parameters."""
        policy = NetworkPolicy.no_network()
        sandbox = SandboxConfig(enabled=True)
        with mock.patch.dict(
            os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True
        ):
            config = IsolationConfig.for_anthropic_api(
                network_policy=policy,
                sandbox=sandbox,
                include_host_env=True,
            )
            assert config.network_policy is policy
            assert config.sandbox is sandbox
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


class TestSkillMount:
    def test_defaults(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill"
        source.mkdir()
        mount = SkillMount(source=source)
        assert mount.source == source
        assert mount.name is None

    def test_with_name(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill"
        source.mkdir()
        mount = SkillMount(source=source, name="custom-name")
        assert mount.name == "custom-name"


class TestResolveSkillName:
    def test_explicit_name(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill"
        source.mkdir()
        mount = SkillMount(source=source, name="explicit")
        assert resolve_skill_name(mount) == "explicit"

    def test_directory_name(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill-dir"
        source.mkdir()
        mount = SkillMount(source=source)
        assert resolve_skill_name(mount) == "my-skill-dir"

    def test_file_name_strips_extension(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill.md"
        source.write_text("# Test")
        mount = SkillMount(source=source)
        assert resolve_skill_name(mount) == "my-skill"


class TestValidateSkillName:
    def test_valid_name(self) -> None:
        validate_skill_name("my-skill")
        validate_skill_name("skill2")
        validate_skill_name("123-test")

    def test_rejects_forward_slash(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("path/traversal")

    def test_rejects_backslash(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("path\\traversal")

    def test_rejects_double_dot(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("..evil")

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(SkillMountError, match="cannot be empty"):
            validate_skill_name("")

    def test_rejects_dot(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name(".")


class TestValidateSkill:
    def test_valid_directory_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: test-skill\n"
            "description: A test skill for validation\n"
            "---\n"
            "\n"
            "# Test Skill\n\nContent"
        )
        validate_skill(skill_dir)  # Should not raise

    def test_directory_missing_skill_md(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        with pytest.raises(SkillValidationError, match=r"missing SKILL\.md"):
            validate_skill(skill_dir)

    def test_valid_file_skill(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "test-skill.md"
        skill_file.write_text(
            "---\n"
            "name: test-skill\n"
            "description: A test skill for validation\n"
            "---\n"
            "\n"
            "# Test Skill\n\nContent"
        )
        validate_skill(skill_file)  # Should not raise

    def test_file_wrong_extension(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "test-skill.txt"
        skill_file.write_text("# Test Skill\n\nContent")
        with pytest.raises(SkillValidationError, match="must be markdown"):
            validate_skill(skill_file)

    def test_file_too_large(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "huge-skill.md"
        # Create a file larger than 1 MiB
        skill_file.write_text("x" * (1024 * 1024 + 1))
        with pytest.raises(SkillValidationError, match="exceeds size limit"):
            validate_skill(skill_file)


class TestCopySkill:
    def test_copy_directory_skill(self, tmp_path: Path) -> None:
        # Create source skill directory
        source = tmp_path / "source-skill"
        source.mkdir()
        skill_content = (
            "---\n"
            "name: source-skill\n"
            "description: A test skill for copying\n"
            "---\n"
            "\n"
            "# Test Skill"
        )
        (source / "SKILL.md").write_text(skill_content)
        (source / "examples").mkdir()
        (source / "examples" / "example.py").write_text("print('hello')")

        dest = tmp_path / "dest-skill"
        bytes_copied = _copy_skill(source, dest)

        assert dest.is_dir()
        assert (dest / "SKILL.md").read_text() == skill_content
        assert (dest / "examples" / "example.py").read_text() == "print('hello')"
        assert bytes_copied > 0

    def test_copy_file_skill_wraps_in_directory(self, tmp_path: Path) -> None:
        # Create source skill file
        source = tmp_path / "skill.md"
        skill_content = (
            "---\n"
            "name: skill\n"
            "description: A single file test skill\n"
            "---\n"
            "\n"
            "# Single File Skill"
        )
        source.write_text(skill_content)

        dest = tmp_path / "dest-skill"
        bytes_copied = _copy_skill(source, dest)

        assert dest.is_dir()
        assert (dest / "SKILL.md").read_text() == skill_content
        assert bytes_copied > 0

    def test_copy_directory_exceeds_size_limit(self, tmp_path: Path) -> None:
        # Create large skill directory
        source = tmp_path / "large-skill"
        source.mkdir()
        (source / "SKILL.md").write_text(
            "---\n"
            "name: large-skill\n"
            "description: A large test skill\n"
            "---\n"
            "\n"
            "# Large Skill"
        )
        (source / "big_file.txt").write_text("x" * 100)

        dest = tmp_path / "dest-skill"
        # Use a very small limit to trigger the error
        with pytest.raises(SkillMountError, match="exceeds total size limit"):
            _copy_skill(source, dest, max_total_bytes=10)

    def test_copy_file_exceeds_size_limit(self, tmp_path: Path) -> None:
        # Create large single-file skill
        source = tmp_path / "large-skill.md"
        source.write_text(
            "---\n"
            "name: large-skill\n"
            "description: A large single file skill\n"
            "---\n"
            "\n"
            "# Large Skill\n" + "x" * 100
        )

        dest = tmp_path / "dest-skill"
        # Use a very small limit to trigger the error
        with pytest.raises(SkillMountError, match="exceeds total size limit"):
            _copy_skill(source, dest, max_total_bytes=10)

    def test_copy_ignores_symlinks_by_default(self, tmp_path: Path) -> None:
        # Create source skill directory with symlink
        source = tmp_path / "source-skill"
        source.mkdir()
        (source / "SKILL.md").write_text(
            "---\n"
            "name: source-skill\n"
            "description: A test skill with symlinks\n"
            "---\n"
            "\n"
            "# Test Skill"
        )
        external_file = tmp_path / "external.txt"
        external_file.write_text("external content")
        (source / "link.txt").symlink_to(external_file)

        dest = tmp_path / "dest-skill"
        _copy_skill(source, dest, follow_symlinks=False)

        assert dest.is_dir()
        assert (dest / "SKILL.md").exists()
        assert not (dest / "link.txt").exists()  # Symlink should be skipped

    def test_copy_raises_on_io_error(self, tmp_path: Path) -> None:
        # Create a source file
        source = tmp_path / "skill.md"
        source.write_text(
            "---\n"
            "name: skill\n"
            "description: A test skill for error handling\n"
            "---\n"
            "\n"
            "# Test Skill"
        )

        dest = tmp_path / "dest-skill"

        # Mock shutil.copy2 to raise OSError
        with mock.patch("shutil.copy2", side_effect=OSError("Disk full")):
            with pytest.raises(SkillMountError, match="Failed to copy skill"):
                _copy_skill(source, dest)


class TestEphemeralHomeMountSkills:
    """Tests for EphemeralHome.mount_skills() method."""

    def test_mounts_directory_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: my-skill\n"
            "description: A test skill for mounting\n"
            "---\n"
            "\n"
            "# My Skill\n"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=skill_dir),))
            assert home.skills_dir.is_dir()
            skill_dest = home.skills_dir / "my-skill"
            assert skill_dest.is_dir()
            content = (skill_dest / "SKILL.md").read_text()
            assert "# My Skill" in content

    def test_mounts_file_skill(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "my-skill.md"
        skill_file.write_text(
            "---\n"
            "name: my-skill\n"
            "description: A file-based test skill\n"
            "---\n"
            "\n"
            "# File Skill\n"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=skill_file),))
            skill_dest = home.skills_dir / "my-skill"
            assert skill_dest.is_dir()
            content = (skill_dest / "SKILL.md").read_text()
            assert "# File Skill" in content

    def test_mounts_skill_with_custom_name(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "original-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: original-name\n"
            "description: A skill with a custom mount name\n"
            "---\n"
            "\n"
            "# Custom Named\n"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=skill_dir, name="custom-name"),))
            assert (home.skills_dir / "custom-name").is_dir()
            assert not (home.skills_dir / "original-name").exists()

    def test_mounts_multiple_skills(self, tmp_path: Path) -> None:
        # Create two skills
        skill1 = tmp_path / "skill-one"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text(
            "---\nname: skill-one\ndescription: First test skill\n---\n\n# Skill One\n"
        )

        skill2 = tmp_path / "skill-two"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text(
            "---\nname: skill-two\ndescription: Second test skill\n---\n\n# Skill Two\n"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home.mount_skills(
                (
                    SkillMount(source=skill1),
                    SkillMount(source=skill2),
                )
            )
            assert (home.skills_dir / "skill-one").is_dir()
            assert (home.skills_dir / "skill-two").is_dir()

    def test_rejects_duplicate_skill_names(self, tmp_path: Path) -> None:
        skill1 = tmp_path / "skill-a"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: First skill\n---\n\n# Skill A\n"
        )

        skill2 = tmp_path / "skill-b"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text(
            "---\nname: skill-b\ndescription: Second skill\n---\n\n# Skill B\n"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            with pytest.raises(SkillMountError, match="Duplicate skill name"):
                home.mount_skills(
                    (
                        SkillMount(source=skill1, name="same-name"),
                        SkillMount(source=skill2, name="same-name"),
                    )
                )

    def test_raises_on_missing_skill_source(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does-not-exist"
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            with pytest.raises(SkillNotFoundError, match="Skill not found"):
                home.mount_skills((SkillMount(source=nonexistent),))

    def test_validates_skill_when_enabled(self, tmp_path: Path) -> None:
        # Directory without SKILL.md
        invalid_skill = tmp_path / "invalid-skill"
        invalid_skill.mkdir()

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            with pytest.raises(SkillValidationError, match=r"missing SKILL\.md"):
                home.mount_skills((SkillMount(source=invalid_skill),), validate=True)

    def test_skips_validation_when_disabled(self, tmp_path: Path) -> None:
        # Directory without SKILL.md (would fail validation)
        invalid_skill = tmp_path / "invalid-skill"
        invalid_skill.mkdir()
        # Create some content to copy
        (invalid_skill / "README.md").write_text("# Not a skill")

        config = IsolationConfig()
        # Should not raise because validation is disabled
        with EphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=invalid_skill),), validate=False)
            assert (home.skills_dir / "invalid-skill").is_dir()

    def test_no_skills_directory_when_no_skills_mounted(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            # skills_dir property should return path but dir shouldn't exist
            assert home.skills_dir == home.claude_dir / "skills"
            assert not home.skills_dir.exists()

    def test_empty_skills_tuple_does_nothing(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home.mount_skills(())  # Empty tuple
            assert home.skills_dir == home.claude_dir / "skills"
            assert not home.skills_dir.exists()

    def test_skills_dir_property(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=skill_dir),))
            assert home.skills_dir == home.claude_dir / "skills"
            assert home.skills_dir.is_dir()

    def test_mount_skills_rejects_second_call(self, tmp_path: Path) -> None:
        """mount_skills() can only be called once per EphemeralHome instance."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            # First call succeeds
            home.mount_skills((SkillMount(source=skill_dir),))

            # Second call raises
            with pytest.raises(SkillMountError, match="Skills already mounted"):
                home.mount_skills((SkillMount(source=skill_dir),))

    def test_mount_skills_rejects_second_call_even_with_empty_first(self) -> None:
        """mount_skills() can only be called once even if first call was empty."""
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            # First call with empty tuple succeeds
            home.mount_skills(())

            # Second call raises even though first was empty
            with pytest.raises(SkillMountError, match="Skills already mounted"):
                home.mount_skills(())


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

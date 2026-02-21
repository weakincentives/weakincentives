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

"""Tests for isolation config classes, factory methods, auth mode, and model IDs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from weakincentives.adapters.claude_agent_sdk._ephemeral_home import (
    AwsConfigResolution,
)
from weakincentives.adapters.claude_agent_sdk._model_utils import (
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_MODEL,
    get_supported_bedrock_models,
    to_anthropic_model_name,
    to_bedrock_model_id,
)
from weakincentives.adapters.claude_agent_sdk.isolation import (
    AuthMode,
    BedrockConfig,
    IsolationAuthError,
    IsolationConfig,
    IsolationOptions,
    NetworkPolicy,
    SandboxConfig,
    get_default_model,
)


class TestNetworkPolicy:
    def test_defaults(self) -> None:
        policy = NetworkPolicy()
        assert policy.allowed_domains == ()
        assert policy.allow_unix_sockets == ()
        assert policy.allow_all_unix_sockets is False
        assert policy.allow_local_binding is False
        assert policy.http_proxy_port is None
        assert policy.socks_proxy_port is None

    def test_no_network_factory(self) -> None:
        policy = NetworkPolicy.no_network()
        assert policy.allowed_domains == ()

    def test_with_domains_factory(self) -> None:
        policy = NetworkPolicy.with_domains("api.github.com", "pypi.org")
        assert policy.allowed_domains == ("api.github.com", "pypi.org")

    def test_with_unix_sockets(self) -> None:
        policy = NetworkPolicy(
            allow_unix_sockets=("/tmp/ssh-agent.sock",),
        )
        assert policy.allow_unix_sockets == ("/tmp/ssh-agent.sock",)

    def test_with_all_unix_sockets(self) -> None:
        policy = NetworkPolicy(allow_all_unix_sockets=True)
        assert policy.allow_all_unix_sockets is True

    def test_with_local_binding(self) -> None:
        policy = NetworkPolicy(allow_local_binding=True)
        assert policy.allow_local_binding is True

    def test_with_proxy_ports(self) -> None:
        policy = NetworkPolicy(http_proxy_port=8080, socks_proxy_port=1080)
        assert policy.http_proxy_port == 8080
        assert policy.socks_proxy_port == 1080


class TestSandboxConfig:
    def test_defaults(self) -> None:
        config = SandboxConfig()
        assert config.enabled is True
        assert config.writable_paths == ()
        assert config.readable_paths == ()
        assert config.excluded_commands == ()
        assert config.allow_unsandboxed_commands is False
        assert config.bash_auto_allow is True
        assert config.enable_weaker_nested_sandbox is False
        assert config.ignore_file_violations == ()
        assert config.ignore_network_violations == ()

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

    def test_enable_weaker_nested_sandbox(self) -> None:
        config = SandboxConfig(enable_weaker_nested_sandbox=True)
        assert config.enable_weaker_nested_sandbox is True

    def test_with_ignore_violations(self) -> None:
        config = SandboxConfig(
            ignore_file_violations=("/tmp/noisy",),
            ignore_network_violations=("localhost",),
        )
        assert config.ignore_file_violations == ("/tmp/noisy",)
        assert config.ignore_network_violations == ("localhost",)


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

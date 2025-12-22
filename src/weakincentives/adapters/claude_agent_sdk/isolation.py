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

"""Hermetic isolation support for Claude Agent SDK adapter.

This module provides configuration and runtime support for fully isolated
SDK execution that doesn't interact with the host's ~/.claude configuration.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...dataclasses import FrozenDataclass

__all__ = [
    "BedrockConfig",
    "EphemeralHome",
    "IsolationConfig",
    "NetworkPolicy",
    "SandboxConfig",
]


@FrozenDataclass()
class NetworkPolicy:
    """Network access policy for Claude Agent SDK sandbox.

    Controls which network resources tools can access during execution.
    All constraints are enforced at the OS level via bubblewrap (Linux) or
    seatbelt (macOS).

    Note: This policy only affects tools making outbound network connections
    (e.g., curl, wget). The MCP bridge for custom weakincentives tools and
    the Claude API connection are not affected by this policy.

    Attributes:
        allowed_domains: Domains tools can access. Empty tuple means no
            network access. Use ("*",) for unrestricted access (not recommended).
    """

    allowed_domains: tuple[str, ...] = ()

    @classmethod
    def no_network(cls) -> NetworkPolicy:
        """Create a policy that blocks all network access."""
        return cls(allowed_domains=())

    @classmethod
    def with_domains(cls, *domains: str) -> NetworkPolicy:
        """Create a policy allowing specific domains."""
        return cls(allowed_domains=domains)


@FrozenDataclass()
class SandboxConfig:
    """Sandbox configuration for Claude Agent SDK.

    Provides programmatic control over OS-level sandboxing that would otherwise
    require manual settings.json configuration.

    Attributes:
        enabled: Enable OS-level sandboxing. Defaults to True for isolation.
        writable_paths: Paths the SDK can write to beyond the workspace.
            Relative paths are resolved against the workspace root.
        readable_paths: Additional paths the SDK can read (beyond workspace).
        excluded_commands: Commands that bypass the sandbox (e.g., "docker").
            Use sparingly—each exclusion is a potential security hole.
        allow_unsandboxed_commands: If True, allow specific commands to run
            outside the sandbox. Requires excluded_commands to be set.
        bash_auto_allow: If True, auto-approve Bash commands in sandbox mode.
            Only safe when network_policy blocks external access.
    """

    enabled: bool = True
    writable_paths: tuple[str, ...] = ()
    readable_paths: tuple[str, ...] = ()
    excluded_commands: tuple[str, ...] = ()
    allow_unsandboxed_commands: bool = False
    bash_auto_allow: bool = True


@FrozenDataclass()
class BedrockConfig:
    """AWS Bedrock authentication configuration.

    Supports multiple AWS authentication methods for accessing Claude models
    via Amazon Bedrock. The authentication method is determined by which
    fields are provided:

    1. **Static credentials**: Set ``access_key_id`` and ``secret_access_key``
    2. **Temporary credentials**: Add ``session_token`` to static credentials
    3. **AWS profile**: Set ``profile`` to use a named profile from
       ``~/.aws/credentials``
    4. **IAM role assumption**: Set ``role_arn`` (optionally with
       ``role_session_name`` and ``external_id``)
    5. **Web identity / OIDC**: Set ``web_identity_token_file`` and ``role_arn``
    6. **Default credentials chain**: Leave all fields as None to use the
       standard AWS credential resolution order (environment variables,
       shared credentials file, IAM role, etc.)

    The ``region`` field is required for all authentication methods.

    Example (static credentials):
        >>> config = BedrockConfig(
        ...     region="us-east-1",
        ...     access_key_id="AKIAIOSFODNN7EXAMPLE",
        ...     secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        ... )

    Example (named profile):
        >>> config = BedrockConfig(
        ...     region="us-west-2",
        ...     profile="my-bedrock-profile",
        ... )

    Example (IAM role assumption):
        >>> config = BedrockConfig(
        ...     region="eu-west-1",
        ...     role_arn="arn:aws:iam::123456789012:role/BedrockAccessRole",
        ...     role_session_name="weakincentives-session",
        ... )

    Example (web identity / OIDC):
        >>> config = BedrockConfig(
        ...     region="us-east-1",
        ...     role_arn="arn:aws:iam::123456789012:role/OIDCRole",
        ...     web_identity_token_file="/var/run/secrets/token",
        ... )

    Example (default credentials chain):
        >>> config = BedrockConfig(region="us-east-1")

    Attributes:
        region: AWS region for Bedrock API (e.g., ``"us-east-1"``). Required.
        access_key_id: AWS access key ID for static credentials.
        secret_access_key: AWS secret access key for static credentials.
        session_token: AWS session token for temporary credentials (STS).
        profile: AWS profile name from ``~/.aws/credentials``.
        role_arn: ARN of IAM role to assume. Used with ``role_session_name``
            for role assumption, or with ``web_identity_token_file`` for
            OIDC/web identity federation.
        role_session_name: Session name for role assumption. Defaults to
            ``"weakincentives"`` if ``role_arn`` is set without this field.
        external_id: External ID for cross-account role assumption.
        web_identity_token_file: Path to file containing OIDC token for
            web identity federation. Requires ``role_arn`` to be set.
        endpoint_url: Custom Bedrock endpoint URL. Use for VPC endpoints
            or testing with LocalStack/moto.
    """

    region: str
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    profile: str | None = None
    role_arn: str | None = None
    role_session_name: str | None = None
    external_id: str | None = None
    web_identity_token_file: str | None = None
    endpoint_url: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        # Check that static credentials are complete if provided
        if (self.access_key_id is None) != (self.secret_access_key is None):
            raise ValueError(
                "Both access_key_id and secret_access_key must be provided together, "
                "or both must be None"
            )

        # Check that session_token is only used with static credentials
        if self.session_token and not self.access_key_id:
            raise ValueError(
                "session_token requires access_key_id and secret_access_key"
            )

        # Check that web_identity_token_file requires role_arn
        if self.web_identity_token_file and not self.role_arn:
            raise ValueError("web_identity_token_file requires role_arn to be set")

        # Check that external_id is only used with role_arn
        if self.external_id and not self.role_arn:
            raise ValueError("external_id requires role_arn to be set")

        # Check that role_session_name is only used with role_arn
        if self.role_session_name and not self.role_arn:
            raise ValueError("role_session_name requires role_arn to be set")

    @classmethod
    def from_static_credentials(
        cls,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        *,
        session_token: str | None = None,
        endpoint_url: str | None = None,
    ) -> BedrockConfig:
        """Create config with static IAM credentials.

        Args:
            region: AWS region for Bedrock API.
            access_key_id: AWS access key ID.
            secret_access_key: AWS secret access key.
            session_token: Optional session token for temporary credentials.
            endpoint_url: Optional custom endpoint URL.

        Returns:
            Configured BedrockConfig instance.
        """
        return cls(
            region=region,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            endpoint_url=endpoint_url,
        )

    @classmethod
    def from_profile(
        cls,
        region: str,
        profile: str,
        *,
        endpoint_url: str | None = None,
    ) -> BedrockConfig:
        """Create config using an AWS profile from ~/.aws/credentials.

        Args:
            region: AWS region for Bedrock API.
            profile: AWS profile name.
            endpoint_url: Optional custom endpoint URL.

        Returns:
            Configured BedrockConfig instance.
        """
        return cls(
            region=region,
            profile=profile,
            endpoint_url=endpoint_url,
        )

    @classmethod
    def from_role(
        cls,
        region: str,
        role_arn: str,
        *,
        role_session_name: str | None = None,
        external_id: str | None = None,
        endpoint_url: str | None = None,
    ) -> BedrockConfig:
        """Create config with IAM role assumption.

        Uses the default credential chain to assume the specified role.
        Suitable for cross-account access or elevated permissions.

        Args:
            region: AWS region for Bedrock API.
            role_arn: ARN of the IAM role to assume.
            role_session_name: Optional session name (defaults to
                "weakincentives").
            external_id: Optional external ID for cross-account access.
            endpoint_url: Optional custom endpoint URL.

        Returns:
            Configured BedrockConfig instance.
        """
        return cls(
            region=region,
            role_arn=role_arn,
            role_session_name=role_session_name,
            external_id=external_id,
            endpoint_url=endpoint_url,
        )

    @classmethod
    def from_web_identity(
        cls,
        region: str,
        role_arn: str,
        web_identity_token_file: str,
        *,
        role_session_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> BedrockConfig:
        """Create config with OIDC/web identity federation.

        Used in Kubernetes (EKS IRSA), GitHub Actions OIDC, or other
        environments that provide web identity tokens.

        Args:
            region: AWS region for Bedrock API.
            role_arn: ARN of the IAM role configured for web identity.
            web_identity_token_file: Path to file containing the OIDC token.
            role_session_name: Optional session name.
            endpoint_url: Optional custom endpoint URL.

        Returns:
            Configured BedrockConfig instance.
        """
        return cls(
            region=region,
            role_arn=role_arn,
            web_identity_token_file=web_identity_token_file,
            role_session_name=role_session_name,
            endpoint_url=endpoint_url,
        )

    @classmethod
    def from_environment(cls, region: str) -> BedrockConfig:
        """Create config using the default AWS credential chain.

        Credentials are resolved in the standard AWS order:
        1. Environment variables (AWS_ACCESS_KEY_ID, etc.)
        2. Shared credentials file (~/.aws/credentials)
        3. AWS config file (~/.aws/config)
        4. Container credentials (ECS/EKS)
        5. Instance profile credentials (EC2/Lambda)

        Args:
            region: AWS region for Bedrock API.

        Returns:
            Configured BedrockConfig instance.
        """
        return cls(region=region)


@FrozenDataclass()
class IsolationConfig:
    """Configuration for hermetic SDK isolation.

    When provided to the adapter, creates an ephemeral home directory with
    generated settings, preventing any interaction with the host's ~/.claude
    configuration, credentials, and session state.

    Supports both Anthropic API and AWS Bedrock authentication. Set ``api_key``
    for direct Anthropic API access, or ``bedrock`` for AWS Bedrock access.
    These are mutually exclusive.

    Example (Anthropic API):
        >>> config = IsolationConfig(api_key="sk-ant-...")

    Example (AWS Bedrock with static credentials):
        >>> config = IsolationConfig(
        ...     bedrock=BedrockConfig.from_static_credentials(
        ...         region="us-east-1",
        ...         access_key_id="AKIA...",
        ...         secret_access_key="...",
        ...     )
        ... )

    Example (AWS Bedrock with profile):
        >>> config = IsolationConfig(
        ...     bedrock=BedrockConfig.from_profile(
        ...         region="us-west-2",
        ...         profile="bedrock-prod",
        ...     )
        ... )

    Example (AWS Bedrock with IAM role):
        >>> config = IsolationConfig(
        ...     bedrock=BedrockConfig.from_role(
        ...         region="us-east-1",
        ...         role_arn="arn:aws:iam::123456789012:role/BedrockRole",
        ...     )
        ... )

    Attributes:
        network_policy: Network access constraints. None means no network access.
        sandbox: Sandbox configuration. None uses secure defaults.
        env: Additional environment variables for the SDK subprocess.
        api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY from
            the host environment. Mutually exclusive with ``bedrock``.
        bedrock: AWS Bedrock authentication configuration. When set, the
            adapter uses Bedrock instead of the Anthropic API. Mutually
            exclusive with ``api_key``.
        include_host_env: If True, inherit non-sensitive host env vars.
            Sensitive vars (HOME, CLAUDE_*, ANTHROPIC_*, AWS_*, GOOGLE_*)
            are always excluded.
    """

    network_policy: NetworkPolicy | None = None
    sandbox: SandboxConfig | None = None
    env: Mapping[str, str] | None = None
    api_key: str | None = None
    bedrock: BedrockConfig | None = None
    include_host_env: bool = False

    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        if self.api_key and self.bedrock:
            raise ValueError(
                "Cannot specify both api_key and bedrock. "
                "Use api_key for Anthropic API or bedrock for AWS Bedrock."
            )


# Prefixes of environment variables that should never be inherited
_SENSITIVE_ENV_PREFIXES: tuple[str, ...] = (
    "HOME",
    "CLAUDE_",
    "ANTHROPIC_",
    "AWS_",
    "GOOGLE_",
    "AZURE_",
    "OPENAI_",
)


class EphemeralHome:
    """Manages temporary home directory for hermetic isolation.

    Creates and manages a temporary directory that serves as HOME for the
    Claude Agent SDK subprocess. This prevents the SDK from reading or
    modifying the user's ~/.claude configuration.

    The ephemeral home contains:
    - .claude/settings.json: Generated from IsolationConfig
    - Any workspace files if a workspace_path is provided

    Example:
        >>> isolation = IsolationConfig(
        ...     network_policy=NetworkPolicy.no_network(),
        ...     api_key="sk-ant-...",
        ... )
        >>> ephemeral = EphemeralHome(isolation)
        >>> try:
        ...     env = ephemeral.get_env()
        ...     # Pass env to SDK subprocess
        ... finally:
        ...     ephemeral.cleanup()
    """

    def __init__(
        self,
        isolation: IsolationConfig,
        *,
        workspace_path: str | None = None,
        temp_dir_prefix: str = "claude-agent-",
    ) -> None:
        """Initialize ephemeral home directory.

        Args:
            isolation: Isolation configuration to apply.
            workspace_path: Optional workspace directory to include.
            temp_dir_prefix: Prefix for the temporary directory name.
        """
        self._isolation = isolation
        self._workspace_path = workspace_path
        self._temp_dir = tempfile.mkdtemp(prefix=temp_dir_prefix)
        self._claude_dir = Path(self._temp_dir) / ".claude"
        self._claude_dir.mkdir(parents=True, exist_ok=True)
        self._generate_settings()
        self._cleaned_up = False

    def _generate_settings(self) -> None:
        """Generate settings.json from IsolationConfig."""
        settings: dict[str, Any] = {}

        # Sandbox settings
        sandbox = self._isolation.sandbox or SandboxConfig()
        settings["sandbox"] = {
            "enabled": sandbox.enabled,
            "autoAllowBashIfSandboxed": sandbox.bash_auto_allow,
        }

        if sandbox.excluded_commands:
            settings["sandbox"]["excludedCommands"] = list(sandbox.excluded_commands)

        if sandbox.allow_unsandboxed_commands:
            settings["sandbox"]["allowUnsandboxedCommands"] = True

        if sandbox.writable_paths:
            settings["sandbox"]["writablePaths"] = list(sandbox.writable_paths)

        if sandbox.readable_paths:
            settings["sandbox"]["readablePaths"] = list(sandbox.readable_paths)

        # Network settings
        network = self._isolation.network_policy or NetworkPolicy.no_network()
        settings["sandbox"]["network"] = {
            "allowedDomains": list(network.allowed_domains),
        }

        # Write settings
        settings_path = self._claude_dir / "settings.json"
        settings_path.write_text(json.dumps(settings, indent=2))

    def get_env(self) -> dict[str, str]:
        """Build environment variables for SDK subprocess.

        Returns:
            Dictionary of environment variables to pass to the SDK subprocess.
            Includes HOME pointing to the ephemeral directory and any
            configured API keys or AWS credentials.
        """
        env: dict[str, str] = {}

        if self._isolation.include_host_env:
            # Copy non-sensitive host env vars
            env.update(
                {
                    k: v
                    for k, v in os.environ.items()
                    if not any(k.startswith(p) for p in _SENSITIVE_ENV_PREFIXES)
                }
            )

        # Override HOME to ephemeral directory
        env["HOME"] = self._temp_dir

        # Set credentials based on configuration
        if self._isolation.bedrock:
            # AWS Bedrock configuration
            self._apply_bedrock_env(env)
        elif self._isolation.api_key:
            # Explicit Anthropic API key
            env["ANTHROPIC_API_KEY"] = self._isolation.api_key
        elif "ANTHROPIC_API_KEY" in os.environ:
            # Fall back to environment API key
            env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

        # Merge user-provided env vars (highest priority)
        if self._isolation.env:
            env.update(self._isolation.env)

        return env

    def _apply_bedrock_env(self, env: dict[str, str]) -> None:
        """Apply AWS Bedrock credentials to environment.

        Precondition: self._isolation.bedrock is not None.

        Args:
            env: Environment dictionary to update in place.
        """
        # Type narrowing: bedrock is guaranteed non-None by caller check
        bedrock = self._isolation.bedrock
        if bedrock is None:  # pragma: no cover
            return

        # Always set the region
        env["AWS_REGION"] = bedrock.region
        env["AWS_DEFAULT_REGION"] = bedrock.region

        # Map optional config fields to environment variables
        optional_mappings: dict[str, str | None] = {
            "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": bedrock.endpoint_url,
            "AWS_SESSION_TOKEN": bedrock.session_token,
            "AWS_PROFILE": bedrock.profile,
            "AWS_ROLE_ARN": bedrock.role_arn,
            "AWS_ROLE_SESSION_NAME": bedrock.role_session_name,
            "AWS_EXTERNAL_ID": bedrock.external_id,
            "AWS_WEB_IDENTITY_TOKEN_FILE": bedrock.web_identity_token_file,
        }

        # Static credentials require both key and secret
        if bedrock.access_key_id and bedrock.secret_access_key:
            env["AWS_ACCESS_KEY_ID"] = bedrock.access_key_id
            env["AWS_SECRET_ACCESS_KEY"] = bedrock.secret_access_key

        # Apply all non-None optional values
        env.update({k: v for k, v in optional_mappings.items() if v is not None})

    @staticmethod
    def get_setting_sources() -> list[str]:
        """Get the setting_sources value for SDK options.

        Returns ["user"] to make the SDK read settings from $HOME/.claude/settings.json.
        Since we redirect HOME to the ephemeral directory, this loads our generated
        settings including sandbox and network policy configuration.

        Note: An empty list would prevent ALL filesystem loading, including our
        ephemeral settings. We need "user" to read from the redirected HOME.

        Returns:
            List containing "user" to load settings from ephemeral HOME.
        """
        return ["user"]

    def cleanup(self) -> None:
        """Remove ephemeral home directory.

        Safe to call multiple times. After cleanup, the ephemeral home
        should not be used.
        """
        if not self._cleaned_up:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._cleaned_up = True

    @property
    def home_path(self) -> str:
        """Absolute path to the ephemeral home directory."""
        return self._temp_dir

    @property
    def claude_dir(self) -> Path:
        """Path to the .claude directory within ephemeral home."""
        return self._claude_dir

    @property
    def settings_path(self) -> Path:
        """Path to the generated settings.json file."""
        return self._claude_dir / "settings.json"

    def __enter__(self) -> EphemeralHome:
        """Context manager entry."""
        return self

    def __exit__(self, *_: object) -> None:
        """Context manager exit with automatic cleanup."""
        self.cleanup()

    def __del__(self) -> None:
        """Destructor that attempts cleanup if not already done."""
        self.cleanup()

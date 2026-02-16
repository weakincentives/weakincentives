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

Troubleshooting
---------------
Common errors and solutions:

**IsolationAuthError: "AWS config directory not found... but Bedrock is configured"**
    The ~/.aws directory doesn't exist but CLAUDE_CODE_USE_BEDROCK=1 is set.
    Solutions:
    - Create ~/.aws with valid config/credentials files
    - Use ``aws_config_path`` parameter to specify an alternate location
    - Use ``IsolationConfig.with_api_key()`` instead of Bedrock

**IsolationAuthError: "HOME not set but Bedrock is configured"**
    Running in an environment without HOME (e.g., some containers).
    Solutions:
    - Set the HOME environment variable
    - Use ``aws_config_path`` parameter to specify AWS config location

**IsolationAuthError: "No authentication configured"**
    Neither ANTHROPIC_API_KEY nor Bedrock environment is configured.
    Solutions:
    - Set ANTHROPIC_API_KEY environment variable
    - Configure Bedrock: CLAUDE_CODE_USE_BEDROCK=1 and AWS_REGION
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from ...dataclasses import FrozenDataclass
from ._model_utils import (
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_MODEL,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "AuthMode",
    "BedrockConfig",
    "IsolationAuthError",
    "IsolationConfig",
    "IsolationOptions",
    "NetworkPolicy",
    "SandboxConfig",
    "get_default_model",
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
        allow_unix_sockets: Unix socket paths accessible within the sandbox
            (e.g., for SSH agents). macOS only; Linux uses seccomp filters.
        allow_all_unix_sockets: If True, allow access to all Unix sockets.
            Less secure than specifying individual paths.
        allow_local_binding: If True, allow binding to localhost ports.
            macOS only.
        http_proxy_port: HTTP proxy port when using a custom proxy. None
            means no custom HTTP proxy.
        socks_proxy_port: SOCKS5 proxy port when using a custom proxy. None
            means no custom SOCKS5 proxy.
    """

    allowed_domains: tuple[str, ...] = ()
    allow_unix_sockets: tuple[str, ...] = ()
    allow_all_unix_sockets: bool = False
    allow_local_binding: bool = False
    http_proxy_port: int | None = None
    socks_proxy_port: int | None = None

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
        enable_weaker_nested_sandbox: If True, use a weaker sandbox that works
            inside unprivileged Docker containers where full bubblewrap
            isolation is unavailable. This is better than disabling the sandbox
            entirely (``enabled=False``) because it still enforces some
            restrictions, but it substantially weakens security compared to
            the full sandbox. Only use when the container itself provides
            additional isolation. Linux only.
        ignore_file_violations: File paths for which sandbox violations should
            be silently ignored rather than flagged.
        ignore_network_violations: Network hosts for which sandbox violations
            should be silently ignored rather than flagged.
    """

    enabled: bool = True
    writable_paths: tuple[str, ...] = ()
    readable_paths: tuple[str, ...] = ()
    excluded_commands: tuple[str, ...] = ()
    allow_unsandboxed_commands: bool = False
    bash_auto_allow: bool = True
    enable_weaker_nested_sandbox: bool = False
    ignore_file_violations: tuple[str, ...] = ()
    ignore_network_violations: tuple[str, ...] = ()


class IsolationAuthError(Exception):
    """Raised when required authentication is not available.

    See module docstring for troubleshooting common errors.
    """


class AuthMode(Enum):
    """Authentication mode for isolated SDK execution.

    Attributes:
        INHERIT_HOST: Inherit authentication from host environment.
            Works with both Anthropic API and AWS Bedrock.
        EXPLICIT_API_KEY: Use an explicitly provided Anthropic API key.
            Disables Bedrock authentication.
        ANTHROPIC_API: Require Anthropic API key from environment.
            Fails fast if ANTHROPIC_API_KEY is not set.
        BEDROCK: Require AWS Bedrock authentication.
            Fails fast if Bedrock is not configured.
    """

    INHERIT_HOST = "inherit_host"
    EXPLICIT_API_KEY = "explicit_api_key"
    ANTHROPIC_API = "anthropic_api"
    BEDROCK = "bedrock"


# ============================================================================
# Auth Configuration Detection
# ============================================================================
# These functions check for valid authentication from multiple sources:
# 1. Shell environment variables (highest priority)
# 2. Host ~/.claude/settings.json (fallback for auth vars)
# This ensures that if 'claude' works on the host, WINK agents will too.


@FrozenDataclass()
class BedrockConfig:
    """Detected Bedrock configuration from the environment.

    Represents the Bedrock authentication configuration found in the
    shell environment and/or host ~/.claude/settings.json.

    Attributes:
        region: AWS region for Bedrock API calls (e.g., "us-east-1").
        profile: AWS profile name, if configured.
        source: Where the configuration was detected from.
    """

    region: str
    profile: str | None = None
    source: str = "shell"

    @classmethod
    def from_environment(
        cls, *, check_host_settings: bool = True
    ) -> BedrockConfig | None:
        """Detect Bedrock configuration from the environment.

        Args:
            check_host_settings: If True, also check ~/.claude/settings.json.

        Returns:
            BedrockConfig if Bedrock is configured, None otherwise.
        """
        host_settings = _read_host_claude_settings() if check_host_settings else {}

        bedrock_enabled = _get_effective_env_value(
            "CLAUDE_CODE_USE_BEDROCK", host_settings
        )
        if bedrock_enabled != "1":
            return None

        region = _get_effective_env_value("AWS_REGION", host_settings)
        if region is None:
            return None

        profile = _get_effective_env_value("AWS_PROFILE", host_settings)

        # Determine source
        source = "shell" if "CLAUDE_CODE_USE_BEDROCK" in os.environ else "host_settings"

        return cls(region=region, profile=profile, source=source)


def _read_host_claude_settings() -> dict[str, Any]:
    """Read the host's ~/.claude/settings.json if it exists.

    Returns an empty dict if the file doesn't exist or can't be parsed.
    This allows inheriting auth settings from the host environment when
    using inherit_host_auth mode.
    """
    real_home = os.environ.get("HOME", "")
    if not real_home:
        return {}

    settings_path = Path(real_home) / ".claude" / "settings.json"
    if not settings_path.exists():
        return {}

    try:
        return json.loads(settings_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        _logger.debug(
            "isolation.read_host_settings.failed",
            extra={"path": str(settings_path), "error": str(e)},
        )
        return {}


def _get_effective_env_value(
    key: str, host_settings: dict[str, Any] | None = None
) -> str | None:
    """Get an environment variable value from shell or host settings.json.

    Priority: shell environment > host settings.json env section.

    Args:
        key: Environment variable name.
        host_settings: Pre-loaded host settings (to avoid re-reading file).

    Returns:
        The value if found, None otherwise.
    """
    # Shell environment has highest priority
    if key in os.environ:
        return os.environ[key]

    # Fall back to host settings.json
    if host_settings is None:
        host_settings = _read_host_claude_settings()
    host_env = host_settings.get("env", {})
    return host_env.get(key)


def _is_anthropic_api_key_set() -> bool:
    """Check if ANTHROPIC_API_KEY is available (shell env only)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _is_bedrock_configured(*, check_host_settings: bool = True) -> bool:
    """Check if Bedrock authentication is configured.

    Args:
        check_host_settings: If True, also check ~/.claude/settings.json.

    Returns:
        True if Bedrock is configured (CLAUDE_CODE_USE_BEDROCK=1 + AWS_REGION).
    """
    return (
        BedrockConfig.from_environment(check_host_settings=check_host_settings)
        is not None
    )


def get_default_model() -> str:
    """Get the default model ID for the current authentication mode.

    Returns the appropriate model ID format based on whether Bedrock
    is configured. Both default to Claude Opus 4.6.

    Returns:
        Model ID string in the appropriate format.
    """
    if _is_bedrock_configured():
        return DEFAULT_BEDROCK_MODEL
    return DEFAULT_MODEL


@FrozenDataclass()
class IsolationOptions:
    """Common optional parameters for IsolationConfig factory methods.

    Groups network, sandbox, environment, and skills configuration to simplify
    factory method signatures.
    """

    network_policy: NetworkPolicy | None = None
    """Network access constraints for tools."""

    sandbox: SandboxConfig | None = None
    """Sandbox configuration for OS-level isolation."""

    env: Mapping[str, str] | None = None
    """Additional environment variables to set."""

    include_host_env: bool = False
    """If True, inherit non-sensitive host environment variables."""


@FrozenDataclass()
class IsolationConfig:
    """Configuration for hermetic SDK isolation.

    When provided to the adapter, creates an ephemeral home directory with
    generated settings, preventing any interaction with the host's ~/.claude
    configuration, credentials, and session state.

    **Authentication modes:**

    1. **Inherit host auth** (default): When ``api_key`` is None, inherits
       authentication from the host environment. This works with both
       Anthropic API (via ANTHROPIC_API_KEY env var) and AWS Bedrock
       (via ~/.aws config and CLAUDE_CODE_USE_BEDROCK env var).

    2. **Explicit API key**: When ``api_key`` is set, uses that key with
       the Anthropic API directly and disables Bedrock.

    **Factory methods** (recommended for explicit intent and validation):

    - ``IsolationConfig.inherit_host_auth()``: Inherit auth, fail if none available
    - ``IsolationConfig.with_api_key(key)``: Use explicit API key
    - ``IsolationConfig.for_anthropic_api()``: Require Anthropic API key from env
    - ``IsolationConfig.for_bedrock()``: Require Bedrock, fail if not configured

    Example (inherit host auth - works with Bedrock or Anthropic):
        >>> config = IsolationConfig.inherit_host_auth()

    Example (explicit API key):
        >>> config = IsolationConfig.with_api_key("sk-ant-...")

    Example (require Anthropic API):
        >>> config = IsolationConfig.for_anthropic_api()

    Example (require Bedrock):
        >>> config = IsolationConfig.for_bedrock()

    Example (Docker container with mounted AWS config):
        >>> config = IsolationConfig.for_bedrock(aws_config_path="/mnt/aws")

    Attributes:
        network_policy: Network access constraints. None means no network access.
        sandbox: Sandbox configuration. None uses secure defaults.
        env: Additional environment variables for the SDK subprocess.
        api_key: Anthropic API key. If set, uses this key and disables Bedrock.
            If None, inherits authentication from the host environment.
        aws_config_path: Path to AWS config directory for Bedrock auth.
            Defaults to ~/.aws. Use this when running in a container where
            AWS config is mounted at a different path.
        include_host_env: If True, inherit non-sensitive host env vars.
            Sensitive vars (HOME, CLAUDE_*, ANTHROPIC_*, AWS_*, GOOGLE_*)
            are always excluded.

    Note:
        Skills are attached to prompt sections, not IsolationConfig.
        See ``specs/SKILLS.md`` for skill attachment patterns.
    """

    network_policy: NetworkPolicy | None = None
    sandbox: SandboxConfig | None = None
    env: Mapping[str, str] | None = None
    api_key: str | None = None
    aws_config_path: Path | str | None = None
    include_host_env: bool = False

    @classmethod
    def inherit_host_auth(
        cls,
        *,
        network_policy: NetworkPolicy | None = None,
        sandbox: SandboxConfig | None = None,
        env: Mapping[str, str] | None = None,
        include_host_env: bool = False,
    ) -> IsolationConfig:
        """Create config that inherits authentication from the host environment.

        Validates that either ANTHROPIC_API_KEY or Bedrock (CLAUDE_CODE_USE_BEDROCK=1
        + AWS_REGION) is configured. Fails fast if no authentication is available.

        Args:
            network_policy: Network access constraints for tools.
            sandbox: Sandbox configuration.
            env: Additional environment variables.
            include_host_env: If True, inherit non-sensitive host env vars.

        Returns:
            IsolationConfig configured to inherit host authentication.

        Raises:
            IsolationAuthError: If no authentication is configured in the environment.
        """
        if not _is_anthropic_api_key_set() and not _is_bedrock_configured():
            msg = (
                "No authentication configured. Set either:\n"
                "  - ANTHROPIC_API_KEY for Anthropic API\n"
                "  - CLAUDE_CODE_USE_BEDROCK=1 and AWS_REGION for AWS Bedrock"
            )
            raise IsolationAuthError(msg)

        return cls(
            network_policy=network_policy,
            sandbox=sandbox,
            env=env,
            api_key=None,
            aws_config_path=None,
            include_host_env=include_host_env,
        )

    @classmethod
    def with_api_key(
        cls,
        api_key: str,
        *,
        options: IsolationOptions | None = None,
    ) -> IsolationConfig:
        """Create config with an explicit Anthropic API key.

        Disables Bedrock authentication and uses the provided key directly.

        Args:
            api_key: Anthropic API key (required).
            options: Optional isolation options (network, sandbox, env).

        Returns:
            IsolationConfig configured with the explicit API key.

        Raises:
            IsolationAuthError: If api_key is empty.
        """
        if not api_key:
            msg = "api_key is required and cannot be empty"
            raise IsolationAuthError(msg)

        opts = options or IsolationOptions()
        return cls(
            network_policy=opts.network_policy,
            sandbox=opts.sandbox,
            env=opts.env,
            api_key=api_key,
            aws_config_path=None,
            include_host_env=opts.include_host_env,
        )

    @classmethod
    def for_anthropic_api(
        cls,
        *,
        network_policy: NetworkPolicy | None = None,
        sandbox: SandboxConfig | None = None,
        env: Mapping[str, str] | None = None,
        include_host_env: bool = False,
    ) -> IsolationConfig:
        """Create config that requires Anthropic API key from environment.

        Validates that ANTHROPIC_API_KEY is set. Fails fast if not available.
        This is useful when you want to ensure Anthropic API is used (not Bedrock).

        Args:
            network_policy: Network access constraints for tools.
            sandbox: Sandbox configuration.
            env: Additional environment variables.
            include_host_env: If True, inherit non-sensitive host env vars.

        Returns:
            IsolationConfig configured for Anthropic API authentication.

        Raises:
            IsolationAuthError: If ANTHROPIC_API_KEY is not set.
        """
        if not _is_anthropic_api_key_set():
            msg = "ANTHROPIC_API_KEY environment variable is not set"
            raise IsolationAuthError(msg)

        return cls(
            network_policy=network_policy,
            sandbox=sandbox,
            env=env,
            api_key=None,
            aws_config_path=None,
            include_host_env=include_host_env,
        )

    @classmethod
    def for_bedrock(
        cls,
        *,
        aws_config_path: Path | str | None = None,
        options: IsolationOptions | None = None,
    ) -> IsolationConfig:
        """Create config that requires AWS Bedrock authentication.

        Validates that Bedrock is configured (CLAUDE_CODE_USE_BEDROCK=1 + AWS_REGION).
        Fails fast if Bedrock authentication is not available.

        Args:
            aws_config_path: Path to AWS config directory. Use this when running
                in a container where AWS config is mounted at a non-standard path.
            options: Optional isolation options (network, sandbox, env).

        Returns:
            IsolationConfig configured for Bedrock authentication.

        Raises:
            IsolationAuthError: If Bedrock is not configured in the environment.
        """
        if not _is_bedrock_configured():
            msg = (
                "Bedrock authentication not configured. Set:\n"
                "  - CLAUDE_CODE_USE_BEDROCK=1\n"
                "  - AWS_REGION (e.g., us-east-1)"
            )
            raise IsolationAuthError(msg)

        opts = options or IsolationOptions()
        return cls(
            network_policy=opts.network_policy,
            sandbox=opts.sandbox,
            env=opts.env,
            api_key=None,
            aws_config_path=aws_config_path,
            include_host_env=opts.include_host_env,
        )

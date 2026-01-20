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
import shutil
import tempfile
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

from ...dataclasses import FrozenDataclass
from ...skills import (
    MAX_SKILL_TOTAL_BYTES,
    SkillConfig,
    SkillMountError,
    SkillNotFoundError,
    resolve_skill_name,
    validate_skill,
    validate_skill_name,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "AuthMode",
    "BedrockConfig",
    "EphemeralHome",
    "IsolationAuthError",
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


# Default model for both Anthropic API and Bedrock
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Model name mappings between Anthropic API and Bedrock
_ANTHROPIC_TO_BEDROCK: dict[str, str] = {
    "claude-opus-4-5-20251101": "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-sonnet-4-5-20250929": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4-20250514": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-3-5-sonnet-20241022": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
}

_BEDROCK_TO_ANTHROPIC: dict[str, str] = {v: k for k, v in _ANTHROPIC_TO_BEDROCK.items()}


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
    is configured. Both default to Claude Sonnet 4.5.

    Returns:
        Model ID string in the appropriate format.
    """
    if _is_bedrock_configured():
        return DEFAULT_BEDROCK_MODEL
    return DEFAULT_MODEL


def get_supported_bedrock_models() -> dict[str, str]:
    """Get the mapping of supported Anthropic models to Bedrock model IDs.

    Returns:
        Dictionary mapping Anthropic model names to Bedrock model IDs.
        Example: {"claude-opus-4-5-20251101": "us.anthropic.claude-opus-4-5-20251101-v1:0"}
    """
    return dict(_ANTHROPIC_TO_BEDROCK)


def to_bedrock_model_id(anthropic_model: str) -> str:
    """Convert an Anthropic model name to Bedrock model ID.

    Args:
        anthropic_model: Anthropic model name (e.g., "claude-opus-4-5-20251101")

    Returns:
        Bedrock model ID with cross-region inference prefix.
        Returns the input unchanged if already a Bedrock ID or not in mapping.
    """
    # Already a Bedrock model ID
    if anthropic_model.startswith(("us.", "anthropic.")):
        return anthropic_model

    # Look up in mapping
    result = _ANTHROPIC_TO_BEDROCK.get(anthropic_model, anthropic_model)
    if result == anthropic_model:
        _logger.debug(
            "isolation.to_bedrock_model_id.unmapped",
            extra={"model": anthropic_model, "returned_unchanged": True},
        )
    return result


def to_anthropic_model_name(bedrock_model_id: str) -> str:
    """Convert a Bedrock model ID to Anthropic model name.

    Args:
        bedrock_model_id: Bedrock model ID (e.g., "us.anthropic.claude-opus-4-5-20251101-v1:0")

    Returns:
        Anthropic model name.
        Returns the input unchanged if not a Bedrock ID or not in mapping.
    """
    # Not a Bedrock model ID
    if not bedrock_model_id.startswith(("us.", "anthropic.")):
        return bedrock_model_id

    # Look up in mapping
    return _BEDROCK_TO_ANTHROPIC.get(bedrock_model_id, bedrock_model_id)


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
        skills: Skills to mount in the hermetic environment. Skills are
            copied to {ephemeral_home}/.claude/skills/ before spawning
            Claude Code.
    """

    network_policy: NetworkPolicy | None = None
    sandbox: SandboxConfig | None = None
    env: Mapping[str, str] | None = None
    api_key: str | None = None
    aws_config_path: Path | str | None = None
    include_host_env: bool = False
    skills: SkillConfig | None = None

    @classmethod
    def inherit_host_auth(
        cls,
        *,
        network_policy: NetworkPolicy | None = None,
        sandbox: SandboxConfig | None = None,
        env: Mapping[str, str] | None = None,
        include_host_env: bool = False,
        skills: SkillConfig | None = None,
    ) -> IsolationConfig:
        """Create config that inherits authentication from the host environment.

        Validates that either ANTHROPIC_API_KEY or Bedrock (CLAUDE_CODE_USE_BEDROCK=1
        + AWS_REGION) is configured. Fails fast if no authentication is available.

        Args:
            network_policy: Network access constraints for tools.
            sandbox: Sandbox configuration.
            env: Additional environment variables.
            include_host_env: If True, inherit non-sensitive host env vars.
            skills: Skills to mount.

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
            skills=skills,
        )

    @classmethod
    def with_api_key(  # noqa: PLR0913
        cls,
        api_key: str,
        *,
        network_policy: NetworkPolicy | None = None,
        sandbox: SandboxConfig | None = None,
        env: Mapping[str, str] | None = None,
        include_host_env: bool = False,
        skills: SkillConfig | None = None,
    ) -> IsolationConfig:
        """Create config with an explicit Anthropic API key.

        Disables Bedrock authentication and uses the provided key directly.

        Args:
            api_key: Anthropic API key (required).
            network_policy: Network access constraints for tools.
            sandbox: Sandbox configuration.
            env: Additional environment variables.
            include_host_env: If True, inherit non-sensitive host env vars.
            skills: Skills to mount.

        Returns:
            IsolationConfig configured with the explicit API key.

        Raises:
            IsolationAuthError: If api_key is empty.
        """
        if not api_key:
            msg = "api_key is required and cannot be empty"
            raise IsolationAuthError(msg)

        return cls(
            network_policy=network_policy,
            sandbox=sandbox,
            env=env,
            api_key=api_key,
            aws_config_path=None,
            include_host_env=include_host_env,
            skills=skills,
        )

    @classmethod
    def for_anthropic_api(
        cls,
        *,
        network_policy: NetworkPolicy | None = None,
        sandbox: SandboxConfig | None = None,
        env: Mapping[str, str] | None = None,
        include_host_env: bool = False,
        skills: SkillConfig | None = None,
    ) -> IsolationConfig:
        """Create config that requires Anthropic API key from environment.

        Validates that ANTHROPIC_API_KEY is set. Fails fast if not available.
        This is useful when you want to ensure Anthropic API is used (not Bedrock).

        Args:
            network_policy: Network access constraints for tools.
            sandbox: Sandbox configuration.
            env: Additional environment variables.
            include_host_env: If True, inherit non-sensitive host env vars.
            skills: Skills to mount.

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
            skills=skills,
        )

    @classmethod
    def for_bedrock(  # noqa: PLR0913
        cls,
        *,
        aws_config_path: Path | str | None = None,
        network_policy: NetworkPolicy | None = None,
        sandbox: SandboxConfig | None = None,
        env: Mapping[str, str] | None = None,
        include_host_env: bool = False,
        skills: SkillConfig | None = None,
    ) -> IsolationConfig:
        """Create config that requires AWS Bedrock authentication.

        Validates that Bedrock is configured (CLAUDE_CODE_USE_BEDROCK=1 + AWS_REGION).
        Fails fast if Bedrock authentication is not available.

        Args:
            aws_config_path: Path to AWS config directory. Use this when running
                in a container where AWS config is mounted at a non-standard path.
            network_policy: Network access constraints for tools.
            sandbox: Sandbox configuration.
            env: Additional environment variables.
            include_host_env: If True, inherit non-sensitive host env vars.
            skills: Skills to mount.

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

        return cls(
            network_policy=network_policy,
            sandbox=sandbox,
            env=env,
            api_key=None,
            aws_config_path=aws_config_path,
            include_host_env=include_host_env,
            skills=skills,
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

# Auth-related vars to include in generated settings.json
# These are safe to write to disk (no credentials)
_SETTINGS_JSON_AUTH_VARS: tuple[str, ...] = (
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "CLAUDE_CODE_USE_BEDROCK",
)

# All AWS-related environment variables to pass through to subprocess
# Includes credentials that should only be passed via env, not written to disk
_AWS_PASSTHROUGH_VARS: tuple[str, ...] = (
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "CLAUDE_CODE_USE_BEDROCK",
)

# Patterns in env var names that indicate sensitive values to redact in logs
_SENSITIVE_KEY_PATTERNS: frozenset[str] = frozenset(
    {"SECRET", "TOKEN", "ACCESS_KEY", "KEY_ID"}
)


def _is_sensitive_key(key: str) -> bool:
    """Check if an environment variable key contains sensitive data."""
    return any(pattern in key for pattern in _SENSITIVE_KEY_PATTERNS)


def _copy_skill(
    source: Path,
    dest_dir: Path,
    *,
    follow_symlinks: bool = False,
    max_total_bytes: int = MAX_SKILL_TOTAL_BYTES,
) -> int:
    """Copy a skill to the destination directory.

    Args:
        source: Path to the skill file or directory.
        dest_dir: Destination directory for the skill.
        follow_symlinks: Whether to follow symlinks during copy.
        max_total_bytes: Maximum total bytes to copy.

    Returns:
        Total bytes copied.

    Raises:
        SkillMountError: If copy fails or exceeds byte limit.
    """
    total_bytes = 0
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        if source.is_dir():
            # Recursive directory copy
            for item in source.rglob("*"):
                if item.is_symlink() and not follow_symlinks:
                    continue
                if item.is_file():
                    rel_path = item.relative_to(source)
                    dest_file = dest_dir / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    size = item.stat().st_size
                    total_bytes += size
                    if total_bytes > max_total_bytes:
                        msg = f"Skill exceeds total size limit ({total_bytes} > {max_total_bytes})"
                        raise SkillMountError(msg)
                    shutil.copy2(item, dest_file)
        else:
            # Single file skill - wrap in directory as SKILL.md
            dest_file = dest_dir / "SKILL.md"
            size = source.stat().st_size
            total_bytes = size
            if total_bytes > max_total_bytes:
                msg = f"Skill exceeds total size limit ({total_bytes} > {max_total_bytes})"
                raise SkillMountError(msg)
            shutil.copy2(source, dest_file)
    except OSError as e:
        msg = f"Failed to copy skill: {e}"
        raise SkillMountError(msg) from e

    return total_bytes


class AwsConfigResolution(NamedTuple):
    """Result of resolving AWS config directory path.

    Attributes:
        path: Path to AWS config directory, or None if not available.
        skip_reason: Reason for skipping, if path is None.
    """

    path: Path | None
    skip_reason: str | None


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

        _logger.debug(
            "isolation.ephemeral_home.init",
            extra={
                "ephemeral_home": self._temp_dir,
                "has_explicit_api_key": isolation.api_key is not None,
                "has_aws_config_path": isolation.aws_config_path is not None,
                "include_host_env": isolation.include_host_env,
                "host_bedrock_enabled": os.environ.get("CLAUDE_CODE_USE_BEDROCK"),
                "host_aws_region": os.environ.get("AWS_REGION"),
                "host_aws_profile": os.environ.get("AWS_PROFILE"),
                "host_has_anthropic_key": "ANTHROPIC_API_KEY" in os.environ,
            },
        )

        self._generate_settings()
        self._mount_skills()
        self._copy_aws_config()
        self._cleaned_up = False

    def _generate_settings(self) -> None:
        """Generate settings.json from IsolationConfig."""
        settings: dict[str, Any] = {}

        self._configure_sandbox_settings(settings)
        self._configure_auth_settings(settings)
        self._write_settings(settings)

    def _configure_sandbox_settings(self, settings: dict[str, Any]) -> None:
        """Configure sandbox and network settings."""
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

    def _configure_auth_settings(self, settings: dict[str, Any]) -> None:
        """Configure env section based on authentication mode."""
        if self._isolation.api_key:
            self._configure_explicit_api_key_mode(settings)
        else:
            self._configure_inherit_host_auth_mode(settings)

    @staticmethod
    def _configure_explicit_api_key_mode(settings: dict[str, Any]) -> None:
        """Configure settings for explicit API key mode (Bedrock disabled)."""
        settings["env"] = {
            "CLAUDE_CODE_USE_BEDROCK": "0",
            "CLAUDE_USE_BEDROCK": "0",
            "DISABLE_AUTOUPDATER": "1",
        }

    @staticmethod
    def _configure_inherit_host_auth_mode(settings: dict[str, Any]) -> None:
        """Configure settings for inherit host auth mode.

        Merges auth settings from host ~/.claude/settings.json to ensure
        that if 'claude' works on the host, our isolated agent will too.

        Note on awsAuthRefresh: This setting is copied from the host's
        settings.json to allow credential refresh in the ephemeral environment.
        It may contain shell commands that run periodically to refresh AWS
        credentials. This is intentional - the security boundary is the host
        system's settings.json, which the user controls.
        """
        settings["env"] = {
            "DISABLE_AUTOUPDATER": "1",
        }

        # Read host settings.json and inherit auth-related env vars
        host_settings = _read_host_claude_settings()
        host_env = host_settings.get("env", {})

        inherited_vars = []
        for key in _SETTINGS_JSON_AUTH_VARS:
            # Prefer shell environment over host settings
            if key in os.environ:
                settings["env"][key] = os.environ[key]
                inherited_vars.append(f"{key}=<from_shell>")
            elif key in host_env:
                settings["env"][key] = host_env[key]
                inherited_vars.append(f"{key}=<from_host_settings>")

        # Copy awsAuthRefresh if present (allows credential refresh).
        # See docstring note about security implications.
        if "awsAuthRefresh" in host_settings:
            settings["awsAuthRefresh"] = host_settings["awsAuthRefresh"]

        _logger.debug(
            "isolation.generate_settings.auth_inherited",
            extra={
                "inherited_vars": inherited_vars,
                "has_aws_auth_refresh": "awsAuthRefresh" in settings,
                "host_settings_found": bool(host_settings),
            },
        )

    def _write_settings(self, settings: dict[str, Any]) -> None:
        """Write settings.json to the ephemeral .claude directory."""
        settings_path = self._claude_dir / "settings.json"
        settings_path.write_text(json.dumps(settings, indent=2))

        _logger.debug(
            "isolation.generate_settings.complete",
            extra={
                "settings_path": str(settings_path),
                "sandbox_enabled": settings.get("sandbox", {}).get("enabled"),
                "env_keys": list(settings.get("env", {}).keys()),
                "bedrock_in_settings": settings.get("env", {}).get(
                    "CLAUDE_CODE_USE_BEDROCK"
                ),
                "allowed_domains_count": len(
                    settings.get("sandbox", {})
                    .get("network", {})
                    .get("allowedDomains", [])
                ),
            },
        )

    def _mount_skills(self) -> None:
        """Mount configured skills into the ephemeral home."""
        skills_config = self._isolation.skills
        if skills_config is None:
            return

        skills_dir = self._claude_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        seen_names: set[str] = set()
        for mount in skills_config.skills:
            if not mount.enabled:
                continue

            name = resolve_skill_name(mount)
            validate_skill_name(name)

            if name in seen_names:
                msg = f"Duplicate skill name: {name}"
                raise SkillMountError(msg)
            seen_names.add(name)

            source = Path(mount.source).resolve()
            if not source.exists():
                msg = f"Skill not found: {mount.source}"
                raise SkillNotFoundError(msg)

            if skills_config.validate_on_mount:
                validate_skill(source)

            dest = skills_dir / name
            _copy_skill(source, dest)

    def _resolve_aws_config_dir(self, bedrock_enabled: bool) -> AwsConfigResolution:
        """Resolve the AWS config directory path.

        Args:
            bedrock_enabled: Whether Bedrock authentication is configured.

        Returns:
            AwsConfigResolution with path and skip_reason.

        Raises:
            IsolationAuthError: If Bedrock is configured but AWS config unavailable.
        """
        if self._isolation.aws_config_path:
            return AwsConfigResolution(
                path=Path(self._isolation.aws_config_path), skip_reason=None
            )

        real_home = os.environ.get("HOME", "")
        if not real_home:
            if bedrock_enabled:
                msg = "HOME not set but Bedrock is configured - cannot copy AWS config"
                raise IsolationAuthError(msg)
            return AwsConfigResolution(path=None, skip_reason="HOME_not_set")

        aws_dir = Path(real_home) / ".aws"
        if not aws_dir.exists():
            if bedrock_enabled:
                msg = f"AWS config directory not found at {aws_dir} but Bedrock is configured"
                raise IsolationAuthError(msg)
            return AwsConfigResolution(path=None, skip_reason="aws_dir_not_found")

        return AwsConfigResolution(path=aws_dir, skip_reason=None)

    def _copy_aws_config(self) -> None:
        """Copy AWS config directory into the ephemeral home.

        When inheriting host auth (api_key is None), AWS SDK needs access to
        ~/.aws/config, ~/.aws/credentials, and ~/.aws/sso/cache/. Since we set
        HOME to an ephemeral directory, we copy these files into the ephemeral
        home so it's fully self-contained.

        Uses aws_config_path if specified, otherwise defaults to ~/.aws.
        This is skipped when an explicit api_key is set since Bedrock is disabled.

        Raises:
            IsolationAuthError: If Bedrock is configured but AWS config copy fails.
        """
        # Skip when using explicit API key (Bedrock is disabled)
        if self._isolation.api_key:
            _logger.debug(
                "isolation.copy_aws_config.skipped_explicit_key",
                extra={"reason": "explicit_api_key_set"},
            )
            return

        bedrock_enabled = _is_bedrock_configured(check_host_settings=True)
        resolution = self._resolve_aws_config_dir(bedrock_enabled)

        if resolution.path is None:
            _logger.debug(
                "isolation.copy_aws_config.skipped",
                extra={"reason": resolution.skip_reason},
            )
            return

        aws_dir = resolution.path

        # Copy AWS config to ephemeral home
        ephemeral_aws_dir = Path(self._temp_dir) / ".aws"
        try:
            shutil.copytree(aws_dir, ephemeral_aws_dir, dirs_exist_ok=True)
            copied_files = list(ephemeral_aws_dir.rglob("*"))
            _logger.debug(
                "isolation.copy_aws_config.success",
                extra={
                    "source": str(aws_dir),
                    "dest": str(ephemeral_aws_dir),
                    "file_count": len([f for f in copied_files if f.is_file()]),
                    "has_config": (ephemeral_aws_dir / "config").exists(),
                    "has_credentials": (ephemeral_aws_dir / "credentials").exists(),
                    "has_sso_cache": (ephemeral_aws_dir / "sso" / "cache").exists(),
                },
            )
        except OSError as e:
            if bedrock_enabled:
                msg = f"Failed to copy AWS config for Bedrock: {e}"
                raise IsolationAuthError(msg) from e
            _logger.warning(
                "isolation.copy_aws_config.failed",
                extra={"source": str(aws_dir), "error": str(e)},
            )

    def get_env(self) -> dict[str, str]:
        """Build environment variables for SDK subprocess.

        Returns:
            Dictionary of environment variables to pass to the SDK subprocess.
            Includes HOME pointing to the ephemeral directory and authentication
            settings based on the configuration mode.
        """
        env: dict[str, str] = {}

        self._apply_host_env_inheritance(env)
        self._apply_base_env(env)

        if self._isolation.api_key:
            self._apply_explicit_api_key_env(env)
        else:
            self._apply_inherit_host_auth_env(env)

        self._apply_user_env_overrides(env)
        self._log_final_env(env)

        return env

    def _apply_host_env_inheritance(self, env: dict[str, str]) -> None:
        """Copy non-sensitive host env vars if include_host_env is enabled."""
        if self._isolation.include_host_env:
            env.update(
                {
                    k: v
                    for k, v in os.environ.items()
                    if not any(k.startswith(p) for p in _SENSITIVE_ENV_PREFIXES)
                }
            )

    def _apply_base_env(self, env: dict[str, str]) -> None:
        """Apply base environment settings common to all modes."""
        # Override HOME to ephemeral directory
        env["HOME"] = self._temp_dir

        # Always pass PATH - SDK needs it to find node/npx for MCP tools
        if "PATH" in os.environ:
            env["PATH"] = os.environ["PATH"]

    def _apply_explicit_api_key_env(self, env: dict[str, str]) -> None:
        """Apply environment for explicit API key mode (Bedrock disabled)."""
        env["CLAUDE_CODE_USE_BEDROCK"] = "0"
        env["CLAUDE_USE_BEDROCK"] = "0"
        env["DISABLE_AUTOUPDATER"] = "1"
        env["ANTHROPIC_API_KEY"] = self._isolation.api_key  # type: ignore[assignment]
        _logger.debug(
            "isolation.get_env.explicit_api_key",
            extra={"auth_mode": "explicit_api_key", "bedrock_disabled": True},
        )

    @staticmethod
    def _apply_inherit_host_auth_env(env: dict[str, str]) -> None:
        """Apply environment for inherit host auth mode."""
        env["DISABLE_AUTOUPDATER"] = "1"

        # Pass through AWS env vars for Bedrock authentication
        # Note: ~/.aws is copied to ephemeral home, so SDK finds config at $HOME/.aws
        # Also inherit auth vars from host settings.json if not in shell env
        host_settings = _read_host_claude_settings()
        host_env = host_settings.get("env", {})

        aws_vars_found = []
        aws_vars_missing = []
        for key in _AWS_PASSTHROUGH_VARS:
            if key in os.environ:
                env[key] = os.environ[key]
                # Don't log sensitive values
                if _is_sensitive_key(key):
                    aws_vars_found.append(f"{key}=<redacted>")
                else:
                    aws_vars_found.append(f"{key}={os.environ[key]}")
            elif key in host_env:
                # Fall back to host settings.json
                env[key] = host_env[key]
                if _is_sensitive_key(key):
                    aws_vars_found.append(f"{key}=<redacted>(from_host_settings)")
                else:
                    aws_vars_found.append(f"{key}={host_env[key]}(from_host_settings)")
            else:
                aws_vars_missing.append(key)

        # Pass through Anthropic API key if present (for non-Bedrock setups)
        has_anthropic_key = "ANTHROPIC_API_KEY" in os.environ
        if has_anthropic_key:
            env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

        _logger.debug(
            "isolation.get_env.inherit_host_auth",
            extra={
                "auth_mode": "inherit_host_auth",
                "aws_vars_found": aws_vars_found,
                "aws_vars_missing": aws_vars_missing,
                "has_anthropic_key": has_anthropic_key,
                "bedrock_enabled": env.get("CLAUDE_CODE_USE_BEDROCK") == "1",
                "host_settings_found": bool(host_settings),
            },
        )

    def _apply_user_env_overrides(self, env: dict[str, str]) -> None:
        """Apply user-provided env vars (highest priority)."""
        if self._isolation.env:
            env.update(self._isolation.env)

    def _log_final_env(self, env: dict[str, str]) -> None:
        """Log the final environment configuration."""
        _logger.debug(
            "isolation.get_env.complete",
            extra={
                "ephemeral_home": self._temp_dir,
                "env_var_count": len(env),
                "env_keys": sorted(
                    k
                    for k in env
                    if "KEY" not in k and "SECRET" not in k and "TOKEN" not in k
                ),
            },
        )

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
        if not getattr(self, "_cleaned_up", True):
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

    @property
    def skills_dir(self) -> Path:
        """Path to the skills directory within ephemeral home."""
        return self._claude_dir / "skills"

    def __enter__(self) -> EphemeralHome:
        """Context manager entry."""
        return self

    def __exit__(self, *_: object) -> None:
        """Context manager exit with automatic cleanup."""
        self.cleanup()

    def __del__(self) -> None:
        """Destructor that attempts cleanup if not already done."""
        self.cleanup()

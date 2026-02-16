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

"""Ephemeral home directory management for hermetic isolation.

Manages temporary home directories for Claude Agent SDK subprocesses,
preventing access to the host's ~/.claude configuration.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from .isolation import IsolationConfig, NetworkPolicy, SandboxConfig

from ...skills import (
    MAX_SKILL_TOTAL_BYTES,
    SkillMount,
    SkillMountError,
    SkillNotFoundError,
    resolve_skill_name,
    validate_skill,
    validate_skill_name,
)

_logger = logging.getLogger(__name__)

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
        self._copy_aws_config()
        self._cleaned_up = False
        self._skills_mounted = False

    def _generate_settings(self) -> None:
        """Generate settings.json from IsolationConfig."""
        settings: dict[str, Any] = {}

        self._configure_sandbox_settings(settings)
        self._configure_auth_settings(settings)
        self._write_settings(settings)

    def _configure_sandbox_settings(self, settings: dict[str, Any]) -> None:
        """Configure sandbox and network settings for settings.json."""
        from .isolation import NetworkPolicy, SandboxConfig

        sandbox = self._isolation.sandbox or SandboxConfig()
        network = self._isolation.network_policy or NetworkPolicy.no_network()

        section: dict[str, Any] = {
            "enabled": sandbox.enabled,
            "autoAllowBashIfSandboxed": sandbox.bash_auto_allow,
            "allowUnsandboxedCommands": sandbox.allow_unsandboxed_commands,
        }
        if sandbox.excluded_commands:
            section["excludedCommands"] = list(sandbox.excluded_commands)
        if sandbox.enable_weaker_nested_sandbox:
            section["enableWeakerNestedSandbox"] = True

        violations: dict[str, list[str]] = {}
        if sandbox.ignore_file_violations:
            violations["file"] = list(sandbox.ignore_file_violations)
        if sandbox.ignore_network_violations:
            violations["network"] = list(sandbox.ignore_network_violations)
        if violations:
            section["ignoreViolations"] = violations

        self._add_sandbox_paths(section, sandbox)
        section["network"] = self._build_network_section(network)
        settings["sandbox"] = section

    @staticmethod
    def _add_sandbox_paths(section: dict[str, Any], sandbox: SandboxConfig) -> None:
        writable_paths: list[str] = list(sandbox.writable_paths)
        if sandbox.enabled:
            claude_temp_dir = f"/tmp/claude-{os.getuid()}"  # nosec B108
            if claude_temp_dir not in writable_paths:
                writable_paths.append(claude_temp_dir)
        if writable_paths:
            section["writablePaths"] = writable_paths
        if sandbox.readable_paths:
            section["readablePaths"] = list(sandbox.readable_paths)

    @staticmethod
    def _build_network_section(network: NetworkPolicy) -> dict[str, Any]:
        net: dict[str, Any] = {}
        if network.allow_unix_sockets:
            net["allowUnixSockets"] = list(network.allow_unix_sockets)
        if network.allow_all_unix_sockets:
            net["allowAllUnixSockets"] = True
        if network.allow_local_binding:
            net["allowLocalBinding"] = True
        if network.http_proxy_port is not None:
            net["httpProxyPort"] = network.http_proxy_port
        if network.socks_proxy_port is not None:
            net["socksProxyPort"] = network.socks_proxy_port
        net["allowedDomains"] = list(network.allowed_domains)
        return net

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
        from .isolation import _read_host_claude_settings

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

    def mount_skills(
        self,
        skills: tuple[SkillMount, ...],
        *,
        validate: bool = True,
    ) -> None:
        """Mount skills into the ephemeral home.

        Skills are typically collected from the rendered prompt and passed
        to this method by the adapter before spawning Claude Code.

        This method can only be called once per EphemeralHome instance.

        Args:
            skills: Tuple of SkillMount instances from RenderedPrompt.skills.
            validate: If True, validate skill structure before copying.

        Raises:
            SkillMountError: If skills have already been mounted, if a skill
                name is duplicated, or if copy fails.
            SkillNotFoundError: If a skill source path does not exist.
            SkillValidationError: If validation is enabled and a skill is invalid.
        """
        if self._skills_mounted:
            raise SkillMountError("Skills already mounted on this ephemeral home")
        self._skills_mounted = True

        if not skills:
            return

        skills_dir = self._claude_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        seen_names: set[str] = set()
        for mount in skills:
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

            if validate:
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
        from .isolation import IsolationAuthError

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
        from .isolation import IsolationAuthError, _is_bedrock_configured

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
        api_key = self._isolation.api_key
        if api_key is None:  # pragma: no cover
            msg = "api_key must not be None when using explicit API key mode"
            raise ValueError(msg)
        env["CLAUDE_CODE_USE_BEDROCK"] = "0"
        env["CLAUDE_USE_BEDROCK"] = "0"
        env["DISABLE_AUTOUPDATER"] = "1"
        env["ANTHROPIC_API_KEY"] = api_key
        _logger.debug(
            "isolation.get_env.explicit_api_key",
            extra={"auth_mode": "explicit_api_key", "bedrock_disabled": True},
        )

    @staticmethod
    def _apply_inherit_host_auth_env(env: dict[str, str]) -> None:
        """Apply environment for inherit host auth mode."""
        from .isolation import _read_host_claude_settings

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

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
from ...skills import (
    MAX_SKILL_TOTAL_BYTES,
    SkillConfig,
    SkillMountError,
    SkillNotFoundError,
    resolve_skill_name,
    validate_skill,
    validate_skill_name,
)

__all__ = [
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
class IsolationConfig:
    """Configuration for hermetic SDK isolation.

    When provided to the adapter, creates an ephemeral home directory with
    generated settings, preventing any interaction with the host's ~/.claude
    configuration, credentials, and session state.

    Attributes:
        network_policy: Network access constraints. None means no network access.
        sandbox: Sandbox configuration. None uses secure defaults.
        env: Additional environment variables for the SDK subprocess.
        api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY from
            the host environment.
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
    include_host_env: bool = False
    skills: SkillConfig | None = None


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
        self._mount_skills()
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

        # Explicitly disable Bedrock and other alternative providers via env section.
        # This is critical for hermetic tests - the host may have Claude configured
        # for AWS Bedrock, and the SDK's settings.json env section can override
        # environment variables. We must explicitly disable these to force Anthropic
        # API usage with the provided ANTHROPIC_API_KEY.
        settings["env"] = {
            "CLAUDE_CODE_USE_BEDROCK": "0",
            "CLAUDE_USE_BEDROCK": "0",
            "DISABLE_AUTOUPDATER": "1",
        }

        # Write settings
        settings_path = self._claude_dir / "settings.json"
        settings_path.write_text(json.dumps(settings, indent=2))

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

    def get_env(self) -> dict[str, str]:
        """Build environment variables for SDK subprocess.

        Returns:
            Dictionary of environment variables to pass to the SDK subprocess.
            Includes HOME pointing to the ephemeral directory and any
            configured API keys.
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

        # Explicitly disable Bedrock and other alternative providers.
        # This is critical for hermetic tests - the host may have Claude
        # configured for AWS Bedrock, and we must force Anthropic API usage.
        # Set these BEFORE the API key so they take precedence.
        env["CLAUDE_CODE_USE_BEDROCK"] = "0"
        env["CLAUDE_USE_BEDROCK"] = "0"
        env["DISABLE_AUTOUPDATER"] = "1"

        # Set API key
        if self._isolation.api_key:
            env["ANTHROPIC_API_KEY"] = self._isolation.api_key
        elif "ANTHROPIC_API_KEY" in os.environ:
            env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

        # Merge user-provided env vars (highest priority)
        if self._isolation.env:
            env.update(self._isolation.env)

        return env

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

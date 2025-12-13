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
    "EphemeralHome",
    "IsolationConfig",
    "NetworkPolicy",
    "SandboxConfig",
]


@FrozenDataclass()
class NetworkPolicy:
    """Network access policy for Claude Agent SDK sandbox.

    Controls which network resources the SDK can access during execution.
    All constraints are enforced at the OS level via bubblewrap (Linux) or
    seatbelt (macOS).

    Attributes:
        allowed_domains: Domains the SDK can access. Empty tuple means no
            network access. Use ("*",) for unrestricted access (not recommended).
        allow_localhost: If True, allow connections to localhost/127.0.0.1.
            Required for local MCP servers or development APIs.
        allow_unix_sockets: If True, allow Unix domain socket connections.
            Security risk—enables Docker socket access if available.
        allowed_ports: Specific ports allowed for outbound connections.
            None means all ports on allowed domains. Empty tuple blocks all.
    """

    allowed_domains: tuple[str, ...] = ()
    allow_localhost: bool = False
    allow_unix_sockets: bool = False
    allowed_ports: tuple[int, ...] | None = None

    @classmethod
    def no_network(cls) -> NetworkPolicy:
        """Create a policy that blocks all network access."""
        return cls(allowed_domains=())

    @classmethod
    def api_only(cls) -> NetworkPolicy:
        """Create a policy allowing only Anthropic API access."""
        return cls(allowed_domains=("api.anthropic.com",))

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
    """

    network_policy: NetworkPolicy | None = None
    sandbox: SandboxConfig | None = None
    env: Mapping[str, str] | None = None
    api_key: str | None = None
    include_host_env: bool = False


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
        ...     network_policy=NetworkPolicy.api_only(),
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

        if network.allow_localhost:
            settings["sandbox"]["network"]["allowLocalBinding"] = True

        if network.allow_unix_sockets:
            settings["sandbox"]["allowUnixSockets"] = True

        if network.allowed_ports is not None:
            settings["sandbox"]["network"]["allowedPorts"] = list(network.allowed_ports)

        # Write settings
        settings_path = self._claude_dir / "settings.json"
        settings_path.write_text(json.dumps(settings, indent=2))

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

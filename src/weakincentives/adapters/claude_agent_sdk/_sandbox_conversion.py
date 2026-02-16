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

"""Conversion between WINK isolation configs and Claude Agent SDK types.

This module is the **single source of truth** for the field mapping between
:class:`~weakincentives.adapters.claude_agent_sdk.isolation.SandboxConfig` /
:class:`~weakincentives.adapters.claude_agent_sdk.isolation.NetworkPolicy`
and the SDK's ``SandboxSettings`` / ``SandboxNetworkConfig`` /
``SandboxIgnoreViolations`` TypedDicts.

Both the ``ClaudeAgentOptions.sandbox`` path and the ``settings.json``
generation path delegate to these converters so the mapping is never
duplicated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from claude_agent_sdk import (
    SandboxIgnoreViolations,
    SandboxNetworkConfig,
    SandboxSettings,
)

if TYPE_CHECKING:
    from .isolation import NetworkPolicy, SandboxConfig


def to_sdk_network_config(network: NetworkPolicy) -> SandboxNetworkConfig:
    """Convert a WINK :class:`NetworkPolicy` to an SDK ``SandboxNetworkConfig``.

    Only non-default values are included so the SDK applies its own defaults
    for omitted keys.
    """
    config: SandboxNetworkConfig = {}
    if network.allow_unix_sockets:
        config["allowUnixSockets"] = list(network.allow_unix_sockets)
    if network.allow_all_unix_sockets:
        config["allowAllUnixSockets"] = True
    if network.allow_local_binding:
        config["allowLocalBinding"] = True
    if network.http_proxy_port is not None:
        config["httpProxyPort"] = network.http_proxy_port
    if network.socks_proxy_port is not None:
        config["socksProxyPort"] = network.socks_proxy_port
    return config


def to_sdk_ignore_violations(
    sandbox: SandboxConfig,
) -> SandboxIgnoreViolations | None:
    """Convert violation fields to an SDK ``SandboxIgnoreViolations``.

    Returns ``None`` when no violations are configured so callers can skip
    the key entirely.
    """
    violations: SandboxIgnoreViolations = {}
    if sandbox.ignore_file_violations:
        violations["file"] = list(sandbox.ignore_file_violations)
    if sandbox.ignore_network_violations:
        violations["network"] = list(sandbox.ignore_network_violations)
    return violations or None


def to_sdk_sandbox_settings(
    sandbox: SandboxConfig | None,
    network: NetworkPolicy | None,
) -> SandboxSettings:
    """Convert WINK isolation configs to an SDK ``SandboxSettings``.

    This is the canonical mapping used by both the ``ClaudeAgentOptions``
    path and the ``settings.json`` generation path.
    """
    from .isolation import NetworkPolicy as NP, SandboxConfig as SC

    sandbox = sandbox or SC()
    network = network or NP.no_network()

    settings: SandboxSettings = {
        "enabled": sandbox.enabled,
        "autoAllowBashIfSandboxed": sandbox.bash_auto_allow,
        "allowUnsandboxedCommands": sandbox.allow_unsandboxed_commands,
    }

    if sandbox.excluded_commands:
        settings["excludedCommands"] = list(sandbox.excluded_commands)
    if sandbox.enable_weaker_nested_sandbox:
        settings["enableWeakerNestedSandbox"] = True

    violations = to_sdk_ignore_violations(sandbox)
    if violations is not None:
        settings["ignoreViolations"] = violations

    net_config = to_sdk_network_config(network)
    if net_config:
        settings["network"] = net_config

    return settings


def to_settings_json_sandbox(
    sandbox: SandboxConfig | None,
    network: NetworkPolicy | None,
) -> dict[str, Any]:
    """Build the ``settings.json`` sandbox section.

    Starts from :func:`to_sdk_sandbox_settings` (shared mapping) and adds
    the settings.json-only keys (``writablePaths``, ``readablePaths``,
    ``allowedDomains``) that are not part of the SDK ``SandboxSettings``
    TypedDict.
    """
    import os

    from .isolation import NetworkPolicy as NP, SandboxConfig as SC

    sandbox = sandbox or SC()
    network = network or NP.no_network()

    # Start with the canonical SDK mapping
    section: dict[str, Any] = dict(to_sdk_sandbox_settings(sandbox, network))

    # --- settings.json-only keys ---

    # Writable paths (not in SandboxSettings TypedDict)
    writable_paths: list[str] = list(sandbox.writable_paths)
    if sandbox.enabled:
        claude_temp_dir = f"/tmp/claude-{os.getuid()}"  # nosec B108
        if claude_temp_dir not in writable_paths:
            writable_paths.append(claude_temp_dir)
    if writable_paths:
        section["writablePaths"] = writable_paths

    # Readable paths (not in SandboxSettings TypedDict)
    if sandbox.readable_paths:
        section["readablePaths"] = list(sandbox.readable_paths)

    # allowedDomains lives under network in settings.json but is not part
    # of the SDK SandboxNetworkConfig TypedDict.
    network_section: dict[str, Any] = section.get("network", {})
    network_section["allowedDomains"] = list(network.allowed_domains)
    section["network"] = network_section

    return section

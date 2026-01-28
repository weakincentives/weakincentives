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

"""Podman connection resolution utilities.

This module handles Podman connection discovery and resolution, including:

- Environment variable-based configuration (PODMAN_BASE_URL, PODMAN_IDENTITY)
- CLI-based connection lookup via ``podman system connection list``
- Default connection fallback

Example usage::

    from weakincentives.contrib.tools.podman_connection import (
        resolve_podman_connection,
        resolve_connection_settings,
    )

    # Resolve using environment or CLI defaults
    conn = resolve_podman_connection()
    if conn is not None:
        print(f"Using connection: {conn.base_url}")

    # Resolve with full validation for section initialization
    base_url, identity, name = resolve_connection_settings(
        base_url=None,
        identity=None,
        connection_name="my-connection",
    )
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec: B404
from typing import Any, Final

from ...dataclasses import FrozenDataclass
from ...errors import ToolValidationError

_PODMAN_BASE_URL_ENV: Final[str] = "PODMAN_BASE_URL"
_PODMAN_IDENTITY_ENV: Final[str] = "PODMAN_IDENTITY"
_PODMAN_CONNECTION_ENV: Final[str] = "PODMAN_CONNECTION"


@FrozenDataclass()
class PodmanConnectionInfo:
    """Resolved Podman connection information.

    Contains the base URL, identity file path, and connection name
    discovered from environment variables or the Podman CLI.
    """

    base_url: str | None
    identity: str | None
    connection_name: str | None


def resolve_podman_connection(
    *,
    preferred_name: str | None = None,
) -> PodmanConnectionInfo | None:
    """Resolve Podman connection from environment or CLI.

    Resolution order:
    1. Environment variables (PODMAN_BASE_URL, PODMAN_IDENTITY)
    2. Named connection via ``podman system connection list``
    3. Default connection from CLI

    Args:
        preferred_name: Optional connection name to look up.

    Returns:
        Resolved connection info, or None if no connection found.
    """
    env_base_url = os.environ.get(_PODMAN_BASE_URL_ENV)
    env_identity = os.environ.get(_PODMAN_IDENTITY_ENV)
    env_connection = os.environ.get(_PODMAN_CONNECTION_ENV)
    if env_base_url or env_identity:
        return PodmanConnectionInfo(
            base_url=env_base_url,
            identity=env_identity,
            connection_name=preferred_name or env_connection,
        )
    resolved_name = preferred_name or env_connection
    return _connection_from_cli(resolved_name)


def find_connection_by_name(
    connections: list[dict[str, Any]], connection_name: str
) -> dict[str, Any] | None:
    """Find a connection by name in a list of connection entries.

    Args:
        connections: List of connection dictionaries from Podman CLI.
        connection_name: Name of the connection to find.

    Returns:
        Matching connection entry, or None if not found.
    """
    for entry in connections:
        if entry.get("Name") == connection_name:
            return entry
    return None


def find_default_connection(
    connections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the default connection or first available.

    Args:
        connections: List of connection dictionaries from Podman CLI.

    Returns:
        Default connection entry, first entry if no default, or None if empty.
    """
    for entry in connections:
        if entry.get("Default"):
            return entry
    return connections[0] if connections else None


def _connection_from_cli(
    connection_name: str | None,
) -> PodmanConnectionInfo | None:
    """Resolve connection information from the Podman CLI.

    Runs ``podman system connection list --format json`` to discover
    available connections.

    Args:
        connection_name: Optional name to look up, otherwise uses default.

    Returns:
        Resolved connection info, or None if CLI fails or no connections.
    """
    try:
        result = subprocess.run(  # nosec B603 B607
            ["podman", "system", "connection", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        connections = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    candidate = (
        find_connection_by_name(connections, connection_name)
        if connection_name
        else find_default_connection(connections)
    )
    if candidate is None:
        return None
    return PodmanConnectionInfo(
        base_url=candidate.get("URI"),
        identity=candidate.get("Identity"),
        connection_name=candidate.get("Name"),
    )


def resolve_connection_settings(
    *,
    base_url: str | None,
    identity: str | os.PathLike[str] | None,
    connection_name: str | None,
) -> tuple[str, str | None, str | None]:
    """Resolve and validate Podman connection settings.

    Combines explicit configuration with environment and CLI discovery
    to produce a complete set of connection settings.

    Args:
        base_url: Explicit base URL, or None to discover.
        identity: Explicit identity file path, or None to discover.
        connection_name: Preferred connection name for lookup.

    Returns:
        Tuple of (base_url, identity, connection_name).

    Raises:
        ToolValidationError: If no valid connection could be resolved.
    """
    env_connection = os.environ.get(_PODMAN_CONNECTION_ENV)
    preferred_connection = connection_name or env_connection
    resolved_connection: PodmanConnectionInfo | None = None
    if base_url is None or identity is None:
        resolved_connection = resolve_podman_connection(
            preferred_name=preferred_connection
        )
    resolved_base_url = base_url or (
        resolved_connection.base_url if resolved_connection is not None else None
    )
    resolved_identity = identity or (
        resolved_connection.identity if resolved_connection is not None else None
    )
    resolved_connection_name = (
        connection_name
        or (
            resolved_connection.connection_name
            if resolved_connection is not None
            else None
        )
        or env_connection
    )
    if resolved_base_url is None:
        message = (
            "Podman connection could not be resolved. Configure `podman system connection` {}"
        ).format("or set PODMAN_BASE_URL/PODMAN_IDENTITY.")
        raise ToolValidationError(message)
    identity_str = str(resolved_identity) if resolved_identity is not None else None
    return resolved_base_url, identity_str, resolved_connection_name


__all__ = [
    "PodmanConnectionInfo",
    "find_connection_by_name",
    "find_default_connection",
    "resolve_connection_settings",
    "resolve_podman_connection",
]

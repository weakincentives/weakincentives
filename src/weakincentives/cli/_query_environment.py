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

"""Environment table builders for the query module.

Builds env_system, env_python, env_git, env_container, env_vars, and
the flat environment key-value table from debug bundle environment data.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import cast

from ..debug import DebugBundle
from ..types import JSONValue
from ._query_helpers import _flatten_json

__all__ = [
    "_build_environment_tables",
]


def _build_environment_tables(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Build environment tables from environment/ directory."""
    env_data = bundle.environment
    if not env_data:
        _create_empty_environment_tables(conn)
        return
    _build_env_system_table(conn, env_data.get("system"))
    _build_env_python_table(conn, env_data.get("python"))
    _build_env_git_table(conn, env_data.get("git"))
    _build_env_container_table(conn, env_data.get("container"))
    _build_env_vars_table(conn, env_data.get("env_vars"))
    _build_environment_flat_table(conn, env_data)


def _create_empty_environment_tables(conn: sqlite3.Connection) -> None:
    """Create empty environment tables when no environment data exists."""
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS env_system (
            os_name TEXT, os_release TEXT, kernel_version TEXT,
            architecture TEXT, processor TEXT, cpu_count INTEGER,
            memory_total_bytes INTEGER, hostname TEXT
        )
    """)
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS env_python (
            version TEXT, version_info TEXT, implementation TEXT,
            executable TEXT, prefix TEXT, base_prefix TEXT,
            is_virtualenv INTEGER
        )
    """)
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS env_git (
            repo_root TEXT, commit_sha TEXT, commit_short TEXT,
            branch TEXT, is_dirty INTEGER, remotes TEXT, tags TEXT
        )
    """)
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS env_container (
            runtime TEXT, container_id TEXT, image TEXT,
            image_digest TEXT, cgroup_path TEXT, is_containerized INTEGER
        )
    """)
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS env_vars (
            rowid INTEGER PRIMARY KEY, name TEXT, value TEXT
        )
    """)
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS environment (
            rowid INTEGER PRIMARY KEY, key TEXT, value TEXT
        )
    """)


def _build_env_system_table(conn: sqlite3.Connection, data: JSONValue | None) -> None:
    """Build env_system table from system.json."""
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS env_system (
            os_name TEXT, os_release TEXT, kernel_version TEXT,
            architecture TEXT, processor TEXT, cpu_count INTEGER,
            memory_total_bytes INTEGER, hostname TEXT
        )
    """)
    if not data or not isinstance(data, Mapping):
        return
    system = cast("Mapping[str, JSONValue]", data)
    _ = conn.execute(
        """INSERT INTO env_system (
            os_name, os_release, kernel_version, architecture,
            processor, cpu_count, memory_total_bytes, hostname
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            system.get("os_name"),
            system.get("os_release"),
            system.get("kernel_version"),
            system.get("architecture"),
            system.get("processor"),
            system.get("cpu_count"),
            system.get("memory_total_bytes"),
            system.get("hostname"),
        ),
    )


def _build_env_python_table(conn: sqlite3.Connection, data: JSONValue | None) -> None:
    """Build env_python table from python.json."""
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS env_python (
            version TEXT, version_info TEXT, implementation TEXT,
            executable TEXT, prefix TEXT, base_prefix TEXT,
            is_virtualenv INTEGER
        )
    """)
    if not data or not isinstance(data, Mapping):
        return
    python = cast("Mapping[str, JSONValue]", data)
    version_info = python.get("version_info")
    version_info_str = json.dumps(version_info) if version_info is not None else None
    is_venv = python.get("is_virtualenv")
    _ = conn.execute(
        """INSERT INTO env_python (
            version, version_info, implementation, executable,
            prefix, base_prefix, is_virtualenv
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            python.get("version"),
            version_info_str,
            python.get("implementation"),
            python.get("executable"),
            python.get("prefix"),
            python.get("base_prefix"),
            1 if is_venv else 0 if is_venv is not None else None,
        ),
    )


def _build_env_git_table(conn: sqlite3.Connection, data: JSONValue | None) -> None:
    """Build env_git table from git.json."""
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS env_git (
            repo_root TEXT, commit_sha TEXT, commit_short TEXT,
            branch TEXT, is_dirty INTEGER, remotes TEXT, tags TEXT
        )
    """)
    if not data or not isinstance(data, Mapping):
        return
    git = cast("Mapping[str, JSONValue]", data)
    remotes = git.get("remotes")
    remotes_str = json.dumps(remotes) if remotes is not None else None
    tags = git.get("tags")
    tags_str = json.dumps(tags) if tags is not None else None
    is_dirty = git.get("is_dirty")
    _ = conn.execute(
        """INSERT INTO env_git (
            repo_root, commit_sha, commit_short, branch,
            is_dirty, remotes, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            git.get("repo_root"),
            git.get("commit_sha"),
            git.get("commit_short"),
            git.get("branch"),
            1 if is_dirty else 0 if is_dirty is not None else None,
            remotes_str,
            tags_str,
        ),
    )


def _build_env_container_table(
    conn: sqlite3.Connection, data: JSONValue | None
) -> None:
    """Build env_container table from container.json."""
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS env_container (
            runtime TEXT, container_id TEXT, image TEXT,
            image_digest TEXT, cgroup_path TEXT, is_containerized INTEGER
        )
    """)
    if not data or not isinstance(data, Mapping):
        return
    container = cast("Mapping[str, JSONValue]", data)
    is_containerized = container.get("is_containerized")
    _ = conn.execute(
        """INSERT INTO env_container (
            runtime, container_id, image, image_digest,
            cgroup_path, is_containerized
        ) VALUES (?, ?, ?, ?, ?, ?)""",
        (
            container.get("runtime"),
            container.get("container_id"),
            container.get("image"),
            container.get("image_digest"),
            container.get("cgroup_path"),
            1 if is_containerized else 0 if is_containerized is not None else None,
        ),
    )


def _build_env_vars_table(conn: sqlite3.Connection, data: JSONValue | None) -> None:
    """Build env_vars table from env_vars.json."""
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS env_vars (
            rowid INTEGER PRIMARY KEY, name TEXT, value TEXT
        )
    """)
    if not data or not isinstance(data, Mapping):
        return
    env_vars = cast("Mapping[str, JSONValue]", data)
    for name, value in env_vars.items():
        _ = conn.execute(
            "INSERT INTO env_vars (name, value) VALUES (?, ?)",
            (name, str(value) if value is not None else None),
        )


def _build_environment_flat_table(
    conn: sqlite3.Connection, env_data: Mapping[str, JSONValue | str | None]
) -> None:
    """Build flat environment table with all data as key-value pairs."""
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS environment (
            rowid INTEGER PRIMARY KEY, key TEXT, value TEXT
        )
    """)
    for category in ("system", "python", "git", "container"):
        category_data = env_data.get(category)
        if category_data and isinstance(category_data, Mapping):
            flattened = _flatten_json(category_data, prefix=category)
            for key, value in flattened.items():
                _ = conn.execute(
                    "INSERT INTO environment (key, value) VALUES (?, ?)",
                    (key, str(value) if value is not None else None),
                )
    packages = env_data.get("packages")
    if packages:
        _ = conn.execute(
            "INSERT INTO environment (key, value) VALUES (?, ?)",
            ("packages", str(packages)),
        )
    command = env_data.get("command")
    if command:
        _ = conn.execute(
            "INSERT INTO environment (key, value) VALUES (?, ?)",
            ("command", str(command)),
        )
    git_diff = env_data.get("git_diff")
    if git_diff:
        _ = conn.execute(
            "INSERT INTO environment (key, value) VALUES (?, ?)",
            ("git_diff", str(git_diff)),
        )

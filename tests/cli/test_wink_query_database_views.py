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

"""Views, schema hints, and environment table tests for the wink query command."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tests.cli._query_fixtures import (
    create_bundle_with_logs,
    create_test_bundle,
)
from weakincentives.cli._query_helpers import _get_table_description
from weakincentives.cli.query import open_query_database


class TestSchemaHints:
    """Tests for schema hints in schema output."""

    def test_schema_includes_hints(self, tmp_path: Path) -> None:
        """Test that schema output includes hints section."""
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            schema = db.get_schema()
            assert schema.hints is not None
            assert len(schema.hints.json_extraction) > 0
            assert len(schema.hints.common_queries) > 0
        finally:
            db.close()

    def test_hints_serializes_to_json(self, tmp_path: Path) -> None:
        """Test that hints are included in JSON output."""
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            schema = db.get_schema()
            json_str = schema.to_json()
            data = json.loads(json_str)
            assert "hints" in data
            assert "json_extraction" in data["hints"]
            assert "common_queries" in data["hints"]
        finally:
            db.close()


class TestQueryViews:
    """Tests for SQL views."""

    def test_tool_timeline_view_exists(self, tmp_path: Path) -> None:
        """Test that tool_timeline view is created."""
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            # Should be able to query the view without error
            results = db.execute_query("SELECT * FROM tool_timeline LIMIT 1")
            assert isinstance(results, list)
        finally:
            db.close()

    def test_native_tool_calls_view_exists(self, tmp_path: Path) -> None:
        """Test that native_tool_calls view is created."""
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            results = db.execute_query("SELECT * FROM native_tool_calls LIMIT 1")
            assert isinstance(results, list)
        finally:
            db.close()

    def test_error_summary_view_exists(self, tmp_path: Path) -> None:
        """Test that error_summary view is created."""
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            results = db.execute_query("SELECT * FROM error_summary LIMIT 1")
            assert isinstance(results, list)
        finally:
            db.close()

    def test_transcript_entries_view_extracts_fields(self, tmp_path: Path) -> None:
        """Test that transcript_entries view extracts key fields from logs."""
        bundle_path = create_bundle_with_logs(tmp_path)
        db = open_query_database(bundle_path)
        try:
            results = db.execute_query(
                """
                SELECT prompt_name, transcript_source, sequence_number, entry_type, role, content
                FROM transcript_entries
                ORDER BY transcript_source, sequence_number
            """
            )
            assert len(results) == 1
            assert results[0]["prompt_name"] == "test-prompt"
            assert results[0]["transcript_source"] == "main"
            assert results[0]["sequence_number"] == 1
            assert results[0]["entry_type"] == "user_message"
            assert results[0]["role"] == "user"
            assert results[0]["content"] == "Hello"
        finally:
            db.close()

    def test_views_included_in_schema(self, tmp_path: Path) -> None:
        """Test that views appear in schema output."""
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            schema = db.get_schema()
            table_names = [t.name for t in schema.tables]
            assert "tool_timeline" in table_names
            assert "native_tool_calls" in table_names
            assert "transcript_entries" in table_names
            assert "error_summary" in table_names
        finally:
            db.close()


class TestEnvironmentTables:
    """Tests for environment data tables."""

    def test_environment_tables_created_without_env_data(self, tmp_path: Path) -> None:
        """Test that environment tables are created even without data."""
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            # Tables should exist but be empty
            for table in [
                "env_system",
                "env_python",
                "env_git",
                "env_container",
                "env_vars",
                "environment",
            ]:
                results = db.execute_query(f"SELECT * FROM {table}")
                assert isinstance(results, list)
        finally:
            db.close()

    def test_environment_tables_populated(self, tmp_path: Path) -> None:
        """Test that environment tables are populated when data exists."""
        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-env",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": ["environment/system.json", "environment/python.json"],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        system_data = {
            "os_name": "Linux",
            "os_release": "5.15.0",
            "kernel_version": "5.15.0-generic",
            "architecture": "x86_64",
            "processor": "x86_64",
            "cpu_count": 8,
            "memory_total_bytes": 16000000000,
            "hostname": "testhost",
        }

        python_data = {
            "version": "3.11.5",
            "version_info": [3, 11, 5],
            "implementation": "CPython",
            "executable": "/usr/bin/python3",
            "prefix": "/usr",
            "base_prefix": "/usr",
            "is_virtualenv": False,
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/environment/system.json", json.dumps(system_data))
            zf.writestr("debug_bundle/environment/python.json", json.dumps(python_data))
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            # Check env_system table
            system_results = db.execute_query("SELECT * FROM env_system")
            assert len(system_results) == 1
            assert system_results[0]["os_name"] == "Linux"
            assert system_results[0]["architecture"] == "x86_64"
            assert system_results[0]["cpu_count"] == 8

            # Check env_python table
            python_results = db.execute_query("SELECT * FROM env_python")
            assert len(python_results) == 1
            assert python_results[0]["implementation"] == "CPython"
            assert python_results[0]["is_virtualenv"] == 0

            # Check flat environment table
            env_results = db.execute_query(
                "SELECT key, value FROM environment WHERE key = 'system_os_name'"
            )
            assert len(env_results) == 1
            assert env_results[0]["value"] == "Linux"
        finally:
            db.close()

    def test_env_git_table_populated(self, tmp_path: Path) -> None:
        """Test that env_git table is populated with git info."""
        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-env-git",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": ["environment/git.json"],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        git_data = {
            "repo_root": "/home/user/project",
            "commit_sha": "abc123def456",
            "commit_short": "abc123de",
            "branch": "main",
            "is_dirty": True,
            "remotes": {"origin": "https://github.com/user/project.git"},
            "tags": ["v1.0.0"],
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/environment/git.json", json.dumps(git_data))
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            git_results = db.execute_query("SELECT * FROM env_git")
            assert len(git_results) == 1
            assert git_results[0]["branch"] == "main"
            assert git_results[0]["is_dirty"] == 1
            assert git_results[0]["commit_short"] == "abc123de"
        finally:
            db.close()

    def test_env_vars_table_populated(self, tmp_path: Path) -> None:
        """Test that env_vars table is populated with environment variables."""
        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-env-vars",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": ["environment/env_vars.json"],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        env_vars_data = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "PYTHON_VERSION": "3.11.5",
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr(
                "debug_bundle/environment/env_vars.json", json.dumps(env_vars_data)
            )
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            env_vars_results = db.execute_query(
                "SELECT name, value FROM env_vars ORDER BY name"
            )
            assert len(env_vars_results) == 3
            names = [r["name"] for r in env_vars_results]
            assert "HOME" in names
            assert "PATH" in names
        finally:
            db.close()

    def test_environment_flat_table_includes_packages(self, tmp_path: Path) -> None:
        """Test that flat environment table includes packages."""
        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-env-packages",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": ["environment/packages.txt"],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        packages = "fastapi==0.100.0\nuvicorn==0.23.0\npytest==7.4.0"

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/environment/packages.txt", packages)
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            pkg_results = db.execute_query(
                "SELECT value FROM environment WHERE key = 'packages'"
            )
            assert len(pkg_results) == 1
            assert "fastapi==0.100.0" in pkg_results[0]["value"]
        finally:
            db.close()

    def test_environment_table_descriptions(self) -> None:
        """Test that environment tables have descriptions."""
        assert "System/OS" in _get_table_description("env_system")
        assert "Python" in _get_table_description("env_python")
        assert "Git" in _get_table_description("env_git")
        assert "Container" in _get_table_description("env_container")
        assert "environment" in _get_table_description("env_vars").lower()
        assert "environment" in _get_table_description("environment").lower()

    def test_schema_includes_environment_queries(self, tmp_path: Path) -> None:
        """Test that schema hints include environment-related queries."""
        bundle_path = create_test_bundle(tmp_path)
        db = open_query_database(bundle_path)
        try:
            schema = db.get_schema()
            assert schema.hints is not None
            assert "system_info" in schema.hints.common_queries
            assert "python_info" in schema.hints.common_queries
            assert "git_info" in schema.hints.common_queries
            assert "env_vars" in schema.hints.common_queries
        finally:
            db.close()

    def test_env_container_table_populated(self, tmp_path: Path) -> None:
        """Test that env_container table is populated with container info."""
        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-env-container",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": ["environment/container.json"],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        container_data = {
            "runtime": "docker",
            "container_id": "abc123def456",
            "image": "python:3.11",
            "image_digest": "sha256:abc123",
            "cgroup_path": "/docker/abc123def456",
            "is_containerized": True,
        }

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr(
                "debug_bundle/environment/container.json", json.dumps(container_data)
            )
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            container_results = db.execute_query("SELECT * FROM env_container")
            assert len(container_results) == 1
            assert container_results[0]["runtime"] == "docker"
            assert container_results[0]["is_containerized"] == 1
        finally:
            db.close()

    def test_environment_flat_table_includes_command_and_git_diff(
        self, tmp_path: Path
    ) -> None:
        """Test that flat environment table includes command and git_diff."""
        manifest = {
            "format_version": "1.0.0",
            "bundle_id": "test-env-command-diff",
            "created_at": "2024-01-01T00:00:00Z",
            "request": {
                "request_id": "req-1",
                "session_id": "sess-1",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:01Z",
            },
            "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
            "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
            "files": ["environment/command.txt", "environment/git.diff"],
            "integrity": {"algorithm": "sha256", "checksums": {}},
            "build": {"version": "1.0.0", "commit": "abc123"},
        }

        command_data = "python script.py --arg value"
        git_diff_data = "diff --git a/file.py b/file.py\n+new line"

        bundle_path = tmp_path / "test.zip"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("debug_bundle/manifest.json", json.dumps(manifest))
            zf.writestr("debug_bundle/environment/command.txt", command_data)
            zf.writestr("debug_bundle/environment/git.diff", git_diff_data)
            zf.writestr("debug_bundle/logs/app.jsonl", "")
            zf.writestr("debug_bundle/session/after.jsonl", "")
            zf.writestr("debug_bundle/request/input.json", "{}")
            zf.writestr("debug_bundle/request/output.json", "{}")

        db = open_query_database(bundle_path)
        try:
            cmd_results = db.execute_query(
                "SELECT value FROM environment WHERE key = 'command'"
            )
            assert len(cmd_results) == 1
            assert "python script.py" in cmd_results[0]["value"]

            diff_results = db.execute_query(
                "SELECT value FROM environment WHERE key = 'git_diff'"
            )
            assert len(diff_results) == 1
            assert "diff --git" in diff_results[0]["value"]
        finally:
            db.close()

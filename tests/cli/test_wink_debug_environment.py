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

"""Tests for the wink debug app — environment endpoint tests."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from weakincentives.cli import debug_app
from weakincentives.debug import BundleWriter
from weakincentives.debug.bundle import (
    BUNDLE_FORMAT_VERSION,
    BUNDLE_ROOT_DIR,
    BundleConfig,
)
from weakincentives.runtime.session import Session


@dataclass(slots=True, frozen=True)
class _ExampleSlice:
    value: str


def _create_test_bundle(
    target_dir: Path,
    values: list[str],
) -> Path:
    """Create a test debug bundle with session data."""
    session = Session()

    for value in values:
        session.dispatch(_ExampleSlice(value))

    with BundleWriter(
        target_dir,
        config=BundleConfig(),
    ) as writer:
        writer.write_session_after(session)
        writer.write_request_input({"task": "test"})
        writer.write_request_output({"status": "ok"})
        writer.write_config({"adapter": "test"})
        writer.write_metrics({"tokens": 100})

    assert writer.path is not None
    return writer.path


def test_api_environment_endpoint(tmp_path: Path) -> None:
    """Test the /api/environment endpoint returns environment data."""
    bundle_id = str(uuid4())
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    zip_name = f"{bundle_id}_{timestamp}.zip"
    bundle_path = tmp_path / zip_name

    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "bundle_id": bundle_id,
        "created_at": datetime.now(UTC).isoformat(),
        "request": {
            "request_id": "req-1",
            "session_id": "sess-1",
            "status": "success",
            "started_at": datetime.now(UTC).isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
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

    env_vars_data = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/user",
    }

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{BUNDLE_ROOT_DIR}/manifest.json", json.dumps(manifest))
        zf.writestr(f"{BUNDLE_ROOT_DIR}/README.txt", "Test bundle")
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/request/input.json", json.dumps({"task": "test"})
        )
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/request/output.json", json.dumps({"status": "ok"})
        )
        zf.writestr(f"{BUNDLE_ROOT_DIR}/session/after.jsonl", "")
        zf.writestr(f"{BUNDLE_ROOT_DIR}/logs/app.jsonl", "")
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/environment/system.json", json.dumps(system_data)
        )
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/environment/python.json", json.dumps(python_data)
        )
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/environment/env_vars.json", json.dumps(env_vars_data)
        )

    logger = debug_app.get_logger("test.environment")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    env_response = client.get("/api/environment")
    assert env_response.status_code == 200
    env = env_response.json()

    # Check system data
    assert env["system"] is not None
    assert env["system"]["os_name"] == "Linux"
    assert env["system"]["architecture"] == "x86_64"
    assert env["system"]["cpu_count"] == 8

    # Check python data
    assert env["python"] is not None
    assert env["python"]["implementation"] == "CPython"
    assert env["python"]["is_virtualenv"] is False

    # Check env_vars
    assert env["env_vars"] is not None
    assert env["env_vars"]["HOME"] == "/home/user"


def test_api_environment_endpoint_empty(tmp_path: Path) -> None:
    """Test the /api/environment endpoint when no environment data exists."""
    bundle_path = _create_test_bundle(tmp_path, ["one"])
    logger = debug_app.get_logger("test.environment.empty")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    env_response = client.get("/api/environment")
    assert env_response.status_code == 200
    env = env_response.json()

    # Should have null values for missing sections
    assert env["system"] is None
    assert env["python"] is None
    assert env["git"] is None
    assert env["env_vars"] == {}


def test_api_environment_with_git_info(tmp_path: Path) -> None:
    """Test the /api/environment endpoint with git info."""
    bundle_id = str(uuid4())
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    zip_name = f"{bundle_id}_{timestamp}.zip"
    bundle_path = tmp_path / zip_name

    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "bundle_id": bundle_id,
        "created_at": datetime.now(UTC).isoformat(),
        "request": {
            "request_id": "req-1",
            "session_id": "sess-1",
            "status": "success",
            "started_at": datetime.now(UTC).isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
        },
        "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
        "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
        "files": ["environment/git.json"],
        "integrity": {"algorithm": "sha256", "checksums": {}},
        "build": {"version": "1.0.0", "commit": "abc123"},
    }

    git_data = {
        "repo_root": "/home/user/project",
        "commit_sha": "abc123def456789012345678901234567890abcd",
        "commit_short": "abc123de",
        "branch": "main",
        "is_dirty": True,
        "remotes": {"origin": "https://github.com/user/project.git"},
        "tags": ["v1.0.0"],
    }

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{BUNDLE_ROOT_DIR}/manifest.json", json.dumps(manifest))
        zf.writestr(f"{BUNDLE_ROOT_DIR}/README.txt", "Test bundle")
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/request/input.json", json.dumps({"task": "test"})
        )
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/request/output.json", json.dumps({"status": "ok"})
        )
        zf.writestr(f"{BUNDLE_ROOT_DIR}/session/after.jsonl", "")
        zf.writestr(f"{BUNDLE_ROOT_DIR}/logs/app.jsonl", "")
        zf.writestr(f"{BUNDLE_ROOT_DIR}/environment/git.json", json.dumps(git_data))

    logger = debug_app.get_logger("test.environment.git")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    env_response = client.get("/api/environment")
    assert env_response.status_code == 200
    env = env_response.json()

    # Check git data
    assert env["git"] is not None
    assert env["git"]["branch"] == "main"
    assert env["git"]["is_dirty"] is True
    assert env["git"]["commit_short"] == "abc123de"


def test_api_environment_with_container_info(tmp_path: Path) -> None:
    """Test the /api/environment endpoint with container info."""
    bundle_id = str(uuid4())
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    zip_name = f"{bundle_id}_{timestamp}.zip"
    bundle_path = tmp_path / zip_name

    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "bundle_id": bundle_id,
        "created_at": datetime.now(UTC).isoformat(),
        "request": {
            "request_id": "req-1",
            "session_id": "sess-1",
            "status": "success",
            "started_at": datetime.now(UTC).isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
        },
        "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
        "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
        "files": ["environment/container.json"],
        "integrity": {"algorithm": "sha256", "checksums": {}},
        "build": {"version": "1.0.0", "commit": "abc123"},
    }

    container_data = {
        "runtime": "docker",
        "container_id": "abc123def456789",
        "image": "python:3.11-slim",
        "image_digest": "sha256:abc123",
        "cgroup_path": "/docker/abc123def456789",
        "is_containerized": True,
    }

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{BUNDLE_ROOT_DIR}/manifest.json", json.dumps(manifest))
        zf.writestr(f"{BUNDLE_ROOT_DIR}/README.txt", "Test bundle")
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/request/input.json", json.dumps({"task": "test"})
        )
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/request/output.json", json.dumps({"status": "ok"})
        )
        zf.writestr(f"{BUNDLE_ROOT_DIR}/session/after.jsonl", "")
        zf.writestr(f"{BUNDLE_ROOT_DIR}/logs/app.jsonl", "")
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/environment/container.json", json.dumps(container_data)
        )

    logger = debug_app.get_logger("test.environment.container")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    env_response = client.get("/api/environment")
    assert env_response.status_code == 200
    env = env_response.json()

    # Check container data
    assert env["container"] is not None
    assert env["container"]["runtime"] == "docker"
    assert env["container"]["is_containerized"] is True
    assert env["container"]["image"] == "python:3.11-slim"


def test_api_environment_with_non_containerized(tmp_path: Path) -> None:
    """Test /api/environment when is_containerized is False."""
    bundle_id = str(uuid4())
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    zip_name = f"{bundle_id}_{timestamp}.zip"
    bundle_path = tmp_path / zip_name

    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "bundle_id": bundle_id,
        "created_at": datetime.now(UTC).isoformat(),
        "request": {
            "request_id": "req-1",
            "session_id": "sess-1",
            "status": "success",
            "started_at": datetime.now(UTC).isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
        },
        "capture": {"mode": "standard", "trigger": "config", "limits_applied": {}},
        "prompt": {"ns": "test", "key": "prompt", "adapter": "test"},
        "files": ["environment/container.json"],
        "integrity": {"algorithm": "sha256", "checksums": {}},
        "build": {"version": "1.0.0", "commit": "abc123"},
    }

    # Container info with is_containerized=False
    container_data = {
        "runtime": None,
        "container_id": None,
        "image": None,
        "image_digest": None,
        "cgroup_path": None,
        "is_containerized": False,
    }

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{BUNDLE_ROOT_DIR}/manifest.json", json.dumps(manifest))
        zf.writestr(f"{BUNDLE_ROOT_DIR}/README.txt", "Test bundle")
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/request/input.json", json.dumps({"task": "test"})
        )
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/request/output.json", json.dumps({"status": "ok"})
        )
        zf.writestr(f"{BUNDLE_ROOT_DIR}/session/after.jsonl", "")
        zf.writestr(f"{BUNDLE_ROOT_DIR}/logs/app.jsonl", "")
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/environment/container.json", json.dumps(container_data)
        )

    logger = debug_app.get_logger("test.environment.not_containerized")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    env_response = client.get("/api/environment")
    assert env_response.status_code == 200
    env = env_response.json()

    # Container should be None when is_containerized is False
    assert env["container"] is None

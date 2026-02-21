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

"""Shared fixtures and helpers for wink debug app tests."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from weakincentives.debug import BundleWriter
from weakincentives.debug.bundle import (
    BUNDLE_FORMAT_VERSION,
    BUNDLE_ROOT_DIR,
    BundleConfig,
    BundleManifest,
)
from weakincentives.runtime.session import Session


@dataclass(slots=True, frozen=True)
class ExampleSlice:
    value: str


@dataclass(slots=True, frozen=True)
class ListSlice:
    value: object


def create_test_bundle(
    target_dir: Path,
    values: list[str],
) -> Path:
    """Create a test debug bundle with session data."""
    session = Session()

    for value in values:
        session.dispatch(ExampleSlice(value))

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


def create_minimal_bundle(
    target_dir: Path,
    session_content: str | None = None,
    manifest_override: dict[str, object] | None = None,
) -> Path:
    """Create a minimal bundle directly for edge case testing."""
    bundle_id = str(uuid4())
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    zip_name = f"{bundle_id}_{timestamp}.zip"
    zip_path = target_dir / zip_name

    manifest = BundleManifest(
        format_version=BUNDLE_FORMAT_VERSION,
        bundle_id=bundle_id,
        created_at=datetime.now(UTC).isoformat(),
    )
    manifest_dict = json.loads(manifest.to_json())
    if manifest_override:
        manifest_dict.update(manifest_override)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/manifest.json",
            json.dumps(manifest_dict, indent=2),
        )
        zf.writestr(f"{BUNDLE_ROOT_DIR}/README.txt", "Test bundle")
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/request/input.json",
            json.dumps({"task": "test"}),
        )
        zf.writestr(
            f"{BUNDLE_ROOT_DIR}/request/output.json",
            json.dumps({"status": "ok"}),
        )
        if session_content:
            zf.writestr(
                f"{BUNDLE_ROOT_DIR}/session/after.jsonl",
                session_content,
            )

    return zip_path

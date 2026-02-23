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

"""Tests for the wink debug app — transcript and file endpoint tests."""

from __future__ import annotations

import base64
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
    BundleManifest,
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


def _create_minimal_bundle(
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


def test_api_transcript_endpoint(tmp_path: Path) -> None:
    """Test the transcript API endpoint."""
    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            transcript_logger = logging.getLogger("test.transcript")
            transcript_logger.setLevel(logging.DEBUG)
            transcript_logger.debug(
                "Transcript entry",
                extra={
                    "event": "transcript.entry",
                    "context": {
                        "prompt_name": "test",
                        "source": "main",
                        "entry_type": "user_message",
                        "sequence_number": 1,
                        "raw": json.dumps(
                            {
                                "type": "user",
                                "message": {"role": "user", "content": "Hi"},
                            }
                        ),
                        "detail": {
                            "type": "user",
                            "message": {"role": "user", "content": "Hi"},
                        },
                    },
                },
            )

    assert writer.path is not None
    logger = debug_app.get_logger("test.transcript.api")
    store = debug_app.BundleStore(writer.path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    meta_response = client.get("/api/meta")
    assert meta_response.json()["has_transcript"] is True

    transcript_response = client.get("/api/transcript")
    assert transcript_response.status_code == 200
    transcript = transcript_response.json()
    assert transcript["total"] >= 1
    assert transcript["entries"][0]["entry_type"] == "user_message"

    facets_response = client.get("/api/transcript/facets")
    assert facets_response.status_code == 200
    facets = facets_response.json()
    assert "sources" in facets


def test_api_transcript_markdown_rendering(tmp_path: Path) -> None:
    """Test that transcript entries with markdown content include rendered HTML."""
    session = Session()
    session.dispatch(_ExampleSlice("test"))

    md_content = "# Heading\n\nSome **bold** text and a [link](http://example.com)."

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            transcript_logger = logging.getLogger("test.transcript.md")
            transcript_logger.setLevel(logging.DEBUG)
            transcript_logger.debug(
                "Transcript entry",
                extra={
                    "event": "transcript.entry",
                    "context": {
                        "prompt_name": "test",
                        "source": "main",
                        "entry_type": "assistant_message",
                        "sequence_number": 1,
                        "raw": json.dumps(
                            {
                                "type": "assistant",
                                "message": {
                                    "role": "assistant",
                                    "content": md_content,
                                },
                            }
                        ),
                        "detail": {
                            "type": "assistant",
                            "message": {
                                "role": "assistant",
                                "content": md_content,
                            },
                        },
                    },
                },
            )

    assert writer.path is not None
    logger = debug_app.get_logger("test.transcript.markdown")
    store = debug_app.BundleStore(writer.path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/transcript")
    assert response.status_code == 200
    data = response.json()
    entry = data["entries"][0]
    assert entry["content"] == md_content
    assert "content_html" in entry
    assert "<h1>" in entry["content_html"]
    assert "<strong>bold</strong>" in entry["content_html"]


def test_api_transcript_no_markdown_for_short_content(tmp_path: Path) -> None:
    """Test that short or plain text content does not get content_html."""
    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            import logging

            transcript_logger = logging.getLogger("test.transcript.plain")
            transcript_logger.setLevel(logging.DEBUG)
            transcript_logger.debug(
                "Transcript entry",
                extra={
                    "event": "transcript.entry",
                    "context": {
                        "prompt_name": "test",
                        "source": "main",
                        "entry_type": "user_message",
                        "sequence_number": 1,
                        "raw": json.dumps(
                            {
                                "type": "user",
                                "message": {"role": "user", "content": "Hi"},
                            }
                        ),
                        "detail": {
                            "type": "user",
                            "message": {"role": "user", "content": "Hi"},
                        },
                    },
                },
            )

    assert writer.path is not None
    logger = debug_app.get_logger("test.transcript.plain")
    store = debug_app.BundleStore(writer.path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/transcript")
    assert response.status_code == 200
    data = response.json()
    entry = data["entries"][0]
    assert entry["content"] == "Hi"
    assert "content_html" not in entry


def test_transcript_endpoints_and_filters(tmp_path: Path) -> None:
    """Test transcript endpoints return entries and support filtering."""
    import logging

    session = Session()
    session.dispatch(_ExampleSlice("test"))

    with BundleWriter(tmp_path, config=BundleConfig()) as writer:
        writer.write_session_after(session)
        writer.write_request_input({})
        writer.write_request_output({})
        with writer.capture_logs():
            transcript_logger = logging.getLogger("test.transcript")
            transcript_logger.setLevel(logging.DEBUG)

            transcript_logger.debug(
                "Transcript entry: user",
                extra={
                    "event": "transcript.entry",
                    "context": {
                        "prompt_name": "test-prompt",
                        "source": "main",
                        "entry_type": "user_message",
                        "sequence_number": 1,
                        "raw": json.dumps(
                            {
                                "type": "user",
                                "message": {"role": "user", "content": "Hello"},
                            }
                        ),
                        "detail": {
                            "type": "user",
                            "message": {"role": "user", "content": "Hello"},
                        },
                    },
                },
            )

            transcript_logger.debug(
                "Transcript entry: assistant",
                extra={
                    "event": "transcript.entry",
                    "context": {
                        "prompt_name": "test-prompt",
                        "source": "main",
                        "entry_type": "assistant_message",
                        "sequence_number": 2,
                        "raw": None,
                        "detail": {
                            "type": "assistant",
                            "message": {"role": "assistant", "content": "Hi"},
                        },
                    },
                },
            )

    assert writer.path is not None
    test_logger = debug_app.get_logger("test.transcript.api")
    store = debug_app.BundleStore(writer.path, logger=test_logger)
    app = debug_app.build_debug_app(store, logger=test_logger)
    client = TestClient(app)

    result = client.get("/api/transcript").json()
    assert result["total"] == 2
    assert len(result["entries"]) == 2
    assert result["entries"][0]["transcript_source"] == "main"

    facets = client.get("/api/transcript/facets").json()
    assert any(item["name"] == "main" for item in facets["sources"])
    assert any(item["name"] == "user_message" for item in facets["entry_types"])
    assert any(item["name"] == "assistant_message" for item in facets["entry_types"])

    # Search (server-side)
    searched = client.get("/api/transcript", params={"search": "Hello"}).json()
    assert searched["total"] == 1
    assert searched["entries"][0]["entry_type"] == "user_message"

    # Include filters
    filtered = client.get(
        "/api/transcript",
        params={"source": "main", "entry_type": "assistant_message"},
    ).json()
    assert filtered["total"] == 1
    assert filtered["entries"][0]["entry_type"] == "assistant_message"

    # Exclude filters
    excluded = client.get(
        "/api/transcript", params={"exclude_entry_type": "assistant_message"}
    ).json()
    assert excluded["total"] == 1
    assert excluded["entries"][0]["entry_type"] == "user_message"

    # Pagination branches
    limited = client.get("/api/transcript", params={"limit": 1}).json()
    assert limited["total"] == 2
    assert len(limited["entries"]) == 1

    offset_only = client.get("/api/transcript", params={"offset": 1}).json()
    assert offset_only["total"] == 2
    assert len(offset_only["entries"]) == 1


def test_file_endpoint_image_file(tmp_path: Path) -> None:
    """Test reading an image file returns base64 content."""
    bundle_path = _create_minimal_bundle(tmp_path, session_content=None)

    # Create a minimal PNG file (1x1 transparent pixel)
    png_data = bytes(
        [
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,  # PNG signature
            0x00,
            0x00,
            0x00,
            0x0D,
            0x49,
            0x48,
            0x44,
            0x52,  # IHDR chunk
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,  # 1x1
            0x08,
            0x06,
            0x00,
            0x00,
            0x00,
            0x1F,
            0x15,
            0xC4,
            0x89,
            0x00,
            0x00,
            0x00,
            0x0A,
            0x49,
            0x44,
            0x41,  # IDAT chunk
            0x54,
            0x78,
            0x9C,
            0x63,
            0x00,
            0x01,
            0x00,
            0x00,
            0x05,
            0x00,
            0x01,
            0x0D,
            0x0A,
            0x2D,
            0xB4,
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4E,
            0x44,
            0xAE,  # IEND chunk
            0x42,
            0x60,
            0x82,
        ]
    )

    with zipfile.ZipFile(bundle_path, "a") as zf:
        zf.writestr("debug_bundle/filesystem/image.png", png_data)

    logger = debug_app.get_logger("test.file.image")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    file_response = client.get("/api/files/filesystem/image.png")
    assert file_response.status_code == 200
    content = file_response.json()
    assert content["type"] == "image"
    assert content["mime_type"] == "image/png"
    assert content["content"] == base64.b64encode(png_data).decode("ascii")


def test_file_endpoint_image_extensions(tmp_path: Path) -> None:
    """Test image detection for various file extensions."""
    bundle_path = _create_minimal_bundle(tmp_path, session_content=None)

    # Test data (just arbitrary bytes, not real images)
    test_data = b"test image data"

    extensions = [
        (".png", "image/png"),
        (".jpg", "image/jpeg"),
        (".jpeg", "image/jpeg"),
        (".gif", "image/gif"),
        (".webp", "image/webp"),
        (".svg", "image/svg+xml"),
        (".ico", "image/x-icon"),
        (".bmp", "image/bmp"),
    ]

    with zipfile.ZipFile(bundle_path, "a") as zf:
        for ext, _ in extensions:
            zf.writestr(f"debug_bundle/filesystem/test{ext}", test_data)

    logger = debug_app.get_logger("test.file.image.ext")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    for ext, expected_mime in extensions:
        file_response = client.get(f"/api/files/filesystem/test{ext}")
        assert file_response.status_code == 200, f"Failed for extension {ext}"
        content = file_response.json()
        assert content["type"] == "image", f"Expected image type for {ext}"
        assert content["mime_type"] == expected_mime, f"Wrong MIME for {ext}"
        assert content["content"] == base64.b64encode(test_data).decode("ascii")


def test_file_endpoint_image_case_insensitive(tmp_path: Path) -> None:
    """Test image detection is case-insensitive for extensions."""
    bundle_path = _create_minimal_bundle(tmp_path, session_content=None)

    test_data = b"test image"

    with zipfile.ZipFile(bundle_path, "a") as zf:
        zf.writestr("debug_bundle/filesystem/upper.PNG", test_data)
        zf.writestr("debug_bundle/filesystem/mixed.JpG", test_data)

    logger = debug_app.get_logger("test.file.image.case")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    for filename in ["upper.PNG", "mixed.JpG"]:
        file_response = client.get(f"/api/files/filesystem/{filename}")
        assert file_response.status_code == 200
        content = file_response.json()
        assert content["type"] == "image"

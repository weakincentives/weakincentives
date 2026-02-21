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

"""Tests for file-related API endpoints: listing, content, images, markdown."""

from __future__ import annotations

import base64
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from tests.cli.conftest import create_minimal_bundle, create_test_bundle
from weakincentives.cli import debug_app


def test_api_files_endpoints(tmp_path: Path) -> None:
    """Test file listing and content endpoints."""
    bundle_path = create_test_bundle(tmp_path, ["test"])
    logger = debug_app.get_logger("test.files")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    files_response = client.get("/api/files")
    assert files_response.status_code == 200
    files = files_response.json()
    assert "manifest.json" in files
    assert "request/input.json" in files

    file_content = client.get("/api/files/manifest.json")
    assert file_content.status_code == 200
    content = file_content.json()
    assert content["type"] == "json"

    missing_file = client.get("/api/files/nonexistent.json")
    assert missing_file.status_code == 404


def test_config_endpoint_missing(tmp_path: Path) -> None:
    """Test config endpoint returns 404 when no config in bundle."""
    bundle_path = create_minimal_bundle(tmp_path, session_content=None)
    logger = debug_app.get_logger("test.config.missing")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    config_response = client.get("/api/config")
    assert config_response.status_code == 404


def test_metrics_endpoint_missing(tmp_path: Path) -> None:
    """Test metrics endpoint returns 404 when no metrics in bundle."""
    bundle_path = create_minimal_bundle(tmp_path, session_content=None)
    logger = debug_app.get_logger("test.metrics.missing")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    metrics_response = client.get("/api/metrics")
    assert metrics_response.status_code == 404


def test_error_endpoint_with_error(tmp_path: Path) -> None:
    """Test error endpoint returns error when present in bundle."""
    bundle_path = create_minimal_bundle(tmp_path, session_content=None)

    # Manually add error.json
    with zipfile.ZipFile(bundle_path, "a") as zf:
        zf.writestr(
            "debug_bundle/error.json",
            '{"type": "ValueError", "message": "test error"}',
        )

    logger = debug_app.get_logger("test.error.present")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    error_response = client.get("/api/error")
    assert error_response.status_code == 200
    error = error_response.json()
    assert error["type"] == "ValueError"


def test_file_endpoint_text_file(tmp_path: Path) -> None:
    """Test reading a non-JSON text file."""
    bundle_path = create_minimal_bundle(tmp_path, session_content=None)

    # Manually add a text file (use unique name to avoid duplicate zip entry)
    with zipfile.ZipFile(bundle_path, "a") as zf:
        zf.writestr("debug_bundle/NOTES.txt", "This is plain text content")

    logger = debug_app.get_logger("test.file.text")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    file_response = client.get("/api/files/NOTES.txt")
    assert file_response.status_code == 200
    content = file_response.json()
    assert content["type"] == "text"
    assert content["content"] == "This is plain text content"


def test_file_endpoint_binary_file(tmp_path: Path) -> None:
    """Test reading a binary file."""
    bundle_path = create_minimal_bundle(tmp_path, session_content=None)

    # Manually add a binary file
    with zipfile.ZipFile(bundle_path, "a") as zf:
        # Add binary data that can't be decoded as UTF-8
        zf.writestr("debug_bundle/binary.dat", b"\x80\x81\x82\x83\xff\xfe")

    logger = debug_app.get_logger("test.file.binary")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    file_response = client.get("/api/files/binary.dat")
    assert file_response.status_code == 200
    content = file_response.json()
    assert content["type"] == "binary"
    assert content["content"] is None


def test_file_endpoint_image_file(tmp_path: Path) -> None:
    """Test reading an image file returns base64 content."""
    bundle_path = create_minimal_bundle(tmp_path, session_content=None)

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
    bundle_path = create_minimal_bundle(tmp_path, session_content=None)

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
    bundle_path = create_minimal_bundle(tmp_path, session_content=None)

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


def test_file_endpoint_markdown(tmp_path: Path) -> None:
    """Test that .md files are returned with rendered HTML."""
    bundle_path = create_minimal_bundle(tmp_path, session_content=None)

    md_content = "# Hello\n\nSome **bold** text."

    with zipfile.ZipFile(bundle_path, "a") as zf:
        zf.writestr("debug_bundle/filesystem/README.md", md_content)

    logger = debug_app.get_logger("test.file.markdown")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/files/filesystem/README.md")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "markdown"
    assert data["content"] == md_content
    assert "<h1>" in data["html"]
    assert "<strong>bold</strong>" in data["html"]


def test_file_endpoint_markdown_case_insensitive(tmp_path: Path) -> None:
    """Test that .MD files (uppercase) are also detected as markdown."""
    bundle_path = create_minimal_bundle(tmp_path, session_content=None)

    md_content = "# Title\n\nParagraph."

    with zipfile.ZipFile(bundle_path, "a") as zf:
        zf.writestr("debug_bundle/filesystem/NOTES.MD", md_content)

    logger = debug_app.get_logger("test.file.markdown.case")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/files/filesystem/NOTES.MD")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "markdown"
    assert data["content"] == md_content
    assert "<h1>" in data["html"]


def test_file_endpoint_markdown_binary_fallback(tmp_path: Path) -> None:
    """Test that a .md file with invalid UTF-8 is returned as binary."""
    bundle_path = create_minimal_bundle(tmp_path, session_content=None)

    # Invalid UTF-8 bytes
    binary_data = b"\x80\x81\x82\xff\xfe"

    with zipfile.ZipFile(bundle_path, "a") as zf:
        zf.writestr("debug_bundle/filesystem/broken.md", binary_data)

    logger = debug_app.get_logger("test.file.markdown.binary")
    store = debug_app.BundleStore(bundle_path, logger=logger)
    app = debug_app.build_debug_app(store, logger=logger)
    client = TestClient(app)

    response = client.get("/api/files/filesystem/broken.md")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "binary"
    assert data["content"] is None

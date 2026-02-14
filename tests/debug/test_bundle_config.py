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

"""Tests for bundle configuration, manifest, errors, and helper functions."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from weakincentives.debug._bundle_writer import _get_compression_type
from weakincentives.debug.bundle import (
    BUNDLE_FORMAT_VERSION,
    BundleConfig,
    BundleError,
    BundleManifest,
    BundleValidationError,
    compute_checksum,
    generate_readme,
)


class TestBundleConfig:
    """Tests for BundleConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BundleConfig()
        assert config.target is None
        assert config.max_file_size == 10_000_000
        assert config.max_total_size == 52_428_800
        assert config.compression == "deflate"

    def test_config_with_string_target(self, tmp_path: Path) -> None:
        """Test config normalizes string target to Path."""
        config = BundleConfig(target=str(tmp_path))
        assert config.target == tmp_path

    def test_enabled_property(self, tmp_path: Path) -> None:
        """Test enabled property."""
        config_disabled = BundleConfig()
        assert config_disabled.enabled is False

        config_enabled = BundleConfig(target=tmp_path)
        assert config_enabled.enabled is True


class TestBundleManifest:
    """Tests for BundleManifest dataclass."""

    def test_default_manifest(self) -> None:
        """Test default manifest values."""
        manifest = BundleManifest()
        assert manifest.format_version == BUNDLE_FORMAT_VERSION
        assert manifest.bundle_id == ""
        assert manifest.files == ()

    def test_manifest_to_json(self) -> None:
        """Test manifest serialization to JSON."""
        manifest = BundleManifest(
            bundle_id="test-123",
            created_at="2024-01-15T10:30:00+00:00",
        )
        json_str = manifest.to_json()
        data = json.loads(json_str)
        assert data["bundle_id"] == "test-123"
        assert data["format_version"] == BUNDLE_FORMAT_VERSION

    def test_manifest_from_json(self) -> None:
        """Test manifest deserialization from JSON."""
        json_str = json.dumps(
            {
                "format_version": BUNDLE_FORMAT_VERSION,
                "bundle_id": "test-456",
                "created_at": "2024-01-15T10:30:00+00:00",
                "request": {"request_id": "req-123"},
                "capture": {"mode": "full", "trigger": "config"},
                "prompt": {"ns": "", "key": "", "adapter": ""},
                "files": [],
                "integrity": {"algorithm": "sha256", "checksums": {}},
                "build": {"version": "", "commit": ""},
            }
        )
        manifest = BundleManifest.from_json(json_str)
        assert manifest.bundle_id == "test-456"

    def test_manifest_from_invalid_json(self) -> None:
        """Test manifest deserialization with invalid JSON."""
        with pytest.raises(BundleValidationError):
            BundleManifest.from_json("[]")  # Not an object


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_compute_checksum(self) -> None:
        """Test SHA-256 checksum computation."""
        content = b"test content"
        checksum = compute_checksum(content)
        assert len(checksum) == 64  # SHA-256 hex digest length
        # Verify deterministic
        assert compute_checksum(content) == checksum

    def test_generate_readme(self) -> None:
        """Test README generation."""
        manifest = BundleManifest(
            bundle_id="test-123",
            created_at="2024-01-15T10:30:00+00:00",
        )
        readme = generate_readme(manifest)
        assert "test-123" in readme
        assert "Debug Bundle" in readme
        assert "manifest.json" in readme

    def test_get_compression_type(self) -> None:
        """Test compression type mapping."""
        assert _get_compression_type("deflate") == zipfile.ZIP_DEFLATED
        assert _get_compression_type("stored") == zipfile.ZIP_STORED
        assert _get_compression_type("bzip2") == zipfile.ZIP_BZIP2
        assert _get_compression_type("lzma") == zipfile.ZIP_LZMA
        # Unknown defaults to deflate
        assert _get_compression_type("unknown") == zipfile.ZIP_DEFLATED


class TestBundleErrors:
    """Tests for bundle exceptions."""

    def test_bundle_error_is_wink_error(self) -> None:
        """Test BundleError is a WinkError."""
        from weakincentives.errors import WinkError

        error = BundleError("test")
        assert isinstance(error, WinkError)
        assert isinstance(error, RuntimeError)

    def test_bundle_validation_error_inheritance(self) -> None:
        """Test BundleValidationError inherits from BundleError."""
        error = BundleValidationError("test")
        assert isinstance(error, BundleError)

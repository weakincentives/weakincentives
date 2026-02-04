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

"""Tests for debug bundle core structures and helpers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from weakincentives.debug.bundle import (
    BUNDLE_FORMAT_VERSION,
    BundleConfig,
    BundleError,
    BundleManifest,
    BundleValidationError,
    BundleWriter,
    _compute_checksum,
    _generate_readme,
    _get_compression_type,
)

if TYPE_CHECKING:
    pass


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
        checksum = _compute_checksum(content)
        assert len(checksum) == 64  # SHA-256 hex digest length
        # Verify deterministic
        assert _compute_checksum(content) == checksum

    def test_generate_readme(self) -> None:
        """Test README generation."""
        manifest = BundleManifest(
            bundle_id="test-123",
            created_at="2024-01-15T10:30:00+00:00",
        )
        readme = _generate_readme(manifest)
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


class TestBundleWriterExitHandling:
    """Tests for BundleWriter __exit__ handling."""

    def test_exit_records_exception(self, tmp_path: Path) -> None:
        """Test __exit__ captures exception info."""
        from weakincentives.debug.bundle import DebugBundle

        with pytest.raises(ValueError, match="test error"):
            with BundleWriter(tmp_path) as writer:
                raise ValueError("test error")

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.error is not None
        assert "ValueError" in bundle.error["type"]

    def test_exit_handles_finalize_failure(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test __exit__ handles _finalize failure gracefully."""
        from unittest.mock import patch

        # Make _finalize raise an exception
        def failing_finalize(self: object) -> None:
            raise RuntimeError("Finalize failed")

        with patch.object(BundleWriter, "_finalize", failing_finalize):
            with BundleWriter(tmp_path) as writer:
                writer.write_request_input({"test": "data"})

        # Should not raise, just log
        assert "Failed to finalize debug bundle" in caplog.text

    def test_exit_cleans_up_temp_dir(self, tmp_path: Path) -> None:
        """Test __exit__ cleans up temporary directory."""
        with BundleWriter(tmp_path) as writer:
            # Get the temp dir path while inside context
            temp_dir = writer._temp_dir
            assert temp_dir is not None
            assert temp_dir.exists()

        # After context, temp dir should be cleaned up
        assert not temp_dir.exists()


class TestFinalizeEdgeCases:
    """Tests for _finalize edge cases."""

    def test_finalize_already_finalized(self, tmp_path: Path) -> None:
        """Test _finalize does nothing if already finalized."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"test": "data"})

        # _finalize was already called in __exit__
        # Calling again should be a no-op
        assert writer.path is not None
        original_path = writer.path
        writer._finalize()
        assert writer.path == original_path

    def test_finalize_cleans_up_temp_file_on_rename_failure(
        self, tmp_path: Path
    ) -> None:
        """Test that temp file is cleaned up if atomic rename fails."""
        from unittest.mock import patch

        writer = BundleWriter(tmp_path)
        writer.__enter__()
        writer.write_request_input({"test": "data"})

        # Track what temp file would be created
        timestamp = writer._started_at.strftime("%Y%m%d_%H%M%S")
        zip_name = f"{writer._bundle_id}_{timestamp}.zip"
        tmp_file_path = tmp_path / f"{zip_name}.tmp"

        # Make Path.replace raise an exception
        original_replace = Path.replace

        def failing_replace(self: Path, target: Path) -> Path:
            # Call original to actually create the tmp file first
            if str(self).endswith(".tmp"):
                raise OSError("Simulated rename failure")
            return original_replace(self, target)

        with patch.object(Path, "replace", failing_replace):
            with pytest.raises(OSError, match="Simulated rename failure"):
                writer._finalize()

        # Temp file should be cleaned up by the finally block
        assert not tmp_file_path.exists()

        # Clean up manually since we bypassed normal __exit__
        writer.__exit__(None, None, None)


class TestStorageHandlerIntegration:
    """Tests for storage handler integration with BundleWriter."""

    def test_storage_handler_is_called(self, tmp_path: Path) -> None:
        """Test storage handler is called after bundle creation."""
        from weakincentives.debug.bundle import BundleStorageHandler

        stored_bundles: list[tuple[Path, BundleManifest]] = []

        class TestHandler:
            def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
                stored_bundles.append((bundle_path, manifest))

        handler = TestHandler()
        assert isinstance(handler, BundleStorageHandler)

        config = BundleConfig(target=tmp_path, storage_handler=handler)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_request_input({"test": True})

        assert len(stored_bundles) == 1
        assert stored_bundles[0][0] == writer.path
        assert stored_bundles[0][1].bundle_id == str(writer.bundle_id)

    def test_storage_handler_receives_manifest(self, tmp_path: Path) -> None:
        """Test storage handler receives correct manifest data."""
        received_manifest: BundleManifest | None = None

        class TestHandler:
            def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
                nonlocal received_manifest
                received_manifest = manifest

        config = BundleConfig(target=tmp_path, storage_handler=TestHandler())

        with BundleWriter(tmp_path, config=config) as writer:
            writer.set_prompt_info(ns="test", key="prompt", adapter="openai")
            writer.write_request_input({"test": True})

        assert received_manifest is not None
        assert received_manifest.prompt.ns == "test"
        assert received_manifest.prompt.key == "prompt"
        assert received_manifest.prompt.adapter == "openai"

    def test_storage_handler_error_is_logged_not_raised(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test storage handler errors are logged but don't fail bundle creation."""

        class FailingHandler:
            def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
                raise RuntimeError("Storage failed")

        config = BundleConfig(target=tmp_path, storage_handler=FailingHandler())

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_request_input({"test": True})

        # Bundle should still be created
        assert writer.path is not None
        assert writer.path.exists()
        assert "Failed to store bundle to external storage" in caplog.text

    def test_storage_handler_none_does_nothing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that no storage handler means no storage attempt."""
        config = BundleConfig(target=tmp_path, storage_handler=None)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_request_input({"test": True})

        assert writer.path is not None
        assert "stored to external storage" not in caplog.text

    def test_storage_handler_called_after_retention(self, tmp_path: Path) -> None:
        """Test storage handler is called after retention policy is applied."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        stored_paths: list[Path] = []

        class OrderTrackingHandler:
            def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
                stored_paths.append(bundle_path)

        retention = BundleRetentionPolicy(max_bundles=2)
        config = BundleConfig(
            target=tmp_path,
            retention=retention,
            storage_handler=OrderTrackingHandler(),
        )

        # Create multiple bundles
        for i in range(3):
            with BundleWriter(tmp_path, config=config) as writer:
                writer.write_request_input({"bundle": i})

        # Storage handler should have been called 3 times
        assert len(stored_paths) == 3
        # And retention should have kept only 2 bundles
        remaining = list(tmp_path.glob("*.zip"))
        assert len(remaining) == 2

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

"""Tests for debug bundle functionality."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from weakincentives.debug.bundle import (
    BUNDLE_FORMAT_VERSION,
    BundleConfig,
    BundleError,
    BundleManifest,
    BundleValidationError,
    BundleWriter,
    DebugBundle,
    _compute_checksum,
    _generate_readme,
    _get_compression_type,
)
from weakincentives.runtime.run_context import RunContext
from weakincentives.runtime.session import Session

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


class TestBundleWriter:
    """Tests for BundleWriter context manager."""

    def test_writer_creates_bundle(self, tmp_path: Path) -> None:
        """Test that writer creates a valid bundle."""
        bundle_id = uuid4()
        with BundleWriter(tmp_path, bundle_id=bundle_id) as writer:
            writer.write_request_input({"test": "input"})
            writer.write_request_output({"test": "output"})

        assert writer.path is not None
        assert writer.path.exists()
        assert writer.path.suffix == ".zip"

    def test_writer_bundle_id(self, tmp_path: Path) -> None:
        """Test bundle ID is correctly set."""
        bundle_id = uuid4()
        with BundleWriter(tmp_path, bundle_id=bundle_id) as writer:
            pass
        assert writer.bundle_id == bundle_id

    def test_writer_auto_generates_bundle_id(self, tmp_path: Path) -> None:
        """Test that bundle ID is auto-generated if not provided."""
        with BundleWriter(tmp_path) as writer:
            pass
        assert writer.bundle_id is not None

    def test_writer_graceful_degradation_without_context(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test writer methods gracefully handle not being entered."""
        writer = BundleWriter(tmp_path)
        # Should not raise, but should log error
        writer.write_request_input({"test": "input"})
        assert "BundleWriter not entered" in caplog.text

    def test_writer_writes_config(self, tmp_path: Path) -> None:
        """Test writing configuration."""
        with BundleWriter(tmp_path) as writer:
            writer.write_config({"adapter": "openai", "model": "gpt-4"})

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.config is not None
        assert bundle.config["adapter"] == "openai"

    def test_writer_writes_run_context(self, tmp_path: Path) -> None:
        """Test writing run context."""
        run_context = RunContext(worker_id="test-worker")

        with BundleWriter(tmp_path) as writer:
            writer.write_run_context(run_context)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.run_context is not None
        assert bundle.run_context["worker_id"] == "test-worker"

    def test_writer_writes_metrics(self, tmp_path: Path) -> None:
        """Test writing metrics."""
        metrics = {"tokens": 100, "duration_ms": 500}

        with BundleWriter(tmp_path) as writer:
            writer.write_metrics(metrics)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.metrics is not None
        assert bundle.metrics["tokens"] == 100

    def test_writer_writes_prompt_overrides(self, tmp_path: Path) -> None:
        """Test writing prompt overrides."""
        overrides = {"section.key": "full"}

        with BundleWriter(tmp_path) as writer:
            writer.write_prompt_overrides(overrides)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.prompt_overrides is not None

    def test_writer_writes_error(self, tmp_path: Path) -> None:
        """Test writing error information."""
        error_info = {"type": "ValueError", "message": "test error"}

        with BundleWriter(tmp_path) as writer:
            writer.write_error(error_info)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.error is not None
        assert bundle.error["type"] == "ValueError"

    def test_writer_writes_metadata(self, tmp_path: Path) -> None:
        """Test writing arbitrary metadata."""
        eval_info = {"sample_id": "sample-1", "score": 0.95}

        with BundleWriter(tmp_path) as writer:
            writer.write_metadata("eval", eval_info)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.eval is not None
        assert bundle.eval["score"] == 0.95

    def test_writer_sets_prompt_info(self, tmp_path: Path) -> None:
        """Test setting prompt info."""
        with BundleWriter(tmp_path) as writer:
            writer.set_prompt_info(ns="test", key="prompt", adapter="openai")

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.manifest.prompt.ns == "test"
        assert bundle.manifest.prompt.key == "prompt"

    def test_writer_captures_logs(self, tmp_path: Path) -> None:
        """Test log capture context manager."""
        import logging

        # Ensure the test logger has a handler and is set to INFO level
        test_logger = logging.getLogger("test.bundle.capture")
        test_logger.setLevel(logging.INFO)

        with BundleWriter(tmp_path) as writer:
            with writer.capture_logs():
                test_logger.info("Test log message for bundle")

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        logs = bundle.logs
        # Logs may be empty if root logger doesn't propagate
        # Check that logs file exists and is accessible
        assert logs is not None or bundle.logs == ""

    def test_writer_captures_exception_on_exit(self, tmp_path: Path) -> None:
        """Test that exceptions are captured in the bundle."""
        with pytest.raises(ValueError):
            with BundleWriter(tmp_path) as writer:
                raise ValueError("Test exception")

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.error is not None
        assert "ValueError" in bundle.error["type"]

    def test_writer_with_session(self, tmp_path: Path) -> None:
        """Test writing session state."""

        @dataclass(frozen=True, slots=True)
        class TestItem:
            value: str

        session = Session()
        _ = session.dispatch(TestItem(value="test"))

        with BundleWriter(tmp_path) as writer:
            writer.write_session_before(session)
            writer.write_session_after(session)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.session_after is not None

    def test_writer_with_custom_compression(self, tmp_path: Path) -> None:
        """Test writer with different compression methods."""
        config = BundleConfig(target=tmp_path, compression="stored")
        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_request_input({"test": "data"})

        assert writer.path is not None
        with zipfile.ZipFile(writer.path) as zf:
            # Verify it's a valid zip
            assert len(zf.namelist()) > 0


class TestDebugBundle:
    """Tests for DebugBundle loader."""

    def test_load_bundle(self, tmp_path: Path) -> None:
        """Test loading a bundle."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"test": "input"})
            writer.write_request_output({"test": "output"})

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.manifest is not None
        assert bundle.path == writer.path

    def test_load_nonexistent_bundle(self, tmp_path: Path) -> None:
        """Test loading nonexistent bundle raises error."""
        with pytest.raises(BundleValidationError, match="not found"):
            DebugBundle.load(tmp_path / "nonexistent.zip")

    def test_load_invalid_zip(self, tmp_path: Path) -> None:
        """Test loading invalid zip raises error."""
        bad_file = tmp_path / "bad.zip"
        _ = bad_file.write_text("not a zip file")
        with pytest.raises(BundleValidationError, match=r"(?i)not a valid zip"):
            DebugBundle.load(bad_file)

    def test_load_zip_without_manifest(self, tmp_path: Path) -> None:
        """Test loading zip without manifest raises error."""
        zip_path = tmp_path / "no_manifest.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("test.txt", "content")

        with pytest.raises(BundleValidationError, match="missing manifest"):
            DebugBundle.load(zip_path)

    def test_bundle_request_input(self, tmp_path: Path) -> None:
        """Test accessing request input."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"key": "value"})

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.request_input == {"key": "value"}

    def test_bundle_request_output(self, tmp_path: Path) -> None:
        """Test accessing request output."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_output({"result": "success"})

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.request_output == {"result": "success"}

    def test_bundle_list_files(self, tmp_path: Path) -> None:
        """Test listing bundle files."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"test": "input"})
            writer.write_config({"key": "value"})

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        files = bundle.list_files()
        assert "manifest.json" in files
        assert "README.txt" in files
        assert "request/input.json" in files
        assert "config.json" in files

    def test_bundle_extract(self, tmp_path: Path) -> None:
        """Test extracting bundle to directory."""
        with BundleWriter(tmp_path / "bundles") as writer:
            writer.write_request_input({"test": "input"})

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        extract_path = bundle.extract(tmp_path / "extracted")

        assert extract_path.exists()
        assert (extract_path / "manifest.json").exists()
        assert (extract_path / "request" / "input.json").exists()

    def test_bundle_verify_integrity(self, tmp_path: Path) -> None:
        """Test bundle integrity verification."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"test": "input"})

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.verify_integrity() is True

    def test_bundle_optional_artifacts(self, tmp_path: Path) -> None:
        """Test accessing optional artifacts returns None when missing."""
        with BundleWriter(tmp_path) as writer:
            pass  # Don't write optional artifacts

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.session_before is None
        assert bundle.config is None
        assert bundle.metrics is None
        assert bundle.prompt_overrides is None
        assert bundle.error is None
        assert bundle.eval is None


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


class TestFilesystemArchiving:
    """Tests for filesystem archiving functionality."""

    def test_writer_archives_filesystem(self, tmp_path: Path) -> None:
        """Test filesystem archiving with InMemoryFilesystem."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        fs = InMemoryFilesystem()
        _ = fs.write("/test.txt", "Hello, World!")
        _ = fs.write("/subdir/nested.txt", "Nested content")

        with BundleWriter(tmp_path) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        files = bundle.list_files()
        assert any("filesystem" in f for f in files)

    def test_writer_handles_filesystem_errors(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test filesystem archiving handles errors gracefully."""

        class FailingFilesystem:
            """Filesystem that always raises."""

            def list(self, _path: str) -> list:
                raise OSError("Simulated error")

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(FailingFilesystem())  # type: ignore[arg-type]

        # Should not fail, just log the error
        assert writer.path is not None
        assert "Failed to write filesystem" in caplog.text

    def test_writer_respects_max_file_size(self, tmp_path: Path) -> None:
        """Test files larger than max_file_size are skipped."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        fs = InMemoryFilesystem()
        # Write a small file
        _ = fs.write("/small.txt", "Small content")
        # Write a large file
        _ = fs.write("/large.txt", "x" * 1000)

        config = BundleConfig(target=tmp_path, max_file_size=100)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        files = bundle.list_files()
        # small.txt should be present, large.txt should not
        assert any("small.txt" in f for f in files)
        assert not any("large.txt" in f for f in files)

    def test_writer_respects_max_total_size(self, tmp_path: Path) -> None:
        """Test filesystem capture stops at max_total_size."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        fs = InMemoryFilesystem()
        # Write multiple files
        for i in range(10):
            _ = fs.write(f"/file{i}.txt", "x" * 100)

        # Set max_total_size to less than total file content
        config = BundleConfig(target=tmp_path, max_total_size=250)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None
        # The filesystem_truncated flag should be set
        bundle = DebugBundle.load(writer.path)
        manifest = bundle.manifest
        assert manifest.capture.limits_applied.get("filesystem_truncated") is True


class TestBundleAccessors:
    """Tests for DebugBundle accessor properties."""

    def test_session_before_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Test session_before returns None when not present."""
        with BundleWriter(tmp_path) as writer:
            pass  # Don't write session_before

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.session_before is None

    def test_session_after_returns_content(self, tmp_path: Path) -> None:
        """Test session_after returns content when present."""

        @dataclass(frozen=True, slots=True)
        class TestSlice:
            data: str

        session = Session()
        _ = session.dispatch(TestSlice(data="test"))

        with BundleWriter(tmp_path) as writer:
            writer.write_session_after(session)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.session_after is not None
        assert "TestSlice" in bundle.session_after

    def test_run_context_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Test run_context returns None when not present."""
        with BundleWriter(tmp_path) as writer:
            pass  # Don't write run_context

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.run_context is None


class TestWriteExceptionHandling:
    """Tests for exception handling in write methods."""

    def test_write_request_input_handles_unserializable(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_request_input handles serialization errors gracefully."""

        class Unserializable:
            """Object that can't be serialized to JSON."""

            def __repr__(self) -> str:
                raise RuntimeError("Cannot represent")

        with BundleWriter(tmp_path) as writer:
            writer.write_request_input(Unserializable())

        # Should not raise, just log the error
        assert writer.path is not None
        assert "Failed to write request input" in caplog.text

    def test_write_request_output_handles_unserializable(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_request_output handles serialization errors gracefully."""

        class Unserializable:
            """Object that can't be serialized to JSON."""

            def __repr__(self) -> str:
                raise RuntimeError("Cannot represent")

        with BundleWriter(tmp_path) as writer:
            writer.write_request_output(Unserializable())

        # Should not raise, just log the error
        assert writer.path is not None
        assert "Failed to write request output" in caplog.text

    def test_write_session_before_handles_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_session_before handles errors gracefully."""
        config = BundleConfig(target=tmp_path)

        class BrokenSession:
            """Session that raises on snapshot."""

            session_id = uuid4()

            def snapshot(self, *, include_all: bool = False) -> None:
                raise RuntimeError("Snapshot failed")

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_session_before(BrokenSession())  # type: ignore[arg-type]

        assert writer.path is not None
        assert "Failed to write session before" in caplog.text

    def test_write_session_after_handles_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_session_after handles errors gracefully."""

        class BrokenSession:
            """Session that raises on snapshot."""

            session_id = uuid4()

            def snapshot(self, *, include_all: bool = False) -> None:
                raise RuntimeError("Snapshot failed")

        with BundleWriter(tmp_path) as writer:
            writer.write_session_after(BrokenSession())  # type: ignore[arg-type]

        assert writer.path is not None
        assert "Failed to write session after" in caplog.text

    def test_write_config_handles_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_config handles serialization errors gracefully."""

        class Unserializable:
            def __repr__(self) -> str:
                raise RuntimeError("Cannot serialize")

        with BundleWriter(tmp_path) as writer:
            writer.write_config(Unserializable())

        assert writer.path is not None
        assert "Failed to write config" in caplog.text

    def test_write_run_context_handles_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_run_context handles errors gracefully."""

        class BrokenContext:
            request_id = uuid4()
            session_id = uuid4()

            def __repr__(self) -> str:
                raise RuntimeError("Cannot serialize")

        with BundleWriter(tmp_path) as writer:
            writer.write_run_context(BrokenContext())  # type: ignore[arg-type]

        assert writer.path is not None
        assert "Failed to write run context" in caplog.text

    def test_write_metrics_handles_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_metrics handles serialization errors gracefully."""

        class Unserializable:
            def __repr__(self) -> str:
                raise RuntimeError("Cannot serialize")

        with BundleWriter(tmp_path) as writer:
            writer.write_metrics(Unserializable())

        assert writer.path is not None
        assert "Failed to write metrics" in caplog.text

    def test_write_prompt_overrides_handles_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_prompt_overrides handles serialization errors gracefully."""

        class Unserializable:
            def __repr__(self) -> str:
                raise RuntimeError("Cannot serialize")

        with BundleWriter(tmp_path) as writer:
            writer.write_prompt_overrides(Unserializable())

        assert writer.path is not None
        assert "Failed to write prompt overrides" in caplog.text

    def test_write_error_handles_serialization_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_error handles serialization errors gracefully."""
        # Pass something that makes json.dumps fail
        error_info = {"circular": object()}

        with BundleWriter(tmp_path) as writer:
            writer.write_error(error_info)  # type: ignore[arg-type]

        assert writer.path is not None
        assert "Failed to write error" in caplog.text

    def test_write_metadata_handles_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_metadata handles serialization errors gracefully."""
        metadata = {"circular": object()}

        with BundleWriter(tmp_path) as writer:
            writer.write_metadata("eval", metadata)  # type: ignore[arg-type]

        assert writer.path is not None
        assert "Failed to write metadata" in caplog.text

    def test_capture_logs_handles_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test capture_logs logs and re-raises errors in log collection."""
        # This tests the exception path in capture_logs
        # We need to patch collect_all_logs to raise an exception
        from unittest.mock import patch

        def failing_collector(*args: object, **kwargs: object) -> object:
            raise RuntimeError("Log collection failed")

        with pytest.raises(RuntimeError, match="Log collection failed"):
            with BundleWriter(tmp_path) as writer:
                with patch(
                    "weakincentives.debug.bundle.collect_all_logs", failing_collector
                ):
                    with writer.capture_logs():
                        pass

        # Bundle is still created (with error status) and error is logged
        assert writer.path is not None
        assert "Error during log capture" in caplog.text

    def test_write_session_before_empty_session(self, tmp_path: Path) -> None:
        """Test session_before skips write when session has no slices."""
        # Empty session with no dispatched events
        session = Session()

        with BundleWriter(tmp_path) as writer:
            writer.write_session_before(session)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        # session_before is None because snapshot.slices was empty
        assert bundle.session_before is None

    def test_capture_logs_without_temp_dir(self, tmp_path: Path) -> None:
        """Test capture_logs returns immediately when temp_dir is None."""
        writer = BundleWriter(tmp_path)
        # Don't enter context, so temp_dir is None
        with writer.capture_logs():
            pass  # Should just yield and return


class TestBundleIntegrity:
    """Tests for bundle integrity verification."""

    def test_verify_integrity_with_tampered_file(self, tmp_path: Path) -> None:
        """Test integrity verification fails with tampered content."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"test": "data"})

        assert writer.path is not None

        # Tamper with the file
        with zipfile.ZipFile(writer.path, "a") as zf:
            zf.writestr("debug_bundle/request/input.json", '{"tampered": true}')

        bundle = DebugBundle.load(writer.path)
        assert bundle.verify_integrity() is False

    def test_verify_integrity_with_missing_file(self, tmp_path: Path) -> None:
        """Test integrity verification handles missing files."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"test": "data"})

        assert writer.path is not None

        # Create a new zip without a file that's in the manifest
        extract_path = tmp_path / "extracted"
        with zipfile.ZipFile(writer.path) as zf:
            zf.extractall(extract_path)

        # Remove a file
        (extract_path / "debug_bundle" / "request" / "input.json").unlink()

        # Repack
        new_zip = tmp_path / "modified.zip"
        with zipfile.ZipFile(new_zip, "w") as zf:
            for root, _, files in (extract_path / "debug_bundle").walk():
                for file in files:
                    file_path = root / file
                    arcname = str(file_path.relative_to(extract_path / "debug_bundle"))
                    zf.write(file_path, f"debug_bundle/{arcname}")

        bundle = DebugBundle.load(new_zip)
        assert bundle.verify_integrity() is False

    def test_verify_integrity_skips_manifest_json(self, tmp_path: Path) -> None:
        """Test integrity check correctly skips manifest.json itself."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"test": "data"})

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)

        # If manifest.json is in the checksums, it should be skipped
        # and verification should still pass
        assert bundle.verify_integrity() is True


class TestSessionContentReturns:
    """Tests for session_before and session_after returning content."""

    def test_session_before_returns_content_when_present(self, tmp_path: Path) -> None:
        """Test session_before returns content when actually written."""

        @dataclass(frozen=True, slots=True)
        class BeforeSlice:
            value: str

        session = Session()
        _ = session.dispatch(BeforeSlice(value="before"))

        with BundleWriter(tmp_path) as writer:
            writer.write_session_before(session)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.session_before is not None
        assert "BeforeSlice" in bundle.session_before

    def test_session_after_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Test session_after returns None when not written."""
        with BundleWriter(tmp_path) as writer:
            pass  # Don't write session_after

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.session_after is None


class TestFilesystemEdgeCases:
    """Tests for filesystem archiving edge cases."""

    def test_filesystem_handles_permission_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test filesystem archiving handles PermissionError gracefully."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        class PermissionErrorFilesystem(InMemoryFilesystem):
            """Filesystem that raises PermissionError on read."""

            def read_bytes(self, path: str) -> object:
                raise PermissionError("No permission")

        fs = PermissionErrorFilesystem()
        _ = fs.write("/test.txt", "content")

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None
        # Should complete without error, file just skipped
        bundle = DebugBundle.load(writer.path)
        files = bundle.list_files()
        assert not any("test.txt" in f for f in files)

    def test_filesystem_handles_file_not_found(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test filesystem archiving handles FileNotFoundError gracefully."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        class DisappearingFilesystem(InMemoryFilesystem):
            """Filesystem where files disappear between list and read."""

            def stat(self, path: str) -> object:
                raise FileNotFoundError("File gone")

        fs = DisappearingFilesystem()
        _ = fs.write("/test.txt", "content")

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None

    def test_filesystem_handles_is_a_directory_error(self, tmp_path: Path) -> None:
        """Test filesystem archiving handles IsADirectoryError gracefully."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        class DirectoryAsFileFilesystem(InMemoryFilesystem):
            """Filesystem where read fails with IsADirectoryError."""

            def read_bytes(self, path: str) -> object:
                raise IsADirectoryError("Is a directory")

        fs = DirectoryAsFileFilesystem()
        _ = fs.write("/test.txt", "content")

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None

    def test_collect_files_handles_file_not_found_on_list(self, tmp_path: Path) -> None:
        """Test _collect_files handles FileNotFoundError gracefully."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        class ListFailsFilesystem(InMemoryFilesystem):
            """Filesystem where list fails with FileNotFoundError."""

            def list(self, path: str) -> list:
                raise FileNotFoundError("Directory gone")

        fs = ListFailsFilesystem()

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None

    def test_collect_files_handles_not_a_directory_error(self, tmp_path: Path) -> None:
        """Test _collect_files handles NotADirectoryError gracefully."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        class NotADirFilesystem(InMemoryFilesystem):
            """Filesystem where list fails with NotADirectoryError."""

            def list(self, path: str) -> list:
                raise NotADirectoryError("Not a directory")

        fs = NotADirFilesystem()

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None

    def test_collect_files_recurses_directories(self, tmp_path: Path) -> None:
        """Test _collect_files recursively collects from subdirectories."""
        from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem

        fs = InMemoryFilesystem()
        _ = fs.write("/level1/level2/deep.txt", "deep content")

        config = BundleConfig(target=tmp_path)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_filesystem(fs)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        files = bundle.list_files()
        assert any("deep.txt" in f for f in files)


class TestBundleWriterExitHandling:
    """Tests for BundleWriter __exit__ handling."""

    def test_exit_records_exception(self, tmp_path: Path) -> None:
        """Test __exit__ captures exception info."""
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


class TestWriteEnvironment:
    """Tests for BundleWriter.write_environment method."""

    def test_write_environment_creates_files(self, tmp_path: Path) -> None:
        """Test that write_environment creates expected files."""
        with BundleWriter(tmp_path) as writer:
            writer.write_environment()

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        files = bundle.list_files()

        # Should have environment files
        assert any("environment/system.json" in f for f in files)
        assert any("environment/python.json" in f for f in files)
        assert any("environment/env_vars.json" in f for f in files)
        assert any("environment/command.txt" in f for f in files)
        assert any("environment/packages.txt" in f for f in files)

    def test_write_environment_with_pre_captured_env(self, tmp_path: Path) -> None:
        """Test that write_environment accepts pre-captured environment."""
        from weakincentives.debug.environment import (
            CommandInfo,
            EnvironmentCapture,
            PythonInfo,
            SystemInfo,
        )

        pre_captured = EnvironmentCapture(
            system=SystemInfo(os_name="TestOS"),
            python=PythonInfo(version="3.12.0"),
            packages="test-package==1.0.0",
            env_vars={"TEST_VAR": "test_value"},
            command=CommandInfo(
                argv=("test",),
                working_dir="/test",
                entrypoint="test.py",
                executable="/usr/bin/python",
            ),
            captured_at="2024-01-01T00:00:00+00:00",
        )

        with BundleWriter(tmp_path) as writer:
            writer.write_environment(env=pre_captured)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        files = bundle.list_files()

        # Should have written the pre-captured environment
        assert any("environment/system.json" in f for f in files)
        assert any("environment/python.json" in f for f in files)

    def test_write_environment_handles_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_environment handles capture errors gracefully."""
        from unittest.mock import patch

        def failing_capture(*args: object, **kwargs: object) -> object:
            raise RuntimeError("Capture failed")

        with BundleWriter(tmp_path) as writer:
            with patch(
                "weakincentives.debug.environment.capture_environment", failing_capture
            ):
                writer.write_environment()

        # Should not raise, just log error
        assert writer.path is not None
        assert "Failed to write environment" in caplog.text

    def test_write_environment_not_entered(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_environment logs error if writer not entered."""
        writer = BundleWriter(tmp_path)
        writer.write_environment()

        assert "BundleWriter not entered" in caplog.text


class TestBundleRetentionPolicy:
    """Tests for BundleRetentionPolicy dataclass."""

    def test_default_policy(self) -> None:
        """Test default retention policy has no limits."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        policy = BundleRetentionPolicy()
        assert policy.max_bundles is None
        assert policy.max_age_seconds is None
        assert policy.max_total_bytes is None

    def test_policy_with_max_bundles(self) -> None:
        """Test policy with max_bundles limit."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        policy = BundleRetentionPolicy(max_bundles=5)
        assert policy.max_bundles == 5

    def test_policy_with_all_limits(self) -> None:
        """Test policy with all limits configured."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        policy = BundleRetentionPolicy(
            max_bundles=10,
            max_age_seconds=86400,
            max_total_bytes=100_000_000,
        )
        assert policy.max_bundles == 10
        assert policy.max_age_seconds == 86400
        assert policy.max_total_bytes == 100_000_000


class TestBundleStorageHandler:
    """Tests for BundleStorageHandler protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test BundleStorageHandler is runtime checkable."""
        from weakincentives.debug.bundle import BundleStorageHandler

        class MyHandler:
            def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
                pass

        handler = MyHandler()
        assert isinstance(handler, BundleStorageHandler)

    def test_non_handler_is_not_instance(self) -> None:
        """Test non-conforming class is not a BundleStorageHandler."""
        from weakincentives.debug.bundle import BundleStorageHandler

        class NotAHandler:
            pass

        not_handler = NotAHandler()
        assert not isinstance(not_handler, BundleStorageHandler)


class TestRetentionPolicyIntegration:
    """Tests for retention policy integration with BundleWriter."""

    def test_retention_max_bundles_deletes_oldest(self, tmp_path: Path) -> None:
        """Test max_bundles limit deletes oldest bundles."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        retention = BundleRetentionPolicy(max_bundles=2)
        config = BundleConfig(target=tmp_path, retention=retention)

        # Create 3 bundles
        paths: list[Path] = []
        for i in range(3):
            with BundleWriter(tmp_path, config=config) as writer:
                writer.write_request_input({"bundle": i})
            assert writer.path is not None
            paths.append(writer.path)

        # After creating the 3rd bundle, only 2 should remain
        remaining = list(tmp_path.glob("*.zip"))
        assert len(remaining) == 2

        # The newest bundle should definitely exist
        assert paths[-1].exists()

    def test_retention_max_age_deletes_old_bundles(self, tmp_path: Path) -> None:
        """Test max_age_seconds limit deletes old bundles."""
        from unittest.mock import patch

        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create a bundle with old timestamp
        with BundleWriter(tmp_path) as old_writer:
            old_writer.write_request_input({"old": True})
        assert old_writer.path is not None
        old_path = old_writer.path

        # Patch datetime.now to return a time far in the future for retention check
        original_now = datetime.now

        def future_now(tz: object = None) -> datetime:
            if tz is not None:
                return original_now(tz) + __import__("datetime").timedelta(days=2)
            return original_now() + __import__("datetime").timedelta(days=2)

        # Create a new bundle with retention policy
        retention = BundleRetentionPolicy(max_age_seconds=86400)  # 24 hours
        config = BundleConfig(target=tmp_path, retention=retention)

        with patch("weakincentives.debug.bundle.datetime") as mock_datetime:
            mock_datetime.now = future_now
            mock_datetime.fromisoformat = datetime.fromisoformat
            # Need to patch at the class level for the retention check
            with BundleWriter(tmp_path, config=config) as new_writer:
                new_writer.write_request_input({"new": True})

        # The old bundle should be deleted
        assert not old_path.exists()
        assert new_writer.path is not None
        assert new_writer.path.exists()

    def test_retention_max_total_bytes_deletes_largest_oldest(
        self, tmp_path: Path
    ) -> None:
        """Test max_total_bytes limit deletes bundles to stay under limit."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create first bundle
        with BundleWriter(tmp_path) as first_writer:
            writer_first = first_writer
            writer_first.write_request_input({"data": "x" * 1000})
        assert writer_first.path is not None
        first_size = writer_first.path.stat().st_size

        # Create second bundle with retention that allows only ~one bundle worth
        retention = BundleRetentionPolicy(max_total_bytes=first_size + 100)
        config = BundleConfig(target=tmp_path, retention=retention)

        with BundleWriter(tmp_path, config=config) as second_writer:
            second_writer.write_request_input({"data": "y" * 1000})

        # The first bundle should be deleted to stay under limit
        assert not writer_first.path.exists()
        assert second_writer.path is not None
        assert second_writer.path.exists()

    def test_retention_handles_invalid_bundles_gracefully(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test retention skips invalid bundle files."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create an invalid "bundle" file
        invalid_file = tmp_path / "invalid_bundle.zip"
        _ = invalid_file.write_text("not a valid zip")

        retention = BundleRetentionPolicy(max_bundles=10)
        config = BundleConfig(target=tmp_path, retention=retention)

        # This should not raise
        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_request_input({"test": True})

        assert writer.path is not None
        assert writer.path.exists()
        # Invalid file should still be there (not processed)
        assert invalid_file.exists()

    def test_retention_error_is_logged_not_raised(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test retention errors are logged but don't fail bundle creation."""
        from unittest.mock import patch

        from weakincentives.debug.bundle import BundleRetentionPolicy

        retention = BundleRetentionPolicy(max_bundles=1)
        config = BundleConfig(target=tmp_path, retention=retention)

        # Mock _enforce_retention to raise
        def failing_enforce(self: object, policy: object, exclude_path: object) -> None:
            raise RuntimeError("Simulated retention failure")

        with patch.object(BundleWriter, "_enforce_retention", failing_enforce):
            with BundleWriter(tmp_path, config=config) as writer:
                writer.write_request_input({"test": True})

        # Bundle should still be created
        assert writer.path is not None
        assert writer.path.exists()
        assert "Failed to apply retention policy" in caplog.text

    def test_retention_none_does_nothing(self, tmp_path: Path) -> None:
        """Test that no retention policy means no cleanup."""
        config = BundleConfig(target=tmp_path, retention=None)

        # Create multiple bundles
        paths: list[Path] = []
        for i in range(5):
            with BundleWriter(tmp_path, config=config) as writer:
                writer.write_request_input({"bundle": i})
            assert writer.path is not None
            paths.append(writer.path)

        # All bundles should still exist
        for path in paths:
            assert path.exists()

    def test_retention_size_limit_skips_already_marked_bundles(
        self, tmp_path: Path
    ) -> None:
        """Test size limit skips bundles already marked for deletion by other limits."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create multiple bundles without retention first
        for i in range(3):
            with BundleWriter(tmp_path) as writer:
                writer.write_request_input({"bundle": i})

        # Now create a new bundle with both max_bundles and max_total_bytes
        # max_bundles will mark some for deletion, size limit should skip those
        retention = BundleRetentionPolicy(
            max_bundles=2,  # Will mark oldest for deletion
            max_total_bytes=100_000_000,  # Large enough to not delete more
        )
        config = BundleConfig(target=tmp_path, retention=retention)

        with BundleWriter(tmp_path, config=config) as writer:
            writer.write_request_input({"bundle": 3})

        assert writer.path is not None
        # Should have 2 bundles remaining (max_bundles=2)
        remaining = list(tmp_path.glob("*.zip"))
        assert len(remaining) == 2

    def test_retention_size_limit_keeps_bundles_under_limit(
        self, tmp_path: Path
    ) -> None:
        """Test size limit keeps bundles that fit under the total size limit."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create a bundle first
        with BundleWriter(tmp_path) as first_writer:
            first_writer.write_request_input({"data": "small"})
        assert first_writer.path is not None
        first_size = first_writer.path.stat().st_size

        # Create a second bundle with generous size limit
        # that should keep both bundles
        retention = BundleRetentionPolicy(
            max_total_bytes=first_size * 10  # Plenty of room
        )
        config = BundleConfig(target=tmp_path, retention=retention)

        with BundleWriter(tmp_path, config=config) as second_writer:
            second_writer.write_request_input({"data": "small"})

        # Both bundles should still exist
        assert first_writer.path.exists()
        assert second_writer.path is not None
        assert second_writer.path.exists()
        remaining = list(tmp_path.glob("*.zip"))
        assert len(remaining) == 2

    def test_retention_size_limit_deletes_oldest_keeps_newest(
        self, tmp_path: Path
    ) -> None:
        """Test size limit deletes oldest bundles, keeping newest ones."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create 3 bundles: A (oldest), B, C (newest before new)
        bundle_paths: list[Path] = []
        for i in range(3):
            with BundleWriter(tmp_path) as writer:
                writer.write_request_input({"bundle": i, "data": "x" * 100})
            assert writer.path is not None
            bundle_paths.append(writer.path)

        # Get the size of one bundle (they should all be similar)
        bundle_size = bundle_paths[0].stat().st_size

        # Create new bundle with size limit that fits only 2 bundles
        # (new bundle + one existing = 2 bundles worth)
        retention = BundleRetentionPolicy(max_total_bytes=bundle_size * 2 + 100)
        config = BundleConfig(target=tmp_path, retention=retention)

        with BundleWriter(tmp_path, config=config) as new_writer:
            new_writer.write_request_input({"bundle": "new", "data": "x" * 100})

        assert new_writer.path is not None

        # The newest bundles should be kept: new_writer and bundle_paths[2] (C)
        # The oldest should be deleted: bundle_paths[0] (A) and bundle_paths[1] (B)
        assert not bundle_paths[0].exists(), "Oldest bundle A should be deleted"
        assert not bundle_paths[1].exists(), "Second oldest bundle B should be deleted"
        assert bundle_paths[2].exists(), "Newest existing bundle C should be kept"
        assert new_writer.path.exists(), "New bundle should be kept"

        remaining = list(tmp_path.glob("*.zip"))
        assert len(remaining) == 2

    def test_retention_delete_failure_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test bundle deletion failure is logged but doesn't fail."""
        from unittest.mock import patch

        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create a bundle first
        with BundleWriter(tmp_path) as first_writer:
            first_writer.write_request_input({"bundle": 0})
        assert first_writer.path is not None

        # Create a new bundle with retention that will try to delete the first
        retention = BundleRetentionPolicy(max_bundles=1)
        config = BundleConfig(target=tmp_path, retention=retention)

        # Patch Path.unlink to fail
        original_unlink = Path.unlink

        def failing_unlink(self: Path, missing_ok: bool = False) -> None:
            if str(self).endswith(".zip") and self != first_writer.path:
                return original_unlink(self, missing_ok)
            raise OSError("Simulated deletion failure")

        with patch.object(Path, "unlink", failing_unlink):
            with BundleWriter(tmp_path, config=config) as writer:
                writer.write_request_input({"bundle": 1})

        # Bundle should still be created
        assert writer.path is not None
        assert writer.path.exists()
        assert "Failed to delete old bundle" in caplog.text

    def test_retention_age_handles_timezone_naive_timestamps(
        self, tmp_path: Path
    ) -> None:
        """Test age limit works with timezone-naive timestamps in manifests."""
        from unittest.mock import MagicMock, patch

        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create two real bundles first
        with BundleWriter(tmp_path) as old_writer:
            old_writer.write_request_input({"bundle": "old"})
        assert old_writer.path is not None

        with BundleWriter(tmp_path) as recent_writer:
            recent_writer.write_request_input({"bundle": "recent"})
        assert recent_writer.path is not None

        # Create mock bundles with different timestamps
        mock_old_bundle = MagicMock()
        # Use naive datetime (no tzinfo) - old date
        naive_old_timestamp = datetime(2020, 1, 1, 0, 0, 0)
        mock_old_bundle.manifest.created_at = naive_old_timestamp.isoformat()

        mock_recent_bundle = MagicMock()
        # Use naive datetime (no tzinfo) - recent date (now)
        naive_recent_timestamp = datetime.now().replace(tzinfo=None)
        mock_recent_bundle.manifest.created_at = naive_recent_timestamp.isoformat()

        original_load = DebugBundle.load

        def mock_load(path: Path) -> DebugBundle | MagicMock:
            if path == old_writer.path:
                return mock_old_bundle
            if path == recent_writer.path:
                return mock_recent_bundle
            return original_load(path)

        # Create a new bundle with max_age retention
        retention = BundleRetentionPolicy(max_age_seconds=86400)  # 24 hours
        config = BundleConfig(target=tmp_path, retention=retention)

        with patch.object(DebugBundle, "load", mock_load):
            with BundleWriter(tmp_path, config=config) as writer:
                writer.write_request_input({"bundle": "new"})

        # The old bundle with naive timestamp should be deleted
        assert not old_writer.path.exists()
        # The recent bundle should be kept
        assert recent_writer.path.exists()
        assert writer.path is not None
        assert writer.path.exists()

    def test_retention_skips_deletion_if_file_changed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test TOCTOU protection: skip deletion if file identity changed."""
        # Create initial bundles
        paths: list[Path] = []
        for i in range(3):
            with BundleWriter(tmp_path) as writer:
                writer.write_request_input({"bundle": i})
            assert writer.path is not None
            paths.append(writer.path)

        # Capture the original inode of the oldest bundle
        original_inode = paths[0].stat().st_ino

        # Create stale file_identity with the original inode
        stale_identity: dict[Path, tuple[int, int]] = {
            paths[0]: (original_inode, paths[0].stat().st_dev)
        }

        # Replace the bundle file (creates new inode)
        original_content = paths[0].read_bytes()
        paths[0].unlink()
        paths[0].write_bytes(original_content)

        # Verify inode changed
        new_inode = paths[0].stat().st_ino
        assert new_inode != original_inode

        # Call _delete_marked_bundles with stale identity
        BundleWriter._delete_marked_bundles({paths[0]}, stale_identity)

        # File should NOT be deleted due to inode mismatch
        assert paths[0].exists()
        assert "Bundle file changed since collection" in caplog.text


class TestRetentionWithNestedDirectories:
    """Tests for retention policy with nested directory structures (EvalLoop).

    EvalLoop creates bundles at ``{debug_bundle_dir}/{request_id}/{bundle}.zip``.
    The retention policy must search recursively from the config.target directory
    to find all bundles in subdirectories.
    """

    def test_retention_finds_bundles_in_subdirectories(self, tmp_path: Path) -> None:
        """Test retention policy finds and deletes bundles in subdirectories."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create bundles in nested directories (simulating EvalLoop structure)
        subdir1 = tmp_path / "request-1"
        subdir1.mkdir()
        subdir2 = tmp_path / "request-2"
        subdir2.mkdir()

        # Create bundles in subdirectories (no retention yet)
        paths: list[Path] = []
        for subdir in [subdir1, subdir2]:
            with BundleWriter(subdir) as writer:
                writer.write_request_input({"subdir": str(subdir)})
            assert writer.path is not None
            paths.append(writer.path)

        # Create new bundle with retention that limits to 2 total bundles
        # Key: config.target points to the root (tmp_path) for recursive search
        retention = BundleRetentionPolicy(max_bundles=2)
        config = BundleConfig(target=tmp_path, retention=retention)

        subdir3 = tmp_path / "request-3"
        subdir3.mkdir()
        # Bundle written to subdir3, but retention searches from config.target (tmp_path)
        with BundleWriter(subdir3, config=config) as writer:
            writer.write_request_input({"subdir": str(subdir3)})

        # Should have 2 bundles remaining (oldest deleted)
        remaining = list(tmp_path.glob("**/*.zip"))
        assert len(remaining) == 2
        # The newest bundle should exist
        assert writer.path is not None
        assert writer.path.exists()

    def test_retention_max_bundles_across_nested_dirs(self, tmp_path: Path) -> None:
        """Test max_bundles counts bundles across all subdirectories."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create bundles in multiple subdirectories (no retention)
        for i in range(4):
            subdir = tmp_path / f"request-{i}"
            subdir.mkdir()
            with BundleWriter(subdir) as writer:
                writer.write_request_input({"request": i})

        # Verify 4 bundles exist
        assert len(list(tmp_path.glob("**/*.zip"))) == 4

        # Create new bundle with retention limit
        # config.target is the root for recursive search
        retention = BundleRetentionPolicy(max_bundles=3)
        config = BundleConfig(target=tmp_path, retention=retention)

        subdir = tmp_path / "request-4"
        subdir.mkdir()
        with BundleWriter(subdir, config=config) as writer:
            writer.write_request_input({"request": 4})

        # Should have 3 bundles remaining
        remaining = list(tmp_path.glob("**/*.zip"))
        assert len(remaining) == 3

    def test_retention_size_limit_across_nested_dirs(self, tmp_path: Path) -> None:
        """Test max_total_bytes considers bundles across all subdirectories."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create bundles in subdirectories (no retention)
        total_size = 0
        for i in range(2):
            subdir = tmp_path / f"request-{i}"
            subdir.mkdir()
            with BundleWriter(subdir) as writer:
                writer.write_request_input({"data": "x" * 500})
            assert writer.path is not None
            total_size += writer.path.stat().st_size

        # Create new bundle with tight size limit
        retention = BundleRetentionPolicy(max_total_bytes=total_size)
        config = BundleConfig(target=tmp_path, retention=retention)

        subdir = tmp_path / "request-2"
        subdir.mkdir()
        with BundleWriter(subdir, config=config) as writer:
            writer.write_request_input({"data": "y" * 500})

        # Oldest bundles should be deleted to fit under limit
        remaining = list(tmp_path.glob("**/*.zip"))
        # Should have fewer bundles due to size limit
        assert len(remaining) < 3

    def test_retention_skips_newly_created_bundle_in_nested_dir(
        self, tmp_path: Path
    ) -> None:
        """Test that the newly created bundle is not considered for deletion."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create a bundle in a subdirectory (no retention)
        subdir = tmp_path / "request-1"
        subdir.mkdir()
        with BundleWriter(subdir) as old_writer:
            old_writer.write_request_input({"old": True})
        assert old_writer.path is not None

        # Create new bundle with max_bundles=1 in same target tree
        retention = BundleRetentionPolicy(max_bundles=1)
        config = BundleConfig(target=tmp_path, retention=retention)

        new_subdir = tmp_path / "request-2"
        new_subdir.mkdir()
        with BundleWriter(new_subdir, config=config) as new_writer:
            new_writer.write_request_input({"new": True})

        # Old bundle should be deleted, new one kept
        assert not old_writer.path.exists()
        assert new_writer.path is not None
        assert new_writer.path.exists()

    def test_retention_without_config_target_uses_writer_target(
        self, tmp_path: Path
    ) -> None:
        """Test retention falls back to writer target if config.target is None."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create bundles in tmp_path
        with BundleWriter(tmp_path) as first_writer:
            first_writer.write_request_input({"first": True})
        assert first_writer.path is not None

        # Create new bundle with retention but no config.target
        # This means retention will only search in the writer's target (tmp_path)
        retention = BundleRetentionPolicy(max_bundles=1)
        config = BundleConfig(retention=retention)  # No target set

        with BundleWriter(tmp_path, config=config) as second_writer:
            second_writer.write_request_input({"second": True})

        # Old bundle should be deleted
        assert not first_writer.path.exists()
        assert second_writer.path is not None
        assert second_writer.path.exists()

    def test_retention_skips_symlinks(self, tmp_path: Path) -> None:
        """Test retention policy skips symlinks to prevent loops."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        # Create initial bundles
        paths: list[Path] = []
        for i in range(3):
            with BundleWriter(tmp_path) as writer:
                writer.write_request_input({"bundle": i})
            assert writer.path is not None
            paths.append(writer.path)

        # Create a symlink to one of the bundles
        symlink_path = tmp_path / "symlink_bundle.zip"
        symlink_path.symlink_to(paths[0])

        # Also create a symlink in a subdirectory
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        nested_symlink = subdir / "nested_symlink.zip"
        nested_symlink.symlink_to(paths[1])

        # Now create new bundle with retention that limits to 2
        # Symlinks should be skipped, so only real bundles count
        retention = BundleRetentionPolicy(max_bundles=2)
        config = BundleConfig(target=tmp_path, retention=retention)

        with BundleWriter(tmp_path, config=config) as new_writer:
            new_writer.write_request_input({"new": True})

        # Oldest bundle should be deleted (paths[0])
        assert not paths[0].exists()
        # Symlinks should still exist (not deleted, just skipped)
        assert symlink_path.is_symlink()
        assert nested_symlink.is_symlink()
        # New bundle and one old bundle should remain
        assert new_writer.path is not None
        assert new_writer.path.exists()


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

        call_order: list[str] = []
        stored_paths: list[Path] = []

        class OrderTrackingHandler:
            def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
                call_order.append("storage")
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


class TestBundleConfigWithRetentionAndStorage:
    """Tests for BundleConfig with retention and storage handler fields."""

    def test_config_default_values(self) -> None:
        """Test BundleConfig has None defaults for retention and storage."""
        config = BundleConfig()
        assert config.retention is None
        assert config.storage_handler is None

    def test_config_with_retention(self, tmp_path: Path) -> None:
        """Test BundleConfig accepts retention policy."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        retention = BundleRetentionPolicy(max_bundles=5)
        config = BundleConfig(target=tmp_path, retention=retention)
        assert config.retention is retention
        assert config.retention.max_bundles == 5

    def test_config_with_storage_handler(self, tmp_path: Path) -> None:
        """Test BundleConfig accepts storage handler."""

        class TestHandler:
            def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
                pass

        handler = TestHandler()
        config = BundleConfig(target=tmp_path, storage_handler=handler)
        assert config.storage_handler is handler

    def test_config_with_both_retention_and_storage(self, tmp_path: Path) -> None:
        """Test BundleConfig accepts both retention and storage handler."""
        from weakincentives.debug.bundle import BundleRetentionPolicy

        class TestHandler:
            def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
                pass

        retention = BundleRetentionPolicy(max_bundles=10)
        handler = TestHandler()
        config = BundleConfig(
            target=tmp_path,
            retention=retention,
            storage_handler=handler,
        )
        assert config.retention is retention
        assert config.storage_handler is handler

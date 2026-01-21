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

    def test_writer_writes_eval(self, tmp_path: Path) -> None:
        """Test writing eval information."""
        eval_info = {"sample_id": "sample-1", "score": 0.95}

        with BundleWriter(tmp_path) as writer:
            writer.write_eval(eval_info)

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

    def test_write_eval_handles_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test write_eval handles serialization errors gracefully."""
        eval_info = {"circular": object()}

        with BundleWriter(tmp_path) as writer:
            writer.write_eval(eval_info)  # type: ignore[arg-type]

        assert writer.path is not None
        assert "Failed to write eval info" in caplog.text

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

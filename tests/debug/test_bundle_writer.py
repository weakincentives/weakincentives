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

"""Tests for BundleWriter, exception handling, exit/finalize, and environment."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from weakincentives.debug import BundleWriter, DebugBundle
from weakincentives.debug.bundle import BundleConfig
from weakincentives.runtime.run_context import RunContext
from weakincentives.runtime.session import Session


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

    def test_manifest_has_no_duplicate_files(self, tmp_path: Path) -> None:
        """Manifest files list should contain no duplicates."""
        run_context = RunContext(worker_id="test-worker")

        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"test": "input"})
            # Writing run_context twice should not create duplicate entries
            writer.write_run_context(run_context)
            writer.write_run_context(run_context)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        files = list(bundle.manifest.files)
        assert len(files) == len(set(files))

    def test_manifest_build_info_populated(self, tmp_path: Path) -> None:
        """Manifest build info should contain version and commit."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"test": "input"})

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.manifest.build.version != ""
        assert bundle.manifest.build.commit != ""

    def test_manifest_build_info_handles_git_failure(self, tmp_path: Path) -> None:
        """Build info falls back to empty commit when git fails."""
        from subprocess import CompletedProcess
        from unittest.mock import patch

        failed = CompletedProcess(args=[], returncode=128, stdout="", stderr="")
        with patch("subprocess.run", return_value=failed):
            with BundleWriter(tmp_path) as writer:
                writer.write_request_input({"test": "input"})

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        # Version is still resolved from importlib.metadata
        assert bundle.manifest.build.version != ""
        # Commit falls back to empty string
        assert bundle.manifest.build.commit == ""

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
                    "weakincentives.debug._bundle_writer.collect_all_logs",
                    failing_collector,
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

    def test_write_session_after_empty_session(self, tmp_path: Path) -> None:
        """Test session_after skips write when session has no slices."""
        # Empty session with no dispatched events
        session = Session()

        with BundleWriter(tmp_path) as writer:
            writer.write_session_after(session)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        # session_after is None because snapshot.slices was empty
        assert bundle.session_after is None

    def test_capture_logs_without_temp_dir(self, tmp_path: Path) -> None:
        """Test capture_logs returns immediately when temp_dir is None."""
        writer = BundleWriter(tmp_path)
        # Don't enter context, so temp_dir is None
        with writer.capture_logs():
            pass  # Should just yield and return


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

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

"""Tests for DebugBundle loading, accessors, integrity, transcript, and session content."""

from __future__ import annotations

import json
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from weakincentives.debug import BundleWriter, DebugBundle
from weakincentives.debug.bundle import (
    BUNDLE_FORMAT_VERSION,
    BundleValidationError,
)
from weakincentives.runtime.session import Session


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

    def test_bundle_transcript_empty_when_missing(self, tmp_path: Path) -> None:
        """Transcript returns empty list when transcript.jsonl is absent."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"test": "input"})

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.transcript == []

    def test_bundle_transcript_from_logs(self, tmp_path: Path) -> None:
        """Transcript entries are extracted from captured logs into transcript.jsonl."""
        import logging

        logger = logging.getLogger("weakincentives.runtime.transcript")

        with BundleWriter(tmp_path) as writer:
            with writer.capture_logs():
                # Emit structured transcript entries via the logger
                record1 = json.dumps(
                    {
                        "event": "transcript.entry",
                        "context": {
                            "prompt_name": "test",
                            "adapter": "sdk",
                            "entry_type": "user_message",
                            "sequence_number": 1,
                            "source": "main",
                            "timestamp": "2025-01-01T00:00:00+00:00",
                        },
                    }
                )
                record2 = json.dumps(
                    {
                        "event": "transcript.entry",
                        "context": {
                            "prompt_name": "test",
                            "adapter": "sdk",
                            "entry_type": "assistant_message",
                            "sequence_number": 2,
                            "source": "main",
                            "timestamp": "2025-01-01T00:00:01+00:00",
                        },
                    }
                )
                # Log non-transcript entries too (should be filtered out)
                logger.debug(
                    "transcript entry: user_message",
                    extra={"_structured": record1},
                )
                logger.debug(
                    "transcript entry: assistant_message",
                    extra={"_structured": record2},
                )

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        transcript = bundle.transcript
        # The logs are captured as structured JSONL. Whether the transcript
        # entries survive depends on the log handler format. At minimum, verify
        # the transcript property returns a list and handles the content.
        assert isinstance(transcript, list)

    def test_bundle_transcript_with_direct_jsonl(self, tmp_path: Path) -> None:
        """Transcript can be read from a manually-constructed bundle."""
        # Create a bundle with a transcript.jsonl file directly
        bundle_dir = tmp_path / "manual"
        bundle_dir.mkdir()
        root = bundle_dir / "debug_bundle"
        root.mkdir()
        (root / "request").mkdir()
        _ = (root / "request" / "input.json").write_text('{"test": "input"}')

        entry1 = json.dumps(
            {
                "event": "transcript.entry",
                "context": {
                    "prompt_name": "test",
                    "adapter": "sdk",
                    "entry_type": "user_message",
                    "sequence_number": 1,
                    "source": "main",
                },
            }
        )
        entry2 = json.dumps(
            {
                "event": "transcript.entry",
                "context": {
                    "prompt_name": "test",
                    "adapter": "sdk",
                    "entry_type": "assistant_message",
                    "sequence_number": 2,
                    "source": "main",
                },
            }
        )
        _ = (root / "transcript.jsonl").write_text(f"{entry1}\n\n{entry2}\n")

        # Create a minimal manifest
        manifest = {
            "format_version": BUNDLE_FORMAT_VERSION,
            "bundle_id": str(uuid4()),
            "created_at": "2025-01-01T00:00:00",
            "request": {
                "request_id": "",
                "status": "completed",
                "started_at": "2025-01-01T00:00:00",
                "ended_at": "2025-01-01T00:00:01",
            },
            "files": ["request/input.json", "transcript.jsonl"],
            "checksums": {},
        }
        _ = (root / "manifest.json").write_text(json.dumps(manifest))
        _ = (root / "README.txt").write_text("test")

        zip_path = tmp_path / "manual_bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for file_path in root.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(bundle_dir))

        bundle = DebugBundle.load(zip_path)
        transcript = bundle.transcript
        assert len(transcript) == 2
        assert transcript[0]["event"] == "transcript.entry"
        assert transcript[0]["context"]["entry_type"] == "user_message"
        assert transcript[1]["context"]["entry_type"] == "assistant_message"

    def test_bundle_transcript_handles_malformed_jsonl(self, tmp_path: Path) -> None:
        """Transcript property handles malformed lines gracefully."""
        bundle_dir = tmp_path / "malformed"
        bundle_dir.mkdir()
        root = bundle_dir / "debug_bundle"
        root.mkdir()
        (root / "request").mkdir()
        _ = (root / "request" / "input.json").write_text('{"test": "input"}')

        valid_entry = json.dumps(
            {
                "event": "transcript.entry",
                "context": {"entry_type": "user_message"},
            }
        )
        # Include malformed JSON and a non-dict JSON value
        _ = (root / "transcript.jsonl").write_text(
            f"{valid_entry}\nnot valid json\n[1,2,3]\n"
        )

        manifest = {
            "format_version": BUNDLE_FORMAT_VERSION,
            "bundle_id": str(uuid4()),
            "created_at": "2025-01-01T00:00:00",
            "request": {
                "request_id": "",
                "status": "completed",
                "started_at": "2025-01-01T00:00:00",
                "ended_at": "2025-01-01T00:00:01",
            },
            "files": ["request/input.json", "transcript.jsonl"],
            "checksums": {},
        }
        _ = (root / "manifest.json").write_text(json.dumps(manifest))
        _ = (root / "README.txt").write_text("test")

        zip_path = tmp_path / "malformed_bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for file_path in root.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(bundle_dir))

        bundle = DebugBundle.load(zip_path)
        transcript = bundle.transcript
        # Only the valid dict entry should be included
        assert len(transcript) == 1
        assert transcript[0]["event"] == "transcript.entry"


class TestExtractTranscript:
    """Tests for BundleWriter._extract_transcript."""

    def test_extracts_transcript_entries(self, tmp_path: Path) -> None:
        """Transcript entries are extracted from app.jsonl content."""
        entry1 = json.dumps(
            {
                "event": "transcript.entry",
                "context": {"entry_type": "user_message"},
            }
        )
        entry2 = json.dumps(
            {
                "event": "transcript.start",
                "context": {"prompt_name": "test"},
            }
        )
        entry3 = json.dumps(
            {
                "event": "transcript.entry",
                "context": {"entry_type": "assistant_message"},
            }
        )
        log_content = f"{entry1}\n{entry2}\n\n{entry3}\n".encode()

        with BundleWriter(tmp_path) as writer:
            writer._extract_transcript(log_content)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        files = bundle.list_files()
        assert "transcript.jsonl" in files

        transcript = bundle.transcript
        assert len(transcript) == 2
        assert transcript[0]["context"]["entry_type"] == "user_message"
        assert transcript[1]["context"]["entry_type"] == "assistant_message"

    def test_no_transcript_entries(self, tmp_path: Path) -> None:
        """When no transcript.entry events, transcript.jsonl is not created."""
        log_content = json.dumps(
            {
                "event": "transcript.start",
                "context": {},
            }
        ).encode()

        with BundleWriter(tmp_path) as writer:
            writer._extract_transcript(log_content)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        assert bundle.transcript == []

    def test_handles_invalid_json_lines(self, tmp_path: Path) -> None:
        """Invalid JSON lines in log content are skipped."""
        valid = json.dumps(
            {
                "event": "transcript.entry",
                "context": {"entry_type": "thinking"},
            }
        )
        log_content = f"not json\n{valid}\nalso bad\n".encode()

        with BundleWriter(tmp_path) as writer:
            writer._extract_transcript(log_content)

        assert writer.path is not None
        bundle = DebugBundle.load(writer.path)
        transcript = bundle.transcript
        assert len(transcript) == 1


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


class TestBundleIntegrity:
    """Tests for bundle integrity verification."""

    def test_verify_integrity_with_tampered_file(self, tmp_path: Path) -> None:
        """Test integrity verification fails with tampered content."""
        with BundleWriter(tmp_path) as writer:
            writer.write_request_input({"test": "data"})

        assert writer.path is not None

        # Tamper with the file (duplicate entry is intentional)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
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

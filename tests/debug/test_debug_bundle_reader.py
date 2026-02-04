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

"""Tests for DebugBundle reader APIs."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.debug._bundle_fixtures import create_bundle, create_bundle_path
from weakincentives.debug.bundle import (
    BundleValidationError,
    BundleWriter,
    DebugBundle,
)
from weakincentives.runtime.session import Session

if TYPE_CHECKING:
    pass


class TestDebugBundle:
    """Tests for DebugBundle loader."""

    def test_load_bundle(self, tmp_path: Path) -> None:
        """Test loading a bundle."""
        bundle_path = create_bundle_path(
            tmp_path,
            write_fn=lambda writer: writer.write_request_input({"test": "input"}),
        )
        bundle = DebugBundle.load(bundle_path)
        assert bundle.manifest is not None
        assert bundle.path == bundle_path

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
        bundle = create_bundle(
            tmp_path, write_fn=lambda writer: writer.write_request_input({"key": "v"})
        )
        assert bundle.request_input == {"key": "v"}

    def test_bundle_request_output(self, tmp_path: Path) -> None:
        """Test accessing request output."""
        bundle = create_bundle(
            tmp_path,
            write_fn=lambda writer: writer.write_request_output({"result": "success"}),
        )
        assert bundle.request_output == {"result": "success"}

    def test_bundle_list_files(self, tmp_path: Path) -> None:
        """Test listing bundle files."""
        bundle = create_bundle(
            tmp_path,
            write_fn=lambda writer: _write_request_and_config(writer),
        )
        files = bundle.list_files()
        assert "manifest.json" in files
        assert "README.txt" in files
        assert "request/input.json" in files
        assert "config.json" in files

    def test_bundle_extract(self, tmp_path: Path) -> None:
        """Test extracting bundle to directory."""
        bundle_path = create_bundle_path(
            tmp_path / "bundles",
            write_fn=lambda writer: writer.write_request_input({"test": "input"}),
        )
        bundle = DebugBundle.load(bundle_path)
        extract_path = bundle.extract(tmp_path / "extracted")

        assert extract_path.exists()
        assert (extract_path / "manifest.json").exists()
        assert (extract_path / "request" / "input.json").exists()

    def test_bundle_verify_integrity(self, tmp_path: Path) -> None:
        """Test bundle integrity verification."""
        bundle = create_bundle(
            tmp_path,
            write_fn=lambda writer: writer.write_request_input({"test": "input"}),
        )
        assert bundle.verify_integrity() is True

    def test_bundle_optional_artifacts(self, tmp_path: Path) -> None:
        """Test accessing optional artifacts returns None when missing."""
        bundle = create_bundle(tmp_path)
        assert bundle.session_before is None
        assert bundle.config is None
        assert bundle.metrics is None
        assert bundle.prompt_overrides is None
        assert bundle.error is None
        assert bundle.eval is None


class TestBundleAccessors:
    """Tests for DebugBundle accessor properties."""

    def test_session_before_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Test session_before returns None when not present."""
        bundle = create_bundle(tmp_path)
        assert bundle.session_before is None

    def test_session_after_returns_content(self, tmp_path: Path) -> None:
        """Test session_after returns content when present."""

        @dataclass(frozen=True, slots=True)
        class TestSlice:
            data: str

        session = Session()
        _ = session.dispatch(TestSlice(data="test"))

        bundle = create_bundle(
            tmp_path, write_fn=lambda writer: writer.write_session_after(session)
        )
        assert bundle.session_after is not None
        assert "TestSlice" in bundle.session_after

    def test_run_context_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Test run_context returns None when not present."""
        bundle = create_bundle(tmp_path)
        assert bundle.run_context is None


class TestBundleIntegrity:
    """Tests for bundle integrity verification."""

    def test_verify_integrity_with_tampered_file(self, tmp_path: Path) -> None:
        """Test integrity verification fails with tampered content."""
        bundle_path = create_bundle_path(
            tmp_path,
            write_fn=lambda writer: writer.write_request_input({"test": "data"}),
        )

        # Tamper with the file
        with zipfile.ZipFile(bundle_path, "a") as zf:
            zf.writestr("debug_bundle/request/input.json", '{"tampered": true}')

        bundle = DebugBundle.load(bundle_path)
        assert bundle.verify_integrity() is False

    def test_verify_integrity_with_missing_file(self, tmp_path: Path) -> None:
        """Test integrity verification handles missing files."""
        bundle_path = create_bundle_path(
            tmp_path,
            write_fn=lambda writer: writer.write_request_input({"test": "data"}),
        )

        # Create a new zip without a file that's in the manifest
        extract_path = tmp_path / "extracted"
        with zipfile.ZipFile(bundle_path) as zf:
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
        bundle = create_bundle(
            tmp_path,
            write_fn=lambda writer: writer.write_request_input({"test": "data"}),
        )

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

        bundle = create_bundle(
            tmp_path, write_fn=lambda writer: writer.write_session_before(session)
        )
        assert bundle.session_before is not None
        assert "BeforeSlice" in bundle.session_before

    def test_session_after_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Test session_after returns None when not written."""
        bundle = create_bundle(tmp_path)
        assert bundle.session_after is None


def _write_request_and_config(writer: BundleWriter) -> None:
    writer.write_request_input({"test": "input"})
    writer.write_config({"key": "value"})

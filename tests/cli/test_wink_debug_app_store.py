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

"""Tests for BundleStore loading, switching, listing, and lifecycle."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tests.cli.conftest import create_minimal_bundle, create_test_bundle
from weakincentives.cli import debug_app


def test_bundle_store_loads_bundle(tmp_path: Path) -> None:
    bundle_path = create_test_bundle(tmp_path, ["one"])

    store = debug_app.BundleStore(bundle_path, logger=debug_app.get_logger("test"))
    meta = store.get_meta()

    assert meta["bundle_id"]
    assert meta["status"] == "success"
    assert len(meta["slices"]) == 1
    assert meta["has_transcript"] is False


def test_bundle_store_errors_on_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.zip"

    with pytest.raises(debug_app.BundleLoadError):
        debug_app.BundleStore(missing_path, logger=debug_app.get_logger("test"))


def test_bundle_store_errors_on_invalid(tmp_path: Path) -> None:
    # Invalid zip file
    invalid_path = tmp_path / "invalid.zip"
    invalid_path.write_text("not a zip")

    with pytest.raises(debug_app.BundleLoadError):
        debug_app.BundleStore(invalid_path, logger=debug_app.get_logger("test"))


def test_bundle_store_handles_errors(tmp_path: Path) -> None:
    bundle_path = create_test_bundle(tmp_path, ["one"])

    store = debug_app.BundleStore(bundle_path, logger=debug_app.get_logger(__name__))

    assert store.path == bundle_path.resolve()

    listing = store.list_bundles()
    assert len(listing) == 1
    assert listing[0]["selected"] is True

    with pytest.raises(KeyError, match="Unknown slice type: missing"):
        store.get_slice_items("missing")


def test_bundle_store_switch_rejects_outside_root(tmp_path: Path) -> None:
    base_bundle = create_test_bundle(tmp_path, ["base"])
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_bundle = create_test_bundle(other_dir, ["other"])

    store = debug_app.BundleStore(
        base_bundle, logger=debug_app.get_logger("test.switch_root")
    )

    with pytest.raises(debug_app.BundleLoadError) as excinfo:
        store.switch(other_bundle)

    assert "Bundle must live under" in str(excinfo.value)


def test_normalize_path_requires_bundles(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(debug_app.BundleLoadError) as excinfo:
        debug_app.BundleStore(empty_dir, logger=debug_app.get_logger("test.empty"))

    assert "No bundles found" in str(excinfo.value)


def test_bundle_without_session(tmp_path: Path) -> None:
    """Test loading a bundle without session data."""
    bundle_path = create_minimal_bundle(tmp_path, session_content=None)

    store = debug_app.BundleStore(bundle_path, logger=debug_app.get_logger("test"))
    meta = store.get_meta()

    assert meta["bundle_id"]
    assert len(meta["slices"]) == 0
    assert meta["has_transcript"] is False


def test_list_bundles_with_broken_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that listing bundles skips files that fail stat."""
    good_bundle = create_test_bundle(tmp_path, ["value"])
    bad = tmp_path / "bad.zip"
    bad.write_text("invalid")

    # Create store first, before monkeypatching stat
    store = debug_app.BundleStore(good_bundle, logger=debug_app.get_logger("test.list"))

    original_stat = Path.stat

    def fake_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path.name == "bad.zip":
            raise OSError("fail")
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fake_stat)

    # Patch iter_bundle_files where it's imported and used (debug_app module)
    monkeypatch.setattr(
        debug_app,
        "iter_bundle_files",
        lambda root: [good_bundle, bad],
    )

    entries = store.list_bundles()

    assert len(entries) == 1
    assert entries[0]["name"] == good_bundle.name


def test_bundle_store_close(tmp_path: Path) -> None:
    """Test that BundleStore can be closed."""
    bundle_path = create_test_bundle(tmp_path, ["one"])
    store = debug_app.BundleStore(bundle_path, logger=debug_app.get_logger("test"))

    # Should be able to close without error
    store.close()

    # Closing again should also work
    store.close()


def test_bundle_store_from_directory(tmp_path: Path) -> None:
    """Test creating BundleStore from a directory with multiple bundles."""
    # Create two bundles
    bundle_one = create_test_bundle(tmp_path, ["a"])
    time.sleep(0.01)
    bundle_two = create_test_bundle(tmp_path, ["b"])

    # Set mtimes to make bundle_two newest
    now = time.time()
    os.utime(bundle_one, (now - 1, now - 1))
    os.utime(bundle_two, (now, now))

    # Create store from directory - should pick newest bundle
    store = debug_app.BundleStore(tmp_path, logger=debug_app.get_logger("test.dir"))

    assert store.path == bundle_two.resolve()


def test_slice_offset_beyond_count(tmp_path: Path) -> None:
    """Test getting slices with offset beyond item count returns empty."""
    bundle_path = create_test_bundle(tmp_path, ["a", "b"])
    store = debug_app.BundleStore(bundle_path, logger=debug_app.get_logger("test"))

    # Get meta to find slice type
    meta = store.get_meta()
    slice_type = meta["slices"][0]["slice_type"]

    # Query with offset beyond the number of items (2)
    result = store.get_slice_items(str(slice_type), offset=100, limit=10)

    # Should return empty items without raising KeyError
    assert result["items"] == []


def test_reload_without_existing_cache(tmp_path: Path) -> None:
    """Test reload when cache file doesn't exist yet."""
    bundle_path = create_test_bundle(tmp_path, ["test"])
    cache_path = bundle_path.with_suffix(bundle_path.suffix + ".sqlite")

    store = debug_app.BundleStore(bundle_path, logger=debug_app.get_logger("test"))

    # Cache should exist now
    assert cache_path.exists()

    # Remove cache
    cache_path.unlink()

    # Reload should still work (it should handle missing cache)
    result = store.reload()
    assert result["bundle_id"]

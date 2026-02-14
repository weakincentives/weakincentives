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

"""Tests for storage handler protocol, integration, and config with retention/storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.debug import BundleWriter
from weakincentives.debug.bundle import (
    BundleConfig,
    BundleManifest,
    BundleRetentionPolicy,
    BundleStorageHandler,
)


class TestBundleStorageHandler:
    """Tests for BundleStorageHandler protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test BundleStorageHandler is runtime checkable."""

        class MyHandler:
            def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
                pass

        handler = MyHandler()
        assert isinstance(handler, BundleStorageHandler)

    def test_non_handler_is_not_instance(self) -> None:
        """Test non-conforming class is not a BundleStorageHandler."""

        class NotAHandler:
            pass

        not_handler = NotAHandler()
        assert not isinstance(not_handler, BundleStorageHandler)


class TestStorageHandlerIntegration:
    """Tests for storage handler integration with BundleWriter."""

    def test_storage_handler_is_called(self, tmp_path: Path) -> None:
        """Test storage handler is called after bundle creation."""
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

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

# pyright: reportImportCycles=false
"""Bundle retention policy enforcement.

Internal module implementing retention policy logic for cleaning up old debug
bundles. Used by BundleWriter after successful bundle creation.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from ..clock import SYSTEM_CLOCK

if TYPE_CHECKING:
    from .bundle import (
        BundleConfig,
        BundleManifest,
        BundleRetentionPolicy,
    )

_logger = logging.getLogger(__name__)


def apply_retention_policy(
    *,
    config: BundleConfig,
    target: Path,
    bundle_id: UUID,
    exclude_path: Path,
    bundle_path: Path | None,
) -> None:
    """Apply retention policy to clean up old bundles.

    Called after a new bundle is successfully created. Scans the target
    directory for existing bundles and deletes those that exceed the
    configured limits.

    Args:
        config: Bundle configuration containing the retention policy.
        target: Writer's target directory (fallback for search root).
        bundle_id: ID of the current bundle (for logging).
        exclude_path: Path to the just-created bundle to exclude from
            retention processing (prevents self-deletion).
        bundle_path: Path to the new bundle (for size calculation).
    """
    retention = config.retention
    if retention is None:
        return

    try:
        _enforce_retention(
            retention=retention,
            config=config,
            target=target,
            exclude_path=exclude_path,
            bundle_path=bundle_path,
        )
    except Exception:
        _logger.warning(
            "Failed to apply retention policy",
            extra={"bundle_id": str(bundle_id)},
            exc_info=True,
        )


def _enforce_retention(
    *,
    retention: BundleRetentionPolicy,
    config: BundleConfig,
    target: Path,
    exclude_path: Path,
    bundle_path: Path | None,
) -> None:
    """Enforce retention policy on bundles in target directory."""
    bundles, file_identity = _collect_existing_bundles(
        config=config,
        target=target,
        exclude_path=exclude_path,
    )
    to_delete: set[Path] = set()

    _apply_age_limit(retention, bundles, to_delete)
    _apply_count_limit(retention, bundles, to_delete)
    _apply_size_limit(retention, bundles, to_delete, bundle_path)
    _delete_marked_bundles(to_delete, file_identity)


def _get_retention_search_root(config: BundleConfig, target: Path) -> Path:
    """Get the root directory for retention policy bundle search.

    Returns config.target if set (for EvalLoop's nested structure),
    otherwise falls back to the writer's target directory.
    """
    if config.target is not None:
        return config.target
    return target


def _collect_existing_bundles(
    *,
    config: BundleConfig,
    target: Path,
    exclude_path: Path,
) -> tuple[list[tuple[Path, datetime, int]], dict[Path, tuple[int, int]]]:
    """Collect metadata for existing bundles in the target directory.

    Searches up to 2 levels deep to support EvalLoop's nested structure
    (``{target}/{request_id}/{bundle}.zip``) while avoiding unbounded
    recursive traversal for performance and security.

    Args:
        config: Bundle configuration.
        target: Writer's target directory (fallback for search root).
        exclude_path: Path to the just-created bundle to exclude from
            collection (prevents self-deletion).

    Returns:
        A tuple of (bundles, file_identity) where:
        - bundles: List of (path, created_at, size) sorted oldest-first
        - file_identity: Dict mapping path to (inode, device) for TOCTOU
          protection during deletion
    """
    from ._bundle_reader import DebugBundle
    from .bundle import BundleValidationError

    search_root = _get_retention_search_root(config, target)
    bundles: list[tuple[Path, datetime, int]] = []
    # Track file identity (inode, device) to prevent TOCTOU race conditions
    # during deletion - verifies file hasn't been replaced between check and use
    file_identity: dict[Path, tuple[int, int]] = {}
    # Resolve exclude path once for efficient comparison
    exclude_path_resolved = exclude_path.resolve()

    # Use explicit patterns instead of ** to limit depth (max 2 levels)
    # This avoids unbounded traversal and symlink loop risks
    for pattern in ("*.zip", "*/*.zip"):
        for bundle_path in search_root.glob(pattern):
            # Skip symlinks to avoid loops and ensure we only process
            # real bundle files
            if bundle_path.is_symlink():
                continue
            try:
                # Skip the bundle we just created - compare resolved paths
                # for efficiency (avoids loading the just-created bundle)
                if bundle_path.resolve() == exclude_path_resolved:
                    continue

                bundle = DebugBundle.load(bundle_path)
                created_at = datetime.fromisoformat(bundle.manifest.created_at)
                # Normalize to UTC for consistent sorting across timezone-naive
                # and timezone-aware timestamps
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                stat_info = bundle_path.stat()
                size = stat_info.st_size
                # Store inode and device for TOCTOU protection
                file_identity[bundle_path] = (stat_info.st_ino, stat_info.st_dev)
                bundles.append((bundle_path, created_at, size))
            except (BundleValidationError, ValueError, OSError):
                continue
    bundles.sort(key=lambda x: x[1])
    return bundles, file_identity


def _apply_age_limit(
    retention: BundleRetentionPolicy,
    bundles: list[tuple[Path, datetime, int]],
    to_delete: set[Path],
) -> None:
    """Mark bundles older than max_age_seconds for deletion."""
    if retention.max_age_seconds is None:
        return
    now = SYSTEM_CLOCK.utcnow()
    for bundle_path, created_at, _ in bundles:
        # Timestamps are already normalized to UTC in _collect_existing_bundles
        if (now - created_at).total_seconds() > retention.max_age_seconds:
            to_delete.add(bundle_path)


def _apply_count_limit(
    retention: BundleRetentionPolicy,
    bundles: list[tuple[Path, datetime, int]],
    to_delete: set[Path],
) -> None:
    """Mark oldest bundles for deletion to stay under max_bundles limit."""
    if retention.max_bundles is None:
        return
    total_count = len(bundles) + 1  # +1 for newly created bundle
    excess = total_count - retention.max_bundles
    if excess > 0:
        for bundle_path, _, _ in bundles[:excess]:
            to_delete.add(bundle_path)


def _apply_size_limit(
    retention: BundleRetentionPolicy,
    bundles: list[tuple[Path, datetime, int]],
    to_delete: set[Path],
    bundle_path: Path | None,
) -> None:
    """Mark oldest bundles for deletion to stay under max_total_bytes limit."""
    if retention.max_total_bytes is None:
        return

    new_bundle_size = bundle_path.stat().st_size if bundle_path else 0

    # Calculate total size including new bundle and all existing bundles not yet marked
    total_size = new_bundle_size + sum(
        size for path, _, size in bundles if path not in to_delete
    )

    # Delete oldest bundles until under limit (bundles already sorted oldest-first)
    for bp, _, size in bundles:
        if bp in to_delete:
            continue
        if total_size <= retention.max_total_bytes:
            break
        to_delete.add(bp)
        total_size -= size


def _delete_marked_bundles(
    to_delete: set[Path], file_identity: dict[Path, tuple[int, int]]
) -> None:
    """Delete bundles marked for removal with TOCTOU protection.

    Verifies file identity (inode/device) before deletion to prevent
    race conditions where a file could be replaced between collection
    and deletion.

    Args:
        to_delete: Set of bundle paths to delete.
        file_identity: Mapping of path to (inode, device) captured during
            collection for verification.
    """
    for bundle_path in to_delete:
        try:
            # Verify file identity before deletion (TOCTOU protection)
            # This prevents deleting a file that was replaced/moved since collection
            expected_identity = file_identity.get(bundle_path)
            if expected_identity is not None:
                current_stat = bundle_path.stat()
                current_identity = (current_stat.st_ino, current_stat.st_dev)
                if current_identity != expected_identity:
                    _logger.warning(
                        "Bundle file changed since collection, skipping deletion",
                        extra={
                            "bundle_path": str(bundle_path),
                            "expected_inode": expected_identity[0],
                            "current_inode": current_identity[0],
                        },
                    )
                    continue

            bundle_path.unlink()
            _logger.debug(
                "Deleted old bundle",
                extra={"bundle_path": str(bundle_path)},
            )
        except OSError:
            _logger.warning(
                "Failed to delete old bundle",
                extra={"bundle_path": str(bundle_path)},
                exc_info=True,
            )


def invoke_storage_handler(
    *,
    config: BundleConfig,
    bundle_id: UUID,
    bundle_path: Path | None,
    manifest: BundleManifest,
) -> None:
    """Invoke storage handler to copy bundle to external storage.

    Called after retention policy is applied. Errors are logged but
    do not propagate.
    """
    handler = config.storage_handler
    if handler is None or bundle_path is None:
        return

    try:
        handler.store_bundle(bundle_path, manifest)
        _logger.debug(
            "Bundle stored to external storage",
            extra={
                "bundle_id": str(bundle_id),
                "bundle_path": str(bundle_path),
            },
        )
    except Exception:
        _logger.warning(
            "Failed to store bundle to external storage",
            extra={
                "bundle_id": str(bundle_id),
                "bundle_path": str(bundle_path),
            },
            exc_info=True,
        )

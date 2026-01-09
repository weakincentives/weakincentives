#!/usr/bin/env python3
"""Build a cached coverage database with test-to-code mappings.

This script runs the full test suite with dynamic context tracking enabled,
which records which test executed which lines of code. The resulting coverage
database can then be queried to determine which tests need to run for a given
set of changed files.

Usage:
    python build/build_coverage_cache.py [--cache-dir PATH]

The cache includes:
- .coverage: SQLite database with test-to-code mappings
- metadata.json: Git commit hash, timestamp, and test statistics
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_git_commit() -> str:
    """Get the current git commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_git_branch() -> str:
    """Get the current git branch name."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def run_tests_with_context() -> int:
    """Run pytest with dynamic context tracking enabled.

    Returns:
        Exit code from pytest (0 on success).
    """
    cmd = [
        "uv",
        "run",
        "--all-extras",
        "pytest",
        "--strict-config",
        "--strict-markers",
        "--cov=src/weakincentives",
        "--cov-context=test",  # Enable test-level context tracking
        "--cov-report=",  # Suppress terminal report (we only need the DB)
        "--maxfail=1",
        "-q",
        "--no-header",
        "tests/",
    ]

    print("Building coverage cache with test-to-code mappings...")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    return result.returncode


def analyze_coverage_database(coverage_path: Path) -> dict[str, int]:
    """Analyze the coverage database to extract statistics.

    Args:
        coverage_path: Path to the .coverage database file.

    Returns:
        Dictionary with statistics (test_count, file_count, etc.).
    """
    try:
        import sqlite3

        conn = sqlite3.connect(coverage_path)
        cursor = conn.cursor()

        # Count unique test contexts
        cursor.execute("SELECT COUNT(*) FROM context")
        test_count = cursor.fetchone()[0]

        # Count measured files
        cursor.execute("SELECT COUNT(*) FROM file")
        file_count = cursor.fetchone()[0]

        # Get schema version
        cursor.execute("SELECT version FROM coverage_schema")
        schema_version = cursor.fetchone()[0]

        conn.close()

        return {
            "test_count": test_count,
            "file_count": file_count,
            "schema_version": schema_version,
        }
    except Exception as e:
        print(f"Warning: Failed to analyze coverage database: {e}")
        return {}


def build_cache(cache_dir: Path) -> int:
    """Build the coverage cache.

    Args:
        cache_dir: Directory to store the cache.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Run tests with context tracking
    returncode = run_tests_with_context()
    if returncode != 0:
        print(f"Error: Tests failed with exit code {returncode}")
        return returncode

    # Check that .coverage was created
    coverage_source = Path(".coverage")
    if not coverage_source.exists():
        print("Error: .coverage file was not created")
        return 1

    # Analyze the database
    stats = analyze_coverage_database(coverage_source)

    # Copy .coverage to cache
    coverage_dest = cache_dir / ".coverage"
    shutil.copy2(coverage_source, coverage_dest)
    print(f"Copied coverage database to {coverage_dest}")

    # Create metadata
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
        "coverage_file": str(coverage_dest),
        **stats,
    }

    # Write metadata
    metadata_path = cache_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to {metadata_path}")

    # Print summary
    print("\n✓ Coverage cache built successfully!")
    print(f"  Commit: {metadata['git_commit'][:8]}")
    print(f"  Branch: {metadata['git_branch']}")
    if stats:
        print(f"  Tests tracked: {stats.get('test_count', 'unknown')}")
        print(f"  Files covered: {stats.get('file_count', 'unknown')}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a cached coverage database with test-to-code mappings"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".coverage-cache"),
        help="Directory to store the coverage cache (default: .coverage-cache)",
    )

    args = parser.parse_args()

    return build_cache(args.cache_dir)


if __name__ == "__main__":
    sys.exit(main())

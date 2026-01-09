#!/usr/bin/env python3
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

"""Select tests to run based on changed files and cached coverage data.

This script queries a cached coverage database (built with build_coverage_cache.py)
to determine which tests need to run for a given set of changed files. This enables
smart test selection that only runs tests affected by code changes.

Usage:
    # Select tests for files changed in current branch vs main
    python build/select_tests.py --base main

    # Select tests for specific files
    python build/select_tests.py --files src/foo.py src/bar.py

    # Run the selected tests
    python build/select_tests.py --base main --run

The script will:
1. Determine which files have changed (from git or explicit file list)
2. Query the coverage cache to find tests that executed those files
3. Output the list of tests to run (or run them if --run is specified)

Exit codes:
    0: Success (tests selected/run successfully)
    1: Error (cache missing, query failed, tests failed, etc.)
    2: All tests should be run (cache miss, too many changes, etc.)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# Maximum number of tests to show in verbose output before truncating
_MAX_TESTS_TO_DISPLAY = 10

# Exit code indicating that all tests should be run (fallback)
_EXIT_CODE_RUN_ALL_TESTS = 2


class CoverageQuery(NamedTuple):
    """Query results from coverage database."""

    tests: set[str]
    """Test node IDs that cover the changed files."""

    covered_files: set[str]
    """Changed files that are covered by tests."""

    uncovered_files: set[str]
    """Changed files with no coverage data."""


def get_changed_files(base_ref: str) -> list[str]:
    """Get list of changed files using git diff.

    Args:
        base_ref: Git reference to compare against (e.g., 'main', 'HEAD~1').

    Returns:
        List of changed file paths relative to repo root.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def normalize_path(path: str, repo_root: Path) -> str:
    """Normalize a file path to be relative to the repo root.

    Args:
        path: File path (absolute or relative).
        repo_root: Repository root directory.

    Returns:
        Path relative to repo root.
    """
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(repo_root))
        except ValueError:
            return str(p)
    return str(p)


def query_coverage_for_files(
    coverage_db: Path, changed_files: list[str], repo_root: Path
) -> CoverageQuery:
    """Query coverage database to find tests covering the changed files.

    Args:
        coverage_db: Path to the .coverage SQLite database.
        changed_files: List of changed file paths.
        repo_root: Repository root directory.

    Returns:
        CoverageQuery with test IDs and coverage information.
    """
    conn = sqlite3.connect(coverage_db)

    # Register coverage.py's custom functions for working with numbits
    try:
        from coverage.sqldata import register_sqlite_functions

        register_sqlite_functions(conn)
    except ImportError:
        print("Warning: Could not import coverage.py functions", file=sys.stderr)

    cursor = conn.cursor()

    # Normalize changed files to match database paths
    # Coverage.py stores absolute paths or paths relative to coverage run location
    normalized_files = [normalize_path(f, repo_root) for f in changed_files]

    # Find file IDs for changed files
    # Need to handle both absolute and relative paths
    placeholders = ",".join("?" * len(normalized_files))

    # Build conditions for both exact matches and basename matches
    basename_placeholders = " OR ".join(["path LIKE '%/' || ?"] * len(normalized_files))

    query = f"""
        SELECT DISTINCT id, path
        FROM file
        WHERE path IN ({placeholders})
    """

    if basename_placeholders:
        query += f" OR {basename_placeholders}"

    # Build parameter list: exact matches + basename matches
    params = normalized_files.copy()
    for f in normalized_files:
        params.append(Path(f).name)

    cursor.execute(query, params)
    file_rows = cursor.fetchall()

    if not file_rows:
        # No files found in coverage database
        conn.close()
        return CoverageQuery(
            tests=set(),
            covered_files=set(),
            uncovered_files=set(changed_files),
        )

    file_ids = [row[0] for row in file_rows]
    covered_paths = {row[1] for row in file_rows}

    # Find which changed files are covered
    covered_files = set()
    uncovered_files = set()
    for changed_file in changed_files:
        norm_file = normalize_path(changed_file, repo_root)
        if any(
            norm_file in path or Path(norm_file).name in path for path in covered_paths
        ):
            covered_files.add(changed_file)
        else:
            uncovered_files.add(changed_file)

    # Find contexts (tests) that executed those files
    placeholders = ",".join("?" * len(file_ids))
    query = f"""
        SELECT DISTINCT c.context
        FROM context c
        JOIN line_bits lb ON c.id = lb.context_id
        WHERE lb.file_id IN ({placeholders})
    """

    cursor.execute(query, file_ids)
    context_rows = cursor.fetchall()

    conn.close()

    # Context names are test node IDs (e.g., "tests/test_foo.py::test_bar")
    tests = {row[0] for row in context_rows if row[0]}  # Filter out empty contexts

    return CoverageQuery(
        tests=tests, covered_files=covered_files, uncovered_files=uncovered_files
    )


def load_cache_metadata(cache_dir: Path) -> dict | None:
    """Load metadata from coverage cache.

    Args:
        cache_dir: Directory containing the cache.

    Returns:
        Metadata dictionary, or None if not found.
    """
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    with metadata_path.open() as f:
        return json.load(f)


def select_tests(  # noqa: C901, PLR0912
    changed_files: list[str],
    cache_dir: Path,
    repo_root: Path,
    verbose: bool = False,
) -> tuple[list[str], int]:
    """Select tests to run based on changed files.

    Args:
        changed_files: List of changed file paths.
        cache_dir: Directory containing the coverage cache.
        repo_root: Repository root directory.
        verbose: Print verbose output.

    Returns:
        Tuple of (test_ids, exit_code).
        Exit code 0 = success, 1 = error, 2 = should run all tests.
    """
    # Check if cache exists
    coverage_db = cache_dir / ".coverage"
    if not coverage_db.exists():
        print("Coverage cache not found. Run 'make build-coverage-cache' first.")
        print("Falling back to running all tests.")
        return [], _EXIT_CODE_RUN_ALL_TESTS

    # Load cache metadata
    metadata = load_cache_metadata(cache_dir)
    if metadata and verbose:
        print(
            f"Using coverage cache from commit {metadata.get('git_commit', 'unknown')[:8]}"
        )
        print(f"Cache created: {metadata.get('created_at', 'unknown')}")
        print(f"Tests tracked: {metadata.get('test_count', 'unknown')}")
        print()

    # Filter to only Python source files (ignore docs, configs, etc.)
    python_files = [
        f for f in changed_files if f.startswith("src/") and f.endswith(".py")
    ]

    if not python_files:
        if verbose:
            print("No Python source files changed in src/. Running all tests.")
        return [], _EXIT_CODE_RUN_ALL_TESTS

    if verbose:
        print(f"Changed Python files ({len(python_files)}):")
        for f in python_files:
            print(f"  - {f}")
        print()

    # Query coverage database
    try:
        result = query_coverage_for_files(coverage_db, python_files, repo_root)
    except Exception as e:
        print(f"Error querying coverage database: {e}", file=sys.stderr)
        print("Falling back to running all tests.")
        return [], _EXIT_CODE_RUN_ALL_TESTS

    # Report uncovered files
    if result.uncovered_files and verbose:
        print(
            f"Warning: {len(result.uncovered_files)} files not found in coverage cache:"
        )
        for f in sorted(result.uncovered_files):
            print(f"  - {f}")
        print("These may be new files or the cache may be stale.")
        print()

    # If we have uncovered files, run all tests to be safe
    if result.uncovered_files:
        if verbose:
            print("Running all tests due to uncovered files.")
        return [], _EXIT_CODE_RUN_ALL_TESTS

    # Convert test set to sorted list
    tests = sorted(result.tests)

    if verbose:
        print(f"Selected {len(tests)} tests to run:")
        for test in tests[:_MAX_TESTS_TO_DISPLAY]:
            print(f"  - {test}")
        if len(tests) > _MAX_TESTS_TO_DISPLAY:
            print(f"  ... and {len(tests) - _MAX_TESTS_TO_DISPLAY} more")
        print()

    return tests, 0


def run_tests(test_ids: list[str]) -> int:
    """Run the specified tests using pytest.

    Args:
        test_ids: List of pytest node IDs to run.

    Returns:
        Exit code from pytest.
    """
    if not test_ids:
        print("No tests to run.")
        return 0

    cmd = [
        "uv",
        "run",
        "--all-extras",
        "pytest",
        "--strict-config",
        "--strict-markers",
        "--cov=src/weakincentives",
        "--cov-report=term-missing",
        "--cov-fail-under=100",
        "--maxfail=1",
        "-v",
        *test_ids,
    ]

    print(f"Running {len(test_ids)} selected tests...")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:  # noqa: C901, PLR0911
    parser = argparse.ArgumentParser(
        description="Select tests to run based on changed files and cached coverage"
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--base",
        type=str,
        help="Git ref to compare against (e.g., 'main', 'HEAD~1')",
    )
    input_group.add_argument(
        "--files",
        nargs="+",
        help="Explicit list of changed files",
    )

    # Options
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".coverage-cache"),
        help="Directory containing the coverage cache (default: .coverage-cache)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the selected tests (otherwise just print them)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    # Get repository root
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    repo_root = Path(result.stdout.strip())

    # Get changed files
    if args.base:
        changed_files = get_changed_files(args.base)
        if args.verbose:
            print(f"Comparing against {args.base}...")
            print(f"Found {len(changed_files)} changed files")
            print()
    else:
        changed_files = args.files

    if not changed_files:
        print("No changed files detected.")
        return 0

    # Select tests
    tests, exit_code = select_tests(
        changed_files, args.cache_dir, repo_root, verbose=args.verbose
    )

    # Handle fallback cases
    if exit_code == _EXIT_CODE_RUN_ALL_TESTS:
        if args.run:
            # Run all tests
            result = subprocess.run(["make", "test"])
            return result.returncode
        print("ALL_TESTS")  # Signal to caller to run all tests
        return _EXIT_CODE_RUN_ALL_TESTS
    if exit_code != 0:
        return exit_code

    # Output or run tests
    if args.run:
        if not tests:
            print(
                "No tests selected. This likely means the changes don't affect tested code."
            )
            print("Running all tests to be safe...")
            result = subprocess.run(["make", "test"])
            return result.returncode
        return run_tests(tests)
    # Print test IDs (one per line for easy parsing)
    for test in tests:
        print(test)
    return 0


if __name__ == "__main__":
    sys.exit(main())

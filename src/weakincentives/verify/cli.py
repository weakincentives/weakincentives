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

"""CLI for the verification toolbox.

Provides the `wink verify` command for running verification checks.

Usage::

    # Run all checkers
    wink verify

    # Run specific category
    wink verify -c architecture

    # Run specific checkers
    wink verify bandit pip_audit

    # List available checkers
    wink verify --list

    # Output as JSON
    wink verify --json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from weakincentives.verify._output import Output, OutputConfig
from weakincentives.verify._paths import find_project_root
from weakincentives.verify._registry import get_all_checkers, get_categories
from weakincentives.verify._runner import run_checkers, run_checkers_async
from weakincentives.verify._types import CheckContext, RunConfig


def _create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="wink verify",
        description="Run verification checks on the codebase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Categories:
  architecture       Module boundaries and layering
  documentation      Docs, specs, examples
  security           Security scanning
  dependencies       Dependency analysis
  types              Type checking
  tests              Test execution

Examples:
  wink verify                    # Run all checkers
  wink verify -c architecture    # Run architecture checkers only
  wink verify bandit pip_audit   # Run specific checkers
  wink verify --list             # Show available checkers
""",
    )

    parser.add_argument(
        "checkers",
        nargs="*",
        metavar="CHECKER",
        help="Specific checker names to run (default: all)",
    )

    parser.add_argument(
        "-c",
        "--category",
        action="append",
        dest="categories",
        metavar="CATEGORY",
        help="Run all checkers in category (can be repeated)",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress success output",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    parser.add_argument(
        "-j",
        "--parallel",
        type=int,
        metavar="N",
        help="Max parallel checkers (default: CPU count)",
    )

    parser.add_argument(
        "--maxfail",
        type=int,
        metavar="N",
        help="Stop after N failures",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_checkers",
        help="List available checkers and exit",
    )

    parser.add_argument(
        "--root",
        type=Path,
        metavar="PATH",
        help="Project root directory (default: auto-detect)",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply auto-fixes where supported",
    )

    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run checkers synchronously (no parallelism)",
    )

    return parser


def _list_checkers(output: Output) -> int:
    """List available checkers and categories."""
    checkers = get_all_checkers()
    categories = get_categories()

    print("Available categories:")
    for category in categories:
        category_checkers = [c for c in checkers if c.category == category]
        print(f"  {category} ({len(category_checkers)} checker(s))")

    print()
    print("Available checkers:")
    for checker in checkers:
        print(f"  {checker.category}.{checker.name}")
        print(f"    {checker.description}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command line arguments (default: sys.argv[1:]).

    Returns:
        Exit code (0 for success, 1 for failures, 2 for errors).
    """
    parser = _create_parser()
    args = parser.parse_args(argv)

    # Configure output
    output_config = OutputConfig(
        quiet=args.quiet,
        color=not args.no_color if args.no_color else None,
        json_output=args.json_output,
    )
    output = Output(output_config)

    # Handle --list
    if args.list_checkers:
        return _list_checkers(output)

    # Find project root
    try:
        if args.root:
            project_root = args.root.resolve()
            if not (project_root / "pyproject.toml").exists():
                output.error(f"No pyproject.toml found at {project_root}")
                return 2
        else:
            project_root = find_project_root()
    except FileNotFoundError as e:
        output.error(str(e))
        return 2

    # Create check context
    ctx = CheckContext.from_project_root(
        project_root,
        quiet=args.quiet,
        fix=args.fix,
    )

    # Create run config
    run_config = RunConfig(
        max_failures=args.maxfail,
        max_parallel=args.parallel,
        categories=frozenset(args.categories) if args.categories else None,
        checkers=frozenset(args.checkers) if args.checkers else None,
    )

    # Get checkers
    all_checkers = get_all_checkers()

    # Filter checkers based on config
    checkers_to_run = [c for c in all_checkers if run_config.should_run(c)]

    if not checkers_to_run:
        output.error("No checkers selected")
        if args.checkers:
            output.info(f"Requested checkers: {', '.join(args.checkers)}")
            output.info("Use --list to see available checkers")
        return 2

    # Run checkers
    if args.sync or args.parallel == 1:
        results = run_checkers(checkers_to_run, ctx, config=run_config)
    else:
        results = asyncio.run(
            run_checkers_async(checkers_to_run, ctx, config=run_config)
        )

    # Report results
    for result in results:
        if result.passed:
            output.checker_success(result)
        else:
            output.checker_failure(result)

    output.summary(results)

    # Return appropriate exit code
    failed = sum(1 for r in results if not r.passed)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

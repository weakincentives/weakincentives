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

"""Verification toolchain for the weakincentives project.

Usage:
    uv run python check.py           # Run all checks (recommended)
    uv run python check.py lint test # Run specific checks
    uv run python check.py --list    # List available checks

Note: Use 'uv run' to ensure Python 3.12+ for PEP 695 syntax support.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add toolchain to path
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from toolchain import (  # noqa: E402
    ConsoleFormatter,
    JSONFormatter,
    QuietFormatter,
    Runner,
)
from toolchain.checkers import create_all_checkers  # noqa: E402
from toolchain.utils import patch_ast_for_bandit  # noqa: E402


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run verification checks on the codebase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check.py                 Run all checks
  python check.py format lint     Run format and lint checks only
  python check.py test -v         Run tests with verbose output
  python check.py --list          List available checkers
""",
    )
    parser.add_argument(
        "checks",
        nargs="*",
        help="Specific checks to run (default: all)",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available checks",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output (show full output on failure)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode (only show failures)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args()

    # Patch AST for bandit compatibility
    patch_ast_for_bandit()

    # Create runner with all checkers
    runner = Runner()
    for checker in create_all_checkers():
        runner.register(checker)

    # List mode
    if args.list:
        print("Available checks:")
        for name, description in runner.list_checkers():
            print(f"  {name:<15} {description}")
        return 0

    # Run checks
    try:
        report = runner.run(args.checks if args.checks else None)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Format output
    if args.json:
        formatter = JSONFormatter()
    elif args.quiet:
        formatter = QuietFormatter(color=not args.no_color)
    else:
        formatter = ConsoleFormatter(verbose=args.verbose, color=not args.no_color)

    output = formatter.format(report)
    if output:
        print(output)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

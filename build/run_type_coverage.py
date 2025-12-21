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

"""Type coverage checker using pyright --verifytypes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

PACKAGE_NAME = "weakincentives"
DEFAULT_THRESHOLD = 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum type completeness score (default: {DEFAULT_THRESHOLD}%).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress output unless check fails.",
    )
    args = parser.parse_args()

    result = subprocess.run(
        [
            "pyright",
            "--verifytypes",
            PACKAGE_NAME,
            "--ignoreexternal",
            "--outputjson",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0 and not result.stdout:
        print(result.stderr, file=sys.stderr)
        return result.returncode

    data = json.loads(result.stdout)
    type_completeness = data.get("typeCompleteness", {})
    score = type_completeness.get("completenessScore", 0.0) * 100

    exported = type_completeness.get("exportedSymbolCounts", {})
    known = exported.get("withKnownType", 0)
    ambiguous = exported.get("withAmbiguousType", 0)
    unknown = exported.get("withUnknownType", 0)
    total = known + ambiguous + unknown

    if not args.quiet:
        print(
            f"Type coverage: {score:.1f}% "
            f"({known}/{total} symbols with known types, "
            f"{ambiguous} ambiguous, {unknown} unknown)"
        )

    if score < args.threshold:
        print(
            f"Type coverage {score:.1f}% is below the required threshold of "
            f"{args.threshold:.1f}%.",
            file=sys.stderr,
        )
        if args.quiet:
            # Re-run without JSON to show detailed errors
            subprocess.run(
                ["pyright", "--verifytypes", PACKAGE_NAME, "--ignoreexternal"],
                check=False,
            )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

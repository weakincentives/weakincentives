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

"""Validate integration tests with pyright (basic mode).

This ensures all imports and references in integration tests are valid
without requiring API keys or running the tests.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

INTEGRATION_TESTS_DIR = "integration-tests"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress output unless check fails.",
    )
    args = parser.parse_args()

    integration_tests_path = Path(INTEGRATION_TESTS_DIR)
    if not integration_tests_path.exists():
        print(f"Directory {INTEGRATION_TESTS_DIR} not found.", file=sys.stderr)
        return 1

    # Create a temporary pyrightconfig.json with basic mode settings
    # Basic mode catches import errors and obvious type issues without being
    # overly strict on test code patterns
    config = {
        "typeCheckingMode": "basic",
        "pythonVersion": "3.12",
        "include": [INTEGRATION_TESTS_DIR],
    }

    # Place config in current directory so relative paths work
    config_path = Path(".pyrightconfig-integration-tests.json")
    config_path.write_text(json.dumps(config))

    try:
        cmd = ["pyright", "--project", str(config_path)]

        if args.quiet:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print("Integration test validation failed:", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                return result.returncode
        else:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                return result.returncode

        if not args.quiet:
            print("Integration tests validated successfully.")

        return 0
    finally:
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())

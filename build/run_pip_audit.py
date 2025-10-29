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

"""Run pip-audit with the repository defaults and quiet successes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent

    command = [
        sys.executable,
        "-m",
        "pip_audit",
        "--progress-spinner",
        "off",
        "--strict",
        "--skip-editable",
        str(project_root),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    output = f"{result.stdout}{result.stderr}".strip()

    if (result.returncode != 0 or "warning" in output.lower()) and output:
        print(output)

    return result.returncode


if __name__ == "__main__":  # pragma: no cover - invoked via Makefile target
    raise SystemExit(main())

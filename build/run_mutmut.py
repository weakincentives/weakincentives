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

"""Mutation test runner with a configurable score gate.

Mutmut reads its configuration from [tool.mutmut] in pyproject.toml.
This wrapper adds a score gate via [tool.mutation-check].minimum_score.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
PYPROJECT_PATH = ROOT / "pyproject.toml"

# Default threshold if not configured
DEFAULT_MINIMUM_SCORE = 80.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Enforce the configured mutation score gate after running mutants.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override the configured minimum mutation score gate.",
    )
    args, unknown_args = parser.parse_known_args()

    threshold = args.threshold if args.threshold is not None else _load_minimum_score()

    run_code = _run_mutmut(unknown_args)
    if run_code != 0:
        return run_code

    stats = _load_results()
    score = _mutation_score(stats)
    summary = _format_summary(stats, score)
    print(summary)

    if args.check and score < threshold:
        message = (
            f"Mutation score {score:.1f}% is below the required threshold of "
            f"{threshold:.1f}%."
        )
        print(message, file=sys.stderr)
        return 1

    return 0


def _load_minimum_score() -> float:
    """Load minimum_score from [tool.mutation-check] in pyproject.toml."""
    if not PYPROJECT_PATH.exists():
        return DEFAULT_MINIMUM_SCORE

    data = tomllib.loads(PYPROJECT_PATH.read_text())
    return float(
        data.get("tool", {})
        .get("mutation-check", {})
        .get("minimum_score", DEFAULT_MINIMUM_SCORE)
    )


def _run_mutmut(extra_args: list[str]) -> int:
    command = ["mutmut", "run"]
    command.extend(extra_args)

    result = subprocess.run(command, check=False)
    return result.returncode


def _load_results() -> dict[str, Any]:
    result = subprocess.run(
        ["mutmut", "results", "--all", "true"],
        check=True,
        capture_output=True,
        text=True,
    )

    stats: dict[str, int] = {
        "killed": 0,
        "survived": 0,
        "timeout": 0,
        "suspicious": 0,
        "incompetent": 0,
        "skipped": 0,
    }

    for line in result.stdout.splitlines():
        if ":" not in line:
            continue

        *_, status_part = line.rsplit(":", maxsplit=1)
        status = status_part.strip().split()[0]
        if status in stats:
            stats[status] += 1

    return stats


def _mutation_score(stats: dict[str, Any]) -> float:
    numeric_values = [value for value in stats.values() if isinstance(value, int)]
    total = sum(numeric_values)
    if total == 0:
        return 0.0

    killed = stats.get("killed", 0)
    timeouts = stats.get("timeout", 0)
    detected = killed + timeouts
    return (detected / total) * 100


def _format_summary(stats: dict[str, Any], score: float) -> str:
    parts = [
        f"killed={stats.get('killed', 0)}",
        f"survived={stats.get('survived', 0)}",
        f"timeout={stats.get('timeout', 0)}",
        f"suspicious={stats.get('suspicious', 0)}",
        f"incompetent={stats.get('incompetent', 0)}",
        f"skipped={stats.get('skipped', 0)}",
        f"score={score:.1f}%",
    ]
    return "Mutation summary: " + " | ".join(parts)


if __name__ == "__main__":  # pragma: no cover - invoked via Makefile
    sys.exit(main())

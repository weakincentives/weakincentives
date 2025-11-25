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

"""Mutation test runner with a configurable score gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "mutation.toml"


@dataclass(frozen=True)
class MutationConfig:
    paths_to_mutate: list[str]
    tests_dir: str
    exclude: list[str]
    runner: str
    use_coverage: bool
    minimum_score: float


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

    config = _load_config()
    threshold = args.threshold if args.threshold is not None else config.minimum_score

    run_code = _run_mutmut(config, unknown_args)
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


def _load_config() -> MutationConfig:
    if not CONFIG_PATH.exists():
        message = f"Missing configuration file at {CONFIG_PATH}"
        raise FileNotFoundError(message)

    data = tomllib.loads(CONFIG_PATH.read_text()).get("mutation", {})
    return MutationConfig(
        paths_to_mutate=list(data.get("paths_to_mutate", [])),
        tests_dir=str(data.get("tests_dir", "tests")),
        exclude=list(data.get("exclude", [])),
        runner=str(data.get("runner", "python -m pytest -q")),
        use_coverage=bool(data.get("use_coverage", True)),
        minimum_score=float(data.get("minimum_score", 0.0)),
    )


def _run_mutmut(config: MutationConfig, extra_args: list[str]) -> int:
    command = [
        "mutmut",
        "run",
        "--paths-to-mutate",
        ",".join(config.paths_to_mutate),
        "--tests-dir",
        config.tests_dir,
        "--runner",
        config.runner,
    ]

    for path in config.exclude:
        command.extend(["--exclude", path])

    if config.use_coverage:
        command.append("--use-coverage")

    command.extend(extra_args)

    result = subprocess.run(command, check=False)
    return result.returncode


def _load_results() -> dict[str, Any]:
    result = subprocess.run(
        ["mutmut", "results", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout or "{}")


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

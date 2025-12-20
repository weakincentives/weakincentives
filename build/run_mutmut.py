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
    also_copy: list[str]
    runner: str
    use_coverage: bool
    minimum_score: float
    score_gates: dict[str, float]


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

    stats, module_stats = _load_results()
    score = _mutation_score(stats)
    summary = _format_summary(stats, score)
    print(summary)

    gate_failures: list[str] = []

    if args.check:
        # Check global threshold
        if score < threshold:
            gate_failures.append(
                f"Overall score {score:.1f}% is below the required threshold of "
                f"{threshold:.1f}%."
            )

        # Check per-module gates
        for gate_pattern, gate_threshold in config.score_gates.items():
            gate_stats = _aggregate_module_stats(module_stats, gate_pattern)
            gate_score = _mutation_score(gate_stats)
            total_mutants = sum(
                v for v in gate_stats.values() if isinstance(v, int)
            )
            if total_mutants > 0 and gate_score < gate_threshold:
                gate_failures.append(
                    f"Module '{gate_pattern}' score {gate_score:.1f}% is below "
                    f"the required threshold of {gate_threshold:.1f}%."
                )

    if gate_failures:
        for failure in gate_failures:
            print(failure, file=sys.stderr)
        return 1

    return 0


def _load_config() -> MutationConfig:
    if not CONFIG_PATH.exists():
        message = f"Missing configuration file at {CONFIG_PATH}"
        raise FileNotFoundError(message)

    data = tomllib.loads(CONFIG_PATH.read_text()).get("mutation", {})
    score_gates_raw = data.get("score_gates", {})
    score_gates = {str(k): float(v) for k, v in score_gates_raw.items()}
    return MutationConfig(
        paths_to_mutate=list(data.get("paths_to_mutate", [])),
        tests_dir=str(data.get("tests_dir", "tests")),
        exclude=list(data.get("exclude", [])),
        also_copy=list(data.get("also_copy", [])),
        runner=str(data.get("runner", "python -m pytest -q")),
        use_coverage=bool(data.get("use_coverage", True)),
        minimum_score=float(data.get("minimum_score", 0.0)),
        score_gates=score_gates,
    )


def _run_mutmut(config: MutationConfig, extra_args: list[str]) -> int:
    command = ["mutmut", "run"]
    command.extend(extra_args)

    result = subprocess.run(command, check=False)
    return result.returncode


def _empty_stats() -> dict[str, int]:
    return {
        "killed": 0,
        "survived": 0,
        "timeout": 0,
        "suspicious": 0,
        "incompetent": 0,
        "skipped": 0,
    }


def _load_results() -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    """Load mutation results with per-module breakdown.

    Returns:
        Tuple of (global_stats, per_module_stats) where per_module_stats maps
        module path patterns to their individual statistics.
    """
    result = subprocess.run(
        ["mutmut", "results", "--all", "true"],
        check=True,
        capture_output=True,
        text=True,
    )

    stats: dict[str, int] = _empty_stats()
    module_stats: dict[str, dict[str, int]] = {}

    for line in result.stdout.splitlines():
        if ":" not in line:
            continue

        # Parse file path and status from line format: "path/to/file.py:line: status ..."
        parts = line.split(":")
        if len(parts) < 3:
            continue

        file_path = parts[0].strip()
        status_part = parts[-1].strip().split()[0]

        if status_part in stats:
            stats[status_part] += 1

            # Track per-module stats
            if file_path not in module_stats:
                module_stats[file_path] = _empty_stats()
            module_stats[file_path][status_part] += 1

    return stats, module_stats


def _aggregate_module_stats(
    module_stats: dict[str, dict[str, int]],
    gate_pattern: str,
) -> dict[str, int]:
    """Aggregate stats for all files matching a gate pattern."""
    aggregated = _empty_stats()
    for file_path, file_stats in module_stats.items():
        if gate_pattern in file_path:
            for key, value in file_stats.items():
                aggregated[key] += value
    return aggregated


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

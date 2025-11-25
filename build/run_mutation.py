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

"""Run mutation testing with mutmut and enforce the configured score gate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import tomllib
from mutmut.__main__ import (
    SourceFileMutationData,
    calculate_summary_stats,
    ensure_config_loaded,
    walk_source_files,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "mutation.toml"


@dataclass
class MutationSummary:
    """Aggregated mutation testing results."""

    score: float
    considered: int
    total: int
    killed: int
    survived: int
    timeout: int
    suspicious: int
    skipped: int
    no_tests: int
    not_checked: int
    interrupted: int
    segfault: int


def load_minimum_score(config_path: Path) -> int:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    gates = data.get("gates", {})
    return int(gates.get("minimum_mutation_score", 0))


def run_mutmut(max_children: int | None) -> int:
    command = [sys.executable, "-m", "mutmut", "run"]
    if max_children is not None:
        command.extend(["--max-children", str(max_children)])

    return subprocess.run(command, check=False).returncode


def collect_mutation_data() -> MutationSummary:
    ensure_config_loaded()
    source_data = _load_source_file_data(walk_source_files())
    stats = calculate_summary_stats(source_data)

    considered = (
        stats.total
        - stats.not_checked
        - stats.no_tests
        - stats.skipped
        - stats.check_was_interrupted_by_user
    )
    score = (stats.killed / considered * 100) if considered else 0.0

    return MutationSummary(
        score=score,
        considered=considered,
        total=stats.total,
        killed=stats.killed,
        survived=stats.survived,
        timeout=stats.timeout,
        suspicious=stats.suspicious,
        skipped=stats.skipped,
        no_tests=stats.no_tests,
        not_checked=stats.not_checked,
        interrupted=stats.check_was_interrupted_by_user,
        segfault=stats.segfault,
    )


def _load_source_file_data(paths: Iterable[Path]):
    source_file_mutation_data_by_path = {}
    for path in paths:
        if not str(path).endswith(".py"):
            continue

        mutation_data = SourceFileMutationData(path=path)
        mutation_data.load()
        source_file_mutation_data_by_path[str(path)] = mutation_data

    return source_file_mutation_data_by_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Enforce the configured minimum mutation score.",
    )
    parser.add_argument(
        "--max-children",
        type=int,
        default=None,
        help="Override the number of child processes mutmut uses.",
    )
    args = parser.parse_args()

    if not CONFIG_PATH.exists():
        print("Missing mutation configuration at mutation.toml", file=sys.stderr)
        return 1

    minimum_score = load_minimum_score(CONFIG_PATH)

    exit_code = run_mutmut(max_children=args.max_children)
    if exit_code != 0:
        return exit_code

    summary = collect_mutation_data()
    print(
        "Mutation score: "
        f"{summary.score:.1f}% ({summary.killed} killed / {summary.considered} considered, "
        f"{summary.total} total)"
    )
    if summary.survived:
        print(f"Survived mutants: {summary.survived}")
    if summary.timeout:
        print(f"Timeouts: {summary.timeout}")
    if summary.suspicious:
        print(f"Suspicious mutants: {summary.suspicious}")
    if summary.interrupted or summary.segfault:
        print(
            "Unaccounted mutants: "
            f"{summary.interrupted} interrupted, {summary.segfault} segfaults"
        )

    if args.check and summary.score < minimum_score:
        print(
            f"Mutation score {summary.score:.1f}% is below the required floor of {minimum_score}%.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - invoked via Makefile target
    raise SystemExit(main())

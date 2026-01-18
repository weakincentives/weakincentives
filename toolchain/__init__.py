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

This package provides a framework for running verification checks on the
codebase. It is designed to be extensible, with a clean separation between:

- Result types: Structured representation of check outcomes
- Checkers: Individual verification passes
- Runner: Orchestration of multiple checkers
- Formatters: Output rendering for different contexts

Usage:
    from toolchain import Runner, ConsoleFormatter
    from toolchain.checkers import create_all_checkers

    runner = Runner()
    for checker in create_all_checkers():
        runner.register(checker)

    report = runner.run()
    print(ConsoleFormatter().format(report))
"""

from .checker import Checker, SubprocessChecker
from .output import ConsoleFormatter, JSONFormatter, QuietFormatter
from .result import CheckResult, Diagnostic, Location, Report
from .runner import Runner

__all__ = [
    # Result types
    "Location",
    "Diagnostic",
    "CheckResult",
    "Report",
    # Checker types
    "Checker",
    "SubprocessChecker",
    # Runner
    "Runner",
    # Formatters
    "ConsoleFormatter",
    "JSONFormatter",
    "QuietFormatter",
]

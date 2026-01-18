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

"""Unified verification toolbox for weakincentives.

This package provides a production-quality verification framework that
consolidates all build-time checks into a single, composable system.

Example usage::

    from weakincentives.verify import run_checkers, get_all_checkers, CheckContext

    ctx = CheckContext.from_project_root(Path.cwd())
    results = run_checkers(get_all_checkers(), ctx)
    for result in results:
        print(f"{result.checker}: {'PASS' if result.passed else 'FAIL'}")
"""

from weakincentives.verify._types import (
    CheckContext,
    Checker,
    CheckResult,
    Finding,
    Severity,
)
from weakincentives.verify._runner import run_checkers, run_checkers_async
from weakincentives.verify._registry import get_all_checkers, get_checker, get_checkers_by_category

__all__ = [
    # Types
    "CheckContext",
    "Checker",
    "CheckResult",
    "Finding",
    "Severity",
    # Runner
    "run_checkers",
    "run_checkers_async",
    # Registry
    "get_all_checkers",
    "get_checker",
    "get_checkers_by_category",
]

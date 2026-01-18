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

"""Checker execution and orchestration."""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING

from weakincentives.verify._types import CheckContext, CheckResult, Checker, RunConfig

if TYPE_CHECKING:
    from collections.abc import Sequence


def run_checkers(
    checkers: Sequence[Checker],
    ctx: CheckContext,
    *,
    config: RunConfig | None = None,
) -> tuple[CheckResult, ...]:
    """Run checkers synchronously.

    For parallel execution, use run_checkers_async instead.

    Args:
        checkers: The checkers to run.
        ctx: The check context.
        config: Optional run configuration.

    Returns:
        A tuple of CheckResult objects, one per checker.
    """
    config = config or RunConfig()
    results: list[CheckResult] = []
    failure_count = 0

    for checker in checkers:
        if not config.should_run(checker):
            continue

        result = checker.check(ctx)
        results.append(result)

        if not result.passed:
            failure_count += 1
            if config.max_failures is not None and failure_count >= config.max_failures:
                break

    return tuple(results)


async def run_checkers_async(
    checkers: Sequence[Checker],
    ctx: CheckContext,
    *,
    config: RunConfig | None = None,
) -> tuple[CheckResult, ...]:
    """Run checkers with bounded parallelism.

    Uses asyncio to run independent checkers concurrently.

    Args:
        checkers: The checkers to run.
        ctx: The check context.
        config: Optional run configuration.

    Returns:
        A tuple of CheckResult objects, one per checker.
    """
    config = config or RunConfig()
    max_parallel = config.max_parallel or os.cpu_count() or 4

    # Filter checkers based on config
    filtered_checkers = [c for c in checkers if config.should_run(c)]

    if not filtered_checkers:
        return ()

    semaphore = asyncio.Semaphore(max_parallel)
    results: list[CheckResult] = []
    failure_count = 0
    stop_flag = asyncio.Event()

    async def run_one(checker: Checker) -> CheckResult | None:
        nonlocal failure_count

        if stop_flag.is_set():
            return None

        async with semaphore:
            if stop_flag.is_set():
                return None

            # Run the synchronous check in a thread pool
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, checker.check, ctx)

            if not result.passed:
                failure_count += 1
                if config.max_failures is not None and failure_count >= config.max_failures:
                    stop_flag.set()

            return result

    # Run all checkers concurrently
    tasks = [run_one(c) for c in filtered_checkers]
    completed = await asyncio.gather(*tasks)

    # Filter out None results (stopped early)
    results = [r for r in completed if r is not None]

    return tuple(results)


def run_checker_with_timing(checker: Checker, ctx: CheckContext) -> CheckResult:
    """Run a single checker and ensure timing is captured.

    This is a utility for checker implementations that want to
    delegate to a core function without timing.

    Args:
        checker: The checker to run.
        ctx: The check context.

    Returns:
        The check result with timing information.
    """
    start_time = time.monotonic()
    result = checker.check(ctx)
    duration_ms = int((time.monotonic() - start_time) * 1000)

    # If the result doesn't have timing, add it
    if result.duration_ms == 0:
        return CheckResult(
            checker=result.checker,
            findings=result.findings,
            duration_ms=duration_ms,
        )

    return result

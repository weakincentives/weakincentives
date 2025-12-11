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

"""Async/sync bridging utilities for Claude Agent SDK adapter."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine

__all__ = [
    "run_sync",
]


def run_sync[T](coro: Coroutine[object, object, T]) -> T:
    """Run an async coroutine synchronously.

    Creates a new event loop if none is running, otherwise runs
    the coroutine in a separate thread to avoid blocking the existing loop.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.
    """
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create a new one
        return asyncio.run(coro)

    # We have a running loop - this is problematic
    # The SDK requires async execution but we're being called synchronously
    # from within an existing loop. This typically happens in tests or
    # when the adapter is used from async code incorrectly.
    #
    # We can't use asyncio.run() here as it would try to create a new loop.
    # We need to schedule the coroutine on a new loop in a separate thread.

    # Create a new thread to run the coroutine
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()

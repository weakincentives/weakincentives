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

"""Async/sync bridging utilities for the Claude Agent SDK adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine

__all__ = [
    "run_async",
]


def run_async[T](coro: Coroutine[object, object, T]) -> T:
    """Run an async coroutine from synchronous code.

    Creates a new event loop, runs the coroutine to completion, and
    cleans up the loop.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.

    Note:
        This creates a new event loop per call. The Claude Agent SDK
        adapter uses this to bridge between the synchronous
        ProviderAdapter.evaluate() interface and the async SDK client.
    """
    return asyncio.run(coro)

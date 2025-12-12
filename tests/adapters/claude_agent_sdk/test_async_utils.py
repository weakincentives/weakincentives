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

"""Tests for Claude Agent SDK async utilities."""

from __future__ import annotations

import pytest

from weakincentives.adapters.claude_agent_sdk._async_utils import run_async


class TestRunAsync:
    def test_runs_simple_coroutine(self) -> None:
        async def simple_coro() -> int:
            return 42

        result = run_async(simple_coro())
        assert result == 42

    def test_runs_async_with_await(self) -> None:
        async def inner() -> str:
            return "inner"

        async def outer() -> str:
            value = await inner()
            return f"outer:{value}"

        result = run_async(outer())
        assert result == "outer:inner"

    def test_propagates_exceptions(self) -> None:
        async def failing_coro() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_async(failing_coro())

    def test_returns_none_from_void_coro(self) -> None:
        async def void_coro() -> None:
            pass

        result = run_async(void_coro())
        assert result is None

    def test_handles_complex_return_types(self) -> None:
        async def complex_coro() -> dict[str, list[int]]:
            return {"values": [1, 2, 3]}

        result = run_async(complex_coro())
        assert result == {"values": [1, 2, 3]}

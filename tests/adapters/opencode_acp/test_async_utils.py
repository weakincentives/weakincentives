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

"""Tests for OpenCode ACP async utilities."""

from __future__ import annotations

from weakincentives.adapters.opencode_acp._async import run_async


class TestRunAsync:
    def test_runs_simple_coroutine(self) -> None:
        async def simple() -> int:
            return 42

        result = run_async(simple())
        assert result == 42

    def test_runs_coroutine_with_awaits(self) -> None:
        import asyncio

        async def with_awaits() -> str:
            await asyncio.sleep(0.001)
            return "completed"

        result = run_async(with_awaits())
        assert result == "completed"

    def test_propagates_exceptions(self) -> None:
        import pytest

        async def failing() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_async(failing())

    def test_returns_typed_result(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class Result:
            value: str

        async def returns_dataclass() -> Result:
            return Result(value="test")

        result = run_async(returns_dataclass())
        assert isinstance(result, Result)
        assert result.value == "test"

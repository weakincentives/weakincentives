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

"""Clock control helpers for tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from weakincentives.runtime.clock import Clock, FakeClock, SystemClock

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def clock(request: pytest.FixtureRequest) -> Generator[Clock, None, None]:
    """Provide a clock for tests.

    By default, returns a FakeClock for deterministic time control.
    When the test is marked with @pytest.mark.allow_system_clock,
    returns a SystemClock for tests that rely on real blocking behavior.
    """
    if request.node.get_closest_marker("allow_system_clock"):
        yield SystemClock()
    else:
        yield FakeClock()


__all__ = ["clock"]

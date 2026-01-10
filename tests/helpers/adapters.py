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

"""Shared adapter name fixtures for tests."""

from __future__ import annotations

from typing import cast

from weakincentives.types import AdapterName

GENERIC_ADAPTER_NAME = cast(AdapterName, "adapter")
"""Adapter name placeholder used in tests that inspect stored events."""

TEST_ADAPTER_NAME = cast(AdapterName, "test")
"""Adapter identifier used when simulating provider interactions in tests."""

DUMMY_ADAPTER_NAME = cast(AdapterName, "dummy")
"""Adapter identifier used by dummy conversation runners in tests."""

UNIT_TEST_ADAPTER_NAME = cast(AdapterName, "unit")
"""Adapter identifier used in thread-safety tests."""

__all__ = [
    "DUMMY_ADAPTER_NAME",
    "GENERIC_ADAPTER_NAME",
    "TEST_ADAPTER_NAME",
    "UNIT_TEST_ADAPTER_NAME",
]

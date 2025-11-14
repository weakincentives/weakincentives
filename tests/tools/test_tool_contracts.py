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

"""Audits enforcing tool dataclass contracts."""

from __future__ import annotations

from tests.plugins.tool_contracts import (
    ToolDataclassContractCase,
    assert_tool_dataclass_contract,
)


def test_tool_dataclasses_satisfy_contract(
    tool_dataclass_case: ToolDataclassContractCase,
) -> None:
    """Ensure dataclasses backing built-in tools satisfy structural invariants."""

    assert_tool_dataclass_contract(tool_dataclass_case)

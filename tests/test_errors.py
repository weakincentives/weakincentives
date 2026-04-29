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

"""Tests for the stable error hierarchy."""

from __future__ import annotations

from weakincentives.core.errors import (
    BudgetExceededError,
    ContractError,
    DeadlineExceededError,
    PolicyDeniedError,
    PromptError,
    SessionError,
    SnapshotError,
    ToolError,
    TransactionError,
    WinkError,
)


def test_all_errors_inherit_from_winkerror() -> None:
    for cls in (
        PromptError,
        ToolError,
        SessionError,
        SnapshotError,
        TransactionError,
        ContractError,
    ):
        assert issubclass(cls, WinkError)


def test_contract_subclasses() -> None:
    for cls in (
        DeadlineExceededError,
        BudgetExceededError,
        PolicyDeniedError,
    ):
        assert issubclass(cls, ContractError)
        assert issubclass(cls, WinkError)


def test_errors_are_raisable() -> None:
    for cls in (
        WinkError,
        PromptError,
        ToolError,
        SessionError,
        SnapshotError,
        TransactionError,
        ContractError,
        DeadlineExceededError,
        BudgetExceededError,
        PolicyDeniedError,
    ):
        try:
            raise cls("boom")
        except WinkError as error:
            assert str(error) == "boom"

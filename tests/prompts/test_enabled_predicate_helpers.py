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

from __future__ import annotations

from dataclasses import dataclass

from weakincentives.prompt._enabled_predicate import (
    callable_requires_positional_argument,
    normalize_enabled_predicate,
)


@dataclass
class ToggleParams:
    include: bool


def test_normalize_enabled_predicate_accepts_zero_arg_callable() -> None:
    predicate = normalize_enabled_predicate(lambda: False, params_type=None)

    assert predicate is not None
    assert predicate(None) is False
    assert callable_requires_positional_argument(lambda: False) is False


def test_normalize_enabled_predicate_passes_positional_value() -> None:
    recorded: list[object | None] = []

    def enabled(value: object | None) -> bool:
        recorded.append(value)
        return value is None

    predicate = normalize_enabled_predicate(enabled, params_type=None)

    assert predicate is not None
    assert predicate(None) is True
    assert recorded == [None]


def test_normalize_enabled_predicate_handles_parameterized_callable() -> None:
    def enabled(params: ToggleParams) -> bool:
        return params.include

    predicate = normalize_enabled_predicate(
        enabled, params_type=ToggleParams
    )

    assert predicate is not None
    assert predicate(ToggleParams(include=True)) is True
    assert predicate(ToggleParams(include=False)) is False

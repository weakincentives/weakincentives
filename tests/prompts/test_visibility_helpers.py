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

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pytest

from weakincentives.prompt._visibility import (
    SectionVisibility,
    _visibility_requires_positional_argument,
    normalize_visibility_selector,
)


@dataclass
class _VisibilityParams:
    flag: bool = False


def test_visibility_selector_rejects_invalid_return_type() -> None:
    selector = normalize_visibility_selector(
        lambda params: cast(SectionVisibility, "not-visibility"), _VisibilityParams
    )

    with pytest.raises(TypeError):
        selector(_VisibilityParams())


def test_visibility_requires_positional_argument_branches() -> None:
    assert (
        _visibility_requires_positional_argument(lambda: SectionVisibility.FULL)
        is False
    )
    assert (
        _visibility_requires_positional_argument(lambda params: SectionVisibility.FULL)
        is True
    )
    assert (
        _visibility_requires_positional_argument(
            cast(Callable[..., SectionVisibility], object())
        )
        is True
    )

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

import pytest

from weakincentives.prompt import Chapter, Section
from weakincentives.prompt.generic_params_specializer import (
    GenericParamsSpecializer,
)


@dataclass
class _Params:
    value: str = "noop"


@pytest.mark.parametrize("generic_cls", [Section, Chapter])
def test_generic_specialization_rejects_multiple_arguments(
    generic_cls: type[GenericParamsSpecializer],
) -> None:
    with pytest.raises(TypeError):
        generic_cls.__class_getitem__((_Params, _Params))


@pytest.mark.parametrize(
    "generic_cls",
    [Section, Chapter],
)
def test_generic_specialization_preserves_metadata(
    generic_cls: type[GenericParamsSpecializer],
) -> None:
    specialized = generic_cls[_Params]

    assert specialized.__module__ == generic_cls.__module__
    assert specialized.__qualname__ == generic_cls.__qualname__
    assert specialized.__name__ == generic_cls.__name__

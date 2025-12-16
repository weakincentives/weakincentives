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

from typing import Literal, TypeVar

from ..dataclasses import FrozenDataclass
from ..types import SupportsDataclass

__all__ = ["StructuredOutputConfig"]

DataclassT = TypeVar("DataclassT", bound=SupportsDataclass)


@FrozenDataclass()
class StructuredOutputConfig[DataclassT]:
    """Resolved structured output declaration for a prompt."""

    dataclass_type: type[DataclassT]
    container: Literal["object", "array"]
    allow_extra_keys: bool

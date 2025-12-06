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

"""Dataclass helper utilities for session modules."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    from ...prompt._types import SupportsDataclass


def is_dataclass_instance(value: object) -> TypeGuard[SupportsDataclass]:
    """Return ``True`` when ``value`` is a dataclass instance."""

    return is_dataclass(value) and not isinstance(value, type)


__all__ = ["is_dataclass_instance"]

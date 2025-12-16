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

"""Internal typing helpers for the :mod:`weakincentives.prompt` package."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types.dataclass import SupportsDataclass


@runtime_checkable
class ToolRenderableResult(SupportsDataclass, Protocol):
    """Protocol implemented by tool result payloads providing render()."""

    def render(self) -> str: ...


__all__ = [
    "ToolRenderableResult",
]

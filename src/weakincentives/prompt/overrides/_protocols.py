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

"""Protocol definitions shared across the prompt overrides package."""

from __future__ import annotations

from typing import Protocol

from .._types import SupportsDataclass
from ..tool import Tool


class SectionLike(Protocol):
    """Structural protocol describing a prompt section exposed to overrides."""

    def original_body_template(self) -> str | None: ...

    def tools(self) -> tuple[Tool[SupportsDataclass, SupportsDataclass], ...]: ...

    accepts_overrides: bool


class SectionNodeLike(Protocol):
    """Structural protocol describing a section node within a prompt tree."""

    path: tuple[str, ...]
    section: SectionLike


class PromptLike(Protocol):
    """Structural protocol describing prompts that support overrides."""

    ns: str
    key: str

    @property
    def sections(self) -> tuple[SectionNodeLike, ...]: ...


__all__ = ["PromptLike", "SectionLike", "SectionNodeLike"]

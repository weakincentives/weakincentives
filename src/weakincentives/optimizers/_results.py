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

"""Result types for optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..adapters.core import PromptResponse


class PersistenceScope(StrEnum):
    """Where optimization artifacts are stored."""

    SESSION = "session"
    GLOBAL = "global"


@dataclass(slots=True, frozen=True)
class OptimizationResult[ArtifactT]:
    """Generic result container for optimization algorithms."""

    response: PromptResponse[object] | None
    """
    The provider response from the optimization prompt, if evaluation
    occurred. None for purely analytical optimizers.
    """

    artifact: ArtifactT
    """The primary optimization artifact (digest, compressed prompt, etc.)."""

    metadata: dict[str, object]
    """
    Algorithm-specific metadata (token counts, compression ratio,
    section paths, etc.).
    """


@dataclass(slots=True, frozen=True)
class WorkspaceDigestResult:
    """Result of workspace digest optimization."""

    response: PromptResponse[object]
    """The provider response from the optimization prompt."""

    summary: str
    """A concise overview of the workspace digest."""

    digest: str
    """The detailed workspace digest text."""

    scope: PersistenceScope
    """Where the digest was persisted."""

    section_key: str
    """The WorkspaceDigestSection key that was updated."""


__all__ = [
    "OptimizationResult",
    "PersistenceScope",
    "WorkspaceDigestResult",
]

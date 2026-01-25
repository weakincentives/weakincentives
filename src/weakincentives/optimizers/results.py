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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..adapters.core import PromptResponse


class PersistenceScope(StrEnum):
    """Enumeration of storage scopes for optimization artifacts.

    Controls where optimization results are persisted, affecting their
    lifetime and visibility across sessions.

    Values:
        SESSION: Artifacts are stored in the current session only. They are
            available immediately but lost when the session ends. Use this
            for transient optimizations or when override stores are unavailable.
        GLOBAL: Artifacts are persisted to the prompt overrides store, making
            them available across sessions. Requires an overrides_store in the
            OptimizationContext. Use this for expensive optimizations that
            should be cached long-term.
    """

    SESSION = "session"
    GLOBAL = "global"


@dataclass(slots=True, frozen=True)
class OptimizationResult[ArtifactT]:
    """Generic result container for optimization algorithms.

    Provides a standardized structure for returning optimization outcomes,
    including the primary artifact, optional provider response, and
    algorithm-specific metadata. Use this for custom optimizers that need
    a flexible result type.

    Type Parameters:
        ArtifactT: The type of the primary optimization artifact (e.g., str
            for text digests, dict for structured summaries).

    Attributes:
        response: The provider response from evaluation, if the optimizer
            invoked the adapter. None for purely analytical optimizers that
            don't require model calls.
        artifact: The primary output of the optimization (digest text,
            compressed prompt, extracted data, etc.).
        metadata: Algorithm-specific details such as token counts, compression
            ratios, processing times, or section paths.

    Example:
        >>> result = OptimizationResult(
        ...     response=provider_response,
        ...     artifact="Optimized prompt text...",
        ...     metadata={"tokens_saved": 1500, "compression_ratio": 0.65},
        ... )
    """

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
    """Result of workspace digest optimization.

    Contains the generated digest along with metadata about where it was
    stored and which section was updated. Use this to verify optimization
    success and access the generated content.

    Attributes:
        response: The full provider response from the optimization prompt,
            including usage statistics and raw model output.
        digest: The extracted workspace digest text summarizing the codebase
            structure, build commands, dependencies, and key details.
        scope: Indicates where the digest was persisted (SESSION for
            in-memory storage, GLOBAL for override store persistence).
        section_key: The key of the WorkspaceDigestSection that was updated
            with the new digest. Use this to locate the section in the prompt.

    Example:
        >>> result = optimizer.optimize(prompt, session=session)
        >>> print(f"Digest stored at scope={result.scope}")
        >>> print(f"Section updated: {result.section_key}")
        >>> print(result.digest[:200])  # Preview the digest
    """

    response: PromptResponse[object]
    """The provider response from the optimization prompt."""

    digest: str
    """The extracted workspace digest text."""

    scope: PersistenceScope
    """Where the digest was persisted."""

    section_key: str
    """The WorkspaceDigestSection key that was updated."""


__all__ = [
    "OptimizationResult",
    "PersistenceScope",
    "WorkspaceDigestResult",
]

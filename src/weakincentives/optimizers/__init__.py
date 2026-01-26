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

"""Prompt optimization framework for transforming and enhancing prompts.

This package provides the core infrastructure for building prompt optimizers
that can analyze, transform, compress, or enhance prompts before they are
sent to a provider. Concrete optimizer implementations live in
``weakincentives.contrib.optimizers``.

Overview
--------

The optimizer framework follows a protocol-based design where:

1. **PromptOptimizer** defines the interface all optimizers must implement
2. **BasePromptOptimizer** provides shared utilities for common patterns
3. **OptimizationContext** bundles dependencies needed during optimization
4. Result types capture optimization artifacts and metadata

Optimizers are parameterized by two type variables:

- ``InputT``: The prompt's output type, allowing optimizers to constrain
  which prompts they accept
- ``OutputT``: The result type returned, enabling type-safe artifact consumption

Exports
-------

Protocol and Base Classes
~~~~~~~~~~~~~~~~~~~~~~~~~

PromptOptimizer
    Protocol defining the optimizer interface. Implementations must provide
    an ``optimize(prompt, *, session)`` method that transforms or analyzes
    the given prompt.

BasePromptOptimizer
    Abstract base class providing shared utilities for optimizer implementations.
    Handles session creation, configuration, and common patterns.

OptimizerConfig
    Base configuration dataclass with settings shared across optimizers,
    such as ``accepts_overrides`` for controlling override store behavior.

Context
~~~~~~~

OptimizationContext
    Immutable context bundle passed to optimizers containing:

    - ``adapter``: Provider adapter for evaluating helper prompts
    - ``dispatcher``: Event dispatcher for optimization telemetry
    - ``deadline``: Optional deadline enforced during optimization
    - ``overrides_store``: Optional store for prompt overrides
    - ``overrides_tag``: Tag for override versioning (default: "latest")
    - ``optimization_session``: Optional pre-configured session

Result Types
~~~~~~~~~~~~

OptimizationResult
    Generic result container parameterized by ``ArtifactT``. Contains:

    - ``response``: The provider response from optimization, if evaluation occurred
    - ``artifact``: The primary optimization artifact (digest, compressed text, etc.)
    - ``metadata``: Algorithm-specific metadata (token counts, ratios, etc.)

WorkspaceDigestResult
    Specialized result for workspace digest optimization containing the
    extracted digest, persistence scope, and updated section key.

PersistenceScope
    Enum indicating where optimization artifacts are stored:

    - ``SESSION``: Scoped to the current session
    - ``GLOBAL``: Persisted globally across sessions

Events
~~~~~~

OptimizationStarted
    Event emitted when an optimizer begins work. Includes optimizer type,
    prompt identifiers, session ID, and timestamp.

OptimizationCompleted
    Event emitted on successful completion. Includes all start fields plus
    a ``result_summary`` dict with algorithm-specific data.

OptimizationFailed
    Event emitted when optimization raises an exception. Includes error
    type and message for diagnostics.

Implementing Custom Optimizers
------------------------------

To create a custom optimizer, either implement the ``PromptOptimizer`` protocol
directly or extend ``BasePromptOptimizer`` for shared utilities::

    from dataclasses import dataclass
    from weakincentives.optimizers import (
        BasePromptOptimizer,
        OptimizationContext,
        OptimizationResult,
        OptimizerConfig,
    )
    from weakincentives.prompt import Prompt
    from weakincentives.runtime.session.protocols import SessionProtocol


    @dataclass(slots=True, frozen=True)
    class CompressionArtifact:
        original_tokens: int
        compressed_tokens: int
        compressed_text: str


    class PromptCompressor(BasePromptOptimizer[object, OptimizationResult[CompressionArtifact]]):
        \"\"\"Example optimizer that compresses prompt content.\"\"\"

        def __init__(
            self,
            context: OptimizationContext,
            *,
            target_ratio: float = 0.5,
        ) -> None:
            super().__init__(context, config=OptimizerConfig(accepts_overrides=False))
            self._target_ratio = target_ratio

        @property
        def _optimizer_scope(self) -> str:
            return "compression"

        def optimize(
            self,
            prompt: Prompt[object],
            *,
            session: SessionProtocol,
        ) -> OptimizationResult[CompressionArtifact]:
            # Create isolated session for internal evaluation
            opt_session = self._create_optimization_session(prompt)

            # Perform compression logic...
            original = prompt.render(session)
            compressed = self._compress(original)

            artifact = CompressionArtifact(
                original_tokens=len(original.split()),
                compressed_tokens=len(compressed.split()),
                compressed_text=compressed,
            )

            return OptimizationResult(
                response=None,  # No provider call needed
                artifact=artifact,
                metadata={"ratio": artifact.compressed_tokens / artifact.original_tokens},
            )

        def _compress(self, text: str) -> str:
            # Implementation details...
            return text

Usage Examples
--------------

Basic optimization with context setup::

    from weakincentives.optimizers import (
        OptimizationContext,
        PersistenceScope,
    )
    from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer

    # Create the optimization context with required dependencies
    context = OptimizationContext(
        adapter=adapter,
        dispatcher=session.dispatcher,
        overrides_store=overrides_store,
    )

    # Instantiate the optimizer
    optimizer = WorkspaceDigestOptimizer(
        context,
        store_scope=PersistenceScope.SESSION,
    )

    # Run optimization
    result = optimizer.optimize(prompt, session=session)
    print(f"Digest stored at: {result.section_key}")

Listening to optimization events::

    from weakincentives.optimizers import (
        OptimizationCompleted,
        OptimizationFailed,
        OptimizationStarted,
    )

    @dispatcher.on(OptimizationStarted)
    def on_start(event: OptimizationStarted) -> None:
        print(f"Starting {event.optimizer_type} on {event.prompt_ns}/{event.prompt_key}")

    @dispatcher.on(OptimizationCompleted)
    def on_complete(event: OptimizationCompleted) -> None:
        print(f"Completed: {event.result_summary}")

    @dispatcher.on(OptimizationFailed)
    def on_failure(event: OptimizationFailed) -> None:
        print(f"Failed: {event.error_type}: {event.error_message}")

Using optimization results::

    result = optimizer.optimize(prompt, session=session)

    # Access the artifact
    if isinstance(result, OptimizationResult):
        print(f"Artifact: {result.artifact}")
        print(f"Metadata: {result.metadata}")

    # For workspace digests specifically
    if isinstance(result, WorkspaceDigestResult):
        print(f"Digest: {result.digest}")
        print(f"Scope: {result.scope}")
"""

from __future__ import annotations

from ._events import OptimizationCompleted, OptimizationFailed, OptimizationStarted
from ._protocol import PromptOptimizer
from .base import BasePromptOptimizer, OptimizerConfig
from .context import OptimizationContext
from .results import OptimizationResult, PersistenceScope, WorkspaceDigestResult

__all__ = [
    "BasePromptOptimizer",
    "OptimizationCompleted",
    "OptimizationContext",
    "OptimizationFailed",
    "OptimizationResult",
    "OptimizationStarted",
    "OptimizerConfig",
    "PersistenceScope",
    "PromptOptimizer",
    "WorkspaceDigestResult",
]

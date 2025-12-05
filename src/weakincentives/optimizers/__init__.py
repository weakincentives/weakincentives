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

"""Prompt optimization algorithms and utilities.

This module provides the ``PromptOptimizer`` protocol and concrete
implementations for transforming, enhancing, or generating content
for ``Prompt`` objects.

Example usage::

    from weakincentives.optimizers import (
        OptimizationContext,
        PersistenceScope,
        WorkspaceDigestOptimizer,
    )

    context = OptimizationContext(
        adapter=adapter,
        event_bus=session.event_bus,
        overrides_store=overrides_store,
    )
    optimizer = WorkspaceDigestOptimizer(
        context,
        store_scope=PersistenceScope.SESSION,
    )
    result = optimizer.optimize(prompt, session=session)
"""

from __future__ import annotations

from ._base import BasePromptOptimizer, OptimizerConfig
from ._context import OptimizationContext
from ._events import OptimizationCompleted, OptimizationFailed, OptimizationStarted
from ._protocol import PromptOptimizer
from ._results import OptimizationResult, PersistenceScope, WorkspaceDigestResult
from .workspace_digest import WorkspaceDigestOptimizer

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
    "WorkspaceDigestOptimizer",
    "WorkspaceDigestResult",
]

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

"""Prompt optimization framework and utilities.

This module provides the ``PromptOptimizer`` protocol and base classes
for building prompt optimizers. Concrete implementations live in
``weakincentives.contrib.optimizers``.

Example usage::

    from weakincentives.optimizers import (
        OptimizationContext,
        PersistenceScope,
    )
    from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer

    context = OptimizationContext(
        adapter=adapter,
        dispatcher=session.dispatcher,
        overrides_store=overrides_store,
    )
    optimizer = WorkspaceDigestOptimizer(
        context,
        store_scope=PersistenceScope.SESSION,
    )
    result = optimizer.optimize(prompt, session=session)
"""

from __future__ import annotations

from .base import BasePromptOptimizer, OptimizerConfig
from .context import OptimizationContext
from ._events import OptimizationCompleted, OptimizationFailed, OptimizationStarted
from ._protocol import PromptOptimizer
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

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

"""Optimization context bundle for optimizer implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..deadlines import Deadline
from ..prompt.overrides import PromptOverridesStore
from ..runtime.events.types import Dispatcher
from ..runtime.session import Session

if TYPE_CHECKING:
    from ..adapters.core import ProviderAdapter


@dataclass(slots=True, frozen=True)
class OptimizationContext:
    """Immutable context bundle for optimization algorithms.

    Optimizers that require provider evaluation receive dependencies through
    this context rather than direct adapter references. This decouples optimizer
    implementations from specific provider configurations and enables consistent
    handling of deadlines, overrides, and telemetry.

    All fields are immutable after construction, ensuring thread-safety and
    predictable behavior when the same context is shared across multiple
    optimization calls.

    Attributes:
        adapter: Provider adapter used to evaluate optimization prompts.
            The adapter handles model invocation and response parsing.
        dispatcher: Event dispatcher for emitting optimization telemetry.
            Optimization events flow through this dispatcher for observability.
        deadline: Optional time limit for the optimization operation. When set,
            the optimizer will abort if the deadline expires.
        overrides_store: Optional store for caching optimization results.
            When provided with GLOBAL scope, results persist across sessions.
        overrides_tag: Version tag for override lookups (default: "latest").
            Use different tags to maintain multiple optimization versions.
        optimization_session: Optional pre-configured session for evaluation.
            When None, optimizers create fresh isolated sessions internally.

    Example:
        >>> context = OptimizationContext(
        ...     adapter=my_adapter,
        ...     dispatcher=session.dispatcher,
        ...     deadline=Deadline.from_timeout(timedelta(seconds=30)),
        ...     overrides_store=my_store,
        ... )
        >>> optimizer = WorkspaceDigestOptimizer(context)
    """

    adapter: ProviderAdapter[object]
    """Provider adapter for evaluating helper prompts."""

    dispatcher: Dispatcher
    """Dispatcher for optimization telemetry."""

    deadline: Deadline | None = None
    """Optional deadline enforced during optimization."""

    overrides_store: PromptOverridesStore | None = None
    """Optional store for persisting or reading prompt overrides."""

    overrides_tag: str = "latest"
    """Tag for override versioning."""

    optimization_session: Session | None = None
    """
    Optional pre-configured session for optimization evaluation.
    When None, optimizers create isolated sessions internally.
    """


__all__ = ["OptimizationContext"]

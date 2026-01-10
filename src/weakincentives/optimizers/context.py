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
    this context rather than direct adapter references.
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

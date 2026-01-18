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

"""Abstract base class for prompt optimizers with shared utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..prompt import Prompt
from ..runtime.session import Session
from ..runtime.session.protocols import SessionProtocol
from .context import OptimizationContext


@dataclass(slots=True, frozen=True)
class OptimizerConfig:
    """Base configuration shared across optimizers."""

    accepts_overrides: bool = True
    """Whether the optimization prompt respects override stores."""


class BasePromptOptimizer[InputT, OutputT](ABC):
    """Abstract base for prompt optimizers with shared utilities."""

    def __init__(
        self,
        context: OptimizationContext,
        *,
        config: OptimizerConfig | None = None,
    ) -> None:
        super().__init__()
        self._context = context
        self._config = config or OptimizerConfig()

    @abstractmethod
    def optimize(
        self,
        prompt: Prompt[InputT],
        *,
        session: SessionProtocol,
    ) -> OutputT:
        """Subclasses implement optimization logic here."""
        ...

    def _create_optimization_session(
        self, prompt: Prompt[InputT], *, session: SessionProtocol
    ) -> Session:
        """Create an isolated session for optimization evaluation.

        Reuses the context's optimization_session if provided; otherwise
        creates a fresh Session tagged for the optimization scope.
        """
        if self._context.optimization_session is not None:
            return self._context.optimization_session
        prompt_name = prompt.name or prompt.key
        return Session(
            tags={
                "scope": f"{self._optimizer_scope}_optimization",
                "prompt": prompt_name,
            },
            clock=session.clock,
        )

    @property
    @abstractmethod
    def _optimizer_scope(self) -> str:
        """Return a short identifier for session tagging."""
        ...


__all__ = ["BasePromptOptimizer", "OptimizerConfig"]

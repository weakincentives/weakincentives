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
    """Base configuration shared across optimizers.

    This configuration controls cross-cutting behavior for all optimizer
    implementations. Subclasses may extend this with algorithm-specific settings.

    Attributes:
        accepts_overrides: When True (default), the optimizer respects prompt
            override stores for caching and retrieving previous optimization
            results. Set to False to force fresh optimization on every call.

    Example:
        >>> config = OptimizerConfig(accepts_overrides=False)
        >>> optimizer = SomeOptimizer(context, config=config)
    """

    accepts_overrides: bool = True
    """Whether the optimization prompt respects override stores."""


class BasePromptOptimizer[InputT, OutputT](ABC):
    """Abstract base class for prompt optimization algorithms.

    Prompt optimizers transform or analyze prompts to improve performance,
    reduce token usage, or extract reusable artifacts (e.g., workspace digests).
    This base class provides shared infrastructure for session management,
    configuration handling, and override store integration.

    Type Parameters:
        InputT: The input type accepted by prompts this optimizer can process.
        OutputT: The result type returned by the optimize() method.

    Subclasses must implement:
        - optimize(): The core optimization logic
        - _optimizer_scope: A short identifier for session tagging

    Example:
        >>> class MyOptimizer(BasePromptOptimizer[str, MyResult]):
        ...     @property
        ...     def _optimizer_scope(self) -> str:
        ...         return "my_optimization"
        ...
        ...     def optimize(self, prompt, *, session):
        ...         # Implementation here
        ...         return MyResult(...)
    """

    def __init__(
        self,
        context: OptimizationContext,
        *,
        config: OptimizerConfig | None = None,
    ) -> None:
        """Initialize the optimizer with required dependencies.

        Args:
            context: The optimization context providing adapter, dispatcher,
                and optional deadline/override store configuration.
            config: Optional configuration overriding default optimizer behavior.
                If None, uses OptimizerConfig defaults.
        """
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
        """Execute the optimization algorithm on the given prompt.

        This method performs the core optimization work: analyzing the prompt,
        potentially invoking the provider adapter for evaluation, and producing
        an optimization artifact. Implementations may persist results to the
        session or override store depending on configuration.

        Args:
            prompt: The prompt to optimize. Must contain any sections required
                by the specific optimizer implementation.
            session: The session context for the optimization. Used for
                dispatching events and storing session-scoped results.

        Returns:
            The optimization result containing artifacts and metadata specific
            to the optimizer implementation.

        Raises:
            PromptEvaluationError: If required sections are missing or
                the optimization process fails.
        """
        ...

    def _create_optimization_session(self, prompt: Prompt[InputT]) -> Session:
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
            }
        )

    @property
    @abstractmethod
    def _optimizer_scope(self) -> str:
        """Return a short identifier for session tagging."""
        ...


__all__ = ["BasePromptOptimizer", "OptimizerConfig"]

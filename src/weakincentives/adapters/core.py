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

"""Core adapter interfaces shared across provider integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ..budget import Budget, BudgetTracker
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..errors import (
    PROMPT_EVALUATION_PHASE_BUDGET,
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
    PromptEvaluationPhase,
)
from ..prompt import Prompt
from ..runtime.session.protocols import SessionProtocol
from ..sandbox import LocalSandboxProvider, SandboxConfig, SandboxProvider

if TYPE_CHECKING:
    from ..runtime.run_context import RunContext
    from ..runtime.watchdog import Heartbeat
    from ..sandbox import Sandbox


@FrozenDataclass()
class PromptResponse[OutputT]:
    """Structured result emitted by an adapter evaluation."""

    prompt_name: str
    text: str | None
    output: OutputT | None


class ProviderAdapter(ABC):
    """Abstract base class describing the synchronous adapter contract.

    The base class owns the sandbox-lease fork: :meth:`evaluate` either
    borrows a sandbox supplied by the caller or opens one for the duration
    of the call via :meth:`open_sandbox`. Concrete adapters implement only
    :meth:`_evaluate`, which always receives an open sandbox and never
    manages its lifecycle.
    """

    _sandbox_provider: SandboxProvider | None = None

    def __init__(self, *, sandbox_provider: SandboxProvider | None = None) -> None:
        """Initialize the adapter.

        Args:
            sandbox_provider: Provider materializing the prompt's sandbox
                config. Defaults to :class:`LocalSandboxProvider`.
        """
        super().__init__()
        self._sandbox_provider = sandbox_provider

    @classmethod
    def __class_getitem__(cls, _: object) -> type[ProviderAdapter[Any]]:
        return cls

    @property
    def adapter_name(self) -> str:
        """Canonical name for this adapter instance.

        Default implementation returns the class name.  Concrete adapters
        should override this to return a stable, well-known identifier
        (e.g. ``CLAUDE_AGENT_SDK_ADAPTER_NAME``).
        """
        return type(self).__name__

    @contextmanager
    def open_sandbox[OutputT](self, prompt: Prompt[OutputT]) -> Generator[Sandbox]:
        """Open a sandbox lease for the prompt's declared environment.

        Materializes the prompt template's
        :class:`~weakincentives.sandbox.SandboxConfig` (an empty config when
        the template declares none) through this adapter's sandbox provider.
        The lease spans the ``with`` block: the sandbox is released —
        closed, for locally provisioned sandboxes — on exit.

        Use this to hold one environment across multiple :meth:`evaluate`
        calls (e.g. visibility-expansion retries) or to inspect the
        sandbox's filesystem before release::

            with adapter.open_sandbox(prompt) as sandbox:
                response = adapter.evaluate(
                    prompt, session=session, sandbox=sandbox
                )
                report = sandbox.filesystem.read("report.md")
        """
        provider = (
            self._sandbox_provider
            if self._sandbox_provider is not None
            else LocalSandboxProvider()
        )
        config = prompt.template.sandbox
        sandbox = provider.open(config if config is not None else SandboxConfig())
        try:
            yield sandbox
        finally:
            sandbox.close()

    def evaluate[OutputT](
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
        sandbox: Sandbox | None = None,
    ) -> PromptResponse[OutputT]:
        """Evaluate the prompt and return a structured response.

        When ``sandbox`` is provided it is **borrowed**: the caller holds
        the lease and the adapter never closes it. When omitted, the
        adapter opens a sandbox via :meth:`open_sandbox` for the duration
        of this call. Either way :meth:`_evaluate` runs against exactly one
        open sandbox and the harness working directory is its root.

        Visibility overrides are managed exclusively via Session state using the
        VisibilityOverrides state slice. Use session[VisibilityOverrides]
        to set visibility overrides before calling evaluate().

        When ``budget`` is provided and ``budget_tracker`` is not, a new tracker
        is created. When ``budget_tracker`` is supplied, it is used directly for
        shared limit enforcement.

        When ``heartbeat`` is provided, the adapter will beat at key execution
        points (LLM calls, tool execution boundaries) to prove liveness. Tool
        handlers receive the heartbeat via ToolContext.beat() for additional
        beats during long-running operations.

        When ``run_context`` is provided, it is threaded through telemetry events
        (PromptRendered, PromptExecuted, ToolInvoked) for distributed tracing.
        """
        if sandbox is not None:
            return self._evaluate(
                prompt,
                session=session,
                deadline=deadline,
                budget=budget,
                budget_tracker=budget_tracker,
                heartbeat=heartbeat,
                run_context=run_context,
                sandbox=sandbox,
            )
        with self.open_sandbox(prompt) as owned:
            return self._evaluate(
                prompt,
                session=session,
                deadline=deadline,
                budget=budget,
                budget_tracker=budget_tracker,
                heartbeat=heartbeat,
                run_context=run_context,
                sandbox=owned,
            )

    @abstractmethod
    def _evaluate[OutputT](
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
        sandbox: Sandbox,
    ) -> PromptResponse[OutputT]:
        """Evaluate against an open sandbox (the explicit core contract).

        Implementations must treat ``sandbox`` as borrowed: use its facets,
        run the harness with ``cwd = sandbox.root``, and never close it —
        the lease is owned by :meth:`evaluate` or by the caller.
        """

        ...


__all__ = [
    "PROMPT_EVALUATION_PHASE_BUDGET",
    "PROMPT_EVALUATION_PHASE_REQUEST",
    "PROMPT_EVALUATION_PHASE_RESPONSE",
    "PROMPT_EVALUATION_PHASE_TOOL",
    "Budget",
    "BudgetTracker",
    "PromptEvaluationError",
    "PromptEvaluationPhase",
    "PromptResponse",
    "ProviderAdapter",
]

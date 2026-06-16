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

"""Core adapter interfaces shared across provider integrations.

The contract has three pieces, all in this module:

- :class:`ProviderAdapter` — the harness translator. Concrete adapters
  implement only :meth:`ProviderAdapter._evaluate`, which always receives
  an open sandbox and never manages its lifecycle.
- :class:`Runtime` — a prompt bound to its adapter and a live
  sandbox for one run. The runtime is the **only** way a sandbox reaches
  evaluation, so a mismatched (adapter, prompt, sandbox) triple is
  unrepresentable: the triple is paired exactly once, inside
  :meth:`ProviderAdapter.runtime`.
- :meth:`ProviderAdapter.evaluate` — one-shot sugar that opens a runtime
  for the duration of a single call.
"""

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
    WinkError,
)
from ..prompt import Prompt
from ..runtime.session.protocols import SessionProtocol
from ..sandbox import LocalSandboxProvider, SandboxProvider, WorkspaceConfig

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


class RuntimeReleasedError(WinkError, RuntimeError):
    """Raised when evaluating through a released :class:`Runtime`."""


class Runtime[OutputT]:
    """A prompt bound to its adapter and a live sandbox for one run.

    Construct via :meth:`ProviderAdapter.runtime` — the single place the
    (adapter, prompt, sandbox) triple is paired. The runtime holds the
    sandbox lease for its lifetime: every :meth:`evaluate` call runs
    against the same environment, so files written in one round are
    visible to the next (e.g. across visibility-expansion retries).

    The sandbox is exposed read-only for inspection and evidence capture
    while the lease is held::

        with adapter.runtime(prompt) as rt:
            response = rt.evaluate(session=session)
            report = rt.sandbox.filesystem.read("report.md")
        # lease released; the runtime can no longer evaluate

    Not thread-safe: use one runtime per request.
    """

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[OutputT],
        prompt: Prompt[OutputT],
        sandbox: Sandbox,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._prompt = prompt
        self._sandbox = sandbox
        self._released = False

    @property
    def adapter(self) -> ProviderAdapter[OutputT]:
        """The adapter this runtime is bound to (telemetry/naming only)."""
        return self._adapter

    @property
    def prompt(self) -> Prompt[OutputT]:
        """The prompt this runtime is bound to."""
        return self._prompt

    @property
    def sandbox(self) -> Sandbox:
        """The live sandbox; valid until the runtime is released."""
        return self._sandbox

    @property
    def released(self) -> bool:
        """True once the sandbox lease has been released."""
        return self._released

    def evaluate(
        self,
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
    ) -> PromptResponse[OutputT]:
        """Evaluate the bound prompt against the bound sandbox.

        Owns the evaluation preamble shared by every adapter: promotes
        ``budget`` to a tracker, resolves the effective deadline, and
        fails fast when the deadline has already expired. The adapter's
        :meth:`ProviderAdapter._evaluate` receives the resolved deadline
        and tracker only.

        Raises:
            RuntimeReleasedError: The runtime's lease was released.
            PromptEvaluationError: The effective deadline already expired.
        """
        if self._released:
            msg = "Runtime has been released; open a new one via adapter.runtime()"
            raise RuntimeReleasedError(msg)

        if budget is not None and budget_tracker is None:
            budget_tracker = BudgetTracker(budget)
        effective_deadline = deadline or (budget.deadline if budget else None)
        if (
            effective_deadline is not None
            and effective_deadline.remaining().total_seconds() <= 0
        ):
            prompt_name = self._prompt.name or f"{self._prompt.ns}:{self._prompt.key}"
            raise PromptEvaluationError(
                message="Deadline expired before evaluation",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            )

        return self._adapter._evaluate(  # pyright: ignore[reportPrivateUsage] - paired in this module
            self._prompt,
            session=session,
            deadline=effective_deadline,
            budget_tracker=budget_tracker,
            heartbeat=heartbeat,
            run_context=run_context,
            sandbox=self._sandbox,
        )

    def _release(self) -> None:
        """Release the sandbox lease. Idempotent; called by the opener."""
        if self._released:
            return
        self._released = True
        self._sandbox.close()


class ProviderAdapter(ABC):
    """Abstract base class describing the synchronous adapter contract.

    Concrete adapters implement only :meth:`_evaluate`; the base class
    owns runtime construction (:meth:`runtime`) and the one-shot
    :meth:`evaluate` sugar. The adapter also owns its sandbox provider:
    the adapter is the authority on which environments its harness can
    attach to, so adapter↔provider coherence lives here while
    adapter↔prompt↔sandbox coherence lives in :class:`Runtime`.
    """

    _sandbox_provider: SandboxProvider | None = None

    def __init__(self, *, sandbox_provider: SandboxProvider | None = None) -> None:
        """Initialize the adapter.

        Args:
            sandbox_provider: Provider materializing the prompt's workspace
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
    def runtime[OutputT](self, prompt: Prompt[OutputT]) -> Generator[Runtime[OutputT]]:
        """Open an :class:`Runtime` for the prompt.

        Materializes the prompt template's
        :class:`~weakincentives.sandbox.WorkspaceConfig` (an empty config
        when the template declares none) through this adapter's sandbox
        provider and pairs it with the prompt for the lifetime of the
        ``with`` block. The lease is released on exit: locally provisioned
        sandboxes are closed and removed.

        Hold a runtime to span one environment across multiple
        evaluations (e.g. visibility-expansion retries) or to inspect the
        sandbox's filesystem before release.
        """
        provider = (
            self._sandbox_provider
            if self._sandbox_provider is not None
            else LocalSandboxProvider()
        )
        config = prompt.template.workspace
        sandbox = provider.open(config if config is not None else WorkspaceConfig())
        rt: Runtime[OutputT] = Runtime(adapter=self, prompt=prompt, sandbox=sandbox)
        try:
            yield rt
        finally:
            rt._release()  # pyright: ignore[reportPrivateUsage] - opener releases

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
    ) -> PromptResponse[OutputT]:
        """Evaluate the prompt in a runtime scoped to this call.

        One-shot sugar over :meth:`runtime`: opens an
        :class:`Runtime`, evaluates once, and releases the sandbox
        lease before returning. To span one environment across multiple
        evaluations, open the runtime explicitly.

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
        with self.runtime(prompt) as rt:
            return rt.evaluate(
                session=session,
                deadline=deadline,
                budget=budget,
                budget_tracker=budget_tracker,
                heartbeat=heartbeat,
                run_context=run_context,
            )

    @abstractmethod
    def _evaluate[OutputT](
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
        sandbox: Sandbox,
    ) -> PromptResponse[OutputT]:
        """Evaluate against an open sandbox (the explicit core contract).

        Called only through :class:`Runtime`, which guarantees the
        sandbox was materialized by this adapter's provider from this
        prompt's config and has already resolved the effective deadline
        and budget tracker. Implementations render with
        ``prompt.render(session=session, sandbox=sandbox)`` so the
        workspace preview reflects the open environment, use the
        sandbox's facets, run the harness with ``cwd = sandbox.root``,
        and never close it.
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
    "Runtime",
    "RuntimeReleasedError",
]

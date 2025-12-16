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

"""Evaluation loop orchestration for dataset evaluation.

EvalLoop mirrors MainLoop's event-driven architecture, enabling dataset
evaluations to be driven via an EventBus and deployed as workers in a cluster.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from ..dataclasses import FrozenDataclass
from ._types import EvalReport, EvalResult, Sample, SampleEvaluated, Score

if TYPE_CHECKING:
    from ..runtime.events._types import ControlBus
    from ..runtime.main_loop import MainLoop


@FrozenDataclass()
class EvalLoopConfig:
    """Configuration for EvalLoop execution defaults.

    Request-level settings override config defaults.
    """

    pass  # Future: max_concurrency, timeout_per_sample, retry_count


@FrozenDataclass()
class EvalLoopRequest[InputT, ExpectedT]:
    """Event requesting EvalLoop execution.

    Note: ``InProcessEventBus`` dispatches by ``type(event)``, not generic alias.
    ``EvalLoopRequest[I, E]`` is for static type checking; at runtime all events
    are ``EvalLoopRequest``. For multiple eval loops on one bus, filter by dataset
    or use separate buses.
    """

    dataset: tuple[Sample[InputT, ExpectedT], ...]
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@FrozenDataclass()
class EvalLoopCompleted:
    """Event published when EvalLoop execution succeeds."""

    request_id: UUID
    report: EvalReport
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@FrozenDataclass()
class EvalLoopFailed:
    """Event published when EvalLoop execution fails."""

    request_id: UUID
    error: Exception
    partial_report: EvalReport | None
    failed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class EvalLoop[InputT, OutputT, ExpectedT]:
    """Event-driven evaluation loop orchestrator.

    EvalLoop standardizes dataset evaluation: receive request, iterate samples,
    execute via MainLoop, score with evaluator, publish progress and results.
    It mirrors MainLoop's architecture for consistent event-driven workflows.

    Execution flow:
        1. Receive ``EvalLoopRequest`` via bus or direct ``execute()`` call
        2. For each sample:
           a. Execute via ``MainLoop.execute()``
           b. Score output with evaluator
           c. Record latency and errors
           d. Publish ``SampleEvaluated`` event
        3. Publish ``EvalLoopCompleted`` or ``EvalLoopFailed``

    Usage::

        class QALoop(MainLoop[str, str]):
            ...

        qa_loop = QALoop(adapter=adapter, bus=control_bus)

        eval_loop = EvalLoop(
            loop=qa_loop,
            evaluator=exact_match,
            bus=control_bus,
        )

        # Bus-driven usage
        bus.subscribe(EvalLoopRequest, eval_loop.handle_request)
        bus.publish(EvalLoopRequest(dataset=dataset))

        # Direct usage
        report = eval_loop.execute(dataset)
    """

    _loop: MainLoop[InputT, OutputT]
    _evaluator: Callable[[OutputT, ExpectedT], Score]
    _bus: ControlBus
    _config: EvalLoopConfig

    def __init__(
        self,
        *,
        loop: MainLoop[InputT, OutputT],
        evaluator: Callable[[OutputT, ExpectedT], Score],
        bus: ControlBus,
        config: EvalLoopConfig | None = None,
    ) -> None:
        """Initialize the EvalLoop with a MainLoop, evaluator, and bus.

        Args:
            loop: MainLoop instance for executing samples.
            evaluator: Scoring function for (output, expected) -> Score.
            bus: Control bus for request/response event routing.
            config: Optional configuration for evaluation defaults.
        """
        super().__init__()
        self._loop = loop
        self._evaluator = evaluator
        self._bus = bus
        self._config = config if config is not None else EvalLoopConfig()
        bus.subscribe(EvalLoopRequest, self.handle_request)

    def execute(
        self,
        dataset: tuple[Sample[InputT, ExpectedT], ...],
    ) -> EvalReport:
        """Execute evaluation for a dataset.

        Iterates through samples, executes each via MainLoop, scores outputs,
        and publishes ``SampleEvaluated`` events for progress tracking.

        Args:
            dataset: Tuple of samples to evaluate.

        Returns:
            EvalReport with all results and aggregate metrics.

        Raises:
            Any exception from MainLoop execution or evaluator (captured in
            EvalResult.error for individual samples; only fatal errors propagate).
        """
        results: list[EvalResult] = []

        for sample in dataset:
            start = time.monotonic()
            try:
                response, _ = self._loop.execute(sample.input)
                latency_ms = int((time.monotonic() - start) * 1000)
                output = response.output
                if output is None:
                    result = EvalResult(
                        sample_id=sample.id,
                        score=Score(
                            value=0.0, passed=False, reason="No output from loop"
                        ),
                        latency_ms=latency_ms,
                        error="No output from loop",
                    )
                else:
                    score = self._evaluator(output, sample.expected)
                    result = EvalResult(
                        sample_id=sample.id,
                        score=score,
                        latency_ms=latency_ms,
                    )
            except Exception as e:
                latency_ms = int((time.monotonic() - start) * 1000)
                result = EvalResult(
                    sample_id=sample.id,
                    score=Score(value=0.0, passed=False, reason=str(e)),
                    latency_ms=latency_ms,
                    error=str(e),
                )

            results.append(result)
            _ = self._bus.publish(SampleEvaluated(sample_id=sample.id, result=result))

        return EvalReport(results=tuple(results))

    def handle_request(self, event: object) -> None:
        """Handle an EvalLoopRequest event from the bus.

        This method is designed to be subscribed to the event bus::

            bus.subscribe(EvalLoopRequest, eval_loop.handle_request)

        On success, publishes ``EvalLoopCompleted``. On failure, publishes
        ``EvalLoopFailed`` and re-raises the exception.

        Args:
            event: An ``EvalLoopRequest`` instance (type is ``object`` for
                compatibility with ``EventHandler`` signature).
        """
        request_event: EvalLoopRequest[InputT, ExpectedT] = event  # type: ignore[assignment]

        try:
            report = self.execute(request_event.dataset)

            completed = EvalLoopCompleted(
                request_id=request_event.request_id,
                report=report,
            )
            _ = self._bus.publish(completed)

        except Exception as exc:
            failed = EvalLoopFailed(
                request_id=request_event.request_id,
                error=exc,
                partial_report=None,
            )
            _ = self._bus.publish(failed)
            raise


__all__ = [
    "EvalLoop",
    "EvalLoopCompleted",
    "EvalLoopConfig",
    "EvalLoopFailed",
    "EvalLoopRequest",
]

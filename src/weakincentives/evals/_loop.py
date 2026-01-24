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

"""Mailbox-driven evaluation loop.

EvalLoop orchestrates evaluation: for each sample, it executes through the
provided MainLoop, scores the output, and aggregates results into a report.
"""

from __future__ import annotations

import contextlib
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, override

from ..dataclasses import FrozenDataclass
from ..runtime.dlq import DeadLetter, DLQPolicy
from ..runtime.lease_extender import LeaseExtenderConfig
from ..runtime.mailbox import (
    Mailbox,
    Message,
    ReceiptHandleExpiredError,
    ReplyNotAvailableError,
)
from ..runtime.mailbox_worker import MailboxWorker
from ._evaluators import is_session_aware
from ._types import EvalRequest, EvalResult, Evaluator, Score, SessionEvaluator

if TYPE_CHECKING:
    from ..runtime import MainLoop

_logger = logging.getLogger(__name__)


@FrozenDataclass()
class EvalLoopConfig:
    """Configuration for EvalLoop execution defaults.

    The ``lease_extender`` field controls automatic message visibility extension
    during evaluation processing. When enabled, heartbeats from tool execution
    and after each sample extend the message lease, preventing timeout during
    long evaluation runs. EvalLoop's heartbeat is passed to MainLoop.execute()
    so that all tool/adapter beats extend the evaluation message's lease.

    The ``debug_bundle_dir`` field enables debug bundle creation for each
    evaluation sample. When set, EvalLoop creates a bundle capturing:
    - Request input (sample and experiment)
    - Response output from MainLoop
    - Session state after execution
    - Application logs during execution
    - Evaluation metadata (score, experiment, latency)
    - Environment information

    Bundles are written to ``{debug_bundle_dir}/{request_id}/{sample_id}_{timestamp}.zip``.
    """

    lease_extender: LeaseExtenderConfig | None = None
    debug_bundle_dir: Path | None = None


class EvalLoop[InputT, OutputT, ExpectedT](
    MailboxWorker[EvalRequest[InputT, ExpectedT], EvalResult]
):
    """Mailbox-driven evaluation loop.

    Receives EvalRequest messages, executes through MainLoop, scores
    with evaluator, and sends EvalResult via Message.reply(). Designed
    to run alongside MainLoop workers in distributed deployments.

    Supports both standard and session-aware evaluators. Session-aware
    evaluators receive a SessionViewProtocol for behavioral assertions.

    The single mailbox pattern with reply_to routing:
    - requests mailbox: receives EvalRequest messages with reply_to
    - Workers use msg.reply(result) to route responses

    Lifecycle:
        - Use ``run()`` to start processing messages
        - Call ``shutdown()`` to request graceful stop
        - In-flight samples complete before exit
        - Supports context manager protocol for automatic cleanup

    Example:
        >>> eval_loop = EvalLoop(
        ...     loop=main_loop,
        ...     evaluator=exact_match,
        ...     requests=requests_mailbox,
        ... )
        >>> eval_loop.run(max_iterations=1)
    """

    _loop: MainLoop[InputT, OutputT]
    _evaluator: Evaluator | SessionEvaluator
    _config: EvalLoopConfig
    _dlq: DLQPolicy[EvalRequest[InputT, ExpectedT], EvalResult] | None

    def __init__(
        self,
        *,
        loop: MainLoop[InputT, OutputT],
        evaluator: Evaluator | SessionEvaluator,
        requests: Mailbox[EvalRequest[InputT, ExpectedT], EvalResult],
        config: EvalLoopConfig | None = None,
        dlq: DLQPolicy[EvalRequest[InputT, ExpectedT], EvalResult] | None = None,
    ) -> None:
        """Initialize the EvalLoop.

        Args:
            loop: MainLoop instance for executing samples.
            evaluator: Scoring function. Can be either:
                - Standard: (output, expected) -> Score
                - Session-aware: (output, expected, session) -> Score
            requests: Mailbox to receive EvalRequest messages from.
                Response routing derives from each message's reply_to field.
            config: Optional configuration for evaluation defaults.
            dlq: Optional dead letter queue policy. When configured, messages
                that fail repeatedly are sent to the DLQ mailbox instead of
                retrying indefinitely.
        """
        effective_config = config if config is not None else EvalLoopConfig()
        super().__init__(
            requests=requests,
            lease_extender_config=effective_config.lease_extender,
        )
        self._loop = loop
        self._evaluator = evaluator
        self._config = effective_config
        self._dlq = dlq

    @override
    def _process_message(
        self, msg: Message[EvalRequest[InputT, ExpectedT], EvalResult]
    ) -> None:
        """Process a single evaluation request message.

        Implements the abstract method from MailboxWorker. Called with lease
        extension already attached by the base class.
        """
        try:
            result = self._evaluate_sample(msg.body)
        except Exception as e:
            self._handle_failure(msg, e)
        else:
            self._reply_and_ack(msg, result)

    def _reply_and_ack(
        self,
        msg: Message[EvalRequest[InputT, ExpectedT], EvalResult],
        result: EvalResult,
    ) -> None:
        """Reply with result and acknowledge message, handling failures gracefully.

        Uses Message.reply() for response routing based on reply_to.
        On reply failure, nack for retry instead of losing successful evaluations.
        """
        _ = self  # Uses self implicitly via Message callbacks
        try:
            _ = msg.reply(result)
            msg.acknowledge()
        except ReplyNotAvailableError:
            # No reply_to specified - log and acknowledge without reply
            _logger.warning(
                "No reply_to for message %s, acknowledging without reply", msg.id
            )
            with contextlib.suppress(ReceiptHandleExpiredError):
                msg.acknowledge()
        except ReceiptHandleExpiredError:
            # Handle expired during processing - message already requeued.
            pass
        except Exception:
            # Reply send failed - nack so message is retried
            with contextlib.suppress(ReceiptHandleExpiredError):
                backoff = min(60 * msg.delivery_count, 900)
                msg.nack(visibility_timeout=backoff)

    def _evaluate_sample(self, request: EvalRequest[InputT, ExpectedT]) -> EvalResult:
        """Execute and score a single sample under an experiment.

        Passes EvalLoop's heartbeat to MainLoop.execute() so that tool execution
        beats extend the evaluation message's lease. Also beats after sample
        execution completes to prove progress between samples.

        The experiment from the request is passed to MainLoop.execute() for
        prompt override resolution and feature flag checking.

        When debug_bundle_dir is configured, creates a debug bundle capturing:
        - Request input (sample and experiment)
        - Response output from MainLoop
        - Session state after execution
        - Application logs during execution
        - Evaluation metadata (score, experiment, latency)
        - Environment information
        """
        if self._config.debug_bundle_dir is not None:
            return self._evaluate_sample_with_bundle(request)
        return self._evaluate_sample_without_bundle(request)

    def _evaluate_sample_without_bundle(
        self, request: EvalRequest[InputT, ExpectedT]
    ) -> EvalResult:
        """Execute and score a sample without debug bundling."""
        sample = request.sample
        experiment = request.experiment
        start = time.monotonic()

        # Pass our heartbeat and experiment to MainLoop
        response, session = self._loop.execute(
            sample.input,
            heartbeat=self._heartbeat,
            experiment=experiment,
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        # Beat after sample execution to prove progress
        self._heartbeat.beat()

        if response.output is None:
            return EvalResult(
                sample_id=sample.id,
                experiment_name=experiment.name,
                score=Score(value=0.0, passed=False, reason="No output from MainLoop"),
                latency_ms=latency_ms,
                error="No output from MainLoop",
            )

        # Invoke evaluator with session if session-aware
        # Type ignore needed: is_session_aware narrows the type at runtime
        if is_session_aware(self._evaluator):
            score = self._evaluator(response.output, sample.expected, session)  # type: ignore[call-arg]
        else:
            score = self._evaluator(response.output, sample.expected)  # type: ignore[call-arg]

        return EvalResult(
            sample_id=sample.id,
            experiment_name=experiment.name,
            score=score,  # pyright: ignore[reportUnknownArgumentType]
            latency_ms=latency_ms,
        )

    def _evaluate_sample_with_bundle(
        self, request: EvalRequest[InputT, ExpectedT]
    ) -> EvalResult:
        """Execute and score a sample with debug bundling enabled.

        Uses MainLoop.execute_with_bundle() to reuse the standard bundle
        creation logic, then adds eval-specific metadata before finalization.

        Exception handling is narrowed to avoid re-executing samples:
        - If bundle setup/execution fails before ctx is created, fall back to
          non-bundled execution
        - If execution succeeds but scoring or bundle writing fails, use the
          results from the successful execution (never re-execute)
        """
        if self._config.debug_bundle_dir is None:  # pragma: no cover - defensive
            return self._evaluate_sample_without_bundle(request)

        # Create per-request directory: {debug_bundle_dir}/{request_id}/
        bundle_target = self._config.debug_bundle_dir / str(request.request_id)

        # Track execution state to avoid re-running on post-execution failures
        ctx = None
        score: Score | None = None
        error: str | None = None

        try:
            with self._loop.execute_with_bundle(
                request.sample.input,
                bundle_target=bundle_target,
                heartbeat=self._heartbeat,
                experiment=request.experiment,
            ) as ctx:
                # Compute score first, then beat to prove progress
                score, error = self._compute_score(
                    ctx.response.output, request.sample.expected, ctx.session
                )

                # Beat after scoring to prove progress
                self._heartbeat.beat()

                # Write eval metadata to bundle
                ctx.write_metadata(
                    "eval", self._build_eval_info(request, score, ctx.latency_ms, error)
                )

            # Normal exit: bundle finalized, retrieve path
            return EvalResult(
                sample_id=request.sample.id,
                experiment_name=request.experiment.name,
                score=score,
                latency_ms=ctx.latency_ms,
                error=error,
                bundle_path=ctx.bundle_path,
            )

        except Exception as exc:
            # Check if execution completed (ctx was created and has results)
            if ctx is not None:
                # Execution succeeded but something failed (scoring, write_metadata,
                # or finalization). Use the results we have - never re-execute.
                _logger.warning(
                    "Bundle creation failed after successful execution",
                    extra={"error": str(exc), "sample_id": request.sample.id},
                )
                # If score wasn't computed before the exception, compute it now.
                # This only happens if _compute_score itself raised, which would
                # likely fail again - but we try anyway as a last resort.
                if score is None:  # pragma: no cover - defensive: evaluator failure
                    try:
                        score, error = self._compute_score(
                            ctx.response.output, request.sample.expected, ctx.session
                        )
                    except Exception as eval_exc:
                        # Evaluator failed twice - provide fallback score
                        score = Score(
                            value=0.0,
                            passed=False,
                            reason=f"Evaluator failed: {eval_exc}",
                        )
                        error = str(eval_exc)
                return EvalResult(
                    sample_id=request.sample.id,
                    experiment_name=request.experiment.name,
                    score=score,
                    latency_ms=ctx.latency_ms,
                    error=error,
                    bundle_path=ctx.bundle_path,  # May be None if finalization failed
                )

            # Execution itself failed - fall back to non-bundled
            _logger.warning(
                "Bundle execution failed, retrying without bundle",
                extra={"error": str(exc), "sample_id": request.sample.id},
            )
            return self._evaluate_sample_without_bundle(request)

    def _compute_score(
        self,
        output: OutputT | None,
        expected: ExpectedT,
        session: object,
    ) -> tuple[Score, str | None]:
        """Compute score for an evaluation result.

        Returns:
            Tuple of (score, error_message). Error is None on success.
        """
        if output is None:
            return (
                Score(value=0.0, passed=False, reason="No output from MainLoop"),
                "No output from MainLoop",
            )

        if is_session_aware(self._evaluator):
            score = self._evaluator(output, expected, session)  # type: ignore[call-arg]
        else:
            score = self._evaluator(output, expected)  # type: ignore[call-arg]
        return score, None  # pyright: ignore[reportUnknownVariableType]

    @staticmethod
    def _build_eval_info(
        request: EvalRequest[object, object],
        score: Score,
        latency_ms: int,
        error: str | None,
    ) -> dict[str, object]:
        """Build eval.json metadata dictionary."""
        eval_info: dict[str, object] = {
            "sample_id": request.sample.id,
            "experiment_name": request.experiment.name,
            "score": {
                "value": score.value,
                "passed": score.passed,
                "reason": score.reason,
            },
            "latency_ms": latency_ms,
        }
        if error is not None:
            eval_info["error"] = error
        return eval_info

    def _handle_failure(
        self,
        msg: Message[EvalRequest[InputT, ExpectedT], EvalResult],
        error: Exception,
    ) -> None:
        """Handle message processing failure.

        When DLQ is configured:
        - Checks DLQ policy and either dead-letters or retries with backoff
        - Only sends error replies on terminal outcomes (DLQ)

        When DLQ is not configured:
        - Sends error reply and acknowledges (original behavior)
        """
        if self._dlq is None:
            # No DLQ configured - use original behavior: error reply + acknowledge
            result = EvalResult(
                sample_id=msg.body.sample.id,
                experiment_name=msg.body.experiment.name,
                score=Score(value=0.0, passed=False, reason=str(error)),
                latency_ms=0,
                error=str(error),
            )
            self._reply_and_ack(msg, result)
            return

        # DLQ is configured - check if we should dead-letter
        if self._dlq.should_dead_letter(msg, error):
            self._dead_letter(msg, error)
            return

        # Retry with backoff - do NOT send error reply here.
        # The message will be redelivered and may succeed on retry.
        # Only send error replies on terminal outcomes (DLQ or final failure).
        backoff = min(60 * msg.delivery_count, 900)
        with contextlib.suppress(ReceiptHandleExpiredError):
            msg.nack(visibility_timeout=backoff)

    def _dead_letter(
        self,
        msg: Message[EvalRequest[InputT, ExpectedT], EvalResult],
        error: Exception,
    ) -> None:
        """Send message to dead letter queue."""
        if self._dlq is None:  # pragma: no cover - defensive check
            return

        dead_letter: DeadLetter[EvalRequest[InputT, ExpectedT]] = DeadLetter(
            message_id=msg.id,
            body=msg.body,
            source_mailbox=self._requests.name,
            delivery_count=msg.delivery_count,
            last_error=str(error),
            last_error_type=f"{type(error).__module__}.{type(error).__qualname__}",
            dead_lettered_at=datetime.now(UTC),
            first_received_at=msg.enqueued_at,
            request_id=msg.body.request_id,
            reply_to=msg.reply_to.name if msg.reply_to else None,
        )

        # Send error reply - this is a terminal outcome
        try:
            if msg.reply_to:
                _ = msg.reply(
                    EvalResult(
                        sample_id=msg.body.sample.id,
                        experiment_name=msg.body.experiment.name,
                        score=Score(value=0.0, passed=False, reason=str(error)),
                        latency_ms=0,
                        error=f"Dead-lettered after {msg.delivery_count} attempts: {error}",
                    )
                )
        except Exception as reply_error:  # nosec B110 - intentional: reply failure should not block dead-lettering
            _logger.debug(
                "Failed to send error reply during dead-lettering",
                extra={"message_id": msg.id, "error": str(reply_error)},
            )

        _ = self._dlq.mailbox.send(dead_letter)
        _logger.warning(
            "Message dead-lettered",
            extra={
                "message_id": msg.id,
                "request_id": str(msg.body.request_id),
                "sample_id": msg.body.sample.id,
                "delivery_count": msg.delivery_count,
                "error_type": dead_letter.last_error_type,
            },
        )

        # Acknowledge to remove from source queue
        with contextlib.suppress(ReceiptHandleExpiredError):
            msg.acknowledge()


__all__ = [
    "EvalLoop",
    "EvalLoopConfig",
]

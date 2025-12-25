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
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Self

from ..runtime.lifecycle import wait_until
from ..runtime.mailbox import Mailbox, Message, ReceiptHandleExpiredError
from ._types import EvalRequest, EvalResult, Score

if TYPE_CHECKING:
    from ..runtime import MainLoop


class EvalLoop[InputT, OutputT, ExpectedT]:
    """Mailbox-driven evaluation loop.

    Receives EvalRequest messages, executes through MainLoop, scores
    with evaluator, and sends EvalResult to results mailbox. Designed
    to run alongside MainLoop workers in distributed deployments.

    The two-mailbox pattern matches MainLoop for distributed deployments:
    - requests mailbox: receives EvalRequest messages
    - results mailbox: sends EvalResult messages

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
        ...     results=results_mailbox,
        ... )
        >>> eval_loop.run(max_iterations=1)
    """

    _loop: MainLoop[InputT, OutputT]
    _evaluator: Callable[[OutputT, ExpectedT], Score]
    _requests: Mailbox[EvalRequest[InputT, ExpectedT]]
    _results: Mailbox[EvalResult]
    _shutdown_event: threading.Event
    _running: bool
    _lock: threading.Lock

    def __init__(
        self,
        *,
        loop: MainLoop[InputT, OutputT],
        evaluator: Callable[[OutputT, ExpectedT], Score],
        requests: Mailbox[EvalRequest[InputT, ExpectedT]],
        results: Mailbox[EvalResult],
    ) -> None:
        """Initialize the EvalLoop.

        Args:
            loop: MainLoop instance for executing samples.
            evaluator: Scoring function (output, expected) -> Score.
            requests: Mailbox to receive EvalRequest messages from.
            results: Mailbox to send EvalResult messages to.
        """
        super().__init__()
        self._loop = loop
        self._evaluator = evaluator
        self._requests = requests
        self._results = results
        self._shutdown_event = threading.Event()
        self._running = False
        self._lock = threading.Lock()

    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> None:
        """Process evaluation requests from mailbox.

        Polls the requests mailbox, evaluates each sample through
        MainLoop, and sends results to the results mailbox.

        The loop exits when:
        - max_iterations is reached
        - shutdown() is called
        - The requests mailbox is closed

        In-flight samples complete before exit. Unprocessed messages from
        the current batch are nacked for redelivery.

        Args:
            max_iterations: Stop after N iterations (None = run forever).
            visibility_timeout: Seconds messages remain invisible during
                processing. Must exceed maximum expected execution time.
            wait_time_seconds: Long poll duration (0-20 seconds).
        """
        with self._lock:
            self._running = True
            self._shutdown_event.clear()

        iterations = 0
        try:
            while max_iterations is None or iterations < max_iterations:
                # Check shutdown before blocking on receive
                if self._shutdown_event.is_set():
                    break

                # Exit if mailbox closed
                if self._requests.closed:
                    break

                for msg in self._requests.receive(
                    visibility_timeout=visibility_timeout,
                    wait_time_seconds=wait_time_seconds,
                ):
                    # Check shutdown between messages
                    if self._shutdown_event.is_set():
                        # Nack unprocessed message for redelivery
                        with contextlib.suppress(ReceiptHandleExpiredError):
                            msg.nack(visibility_timeout=0)
                        break

                    try:
                        result = self._evaluate_sample(msg.body)
                    except Exception as e:
                        result = EvalResult(
                            sample_id=msg.body.sample.id,
                            score=Score(value=0.0, passed=False, reason=str(e)),
                            latency_ms=0,
                            error=str(e),
                        )
                    self._send_and_ack(msg, result)
                iterations += 1
        finally:
            with self._lock:
                self._running = False

    def shutdown(self, *, timeout: float = 30.0) -> bool:
        """Request graceful shutdown and wait for completion.

        Sets the shutdown flag. If the loop is running, waits up to timeout
        seconds for it to stop.

        Args:
            timeout: Maximum seconds to wait for the loop to stop.

        Returns:
            True if loop stopped cleanly, False if timeout expired.
        """
        self._shutdown_event.set()
        return wait_until(lambda: not self.running, timeout=timeout)

    @property
    def running(self) -> bool:
        """True if the loop is currently processing messages."""
        with self._lock:
            return self._running

    def __enter__(self) -> Self:
        """Context manager entry. Returns self for use in with statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit. Triggers shutdown and waits for completion."""
        _ = (exc_type, exc_val, exc_tb)
        _ = self.shutdown()

    def _send_and_ack(
        self,
        msg: Message[EvalRequest[InputT, ExpectedT]],
        result: EvalResult,
    ) -> None:
        """Send result and acknowledge message, handling failures gracefully.

        Mirrors MainLoop._send_and_ack: on send failure, nack for retry instead
        of fabricating an error result that would lose successful evaluations.
        """
        try:
            _ = self._results.send(result)
            msg.acknowledge()
        except ReceiptHandleExpiredError:
            # Handle expired during processing - message already requeued.
            pass
        except Exception:
            # Response send failed - nack so message is retried
            try:
                backoff = min(60 * msg.delivery_count, 900)
                msg.nack(visibility_timeout=backoff)
            except ReceiptHandleExpiredError:
                # Handle expired - message already requeued
                pass

    def _evaluate_sample(self, request: EvalRequest[InputT, ExpectedT]) -> EvalResult:
        """Execute and score a single sample."""
        sample = request.sample
        start = time.monotonic()

        response, _ = self._loop.execute(sample.input)
        latency_ms = int((time.monotonic() - start) * 1000)

        if response.output is None:
            return EvalResult(
                sample_id=sample.id,
                score=Score(value=0.0, passed=False, reason="No output from MainLoop"),
                latency_ms=latency_ms,
                error="No output from MainLoop",
            )

        score = self._evaluator(response.output, sample.expected)

        return EvalResult(
            sample_id=sample.id,
            score=score,
            latency_ms=latency_ms,
        )


__all__ = [
    "EvalLoop",
]

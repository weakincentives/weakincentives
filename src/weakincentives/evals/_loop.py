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

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

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

    def run(self, *, max_iterations: int | None = None) -> None:
        """Process evaluation requests from mailbox.

        Polls the requests mailbox, evaluates each sample through
        MainLoop, and sends results to the results mailbox.

        The loop exits when:
        - max_iterations is reached
        - The requests mailbox is closed

        Args:
            max_iterations: Stop after N iterations (None = run forever).
        """
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            # Exit if mailbox closed
            if self._requests.closed:
                break

            for msg in self._requests.receive(
                visibility_timeout=300,  # 5 min - must exceed max execution time
                wait_time_seconds=20,  # Long poll for efficiency
            ):
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

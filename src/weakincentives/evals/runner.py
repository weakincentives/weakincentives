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

"""Evaluation runner."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from ._types import EvalReport, EvalResult, Sample, SampleEvaluated, Score

if TYPE_CHECKING:
    from ..runtime.events._types import EventBus
    from ..runtime.main_loop import MainLoop


def run_eval[I, O, E](
    loop: MainLoop[I, O],
    dataset: tuple[Sample[I, E], ...],
    evaluator: Callable[[O, E], Score],
    *,
    bus: EventBus | None = None,
) -> EvalReport:
    """Run evaluation using MainLoop.

    For each sample in the dataset:
    1. Execute the sample input through MainLoop
    2. Score the output using the evaluator
    3. Record timing
    4. Publish SampleEvaluated event (if bus provided)

    Args:
        loop: MainLoop instance to run samples through
        dataset: Tuple of samples to evaluate
        evaluator: Scoring function for outputs
        bus: Optional EventBus for progress notifications

    Returns:
        EvalReport with all results and aggregate metrics
    """
    results: list[EvalResult] = []

    for sample in dataset:
        start = time.monotonic()
        try:
            response, _ = loop.execute(sample.input)
            latency_ms = int((time.monotonic() - start) * 1000)
            output = response.output
            if output is None:
                result = EvalResult(
                    sample_id=sample.id,
                    score=Score(value=0.0, passed=False, reason="No output from loop"),
                    latency_ms=latency_ms,
                    error="No output from loop",
                )
            else:
                score = evaluator(output, sample.expected)
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
        if bus is not None:
            _ = bus.publish(SampleEvaluated(sample_id=sample.id, result=result))

    return EvalReport(results=tuple(results))


__all__ = ["run_eval"]

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

"""Helper functions for submitting samples and collecting results.

This module provides high-level functions for common evaluation workflows:
- ``submit_dataset()`` - Submit all samples from a dataset for evaluation
- ``submit_experiments()`` - Submit samples under multiple experiments for A/B testing
- ``collect_results()`` - Collect evaluation results into an EvalReport

These functions work with the mailbox infrastructure to enable distributed
evaluation across multiple workers.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TypeVar

from ..experiment import Experiment
from ..runtime.mailbox import Mailbox
from ._types import Dataset, EvalReport, EvalRequest, EvalResult

InputT = TypeVar("InputT")
ExpectedT = TypeVar("ExpectedT")


def submit_dataset(
    dataset: Dataset[InputT, ExpectedT],
    experiment: Experiment,
    requests: Mailbox[EvalRequest[InputT, ExpectedT], None],
) -> int:
    """Submit all samples in a dataset for evaluation under an experiment.

    Sends each sample to the requests mailbox as an EvalRequest with the
    specified experiment. This function is synchronous and blocks until
    all samples are enqueued.

    Args:
        dataset: The dataset containing samples to evaluate.
        experiment: The experiment to run samples under.
        requests: Mailbox to send EvalRequest messages to.

    Returns:
        Number of samples submitted.

    Example:
        >>> from weakincentives.evals import BASELINE
        >>> dataset = Dataset.load(Path("qa.jsonl"), str, str)
        >>> count = submit_dataset(dataset, BASELINE, requests_mailbox)
        >>> print(f"Submitted {count} samples")
    """
    count = 0
    for sample in dataset:
        _ = requests.send(EvalRequest(sample=sample, experiment=experiment))
        count += 1
    return count


def submit_experiments(
    dataset: Dataset[InputT, ExpectedT],
    experiments: Sequence[Experiment],
    requests: Mailbox[EvalRequest[InputT, ExpectedT], None],
) -> int:
    """Submit dataset under multiple experiments for comparison.

    Submits each sample under each experiment, allowing A/B testing
    and multi-variant comparison. The order is experiments-then-samples,
    meaning all samples for the first experiment are submitted before
    moving to the next experiment.

    Args:
        dataset: The dataset containing samples to evaluate.
        experiments: Sequence of experiments to run samples under.
        requests: Mailbox to send EvalRequest messages to.

    Returns:
        Total number of requests submitted (len(dataset) * len(experiments)).

    Example:
        >>> from weakincentives.evals import BASELINE, Experiment
        >>> baseline = BASELINE
        >>> treatment = Experiment(name="v2-prompts", overrides_tag="v2")
        >>> count = submit_experiments(dataset, [baseline, treatment], requests)
        >>> print(f"Submitted {count} total requests")
    """
    count = 0
    for experiment in experiments:
        for sample in dataset:
            _ = requests.send(EvalRequest(sample=sample, experiment=experiment))
            count += 1
    return count


def collect_results(
    results: Mailbox[EvalResult, None],
    expected_count: int,
    *,
    timeout_seconds: float = 300,
) -> EvalReport:
    """Collect evaluation results into a report.

    Polls the results mailbox until all expected results are collected
    or the timeout expires. Each collected message is acknowledged
    immediately upon receipt.

    Note:
        If the timeout expires before all results are collected, the
        returned report will contain only the results received so far.
        Check ``report.total`` against ``expected_count`` to detect
        incomplete collection.

    Args:
        results: Mailbox to receive EvalResult messages from.
        expected_count: Number of results to collect before returning.
        timeout_seconds: Maximum time to wait for all results. Defaults
            to 300 seconds (5 minutes).

    Returns:
        EvalReport containing all collected results, which may be fewer
        than expected_count if the timeout was reached.

    Example:
        >>> report = collect_results(
        ...     results_mailbox,
        ...     expected_count=len(dataset),
        ...     timeout_seconds=600,
        ... )
        >>> if report.total < len(dataset):
        ...     print(f"Warning: only collected {report.total}/{len(dataset)}")
        >>> print(f"Pass rate: {report.pass_rate:.1%}")
    """
    collected: list[EvalResult] = []
    deadline = time.time() + timeout_seconds

    while len(collected) < expected_count and time.time() < deadline:
        remaining = deadline - time.time()
        wait_time = min(20, max(1, int(remaining)))

        for msg in results.receive(wait_time_seconds=wait_time):
            collected.append(msg.body)
            msg.acknowledge()

    return EvalReport(results=tuple(collected))


__all__ = [
    "collect_results",
    "submit_dataset",
    "submit_experiments",
]

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

"""Helper functions for submitting samples and collecting results."""

from __future__ import annotations

import time
from typing import TypeVar

from ..runtime.mailbox import Mailbox
from ._types import Dataset, EvalReport, EvalRequest, EvalResult

InputT = TypeVar("InputT")
ExpectedT = TypeVar("ExpectedT")


def submit_dataset(
    dataset: Dataset[InputT, ExpectedT],
    requests: Mailbox[EvalRequest[InputT, ExpectedT]],
) -> None:
    """Submit all samples in a dataset for evaluation.

    Sends each sample to the requests mailbox as an EvalRequest.
    This function is synchronous and blocks until all samples
    are enqueued.

    Args:
        dataset: The dataset containing samples to evaluate.
        requests: Mailbox to send EvalRequest messages to.

    Example:
        >>> dataset = Dataset.load(Path("qa.jsonl"), str, str)
        >>> submit_dataset(dataset, requests_mailbox)
    """
    for sample in dataset:
        _ = requests.send(EvalRequest(sample=sample))


def collect_results(
    results: Mailbox[EvalResult],
    expected_count: int,
    *,
    timeout_seconds: float = 300,
) -> EvalReport:
    """Collect evaluation results into a report.

    Polls the results mailbox until all expected results are collected
    or the timeout expires. Each collected message is acknowledged.

    Args:
        results: Mailbox to receive results from.
        expected_count: Number of results to collect.
        timeout_seconds: Maximum time to wait for all results.

    Returns:
        EvalReport with all collected results.

    Example:
        >>> report = collect_results(
        ...     results_mailbox,
        ...     expected_count=len(dataset),
        ...     timeout_seconds=600,
        ... )
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
]

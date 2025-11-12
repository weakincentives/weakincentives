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

"""Pytest plugin that stress-tests thread-sensitive workloads."""

from __future__ import annotations

import os
import random
from collections.abc import Iterable, Iterator

import pytest

_MARKER = "threadstress"
_DEFAULT_MIN_WORKERS = 1
_DEFAULT_MAX_WORKERS = max(os.cpu_count() or 1, 4)


def pytest_configure(config: pytest.Config) -> None:
    """Register the threadstress marker."""

    config.addinivalue_line(
        "markers",
        "threadstress(min_workers=1, max_workers=None): repeat the test multiple times "
        "with randomized thread-pool sizes.",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Expose CLI toggles for the threadstress plugin."""

    group = parser.getgroup("threadstress")
    group.addoption(
        "--threadstress-iterations",
        action="store",
        type=int,
        default=1,
        help=(
            "Number of times tests marked with @pytest.mark.threadstress should be "
            "executed."
        ),
    )
    group.addoption(
        "--threadstress-max-workers",
        action="store",
        type=int,
        default=0,
        help=(
            "Upper bound for randomized worker counts when running threadstress "
            "tests. Defaults to the greater of the marker-provided max or the CPU "
            "count."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Duplicate threadstress tests for the configured number of iterations."""

    iterations = max(config.getoption("--threadstress-iterations"), 1)
    max_override = config.getoption("--threadstress-max-workers")

    if iterations == 1 and not max_override:
        for item in _iter_threadstress_functions(items):
            worker_count = _resolve_worker_count(config, item)
            item.add_marker(
                pytest.mark.parametrize(
                    "threadstress_workers",
                    [
                        pytest.param(
                            worker_count,
                            id=f"threadstress-workers-{worker_count}",
                        )
                    ],
                    scope="function",
                )
            )
        return

    for item in _iter_threadstress_functions(items):
        worker_counts = [
            _resolve_worker_count(config, item, salt=index)
            for index in range(iterations)
        ]
        params = [
            pytest.param(
                count,
                id=f"threadstress-{index}-workers-{count}",
            )
            for index, count in enumerate(worker_counts, start=1)
        ]
        item.add_marker(
            pytest.mark.parametrize(
                "threadstress_workers",
                params,
                scope="function",
            )
        )


@pytest.fixture
def threadstress_workers(request: pytest.FixtureRequest) -> int:
    """Return the worker count chosen for the current threadstress iteration."""

    param = getattr(request, "param", None)
    if param is not None:
        return int(param)

    marker = request.node.get_closest_marker(_MARKER)
    if marker is None:
        return _DEFAULT_MIN_WORKERS

    min_workers, max_workers = _marker_bounds(marker, request.config)
    rng = random.Random(_rng_seed(request.node.name))
    return rng.randint(min_workers, max_workers)


def _iter_threadstress_functions(
    items: Iterable[pytest.Item],
) -> Iterator[pytest.Function]:
    for item in items:
        if not isinstance(item, pytest.Function):
            continue
        if item.get_closest_marker(_MARKER) is None:
            continue
        yield item


def _resolve_worker_count(
    config: pytest.Config, item: pytest.Function, *, salt: int | None = None
) -> int:
    marker = item.get_closest_marker(_MARKER)
    if marker is None:
        return _DEFAULT_MIN_WORKERS
    min_workers, max_workers = _marker_bounds(marker, config)
    identifier = item.nodeid if salt is None else f"{item.nodeid}-{salt}"
    rng = random.Random(_rng_seed(identifier))
    return rng.randint(min_workers, max_workers)


def _marker_bounds(
    marker: pytest.Mark, config: pytest.Config | None = None
) -> tuple[int, int]:
    min_workers = int(marker.kwargs.get("min_workers", _DEFAULT_MIN_WORKERS))
    min_workers = max(min_workers, _DEFAULT_MIN_WORKERS)

    marker_max = marker.kwargs.get("max_workers")
    max_workers = int(marker_max) if marker_max is not None else _DEFAULT_MAX_WORKERS

    override = None
    if config is not None:
        override = config.getoption("--threadstress-max-workers")
    if override:
        max_workers = max(int(override), min_workers)

    if max_workers < min_workers:
        max_workers = min_workers

    return min_workers, max_workers


def _rng_seed(identifier: str) -> int:
    return hash(identifier) & 0xFFFFFFFF

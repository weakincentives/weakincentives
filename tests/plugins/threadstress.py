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
import warnings

import pytest
from _pytest.mark.structures import ParameterSet

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


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parameterize threadstress tests early so iterations actually duplicate."""

    marker = metafunc.definition.get_closest_marker(_MARKER)
    if marker is None:
        return

    iterations = max(metafunc.config.getoption("--threadstress-iterations"), 1)
    max_override = metafunc.config.getoption("--threadstress-max-workers")

    if iterations == 1 and not max_override:
        return

    if "threadstress_workers" not in metafunc.fixturenames:
        if iterations > 1:
            warnings.warn(
                pytest.PytestWarning(
                    "Test marked with @pytest.mark.threadstress does not request the "
                    "'threadstress_workers' fixture; --threadstress-iterations will "
                    "have no effect.",
                ),
                stacklevel=2,
            )
        return

    nodeid = _metafunc_nodeid(metafunc)
    params = _threadstress_params(
        marker=marker,
        config=metafunc.config,
        nodeid=nodeid,
        iterations=iterations,
    )
    metafunc.parametrize("threadstress_workers", params, scope="function")


@pytest.fixture
def threadstress_workers(request: pytest.FixtureRequest) -> int:
    """Return the worker count chosen for the current threadstress iteration."""

    param = getattr(request, "param", None)
    if param is not None:
        return int(param)

    marker = request.node.get_closest_marker(_MARKER)
    min_workers, max_workers = _marker_bounds(marker, request.config)
    rng = random.Random(_rng_seed(request.node.name))
    return rng.randint(min_workers, max_workers)


def _threadstress_params(
    *,
    marker: pytest.Mark,
    config: pytest.Config,
    nodeid: str,
    iterations: int,
) -> list[ParameterSet]:
    worker_counts = [
        _resolve_worker_count(
            marker=marker,
            config=config,
            nodeid=nodeid,
            iteration=index,
        )
        for index in range(iterations)
    ]
    return [
        pytest.param(
            count,
            id=f"threadstress-{index}-workers-{count}",
        )
        for index, count in enumerate(worker_counts, start=1)
    ]


def _resolve_worker_count(
    *,
    marker: pytest.Mark | None,
    config: pytest.Config,
    nodeid: str,
    iteration: int | None = None,
) -> int:
    min_workers, max_workers = _marker_bounds(marker, config)
    identifier = nodeid if iteration is None else f"{nodeid}-{iteration}"
    rng = random.Random(_rng_seed(identifier))
    return rng.randint(min_workers, max_workers)


def _marker_bounds(
    marker: pytest.Mark | None, config: pytest.Config | None = None
) -> tuple[int, int]:
    if marker is None:
        min_workers = _DEFAULT_MIN_WORKERS
        max_workers = _DEFAULT_MAX_WORKERS
    else:
        min_workers = int(marker.kwargs.get("min_workers", _DEFAULT_MIN_WORKERS))
        min_workers = max(min_workers, _DEFAULT_MIN_WORKERS)

        marker_max = marker.kwargs.get("max_workers")
        max_workers = (
            int(marker_max) if marker_max is not None else _DEFAULT_MAX_WORKERS
        )

    override = None
    if config is not None:
        override = config.getoption("--threadstress-max-workers")
    if override:
        max_workers = max(int(override), min_workers)

    max_workers = max(max_workers, min_workers)

    return min_workers, max_workers


def _metafunc_nodeid(metafunc: pytest.Metafunc) -> str:
    node = metafunc.definition
    nodeid = getattr(node, "nodeid", "")
    if nodeid:
        return str(nodeid)
    module_name = getattr(metafunc.module, "__name__", "unknown")
    function_name = getattr(node, "originalname", getattr(node, "name", "test"))
    return f"{module_name}::{function_name}"


def _rng_seed(identifier: str) -> int:
    return hash(identifier) & 0xFFFFFFFF

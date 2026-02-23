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

"""Pytest configuration for ACK integration tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.runtime.session import Session

from .adapters import (
    AdapterFixture,
    ClaudeAgentSDKFixture,
    CodexAppServerFixture,
    GeminiACPFixture,
    OpenCodeACPFixture,
)

_ACK_TIMEOUT_SECONDS = int(os.environ.get("ACK_TIMEOUT", "120"))
_ACK_ADAPTERS_ENV = "ACK_ADAPTERS"

_ALL_FIXTURES: tuple[AdapterFixture, ...] = (
    ClaudeAgentSDKFixture(),
    CodexAppServerFixture(),
    GeminiACPFixture(),
    OpenCodeACPFixture(),
)


def _selected_adapter_names() -> frozenset[str] | None:
    raw = os.environ.get(_ACK_ADAPTERS_ENV)
    if raw is None:
        return None
    selected = frozenset(part.strip() for part in raw.split(",") if part.strip())
    return selected if selected else None


def _selected_fixtures() -> tuple[AdapterFixture, ...]:
    selected = _selected_adapter_names()
    if selected is None:
        return _ALL_FIXTURES
    return tuple(f for f in _ALL_FIXTURES if f.adapter_name in selected)


def _available_fixtures() -> tuple[AdapterFixture, ...]:
    return tuple(f for f in _selected_fixtures() if f.is_available())


_AVAILABLE_FIXTURES = _available_fixtures()
_ADAPTER_PARAMS: tuple[object, ...]
if _AVAILABLE_FIXTURES:
    _ADAPTER_PARAMS = _AVAILABLE_FIXTURES
else:
    _ADAPTER_PARAMS = (
        pytest.param(
            None,
            marks=pytest.mark.skip(
                reason="No ACK adapters available in this environment."
            ),
            id="no-adapters",
        ),
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply ACK defaults: integration marker + timeout marker."""
    for item in items:
        if item.get_closest_marker("integration") is None:
            item.add_marker(pytest.mark.integration)
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(_ACK_TIMEOUT_SECONDS))


@pytest.fixture(
    params=_ADAPTER_PARAMS, ids=lambda f: "no-adapters" if f is None else f.adapter_name
)
def adapter_fixture(request: pytest.FixtureRequest) -> AdapterFixture:
    """Parameterized fixture that yields each available ACK adapter fixture."""
    fixture = cast("AdapterFixture | None", request.param)
    if fixture is None:
        pytest.skip("No ACK adapters available in this environment.")
    return fixture


@pytest.fixture
def adapter(
    adapter_fixture: AdapterFixture,
    tmp_path: Path,
) -> ProviderAdapter[object]:
    """Create a provider adapter configured for this test's temp workspace."""
    return adapter_fixture.create_adapter(tmp_path)


@pytest.fixture
def session(adapter_fixture: AdapterFixture) -> Session:
    """Create a fresh session for ACK integration scenarios."""
    return adapter_fixture.create_session()


@pytest.fixture(autouse=True)
def _enforce_ack_capabilities(request: pytest.FixtureRequest) -> None:
    """Skip tests whose required capabilities are not supported."""
    if "adapter_fixture" not in request.fixturenames:
        return

    fixture = cast(AdapterFixture, request.getfixturevalue("adapter_fixture"))
    for marker in request.node.iter_markers("ack_capability"):
        if not marker.args:
            raise RuntimeError("ack_capability marker requires a capability name")

        capability = marker.args[0]
        if not isinstance(capability, str):
            raise TypeError("ack_capability marker argument must be a string")

        if not getattr(fixture.capabilities, capability, False):
            pytest.skip(f"{fixture.adapter_name} does not support {capability}")

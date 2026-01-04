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

"""Tests for session slice management (query, seed, clear, register)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tests.helpers.session import ExampleOutput, make_prompt_event
from weakincentives.dbc import dbc_enabled
from weakincentives.runtime.session import (
    SliceAccessor,
    replace_latest,
)

if TYPE_CHECKING:
    from tests.conftest import SessionFactory

import pytest


def test_indexing_returns_slice_accessor(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    accessor = session[ExampleOutput]

    assert isinstance(accessor, SliceAccessor)


def test_query_all_returns_empty_tuple_when_no_values(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    result = session[ExampleOutput].all()

    assert result == ()


def test_query_latest_returns_none_when_no_values(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    result = session[ExampleOutput].latest()

    assert result is None


def test_query_where_returns_empty_tuple_when_no_values(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    result = session[ExampleOutput].where(lambda x: True)

    assert result == ()


def test_query_all_returns_all_values(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="second")))

    result = session[ExampleOutput].all()

    assert result == (ExampleOutput(text="first"), ExampleOutput(text="second"))


def test_query_latest_returns_most_recent_value(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="second")))

    result = session[ExampleOutput].latest()

    assert result == ExampleOutput(text="second")


def test_query_where_filters_values(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="apple")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="banana")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="apricot")))

    result = session[ExampleOutput].where(lambda x: x.text.startswith("a"))

    assert result == (ExampleOutput(text="apple"), ExampleOutput(text="apricot"))


def test_query_respects_dbc_purity(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="second")))

    with dbc_enabled():
        assert session[ExampleOutput].all() == (
            ExampleOutput(text="first"),
            ExampleOutput(text="second"),
        )
        assert session[ExampleOutput].latest() == ExampleOutput(text="second")
        assert session[ExampleOutput].where(lambda x: x.text.startswith("f")) == (
            ExampleOutput(text="first"),
        )


def test_query_where_logs_violate_purity_contract(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))

    logger = logging.getLogger(__name__)

    def predicate(value: ExampleOutput) -> bool:
        logger.warning("Saw %s", value)
        return True

    with dbc_enabled(), pytest.raises(AssertionError):
        session[ExampleOutput].where(predicate)


def test_seed_single_value(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session[ExampleOutput].seed(ExampleOutput(text="seeded"))

    assert session[ExampleOutput].all() == (ExampleOutput(text="seeded"),)


def test_mutate_seed_iterable_values(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session[ExampleOutput].seed(
        [
            ExampleOutput(text="first"),
            ExampleOutput(text="second"),
        ]
    )

    assert session[ExampleOutput].all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )


def test_mutate_clear_removes_all_values(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="second")))
    assert session[ExampleOutput].all()

    session[ExampleOutput].clear()

    assert session[ExampleOutput].all() == ()


def test_mutate_clear_with_predicate(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="apple")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="banana")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="apricot")))

    session[ExampleOutput].clear(lambda x: x.text.startswith("a"))

    assert session[ExampleOutput].all() == (ExampleOutput(text="banana"),)


def test_mutate_append_uses_default_reducer(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session[ExampleOutput].append(ExampleOutput(text="first"))
    session[ExampleOutput].append(ExampleOutput(text="second"))

    assert session[ExampleOutput].all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )


def test_mutate_register_adds_reducer(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session[ExampleOutput].register(ExampleOutput, replace_latest)

    session[ExampleOutput].append(ExampleOutput(text="first"))
    session[ExampleOutput].append(ExampleOutput(text="second"))

    assert session[ExampleOutput].all() == (ExampleOutput(text="second"),)


def test_getitem_returns_slice_accessor(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    accessor = session[ExampleOutput]

    assert isinstance(accessor, SliceAccessor)


def test_slice_accessor_query_methods_work(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="second")))

    # Test all(), latest(), and where() work via SliceAccessor
    assert session[ExampleOutput].all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )
    assert session[ExampleOutput].latest() == ExampleOutput(text="second")
    assert session[ExampleOutput].where(lambda x: x.text.startswith("f")) == (
        ExampleOutput(text="first"),
    )

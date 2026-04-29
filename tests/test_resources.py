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

"""Tests for ``weakincentives.resources``."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytest

from weakincentives.core import ResourceProvider
from weakincentives.filesystem import InMemoryFilesystem
from weakincentives.resources import (
    Binding,
    ResourceRegistry,
    ResourceScope,
)


@runtime_checkable
class _Greeter(Protocol):
    def greet(self) -> str: ...


class _ConcreteGreeter:
    def __init__(self) -> None:
        self.greeted = 0

    def greet(self) -> str:
        self.greeted += 1
        return "hi"


def test_singleton_resource_is_reused() -> None:
    registry = ResourceRegistry()
    registry.bind(Binding(_Greeter, lambda _: _ConcreteGreeter()))
    with registry.context() as ctx:
        first = ctx.get(_Greeter)
        second = ctx.get(_Greeter)
        assert first is second
        assert first.greet() == "hi"


def test_tool_call_resource_is_per_call() -> None:
    registry = ResourceRegistry()
    registry.bind(
        Binding(
            _Greeter,
            lambda _: _ConcreteGreeter(),
            scope=ResourceScope.TOOL_CALL,
        )
    )
    with registry.context() as ctx:
        with ctx.tool_call_scope():
            a = ctx.get(_Greeter)
            assert ctx.get(_Greeter) is a
        with ctx.tool_call_scope():
            b = ctx.get(_Greeter)
            assert b is not a


def test_prototype_resource_is_always_fresh() -> None:
    registry = ResourceRegistry()
    registry.bind(
        Binding(
            _Greeter,
            lambda _: _ConcreteGreeter(),
            scope=ResourceScope.PROTOTYPE,
        )
    )
    with registry.context() as ctx:
        assert ctx.get(_Greeter) is not ctx.get(_Greeter)


def test_get_unbound_protocol_raises() -> None:
    registry = ResourceRegistry()
    with registry.context() as ctx:
        with pytest.raises(LookupError, match="no binding"):
            ctx.get(_Greeter)


def test_tool_call_scope_required_for_tool_call_resources() -> None:
    registry = ResourceRegistry()
    registry.bind(
        Binding(
            _Greeter,
            lambda _: _ConcreteGreeter(),
            scope=ResourceScope.TOOL_CALL,
        )
    )
    with registry.context() as ctx:
        with pytest.raises(LookupError, match="enter_tool_call"):
            ctx.get(_Greeter)


def test_tool_call_scope_cannot_nest() -> None:
    registry = ResourceRegistry()
    with registry.context() as ctx:
        with ctx.tool_call_scope():
            with pytest.raises(LookupError, match="already active"):
                with ctx.tool_call_scope():
                    pass


def test_post_construct_hook_runs() -> None:
    constructed: list[str] = []

    class Hooked:
        def post_construct(self) -> None:
            constructed.append("ok")

    @runtime_checkable
    class Marker(Protocol):
        def post_construct(self) -> None: ...

    registry = ResourceRegistry()
    registry.bind(Binding(Marker, lambda _: Hooked()))
    with registry.context() as ctx:
        ctx.get(Marker)
    assert constructed == ["ok"]


def test_close_hook_runs_on_context_exit() -> None:
    closed: list[str] = []

    class Hooked:
        def close(self) -> None:
            closed.append("closed")

    @runtime_checkable
    class Marker(Protocol):
        def close(self) -> None: ...

    registry = ResourceRegistry()
    registry.bind(Binding(Marker, lambda _: Hooked()))
    with registry.context() as ctx:
        ctx.get(Marker)
    assert closed == ["closed"]


def test_close_hook_runs_on_tool_call_exit() -> None:
    closed: list[str] = []

    class Hooked:
        def close(self) -> None:
            closed.append("c")

    @runtime_checkable
    class Marker(Protocol):
        def close(self) -> None: ...

    registry = ResourceRegistry()
    registry.bind(Binding(Marker, lambda _: Hooked(), scope=ResourceScope.TOOL_CALL))
    with registry.context() as ctx:
        with ctx.tool_call_scope():
            ctx.get(Marker)
        assert closed == ["c"]


def test_singletons_view_is_a_copy() -> None:
    registry = ResourceRegistry()
    registry.bind(Binding(_Greeter, lambda _: _ConcreteGreeter()))
    with registry.context() as ctx:
        ctx.get(_Greeter)
        view = ctx.singletons
        view.clear()
        assert _Greeter in ctx.singletons


def test_snapshotable_resources_collected() -> None:
    @runtime_checkable
    class FsProto(Protocol):
        def read_text(self, path: str) -> str: ...

    registry = ResourceRegistry()
    registry.bind(Binding(FsProto, lambda _: InMemoryFilesystem()))
    with registry.context() as ctx:
        ctx.get(FsProto)
        snapshotables = ctx.snapshotable_resources()
        assert len(snapshotables) == 1


def test_registry_has_method() -> None:
    registry = ResourceRegistry()
    assert not registry.has(_Greeter)
    registry.bind(Binding(_Greeter, lambda _: _ConcreteGreeter()))
    assert registry.has(_Greeter)


def test_context_satisfies_resource_provider() -> None:
    registry = ResourceRegistry()
    with registry.context() as ctx:
        assert isinstance(ctx, ResourceProvider)

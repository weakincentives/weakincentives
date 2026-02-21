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

"""Tests for scope behavior (SINGLETON, TOOL_CALL, PROTOTYPE)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from weakincentives.resources import (
    Binding,
    ResourceRegistry,
    ResourceResolver,
    Scope,
)

from .conftest import (
    CloseableResource,
    ConcreteConfig,
    Config,
)

# === Scope Behavior Tests ===


class TestScopeBehavior:
    def test_prototype_never_cached(self) -> None:
        counter = itertools.count()

        @dataclass
        class Numbered:
            n: int

        registry = ResourceRegistry.build(
            {
                Numbered: Binding(
                    Numbered, lambda r: Numbered(n=next(counter)), scope=Scope.PROTOTYPE
                )
            }
        )
        with registry.open() as ctx:
            n1 = ctx.get(Numbered)
            n2 = ctx.get(Numbered)
            n3 = ctx.get(Numbered)
            assert n1.n == 0
            assert n2.n == 1
            assert n3.n == 2

    def test_singleton_shared_across_tool_scopes(self) -> None:
        call_count = 0

        def make_config(r: ResourceResolver) -> ConcreteConfig:
            nonlocal call_count
            call_count += 1
            return ConcreteConfig()

        registry = ResourceRegistry.build(
            {Config: Binding(Config, make_config, scope=Scope.SINGLETON)}
        )
        with registry.open() as ctx:
            with ctx.tool_scope() as r1:
                c1 = r1.get(Config)

            with ctx.tool_scope() as r2:
                c2 = r2.get(Config)

            assert c1 is c2
            assert call_count == 1

    def test_tool_call_fresh_per_scope(self) -> None:
        counter = itertools.count()

        @dataclass
        class Tracer:
            id: int

        registry = ResourceRegistry.build(
            {
                Tracer: Binding(
                    Tracer, lambda r: Tracer(id=next(counter)), scope=Scope.TOOL_CALL
                )
            }
        )
        with registry.open() as ctx:
            with ctx.tool_scope() as r1:
                t1 = r1.get(Tracer)
                t1_again = r1.get(Tracer)
                assert t1 is t1_again  # Same within scope

            with ctx.tool_scope() as r2:
                t2 = r2.get(Tracer)

            assert t1.id == 0
            assert t2.id == 1  # Fresh instance

    def test_tool_call_closed_on_scope_exit(self) -> None:
        registry = ResourceRegistry.build(
            {
                CloseableResource: Binding(
                    CloseableResource,
                    lambda r: CloseableResource(),
                    scope=Scope.TOOL_CALL,
                )
            }
        )
        with registry.open() as ctx:
            with ctx.tool_scope() as r:
                resource = r.get(CloseableResource)
                assert resource.closed is False

            assert resource.closed is True

    def test_tool_scope_does_not_leak(self) -> None:
        counter = itertools.count()

        @dataclass
        class Tracer:
            id: int

        registry = ResourceRegistry.build(
            {
                Tracer: Binding(
                    Tracer, lambda r: Tracer(id=next(counter)), scope=Scope.TOOL_CALL
                )
            }
        )
        with registry.open() as ctx:
            # First scope
            with ctx.tool_scope() as r:
                _ = r.get(Tracer)

            # Second scope should start fresh
            with ctx.tool_scope() as r:
                t = r.get(Tracer)
                assert t.id == 1

    def test_nested_tool_scopes(self) -> None:
        """Test nested tool scopes maintain proper isolation."""
        counter = itertools.count()
        close_order: list[int] = []

        @dataclass
        class NestedTracer:
            id: int
            closed: bool = False

            def close(self) -> None:
                self.closed = True
                close_order.append(self.id)

        registry = ResourceRegistry.build(
            {
                NestedTracer: Binding(
                    NestedTracer,
                    lambda r: NestedTracer(id=next(counter)),
                    scope=Scope.TOOL_CALL,
                )
            }
        )
        with registry.open() as ctx:
            with ctx.tool_scope() as outer:
                t_outer = outer.get(NestedTracer)
                assert t_outer.id == 0

                # Nested scope
                with ctx.tool_scope() as inner:
                    t_inner = inner.get(NestedTracer)
                    assert t_inner.id == 1
                    assert t_inner is not t_outer

                # Inner tracer closed on exit
                assert t_inner.closed is True
                assert t_outer.closed is False

                # Outer scope still works
                t_outer_again = outer.get(NestedTracer)
                assert t_outer_again is t_outer

            # Outer tracer closed on exit
            assert t_outer.closed is True
            # Inner closed first, then outer
            assert close_order == [1, 0]

    def test_nested_tool_scopes_with_singleton(self) -> None:
        """Test nested tool scopes share singleton resources."""
        singleton_count = 0
        tool_count = itertools.count()

        @dataclass
        class SharedConfig:
            id: int

        @dataclass
        class ScopedTracer:
            config: SharedConfig
            id: int

        def make_config(r: ResourceResolver) -> SharedConfig:
            nonlocal singleton_count
            singleton_count += 1
            return SharedConfig(id=singleton_count)

        registry = ResourceRegistry.build(
            {
                SharedConfig: Binding(SharedConfig, make_config),
                ScopedTracer: Binding(
                    ScopedTracer,
                    lambda r: ScopedTracer(
                        config=r.get(SharedConfig), id=next(tool_count)
                    ),
                    scope=Scope.TOOL_CALL,
                ),
            }
        )
        with registry.open() as ctx:
            with ctx.tool_scope() as outer:
                t_outer = outer.get(ScopedTracer)

                with ctx.tool_scope() as inner:
                    t_inner = inner.get(ScopedTracer)

                    # Both refer to same singleton
                    assert t_outer.config is t_inner.config
                    # But different tool-scoped instances
                    assert t_outer.id != t_inner.id

            # Singleton created only once
            assert singleton_count == 1

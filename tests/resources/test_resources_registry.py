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

"""Tests for ResourceRegistry.of(), error messages, and integration scenarios."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Protocol

import pytest

from weakincentives.resources import (
    Binding,
    CircularDependencyError,
    DuplicateBindingError,
    ProviderError,
    ResourceRegistry,
    ResourceResolver,
    Scope,
    UnboundResourceError,
)

# === Test Fixtures ===


class Config(Protocol):
    @property
    def value(self) -> int: ...


class HTTPClient(Protocol):
    @property
    def config(self) -> Config: ...


class Service(Protocol):
    @property
    def http(self) -> HTTPClient: ...


@dataclass
class ConcreteConfig:
    value: int = 42


@dataclass
class ConcreteHTTPClient:
    config: Config


@dataclass
class ConcreteService:
    http: HTTPClient


@dataclass
class CloseableResource:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass
class PostConstructResource:
    initialized: bool = False

    def post_construct(self) -> None:
        self.initialized = True


@dataclass
class FailingPostConstruct:
    def post_construct(self) -> None:
        raise RuntimeError("Initialization failed")


@dataclass
class CloseableFailingPostConstruct:
    closed: bool = False

    def post_construct(self) -> None:
        raise RuntimeError("Initialization failed")

    def close(self) -> None:
        self.closed = True


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


# === ResourceRegistry.of() Tests ===


class TestResourceRegistryOf:
    def test_of_creates_registry_from_bindings(self) -> None:
        """ResourceRegistry.of() creates registry from explicit bindings."""
        registry = ResourceRegistry.of(
            Binding(Config, lambda r: ConcreteConfig(value=42)),
            Binding(HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))),
        )
        assert len(registry) == 2
        assert Config in registry
        assert HTTPClient in registry

        with registry.open() as ctx:
            http = ctx.get(HTTPClient)
            assert http.config.value == 42

    def test_of_empty(self) -> None:
        """ResourceRegistry.of() with no bindings creates empty registry."""
        registry = ResourceRegistry.of()
        assert len(registry) == 0

    def test_of_raises_on_duplicate(self) -> None:
        """ResourceRegistry.of() raises DuplicateBindingError on duplicate protocol."""
        with pytest.raises(DuplicateBindingError) as exc:
            ResourceRegistry.of(
                Binding(Config, lambda r: ConcreteConfig(value=1)),
                Binding(Config, lambda r: ConcreteConfig(value=2)),
            )
        assert exc.value.protocol is Config


# === Error Message Tests ===


class TestErrorMessages:
    def test_unbound_error_message(self) -> None:
        err = UnboundResourceError(Config)
        assert "Config" in str(err)

    def test_circular_error_message(self) -> None:
        @dataclass
        class A:
            pass

        @dataclass
        class B:
            pass

        err = CircularDependencyError((A, B, A))
        msg = str(err)
        assert "A" in msg
        assert "B" in msg
        assert "->" in msg

    def test_duplicate_error_message(self) -> None:
        err = DuplicateBindingError(Config)
        assert "Config" in str(err)

    def test_provider_error_message(self) -> None:
        cause = ValueError("bad value")
        err = ProviderError(Config, cause)
        assert "Config" in str(err)
        assert "ValueError" in str(err)
        assert "bad value" in str(err)


# === Integration Tests ===


class TestIntegration:
    def test_full_dependency_chain(self) -> None:
        registry = ResourceRegistry.build(
            {
                Config: Binding(Config, lambda r: ConcreteConfig(value=100)),
                HTTPClient: Binding(
                    HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))
                ),
                Service: Binding(Service, lambda r: ConcreteService(r.get(HTTPClient))),
            }
        )
        with registry.open() as ctx:
            service = ctx.get(Service)
            assert service.http.config.value == 100

    def test_deeply_nested_dependency_chain(self) -> None:
        """Test resolving a 5-level dependency chain: A → B → C → D → E."""

        @dataclass
        class LevelE:
            value: int

        @dataclass
        class LevelD:
            e: LevelE

        @dataclass
        class LevelC:
            d: LevelD

        @dataclass
        class LevelB:
            c: LevelC

        @dataclass
        class LevelA:
            b: LevelB

        registry = ResourceRegistry.build(
            {
                LevelE: Binding(LevelE, lambda r: LevelE(value=42)),
                LevelD: Binding(LevelD, lambda r: LevelD(e=r.get(LevelE))),
                LevelC: Binding(LevelC, lambda r: LevelC(d=r.get(LevelD))),
                LevelB: Binding(LevelB, lambda r: LevelB(c=r.get(LevelC))),
                LevelA: Binding(LevelA, lambda r: LevelA(b=r.get(LevelB))),
            }
        )
        with registry.open() as ctx:
            a = ctx.get(LevelA)
            assert a.b.c.d.e.value == 42

    def test_deep_circular_dependency(self) -> None:
        """Test circular dependency detection in deep chain: A → B → C → A."""

        @dataclass
        class DeepA:
            b: object

        @dataclass
        class DeepB:
            c: object

        @dataclass
        class DeepC:
            a: object  # Cycle back to A

        registry = ResourceRegistry.build(
            {
                DeepA: Binding(DeepA, lambda r: DeepA(b=r.get(DeepB))),
                DeepB: Binding(DeepB, lambda r: DeepB(c=r.get(DeepC))),
                DeepC: Binding(DeepC, lambda r: DeepC(a=r.get(DeepA))),
            }
        )
        with registry.open() as ctx:
            with pytest.raises(CircularDependencyError) as exc:
                ctx.get(DeepA)
            # Cycle should include all three types
            assert DeepA in exc.value.cycle
            assert DeepB in exc.value.cycle
            assert DeepC in exc.value.cycle

    def test_deep_chain_with_mixed_scopes(self) -> None:
        """Test deep chain with SINGLETON depending on TOOL_CALL (and vice versa)."""
        counter = itertools.count()

        @dataclass
        class DeepConfig:
            id: int

        @dataclass
        class DeepClient:
            config: DeepConfig
            id: int

        @dataclass
        class DeepService:
            client: DeepClient
            id: int

        registry = ResourceRegistry.build(
            {
                # SINGLETON at the root
                DeepConfig: Binding(DeepConfig, lambda r: DeepConfig(id=next(counter))),
                # TOOL_CALL depends on SINGLETON
                DeepClient: Binding(
                    DeepClient,
                    lambda r: DeepClient(config=r.get(DeepConfig), id=next(counter)),
                    scope=Scope.TOOL_CALL,
                ),
                # Another TOOL_CALL depends on first TOOL_CALL
                DeepService: Binding(
                    DeepService,
                    lambda r: DeepService(client=r.get(DeepClient), id=next(counter)),
                    scope=Scope.TOOL_CALL,
                ),
            }
        )
        with registry.open() as ctx:
            # First tool scope
            with ctx.tool_scope() as r1:
                s1 = r1.get(DeepService)
                config_id = s1.client.config.id

            # Second tool scope - same config, fresh client and service
            with ctx.tool_scope() as r2:
                s2 = r2.get(DeepService)
                assert s2.client.config.id == config_id  # Same singleton
                assert s2.client.id != s1.client.id  # Fresh tool-call resource
                assert s2.id != s1.id  # Fresh tool-call resource

    def test_mixed_scopes(self) -> None:
        counter = itertools.count()

        @dataclass
        class RequestId:
            id: int

        registry = ResourceRegistry.build(
            {
                Config: Binding(Config, lambda r: ConcreteConfig()),  # SINGLETON
                RequestId: Binding(
                    RequestId,
                    lambda r: RequestId(id=next(counter)),
                    scope=Scope.TOOL_CALL,
                ),
            }
        )
        with registry.open() as ctx:
            configs = []
            request_ids = []

            for _ in range(3):
                with ctx.tool_scope() as r:
                    configs.append(r.get(Config))
                    request_ids.append(r.get(RequestId))

            # Same config instance
            assert configs[0] is configs[1] is configs[2]
            # Different request IDs
            assert request_ids[0].id == 0
            assert request_ids[1].id == 1
            assert request_ids[2].id == 2

    def test_context_manager_pattern(self) -> None:
        """Test manual lifecycle control using _create_context."""
        registry = ResourceRegistry.build(
            {
                CloseableResource: Binding(
                    CloseableResource, lambda r: CloseableResource()
                )
            }
        )

        ctx = registry._create_context()  # pyright: ignore[reportPrivateUsage]
        ctx.start()
        try:
            resource = ctx.get(CloseableResource)
            assert resource.closed is False
        finally:
            ctx.close()

        assert resource.closed is True

    def test_registry_open_context_manager(self) -> None:
        """Test ResourceRegistry.open() context manager."""
        registry = ResourceRegistry.build(
            {
                CloseableResource: Binding(
                    CloseableResource, lambda r: CloseableResource()
                )
            }
        )

        with registry.open() as ctx:
            resource = ctx.get(CloseableResource)
            assert resource.closed is False

        assert resource.closed is True

    def test_registry_open_handles_exception(self) -> None:
        """Test that open() cleans up resources on exception."""
        registry = ResourceRegistry.build(
            {
                CloseableResource: Binding(
                    CloseableResource, lambda r: CloseableResource()
                )
            }
        )
        resources: list[CloseableResource] = []

        with pytest.raises(ValueError, match="test error"):
            with registry.open() as ctx:
                resource = ctx.get(CloseableResource)
                resources.append(resource)
                raise ValueError("test error")

        assert len(resources) == 1
        assert resources[0].closed is True

    def test_registry_open_starts_eager_bindings(self) -> None:
        """Test that open() starts eager bindings."""
        constructed = []

        def make_config(r: ResourceResolver) -> ConcreteConfig:
            constructed.append("config")
            return ConcreteConfig()

        registry = ResourceRegistry.build(
            {Config: Binding(Config, make_config, eager=True)}
        )

        assert constructed == []
        with registry.open():
            assert constructed == ["config"]

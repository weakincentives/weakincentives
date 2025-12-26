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

"""Comprehensive tests for the resources module."""

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


# === Scope Tests ===


class TestScope:
    def test_scope_values(self) -> None:
        assert Scope.SINGLETON.value == "singleton"
        assert Scope.TOOL_CALL.value == "tool_call"
        assert Scope.PROTOTYPE.value == "prototype"

    def test_scope_is_enum(self) -> None:
        assert len(Scope) == 3


# === Binding Tests ===


class TestBinding:
    def test_binding_default_scope(self) -> None:
        binding = Binding(Config, lambda r: ConcreteConfig())
        assert binding.scope == Scope.SINGLETON
        assert binding.eager is False

    def test_binding_custom_scope(self) -> None:
        binding = Binding(Config, lambda r: ConcreteConfig(), scope=Scope.TOOL_CALL)
        assert binding.scope == Scope.TOOL_CALL

    def test_binding_eager(self) -> None:
        binding = Binding(Config, lambda r: ConcreteConfig(), eager=True)
        assert binding.eager is True

    def test_binding_is_frozen(self) -> None:
        binding = Binding(Config, lambda r: ConcreteConfig())
        with pytest.raises(AttributeError):
            binding.scope = Scope.PROTOTYPE  # type: ignore[misc]


# === ResourceRegistry Tests ===


class TestResourceRegistry:
    def test_empty_registry(self) -> None:
        registry = ResourceRegistry.of()
        assert len(registry) == 0
        assert Config not in registry

    def test_single_binding(self) -> None:
        registry = ResourceRegistry.of(Binding(Config, lambda r: ConcreteConfig()))
        assert len(registry) == 1
        assert Config in registry

    def test_multiple_bindings(self) -> None:
        registry = ResourceRegistry.of(
            Binding(Config, lambda r: ConcreteConfig()),
            Binding(HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))),
        )
        assert len(registry) == 2
        assert Config in registry
        assert HTTPClient in registry

    def test_duplicate_binding_raises(self) -> None:
        with pytest.raises(DuplicateBindingError) as exc:
            ResourceRegistry.of(
                Binding(Config, lambda r: ConcreteConfig()),
                Binding(Config, lambda r: ConcreteConfig(value=99)),
            )
        assert exc.value.protocol is Config

    def test_binding_for(self) -> None:
        binding = Binding(Config, lambda r: ConcreteConfig())
        registry = ResourceRegistry.of(binding)
        assert registry.binding_for(Config) is binding
        assert registry.binding_for(HTTPClient) is None

    def test_merge(self) -> None:
        base = ResourceRegistry.of(Binding(Config, lambda r: ConcreteConfig(value=1)))
        override = ResourceRegistry.of(
            Binding(Config, lambda r: ConcreteConfig(value=2))
        )
        merged = base.merge(override)

        ctx = merged.create_context()
        config = ctx.get(Config)
        assert config.value == 2

    def test_merge_adds_new(self) -> None:
        base = ResourceRegistry.of(Binding(Config, lambda r: ConcreteConfig()))
        additional = ResourceRegistry.of(
            Binding(HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config)))
        )
        merged = base.merge(additional)
        assert Config in merged
        assert HTTPClient in merged

    def test_iter(self) -> None:
        registry = ResourceRegistry.of(
            Binding(Config, lambda r: ConcreteConfig()),
            Binding(HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))),
        )
        protocols = set(registry)
        assert protocols == {Config, HTTPClient}

    def test_build_creates_registry_with_instances(self) -> None:
        config = ConcreteConfig(value=42)
        registry = ResourceRegistry.build({Config: config})
        assert Config in registry
        assert len(registry) == 1
        assert registry.get(Config) is config

    def test_build_filters_none_values(self) -> None:
        config = ConcreteConfig(value=42)
        registry = ResourceRegistry.build({Config: config, HTTPClient: None})
        assert Config in registry
        assert HTTPClient not in registry
        assert len(registry) == 1

    def test_iter_with_instances_only(self) -> None:
        """Test iteration when registry has only instances (no bindings)."""
        config = ConcreteConfig()
        client = ConcreteHTTPClient(config=config)
        registry = ResourceRegistry.build({Config: config, HTTPClient: client})
        protocols = set(registry)
        assert protocols == {Config, HTTPClient}

    def test_iter_with_bindings_and_instances(self) -> None:
        """Test iteration yields both bindings and non-overlapping instances."""
        config = ConcreteConfig()
        # Create registry with a binding for Service and instance for Config
        base = ResourceRegistry.of(
            Binding(Service, lambda r: ConcreteService(r.get(HTTPClient)))
        )
        instances = ResourceRegistry.build({Config: config})
        merged = base.merge(instances)
        protocols = set(merged)
        # Should yield both Service (from binding) and Config (from instances)
        assert protocols == {Service, Config}

    def test_iter_skips_duplicate_protocols(self) -> None:
        """Test iteration doesn't yield the same protocol twice."""
        config = ConcreteConfig()
        # Create registry with a binding for Config and also an instance for Config
        bindings = ResourceRegistry.of(
            Binding(Config, lambda r: ConcreteConfig(value=1))
        )
        instances = ResourceRegistry.build({Config: config})
        merged = bindings.merge(instances)
        # Iterate and check Config only appears once
        protocols = list(merged)
        assert protocols.count(Config) == 1
        assert len(protocols) == 1

    def test_eager_bindings(self) -> None:
        registry = ResourceRegistry.of(
            Binding(Config, lambda r: ConcreteConfig(), eager=True),
            Binding(HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))),
        )
        eager = registry.eager_bindings()
        assert len(eager) == 1
        assert eager[0].protocol is Config

    def test_get_with_provider_binding(self) -> None:
        """registry.get() resolves provider bindings via context."""
        registry = ResourceRegistry.of(
            Binding(Config, lambda r: ConcreteConfig(value=42))
        )
        # get() should resolve the provider and return the constructed instance
        config = registry.get(Config)
        assert config is not None
        assert config.value == 42

    def test_get_all_with_predicate(self) -> None:
        """get_all() returns instances matching a predicate."""
        config = ConcreteConfig(value=10)
        registry = ResourceRegistry.build({Config: config})
        # Match instances where value > 5
        result = registry.get_all(
            lambda x: isinstance(x, ConcreteConfig) and x.value > 5
        )
        assert Config in result
        assert result[Config] is config

    def test_get_all_excludes_non_matching(self) -> None:
        """get_all() excludes instances that don't match."""
        config = ConcreteConfig(value=3)
        registry = ResourceRegistry.build({Config: config})
        # Match instances where value > 5 (config.value is 3)
        result = registry.get_all(
            lambda x: isinstance(x, ConcreteConfig) and x.value > 5
        )
        assert len(result) == 0

    def test_get_all_only_includes_eager_bindings(self) -> None:
        """get_all() only includes eager (instance) bindings, not lazy providers."""
        # Lazy provider binding - not resolved until accessed
        registry = ResourceRegistry.of(
            Binding(Config, lambda r: ConcreteConfig()),
        )
        result = registry.get_all(lambda x: isinstance(x, ConcreteConfig))
        # Lazy bindings aren't in singleton cache after start()
        assert len(result) == 0


# === ScopedResourceContext Tests ===


class TestScopedResourceContext:
    def test_get_constructs_lazily(self) -> None:
        constructed = []

        def make_config(r: ResourceResolver) -> ConcreteConfig:
            constructed.append("config")
            return ConcreteConfig()

        registry = ResourceRegistry.of(Binding(Config, make_config))
        ctx = registry.create_context()

        assert constructed == []
        _ = ctx.get(Config)
        assert constructed == ["config"]

    def test_get_caches_singleton(self) -> None:
        call_count = 0

        def make_config(r: ResourceResolver) -> ConcreteConfig:
            nonlocal call_count
            call_count += 1
            return ConcreteConfig()

        registry = ResourceRegistry.of(Binding(Config, make_config))
        ctx = registry.create_context()

        c1 = ctx.get(Config)
        c2 = ctx.get(Config)
        assert c1 is c2
        assert call_count == 1

    def test_get_returns_preconstructed_instance(self) -> None:
        """Context.get() returns pre-constructed instances from registry."""
        config = ConcreteConfig(value=99)
        registry = ResourceRegistry.build({Config: config})
        ctx = registry.create_context()

        # Should return the pre-constructed instance directly
        result = ctx.get(Config)
        assert result is config
        assert result.value == 99

    def test_get_resolves_dependencies(self) -> None:
        registry = ResourceRegistry.of(
            Binding(Config, lambda r: ConcreteConfig(value=99)),
            Binding(HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))),
        )
        ctx = registry.create_context()
        http = ctx.get(HTTPClient)
        assert http.config.value == 99

    def test_get_unbound_raises(self) -> None:
        registry = ResourceRegistry.of()
        ctx = registry.create_context()
        with pytest.raises(UnboundResourceError) as exc:
            ctx.get(Config)
        assert exc.value.protocol is Config

    def test_get_optional_returns_none(self) -> None:
        registry = ResourceRegistry.of()
        ctx = registry.create_context()
        assert ctx.get_optional(Config) is None

    def test_get_optional_returns_value(self) -> None:
        registry = ResourceRegistry.of(Binding(Config, lambda r: ConcreteConfig()))
        ctx = registry.create_context()
        assert ctx.get_optional(Config) is not None

    def test_circular_dependency_raises(self) -> None:
        @dataclass
        class A:
            b: object

        @dataclass
        class B:
            a: object

        registry = ResourceRegistry.of(
            Binding(A, lambda r: A(b=r.get(B))),
            Binding(B, lambda r: B(a=r.get(A))),
        )
        ctx = registry.create_context()
        with pytest.raises(CircularDependencyError) as exc:
            ctx.get(A)
        assert A in exc.value.cycle
        assert B in exc.value.cycle

    def test_provider_error_wrapped(self) -> None:
        def failing_provider(r: ResourceResolver) -> ConcreteConfig:
            raise ValueError("Bad config")

        registry = ResourceRegistry.of(Binding(Config, failing_provider))
        ctx = registry.create_context()
        with pytest.raises(ProviderError) as exc:
            ctx.get(Config)
        assert exc.value.protocol is Config
        assert isinstance(exc.value.cause, ValueError)


# === Lifecycle Tests ===


class TestLifecycle:
    def test_post_construct_called(self) -> None:
        registry = ResourceRegistry.of(
            Binding(PostConstructResource, lambda r: PostConstructResource())
        )
        ctx = registry.create_context()
        resource = ctx.get(PostConstructResource)
        assert resource.initialized is True

    def test_post_construct_failure_wrapped(self) -> None:
        registry = ResourceRegistry.of(
            Binding(FailingPostConstruct, lambda r: FailingPostConstruct())
        )
        ctx = registry.create_context()
        with pytest.raises(ProviderError) as exc:
            ctx.get(FailingPostConstruct)
        assert "Initialization failed" in str(exc.value.cause)

    def test_post_construct_failure_closes_resource(self) -> None:
        instances: list[CloseableFailingPostConstruct] = []

        def make(r: ResourceResolver) -> CloseableFailingPostConstruct:
            inst = CloseableFailingPostConstruct()
            instances.append(inst)
            return inst

        registry = ResourceRegistry.of(Binding(CloseableFailingPostConstruct, make))
        ctx = registry.create_context()
        with pytest.raises(ProviderError):
            ctx.get(CloseableFailingPostConstruct)
        assert len(instances) == 1
        assert instances[0].closed is True

    def test_close_disposes_singletons(self) -> None:
        registry = ResourceRegistry.of(
            Binding(CloseableResource, lambda r: CloseableResource())
        )
        ctx = registry.create_context()
        resource = ctx.get(CloseableResource)
        assert resource.closed is False
        ctx.close()
        assert resource.closed is True

    def test_close_reverse_order(self) -> None:
        closed_order: list[str] = []

        @dataclass
        class ResourceA:
            def close(self) -> None:
                closed_order.append("A")

        @dataclass
        class ResourceB:
            a: ResourceA

            def close(self) -> None:
                closed_order.append("B")

        registry = ResourceRegistry.of(
            Binding(ResourceA, lambda r: ResourceA()),
            Binding(ResourceB, lambda r: ResourceB(a=r.get(ResourceA))),
        )
        ctx = registry.create_context()
        _ = ctx.get(ResourceB)  # Constructs A, then B
        ctx.close()
        # B was instantiated after A, so B closes first
        assert closed_order == ["B", "A"]

    def test_start_instantiates_eager(self) -> None:
        constructed = []

        def make_config(r: ResourceResolver) -> ConcreteConfig:
            constructed.append("config")
            return ConcreteConfig()

        registry = ResourceRegistry.of(Binding(Config, make_config, eager=True))
        ctx = registry.create_context()
        assert constructed == []
        ctx.start()
        assert constructed == ["config"]

    def test_post_construct_failure_with_close_failure(self) -> None:
        """Test that close() failure during post_construct cleanup is logged."""

        @dataclass
        class FailingClose:
            def post_construct(self) -> None:
                raise RuntimeError("post_construct failed")

            def close(self) -> None:
                raise RuntimeError("close also failed")

        registry = ResourceRegistry.of(Binding(FailingClose, lambda r: FailingClose()))
        ctx = registry.create_context()
        with pytest.raises(ProviderError) as exc:
            ctx.get(FailingClose)
        assert "post_construct failed" in str(exc.value.cause)

    def test_close_skips_non_closeable_resources(self) -> None:
        """Test that close() skips resources that don't implement Closeable."""

        @dataclass
        class NonCloseableResource:
            pass

        @dataclass
        class CloseableResource:
            closed: bool = False

            def close(self) -> None:
                self.closed = True

        registry = ResourceRegistry.of(
            Binding(NonCloseableResource, lambda r: NonCloseableResource()),
            Binding(CloseableResource, lambda r: CloseableResource()),
        )
        ctx = registry.create_context()
        _ = ctx.get(NonCloseableResource)
        closeable = ctx.get(CloseableResource)

        ctx.close()  # Should not raise
        assert closeable.closed is True

    def test_close_with_failing_resource(self) -> None:
        """Test that close continues after a resource fails to close."""

        @dataclass
        class FailingCloseResource:
            def close(self) -> None:
                raise RuntimeError("close failed")

        @dataclass
        class GoodResource:
            closed: bool = False

            def close(self) -> None:
                self.closed = True

        registry = ResourceRegistry.of(
            Binding(GoodResource, lambda r: GoodResource()),
            Binding(FailingCloseResource, lambda r: FailingCloseResource()),
        )
        ctx = registry.create_context()
        good = ctx.get(GoodResource)
        _ = ctx.get(FailingCloseResource)

        # close() should not raise, but log the error
        ctx.close()
        assert good.closed is True

    def test_tool_scope_close_with_failing_resource(self) -> None:
        """Test that tool_scope cleanup continues after a resource fails to close."""

        @dataclass
        class FailingCloseTracer:
            def close(self) -> None:
                raise RuntimeError("close failed")

        registry = ResourceRegistry.of(
            Binding(
                FailingCloseTracer,
                lambda r: FailingCloseTracer(),
                scope=Scope.TOOL_CALL,
            )
        )
        ctx = registry.create_context()

        # Should not raise despite close() failure
        with ctx.tool_scope() as r:
            _ = r.get(FailingCloseTracer)


# === Scope Behavior Tests ===


class TestScopeBehavior:
    def test_prototype_never_cached(self) -> None:
        counter = itertools.count()

        @dataclass
        class Numbered:
            n: int

        registry = ResourceRegistry.of(
            Binding(
                Numbered, lambda r: Numbered(n=next(counter)), scope=Scope.PROTOTYPE
            )
        )
        ctx = registry.create_context()
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

        registry = ResourceRegistry.of(
            Binding(Config, make_config, scope=Scope.SINGLETON)
        )
        ctx = registry.create_context()

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

        registry = ResourceRegistry.of(
            Binding(Tracer, lambda r: Tracer(id=next(counter)), scope=Scope.TOOL_CALL)
        )
        ctx = registry.create_context()

        with ctx.tool_scope() as r1:
            t1 = r1.get(Tracer)
            t1_again = r1.get(Tracer)
            assert t1 is t1_again  # Same within scope

        with ctx.tool_scope() as r2:
            t2 = r2.get(Tracer)

        assert t1.id == 0
        assert t2.id == 1  # Fresh instance

    def test_tool_call_closed_on_scope_exit(self) -> None:
        registry = ResourceRegistry.of(
            Binding(
                CloseableResource, lambda r: CloseableResource(), scope=Scope.TOOL_CALL
            )
        )
        ctx = registry.create_context()

        with ctx.tool_scope() as r:
            resource = r.get(CloseableResource)
            assert resource.closed is False

        assert resource.closed is True

    def test_tool_scope_does_not_leak(self) -> None:
        counter = itertools.count()

        @dataclass
        class Tracer:
            id: int

        registry = ResourceRegistry.of(
            Binding(Tracer, lambda r: Tracer(id=next(counter)), scope=Scope.TOOL_CALL)
        )
        ctx = registry.create_context()

        # First scope
        with ctx.tool_scope() as r:
            _ = r.get(Tracer)

        # Second scope should start fresh
        with ctx.tool_scope() as r:
            t = r.get(Tracer)
            assert t.id == 1


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
        registry = ResourceRegistry.of(
            Binding(Config, lambda r: ConcreteConfig(value=100)),
            Binding(HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))),
            Binding(Service, lambda r: ConcreteService(r.get(HTTPClient))),
        )
        ctx = registry.create_context()
        service = ctx.get(Service)
        assert service.http.config.value == 100

    def test_mixed_scopes(self) -> None:
        counter = itertools.count()

        @dataclass
        class RequestId:
            id: int

        registry = ResourceRegistry.of(
            Binding(Config, lambda r: ConcreteConfig()),  # SINGLETON
            Binding(
                RequestId, lambda r: RequestId(id=next(counter)), scope=Scope.TOOL_CALL
            ),
        )
        ctx = registry.create_context()

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
        registry = ResourceRegistry.of(
            Binding(CloseableResource, lambda r: CloseableResource())
        )

        ctx = registry.create_context()
        ctx.start()
        try:
            resource = ctx.get(CloseableResource)
            assert resource.closed is False
        finally:
            ctx.close()

        assert resource.closed is True

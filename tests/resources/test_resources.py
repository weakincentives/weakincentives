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

    def test_binding_instance_creates_eager_singleton(self) -> None:
        """Binding.instance() creates an eager SINGLETON binding."""
        config = ConcreteConfig(value=99)
        binding = Binding.instance(Config, config)

        assert binding.protocol is Config
        assert binding.scope == Scope.SINGLETON
        assert binding.eager is True

    def test_binding_instance_resolved_on_start(self) -> None:
        """Instances from Binding.instance() are available immediately after start()."""
        config = ConcreteConfig(value=42)
        registry = ResourceRegistry.build({Config: config})

        ctx = registry._create_context()  # pyright: ignore[reportPrivateUsage]
        # Before start, nothing in cache
        assert Config not in ctx.singleton_cache

        ctx.start()
        # After start, instance is in cache
        assert Config in ctx.singleton_cache
        assert ctx.singleton_cache[Config] is config


# === ResourceRegistry Tests ===


class TestResourceRegistry:
    def test_empty_registry(self) -> None:
        registry = ResourceRegistry.build({})
        assert len(registry) == 0
        assert Config not in registry

    def test_single_binding(self) -> None:
        registry = ResourceRegistry.build(
            {Config: Binding(Config, lambda r: ConcreteConfig())}
        )
        assert len(registry) == 1
        assert Config in registry

    def test_multiple_bindings(self) -> None:
        registry = ResourceRegistry.build(
            {
                Config: Binding(Config, lambda r: ConcreteConfig()),
                HTTPClient: Binding(
                    HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))
                ),
            }
        )
        assert len(registry) == 2
        assert Config in registry
        assert HTTPClient in registry

    def test_duplicate_binding_raises(self) -> None:
        # Note: With dict-based build(), duplicates are detected at dict level
        # (last value wins) so we test with same key twice using update
        base = ResourceRegistry.build(
            {Config: Binding(Config, lambda r: ConcreteConfig())}
        )
        override = ResourceRegistry.build(
            {Config: Binding(Config, lambda r: ConcreteConfig(value=99))}
        )
        with pytest.raises(DuplicateBindingError) as exc:
            base.merge(override, strict=True)
        assert exc.value.protocol is Config

    def test_build_raises_on_duplicate_from_custom_mapping(self) -> None:
        """build() raises DuplicateBindingError if mapping yields same key twice."""
        from collections.abc import Iterator, Mapping

        class DuplicateKeyMapping(Mapping[type[object], object]):
            """Custom Mapping that yields same key twice during iteration."""

            def __getitem__(self, key: type[object]) -> object:
                if key is Config:
                    return ConcreteConfig()
                raise KeyError(key)

            def __len__(self) -> int:
                return 1

            def __iter__(self) -> Iterator[type[object]]:
                # Yield Config twice to trigger duplicate detection
                yield Config
                yield Config

        with pytest.raises(DuplicateBindingError) as exc:
            ResourceRegistry.build(DuplicateKeyMapping())
        assert exc.value.protocol is Config

    def test_binding_for(self) -> None:
        binding = Binding(Config, lambda r: ConcreteConfig())
        registry = ResourceRegistry.build({Config: binding})
        assert registry.binding_for(Config) is binding
        assert registry.binding_for(HTTPClient) is None

    def test_merge(self) -> None:
        base = ResourceRegistry.build(
            {Config: Binding(Config, lambda r: ConcreteConfig(value=1))}
        )
        override = ResourceRegistry.build(
            {Config: Binding(Config, lambda r: ConcreteConfig(value=2))}
        )
        merged = base.merge(override, strict=False)

        with merged.open() as ctx:
            config = ctx.get(Config)
            assert config.value == 2

    def test_merge_adds_new(self) -> None:
        base = ResourceRegistry.build(
            {Config: Binding(Config, lambda r: ConcreteConfig())}
        )
        additional = ResourceRegistry.build(
            {
                HTTPClient: Binding(
                    HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))
                )
            }
        )
        merged = base.merge(additional)
        assert Config in merged
        assert HTTPClient in merged

    def test_iter(self) -> None:
        registry = ResourceRegistry.build(
            {
                Config: Binding(Config, lambda r: ConcreteConfig()),
                HTTPClient: Binding(
                    HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))
                ),
            }
        )
        protocols = set(registry)
        assert protocols == {Config, HTTPClient}

    def test_build_creates_registry_with_instances(self) -> None:
        config = ConcreteConfig(value=42)
        registry = ResourceRegistry.build({Config: config})
        assert Config in registry
        assert len(registry) == 1
        with registry.open() as ctx:
            assert ctx.get(Config) is config

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
        base = ResourceRegistry.build(
            {Service: Binding(Service, lambda r: ConcreteService(r.get(HTTPClient)))}
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
        bindings = ResourceRegistry.build(
            {Config: Binding(Config, lambda r: ConcreteConfig(value=1))}
        )
        instances = ResourceRegistry.build({Config: config})
        merged = bindings.merge(instances, strict=False)
        # Iterate and check Config only appears once
        protocols = list(merged)
        assert protocols.count(Config) == 1
        assert len(protocols) == 1

    def test_eager_bindings(self) -> None:
        registry = ResourceRegistry.build(
            {
                Config: Binding(Config, lambda r: ConcreteConfig(), eager=True),
                HTTPClient: Binding(
                    HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))
                ),
            }
        )
        eager = registry.eager_bindings()
        assert len(eager) == 1
        assert eager[0].protocol is Config

    def test_conflicts_returns_shared_protocols(self) -> None:
        """conflicts() returns protocols bound in both registries."""
        base = ResourceRegistry.build(
            {
                Config: Binding(Config, lambda r: ConcreteConfig()),
                HTTPClient: Binding(
                    HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))
                ),
            }
        )
        override = ResourceRegistry.build(
            {
                Config: Binding(Config, lambda r: ConcreteConfig(value=2)),  # Conflicts
            }
        )
        conflicts = base.conflicts(override)
        assert conflicts == frozenset({Config})
        assert HTTPClient not in conflicts

    def test_conflicts_returns_empty_when_disjoint(self) -> None:
        """conflicts() returns empty set when no protocols overlap."""
        base = ResourceRegistry.build(
            {Config: Binding(Config, lambda r: ConcreteConfig())}
        )
        other = ResourceRegistry.build(
            {
                HTTPClient: Binding(
                    HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))
                )
            }
        )
        conflicts = base.conflicts(other)
        assert len(conflicts) == 0

    def test_merge_strict_raises_on_conflict(self) -> None:
        """merge(strict=True) raises DuplicateBindingError on conflict."""
        base = ResourceRegistry.build(
            {Config: Binding(Config, lambda r: ConcreteConfig())}
        )
        override = ResourceRegistry.build(
            {Config: Binding(Config, lambda r: ConcreteConfig(value=2))}
        )
        with pytest.raises(DuplicateBindingError) as exc:
            base.merge(override, strict=True)
        assert exc.value.protocol is Config

    def test_merge_strict_allows_disjoint_registries(self) -> None:
        """merge(strict=True) succeeds when registries are disjoint."""
        base = ResourceRegistry.build(
            {Config: Binding(Config, lambda r: ConcreteConfig())}
        )
        other = ResourceRegistry.build(
            {
                HTTPClient: Binding(
                    HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))
                )
            }
        )
        # Should not raise
        merged = base.merge(other, strict=True)
        assert Config in merged
        assert HTTPClient in merged


# === ScopedResourceContext Tests ===


class TestScopedResourceContext:
    def test_get_constructs_lazily(self) -> None:
        constructed = []

        def make_config(r: ResourceResolver) -> ConcreteConfig:
            constructed.append("config")
            return ConcreteConfig()

        registry = ResourceRegistry.build({Config: Binding(Config, make_config)})

        with registry.open() as ctx:
            assert constructed == []
            _ = ctx.get(Config)
            assert constructed == ["config"]

    def test_get_caches_singleton(self) -> None:
        call_count = 0

        def make_config(r: ResourceResolver) -> ConcreteConfig:
            nonlocal call_count
            call_count += 1
            return ConcreteConfig()

        registry = ResourceRegistry.build({Config: Binding(Config, make_config)})

        with registry.open() as ctx:
            c1 = ctx.get(Config)
            c2 = ctx.get(Config)
            assert c1 is c2
            assert call_count == 1

    def test_get_returns_preconstructed_instance(self) -> None:
        """Context.get() returns pre-constructed instances from registry."""
        config = ConcreteConfig(value=99)
        registry = ResourceRegistry.build({Config: config})

        with registry.open() as ctx:
            # Should return the pre-constructed instance directly
            result = ctx.get(Config)
            assert result is config
            assert result.value == 99

    def test_get_resolves_dependencies(self) -> None:
        registry = ResourceRegistry.build(
            {
                Config: Binding(Config, lambda r: ConcreteConfig(value=99)),
                HTTPClient: Binding(
                    HTTPClient, lambda r: ConcreteHTTPClient(r.get(Config))
                ),
            }
        )
        with registry.open() as ctx:
            http = ctx.get(HTTPClient)
            assert http.config.value == 99

    def test_get_unbound_raises(self) -> None:
        registry = ResourceRegistry.build({})
        with registry.open() as ctx:
            with pytest.raises(UnboundResourceError) as exc:
                ctx.get(Config)
            assert exc.value.protocol is Config

    def test_get_optional_returns_none(self) -> None:
        registry = ResourceRegistry.build({})
        with registry.open() as ctx:
            assert ctx.get_optional(Config) is None

    def test_get_optional_returns_value(self) -> None:
        registry = ResourceRegistry.build(
            {Config: Binding(Config, lambda r: ConcreteConfig())}
        )
        with registry.open() as ctx:
            assert ctx.get_optional(Config) is not None

    def test_circular_dependency_raises(self) -> None:
        @dataclass
        class A:
            b: object

        @dataclass
        class B:
            a: object

        registry = ResourceRegistry.build(
            {
                A: Binding(A, lambda r: A(b=r.get(B))),
                B: Binding(B, lambda r: B(a=r.get(A))),
            }
        )
        with registry.open() as ctx:
            with pytest.raises(CircularDependencyError) as exc:
                ctx.get(A)
            assert A in exc.value.cycle
            assert B in exc.value.cycle

    def test_provider_error_wrapped(self) -> None:
        def failing_provider(r: ResourceResolver) -> ConcreteConfig:
            raise ValueError("Bad config")

        registry = ResourceRegistry.build({Config: Binding(Config, failing_provider)})
        with registry.open() as ctx:
            with pytest.raises(ProviderError) as exc:
                ctx.get(Config)
            assert exc.value.protocol is Config
            assert isinstance(exc.value.cause, ValueError)


# === Lifecycle Tests ===


class TestLifecycle:
    def test_post_construct_called(self) -> None:
        registry = ResourceRegistry.build(
            {
                PostConstructResource: Binding(
                    PostConstructResource, lambda r: PostConstructResource()
                )
            }
        )
        with registry.open() as ctx:
            resource = ctx.get(PostConstructResource)
            assert resource.initialized is True

    def test_post_construct_failure_wrapped(self) -> None:
        registry = ResourceRegistry.build(
            {
                FailingPostConstruct: Binding(
                    FailingPostConstruct, lambda r: FailingPostConstruct()
                )
            }
        )
        with registry.open() as ctx:
            with pytest.raises(ProviderError) as exc:
                ctx.get(FailingPostConstruct)
            assert "Initialization failed" in str(exc.value.cause)

    def test_post_construct_failure_closes_resource(self) -> None:
        instances: list[CloseableFailingPostConstruct] = []

        def make(r: ResourceResolver) -> CloseableFailingPostConstruct:
            inst = CloseableFailingPostConstruct()
            instances.append(inst)
            return inst

        registry = ResourceRegistry.build(
            {
                CloseableFailingPostConstruct: Binding(
                    CloseableFailingPostConstruct, make
                )
            }
        )
        with registry.open() as ctx:
            with pytest.raises(ProviderError):
                ctx.get(CloseableFailingPostConstruct)
            assert len(instances) == 1
            assert instances[0].closed is True

    def test_close_disposes_singletons(self) -> None:
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
        # Context exits, close() is called
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

        registry = ResourceRegistry.build(
            {
                ResourceA: Binding(ResourceA, lambda r: ResourceA()),
                ResourceB: Binding(ResourceB, lambda r: ResourceB(a=r.get(ResourceA))),
            }
        )
        with registry.open() as ctx:
            _ = ctx.get(ResourceB)  # Constructs A, then B
        # B was instantiated after A, so B closes first
        assert closed_order == ["B", "A"]

    def test_start_instantiates_eager(self) -> None:
        constructed = []

        def make_config(r: ResourceResolver) -> ConcreteConfig:
            constructed.append("config")
            return ConcreteConfig()

        registry = ResourceRegistry.build(
            {Config: Binding(Config, make_config, eager=True)}
        )
        # Use _create_context to test start() behavior explicitly
        ctx = registry._create_context()  # pyright: ignore[reportPrivateUsage]
        assert constructed == []
        ctx.start()
        assert constructed == ["config"]
        ctx.close()

    def test_post_construct_failure_with_close_failure(self) -> None:
        """Test that close() failure during post_construct cleanup is logged."""

        @dataclass
        class FailingClose:
            def post_construct(self) -> None:
                raise RuntimeError("post_construct failed")

            def close(self) -> None:
                raise RuntimeError("close also failed")

        registry = ResourceRegistry.build(
            {FailingClose: Binding(FailingClose, lambda r: FailingClose())}
        )
        with registry.open() as ctx:
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

        registry = ResourceRegistry.build(
            {
                NonCloseableResource: Binding(
                    NonCloseableResource, lambda r: NonCloseableResource()
                ),
                CloseableResource: Binding(
                    CloseableResource, lambda r: CloseableResource()
                ),
            }
        )
        with registry.open() as ctx:
            _ = ctx.get(NonCloseableResource)
            closeable = ctx.get(CloseableResource)
        # close() should not raise
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

        registry = ResourceRegistry.build(
            {
                GoodResource: Binding(GoodResource, lambda r: GoodResource()),
                FailingCloseResource: Binding(
                    FailingCloseResource, lambda r: FailingCloseResource()
                ),
            }
        )
        with registry.open() as ctx:
            good = ctx.get(GoodResource)
            _ = ctx.get(FailingCloseResource)
        # close() should not raise, but log the error
        assert good.closed is True

    def test_tool_scope_close_with_failing_resource(self) -> None:
        """Test that tool_scope cleanup continues after a resource fails to close."""

        @dataclass
        class FailingCloseTracer:
            def close(self) -> None:
                raise RuntimeError("close failed")

        registry = ResourceRegistry.build(
            {
                FailingCloseTracer: Binding(
                    FailingCloseTracer,
                    lambda r: FailingCloseTracer(),
                    scope=Scope.TOOL_CALL,
                )
            }
        )
        with registry.open() as ctx:
            # Should not raise despite close() failure
            with ctx.tool_scope() as r:
                _ = r.get(FailingCloseTracer)

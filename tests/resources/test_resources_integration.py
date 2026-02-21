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

"""Tests for error messages and integration scenarios."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

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

from .conftest import (
    CloseableResource,
    ConcreteConfig,
    ConcreteHTTPClient,
    ConcreteService,
    Config,
    HTTPClient,
    Service,
)

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
        """Test resolving a 5-level dependency chain: A -> B -> C -> D -> E."""

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
        """Test circular dependency detection in deep chain: A -> B -> C -> A."""

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

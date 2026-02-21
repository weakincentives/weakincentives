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

"""Tests for ScopedResourceContext."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.resources import (
    Binding,
    CircularDependencyError,
    ProviderError,
    ResourceRegistry,
    ResourceResolver,
    UnboundResourceError,
)

from .conftest import (
    ConcreteConfig,
    ConcreteHTTPClient,
    Config,
    HTTPClient,
)

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

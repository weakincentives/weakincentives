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

"""Tests for resource lifecycle (post_construct, close, eager instantiation)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.resources import (
    Binding,
    ProviderError,
    ResourceRegistry,
    ResourceResolver,
    Scope,
)

from .conftest import (
    CloseableFailingPostConstruct,
    CloseableResource,
    ConcreteConfig,
    Config,
    FailingPostConstruct,
    PostConstructResource,
)

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

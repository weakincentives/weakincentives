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

"""Tests for ResourceRegistry."""

from __future__ import annotations

from dataclasses import dataclass

from weakincentives.resources import ResourceRegistry


@dataclass
class MockResource:
    """A mock resource for testing."""

    value: str


@dataclass
class AnotherResource:
    """Another mock resource for testing."""

    count: int


class TestResourceRegistryContains:
    """Tests for ResourceRegistry.__contains__ method."""

    def test_contains_true_when_present(self) -> None:
        """Test that __contains__ returns True when resource is registered."""
        resource = MockResource(value="test")
        registry = ResourceRegistry.build({MockResource: resource})
        assert MockResource in registry

    def test_contains_false_when_absent(self) -> None:
        """Test that __contains__ returns False when resource is not registered."""
        registry = ResourceRegistry()
        assert MockResource not in registry


class TestResourceRegistryFromMapping:
    """Tests for ResourceRegistry.build factory method."""

    def test_build_stores_by_explicit_type(self) -> None:
        """Test that build stores by the mapping key type."""
        resource = MockResource(value="test")
        registry = ResourceRegistry.build({MockResource: resource})
        with registry.open() as ctx:
            assert ctx.get(MockResource) is resource

    def test_build_skips_none_values(self) -> None:
        """Test that build filters out None values."""
        resource = MockResource(value="test")
        registry = ResourceRegistry.build(
            {
                MockResource: resource,
                AnotherResource: None,
            }
        )
        with registry.open() as ctx:
            assert ctx.get(MockResource) is resource
            assert ctx.get_optional(AnotherResource) is None


class TestResourceRegistryMerge:
    """Tests for ResourceRegistry.merge method."""

    def test_merge_combines_disjoint_registries(self) -> None:
        """Test that merge combines resources from both registries."""
        resource1 = MockResource(value="first")
        resource2 = AnotherResource(count=42)
        registry1 = ResourceRegistry.build({MockResource: resource1})
        registry2 = ResourceRegistry.build({AnotherResource: resource2})

        merged = registry1.merge(registry2)

        with merged.open() as ctx:
            assert ctx.get(MockResource) is resource1
            assert ctx.get(AnotherResource) is resource2

    def test_merge_other_takes_precedence(self) -> None:
        """Test that 'other' registry values take precedence on conflicts."""
        resource1 = MockResource(value="first")
        resource2 = MockResource(value="second")
        registry1 = ResourceRegistry.build({MockResource: resource1})
        registry2 = ResourceRegistry.build({MockResource: resource2})

        merged = registry1.merge(registry2, strict=False)

        # The 'other' registry (registry2) should win
        with merged.open() as ctx:
            result = ctx.get(MockResource)
            assert result is resource2
            assert result.value == "second"

    def test_merge_does_not_mutate_original(self) -> None:
        """Test that merge returns a new registry without mutating originals."""
        resource1 = MockResource(value="first")
        resource2 = AnotherResource(count=42)
        registry1 = ResourceRegistry.build({MockResource: resource1})
        registry2 = ResourceRegistry.build({AnotherResource: resource2})

        merged = registry1.merge(registry2)

        # Original registries should be unchanged
        with registry1.open() as ctx1:
            assert ctx1.get_optional(AnotherResource) is None

        with registry2.open() as ctx2:
            assert ctx2.get_optional(MockResource) is None

        # Merged should have both
        with merged.open() as ctx:
            assert ctx.get(MockResource) is resource1
            assert ctx.get(AnotherResource) is resource2

    def test_merge_with_empty_registry(self) -> None:
        """Test merging with an empty registry."""
        resource = MockResource(value="test")
        registry = ResourceRegistry.build({MockResource: resource})
        empty = ResourceRegistry()

        # Merge non-empty with empty
        merged1 = registry.merge(empty)
        with merged1.open() as ctx1:
            assert ctx1.get(MockResource) is resource

        # Merge empty with non-empty
        merged2 = empty.merge(registry)
        with merged2.open() as ctx2:
            assert ctx2.get(MockResource) is resource

    def test_merge_both_empty(self) -> None:
        """Test merging two empty registries."""
        empty1 = ResourceRegistry()
        empty2 = ResourceRegistry()

        merged = empty1.merge(empty2)

        with merged.open() as ctx:
            assert ctx.get_optional(MockResource) is None

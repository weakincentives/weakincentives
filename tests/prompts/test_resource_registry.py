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


class TestResourceRegistryGet:
    """Tests for ResourceRegistry.get method."""

    def test_get_returns_resource_when_present(self) -> None:
        """Test that get returns the resource when it's registered."""
        resource = MockResource(value="test")
        registry = ResourceRegistry.build({MockResource: resource})
        assert registry.get(MockResource) is resource

    def test_get_returns_none_when_absent(self) -> None:
        """Test that get returns None when resource is not registered."""
        registry = ResourceRegistry()
        assert registry.get(MockResource) is None

    def test_get_with_default_when_absent(self) -> None:
        """Test that get returns default when resource is absent."""
        default = MockResource(value="default")
        registry = ResourceRegistry()
        assert registry.get(MockResource, default) is default

    def test_get_ignores_default_when_present(self) -> None:
        """Test that get ignores default when resource is present."""
        resource = MockResource(value="actual")
        default = MockResource(value="default")
        registry = ResourceRegistry.build({MockResource: resource})
        assert registry.get(MockResource, default) is resource


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
        assert registry.get(MockResource) is resource

    def test_build_skips_none_values(self) -> None:
        """Test that build filters out None values."""
        resource = MockResource(value="test")
        registry = ResourceRegistry.build(
            {
                MockResource: resource,
                AnotherResource: None,
            }
        )
        assert registry.get(MockResource) is resource
        assert registry.get(AnotherResource) is None


class TestResourceRegistryMerge:
    """Tests for ResourceRegistry.merge method."""

    def test_merge_combines_disjoint_registries(self) -> None:
        """Test that merge combines resources from both registries."""
        resource1 = MockResource(value="first")
        resource2 = AnotherResource(count=42)
        registry1 = ResourceRegistry.build({MockResource: resource1})
        registry2 = ResourceRegistry.build({AnotherResource: resource2})

        merged = registry1.merge(registry2)

        assert merged.get(MockResource) is resource1
        assert merged.get(AnotherResource) is resource2

    def test_merge_other_takes_precedence(self) -> None:
        """Test that 'other' registry values take precedence on conflicts."""
        resource1 = MockResource(value="first")
        resource2 = MockResource(value="second")
        registry1 = ResourceRegistry.build({MockResource: resource1})
        registry2 = ResourceRegistry.build({MockResource: resource2})

        merged = registry1.merge(registry2)

        # The 'other' registry (registry2) should win
        assert merged.get(MockResource) is resource2
        assert merged.get(MockResource).value == "second"

    def test_merge_does_not_mutate_original(self) -> None:
        """Test that merge returns a new registry without mutating originals."""
        resource1 = MockResource(value="first")
        resource2 = AnotherResource(count=42)
        registry1 = ResourceRegistry.build({MockResource: resource1})
        registry2 = ResourceRegistry.build({AnotherResource: resource2})

        merged = registry1.merge(registry2)

        # Original registries should be unchanged
        assert registry1.get(AnotherResource) is None
        assert registry2.get(MockResource) is None
        # Merged should have both
        assert merged.get(MockResource) is resource1
        assert merged.get(AnotherResource) is resource2

    def test_merge_with_empty_registry(self) -> None:
        """Test merging with an empty registry."""
        resource = MockResource(value="test")
        registry = ResourceRegistry.build({MockResource: resource})
        empty = ResourceRegistry()

        # Merge non-empty with empty
        merged1 = registry.merge(empty)
        assert merged1.get(MockResource) is resource

        # Merge empty with non-empty
        merged2 = empty.merge(registry)
        assert merged2.get(MockResource) is resource

    def test_merge_both_empty(self) -> None:
        """Test merging two empty registries."""
        empty1 = ResourceRegistry()
        empty2 = ResourceRegistry()

        merged = empty1.merge(empty2)

        assert merged.get(MockResource) is None

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

from weakincentives.prompt.tool import ResourceRegistry


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
        registry = ResourceRegistry.from_mapping({MockResource: resource})
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
        registry = ResourceRegistry.from_mapping({MockResource: resource})
        assert registry.get(MockResource, default) is resource


class TestResourceRegistryContains:
    """Tests for ResourceRegistry.__contains__ method."""

    def test_contains_true_when_present(self) -> None:
        """Test that __contains__ returns True when resource is registered."""
        resource = MockResource(value="test")
        registry = ResourceRegistry.from_mapping({MockResource: resource})
        assert MockResource in registry

    def test_contains_false_when_absent(self) -> None:
        """Test that __contains__ returns False when resource is not registered."""
        registry = ResourceRegistry()
        assert MockResource not in registry


class TestResourceRegistryBuild:
    """Tests for ResourceRegistry.build factory method."""

    def test_build_with_single_resource(self) -> None:
        """Test build with a single resource stores by concrete type."""
        resource = MockResource(value="test")
        registry = ResourceRegistry.build(my_resource=resource)
        # build() stores by type(value), so lookup by concrete type works
        assert registry.get(MockResource) is resource

    def test_build_with_multiple_resources(self) -> None:
        """Test build with multiple resources."""
        resource1 = MockResource(value="first")
        resource2 = AnotherResource(count=42)
        registry = ResourceRegistry.build(resource1=resource1, resource2=resource2)
        assert registry.get(MockResource) is resource1
        assert registry.get(AnotherResource) is resource2

    def test_build_skips_none_values(self) -> None:
        """Test that build skips None values."""
        resource = MockResource(value="test")
        registry = ResourceRegistry.build(
            present=resource,
            absent=None,
        )
        assert registry.get(MockResource) is resource
        assert AnotherResource not in registry

    def test_build_with_no_resources(self) -> None:
        """Test build with no resources creates empty registry."""
        registry = ResourceRegistry.build()
        assert MockResource not in registry


class TestResourceRegistryFromMapping:
    """Tests for ResourceRegistry.from_mapping factory method."""

    def test_from_mapping_stores_by_explicit_type(self) -> None:
        """Test that from_mapping stores by the mapping key type."""
        resource = MockResource(value="test")
        registry = ResourceRegistry.from_mapping({MockResource: resource})
        assert registry.get(MockResource) is resource

    def test_from_mapping_skips_none_values(self) -> None:
        """Test that from_mapping filters out None values."""
        resource = MockResource(value="test")
        registry = ResourceRegistry.from_mapping(
            {
                MockResource: resource,
                AnotherResource: None,
            }
        )
        assert registry.get(MockResource) is resource
        assert registry.get(AnotherResource) is None

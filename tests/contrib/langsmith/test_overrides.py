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

"""Tests for LangSmithPromptOverridesStore."""

from __future__ import annotations

import pytest

from weakincentives.contrib.langsmith import (
    LangSmithConfig,
    LangSmithPromptOverridesStore,
)
from weakincentives.contrib.langsmith.testing import MockLangSmithHub
from weakincentives.prompt.overrides.versioning import (
    HexDigest,
    PromptDescriptor,
    PromptOverride,
    SectionDescriptor,
    SectionOverride,
)


@pytest.fixture
def mock_hub() -> MockLangSmithHub:
    """Create a mock Hub client."""
    return MockLangSmithHub()


@pytest.fixture
def config() -> LangSmithConfig:
    """Create test config."""
    return LangSmithConfig(
        api_key="test-key",
        project="test-project",
        hub_enabled=True,
        cache_ttl_seconds=0.0,  # Disable caching for tests
    )


@pytest.fixture
def store(
    config: LangSmithConfig, mock_hub: MockLangSmithHub
) -> LangSmithPromptOverridesStore:
    """Create store with mock hub."""
    return LangSmithPromptOverridesStore(config, client=mock_hub)


@pytest.fixture
def descriptor() -> PromptDescriptor:
    """Create test descriptor."""
    return PromptDescriptor(
        ns="test-ns",
        key="test-key",
        sections=[
            SectionDescriptor(
                path=("main",),
                content_hash=HexDigest("a" * 64),  # Valid 64-char hex
                number="1",
            )
        ],
        tools=[],
    )


class TestLangSmithPromptOverridesStore:
    """Tests for LangSmithPromptOverridesStore."""

    def test_resolve_returns_none_for_missing(
        self,
        store: LangSmithPromptOverridesStore,
        descriptor: PromptDescriptor,
    ) -> None:
        """resolve returns None when prompt not in hub."""
        result = store.resolve(descriptor, tag="latest")
        assert result is None

    def test_resolve_returns_override_from_hub(
        self,
        store: LangSmithPromptOverridesStore,
        mock_hub: MockLangSmithHub,
        descriptor: PromptDescriptor,
    ) -> None:
        """resolve returns override when prompt exists in hub."""
        mock_hub.add_prompt(
            "test-ns-test-key",
            "Template content from Hub",
        )

        result = store.resolve(descriptor, tag="latest")

        assert result is not None
        assert result.ns == "test-ns"
        assert result.prompt_key == "test-key"

    def test_resolve_caches_result(
        self,
        mock_hub: MockLangSmithHub,
        descriptor: PromptDescriptor,
    ) -> None:
        """resolve caches results for non-latest tags."""
        config = LangSmithConfig(
            api_key="test",
            hub_enabled=True,
            cache_ttl_seconds=300.0,  # Enable caching
        )
        store = LangSmithPromptOverridesStore(config, client=mock_hub)

        mock_hub.add_prompt("test-ns-test-key", "Template")

        # First call
        store.resolve(descriptor, tag="v1")
        # Second call (should use cache)
        store.resolve(descriptor, tag="v1")

        # Only one pull call should be made
        assert len(mock_hub.pull_calls) == 1

    def test_resolve_skips_cache_for_latest(
        self,
        mock_hub: MockLangSmithHub,
        descriptor: PromptDescriptor,
    ) -> None:
        """resolve bypasses cache for 'latest' tag."""
        config = LangSmithConfig(
            api_key="test",
            hub_enabled=True,
            cache_ttl_seconds=300.0,
        )
        store = LangSmithPromptOverridesStore(config, client=mock_hub)

        mock_hub.add_prompt("test-ns-test-key", "Template")

        # Multiple calls with "latest"
        store.resolve(descriptor, tag="latest")
        store.resolve(descriptor, tag="latest")

        # Both calls should hit the hub
        assert len(mock_hub.pull_calls) == 2

    def test_upsert_pushes_to_hub(
        self,
        store: LangSmithPromptOverridesStore,
        mock_hub: MockLangSmithHub,
        descriptor: PromptDescriptor,
    ) -> None:
        """upsert pushes override to hub."""
        override = PromptOverride(
            ns="test-ns",
            prompt_key="test-key",
            tag="latest",
            sections={
                ("main",): SectionOverride(
                    expected_hash=HexDigest("a" * 64),
                    body="Updated template",
                )
            },
        )

        result = store.upsert(descriptor, override)

        assert len(mock_hub.push_calls) == 1
        # Result should have updated tag (commit hash)
        assert len(result.tag) == 64  # SHA-256 hash

    def test_upsert_invalidates_cache(
        self,
        mock_hub: MockLangSmithHub,
        descriptor: PromptDescriptor,
    ) -> None:
        """upsert invalidates cached entries."""
        config = LangSmithConfig(
            api_key="test",
            hub_enabled=True,
            cache_ttl_seconds=300.0,
        )
        store = LangSmithPromptOverridesStore(config, client=mock_hub)

        mock_hub.add_prompt("test-ns-test-key", "Original")

        # Populate cache
        store.resolve(descriptor, tag="v1")

        # Upsert should invalidate
        override = PromptOverride(
            ns="test-ns",
            prompt_key="test-key",
            tag="v1",
        )
        store.upsert(descriptor, override)

        # Next resolve should hit hub again
        store.resolve(descriptor, tag="v1")

        # Should have 2 pull calls (initial + after invalidation)
        assert len(mock_hub.pull_calls) == 2

    def test_delete_invalidates_cache(
        self,
        store: LangSmithPromptOverridesStore,
        mock_hub: MockLangSmithHub,
        descriptor: PromptDescriptor,
    ) -> None:
        """delete removes cached entry."""
        mock_hub.add_prompt("test-ns-test-key", "Template")

        # Populate cache with explicit tag
        config = LangSmithConfig(
            api_key="test",
            hub_enabled=True,
            cache_ttl_seconds=300.0,
        )
        store = LangSmithPromptOverridesStore(config, client=mock_hub)

        store.resolve(descriptor, tag="v1")
        store.delete(ns="test-ns", prompt_key="test-key", tag="v1")

        # Cache should be cleared
        # Next resolve would need to hit hub again

    def test_pull_without_descriptor(
        self,
        store: LangSmithPromptOverridesStore,
        mock_hub: MockLangSmithHub,
    ) -> None:
        """pull works without a full descriptor."""
        mock_hub.add_prompt("test-ns-test-key", "Template")

        result = store.pull(ns="test-ns", prompt_key="test-key")

        assert result is not None

    def test_push_returns_commit_hash(
        self,
        store: LangSmithPromptOverridesStore,
        mock_hub: MockLangSmithHub,
    ) -> None:
        """push returns commit hash."""

        # Create minimal prompt-like object
        class MinimalPrompt:
            ns = "test-ns"
            key = "test-key"

            @property
            def sections(self) -> tuple[object, ...]:
                return ()

        prompt = MinimalPrompt()
        commit_hash = store.push(prompt, tag="latest")

        assert len(commit_hash) == 64
        assert len(mock_hub.push_calls) == 1

    def test_hub_disabled_falls_back(
        self,
        descriptor: PromptDescriptor,
    ) -> None:
        """Disabled hub falls back to fallback store."""

        class FallbackStore:
            def resolve(
                self, descriptor: PromptDescriptor, tag: str = "latest"
            ) -> PromptOverride | None:
                return PromptOverride(
                    ns=descriptor.ns,
                    prompt_key=descriptor.key,
                    tag=tag,
                )

            def upsert(
                self, descriptor: PromptDescriptor, override: PromptOverride
            ) -> PromptOverride:
                return override

        config = LangSmithConfig(hub_enabled=False)
        store = LangSmithPromptOverridesStore(config, fallback_store=FallbackStore())

        result = store.resolve(descriptor, tag="latest")
        assert result is not None
        assert result.ns == "test-ns"

    def test_hub_disabled_no_fallback_returns_none(
        self,
        descriptor: PromptDescriptor,
    ) -> None:
        """Disabled hub with no fallback returns None."""
        config = LangSmithConfig(hub_enabled=False)
        store = LangSmithPromptOverridesStore(config)

        result = store.resolve(descriptor, tag="latest")
        assert result is None

    def test_hub_name_generation(
        self,
        store: LangSmithPromptOverridesStore,
    ) -> None:
        """Hub prompt names are generated correctly."""
        name = store._hub_prompt_name("my-ns", "my-key")
        assert name == "my-ns-my-key"

        # Slashes are replaced
        name = store._hub_prompt_name("ns/sub", "key/path")
        assert name == "ns-sub-key-path"

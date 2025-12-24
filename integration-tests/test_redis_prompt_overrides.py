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

"""Integration tests for RedisPromptOverridesStore.

These tests require a running Redis instance and use the redis_utils
context managers to start containerized Redis instances.

Run with:
    uv run pytest integration-tests/test_redis_prompt_overrides.py -v
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from redis_utils import (
    REDIS_CLUSTER_TESTS_ENABLED,
    redis_cluster,
    redis_standalone,
    skip_if_no_redis,
)

from weakincentives.contrib.prompt.overrides import (
    RedisPromptOverridesError,
    RedisPromptOverridesStore,
)
from weakincentives.prompt import MarkdownSection, PromptTemplate, Tool
from weakincentives.prompt.overrides import (
    HexDigest,
    PromptDescriptor,
    PromptOverride,
    SectionOverride,
    ToolOverride,
)


skip_reason = skip_if_no_redis()
pytestmark = pytest.mark.skipif(bool(skip_reason), reason=skip_reason or "Redis unavailable")


@dataclass
class _GreetingParams:
    subject: str


@dataclass
class _ToolParams:
    query: str = field(metadata={"description": "User provided keywords."})


@dataclass
class _ToolResult:
    result: str


def _build_prompt() -> PromptTemplate:
    return PromptTemplate(
        ns="integration/tests",
        key="redis-greeting",
        sections=[
            MarkdownSection[_GreetingParams](
                title="Greeting",
                template="Greet ${subject} warmly.",
                key="greeting",
            )
        ],
    )


def _build_prompt_with_tool() -> PromptTemplate:
    tool = Tool[_ToolParams, _ToolResult](
        name="search",
        description="Search stored notes.",
        handler=None,
    )
    return PromptTemplate(
        ns="integration/tests",
        key="redis-greeting-tools",
        sections=[
            MarkdownSection[_GreetingParams](
                title="Greeting",
                template="Greet ${subject} warmly.",
                key="greeting",
                tools=[tool],
            )
        ],
    )


OTHER_DIGEST = HexDigest("b" * 64)


class TestRedisStandalone:
    """Integration tests with standalone Redis."""

    def test_upsert_resolve_delete_roundtrip(self) -> None:
        with redis_standalone() as client:
            store = RedisPromptOverridesStore(client=client)
            prompt = _build_prompt()
            descriptor = PromptDescriptor.from_prompt(prompt)
            section = descriptor.sections[0]

            # Initially no override
            assert store.resolve(descriptor) is None

            # Create override
            override = PromptOverride(
                ns=descriptor.ns,
                prompt_key=descriptor.key,
                tag="latest",
                sections={
                    section.path: SectionOverride(
                        expected_hash=section.content_hash,
                        body="Cheer loudly for ${subject}.",
                    )
                },
            )

            persisted = store.upsert(descriptor, override)
            assert (
                persisted.sections[section.path].body == "Cheer loudly for ${subject}."
            )

            # Resolve returns the override
            resolved = store.resolve(descriptor)
            assert resolved is not None
            assert (
                resolved.sections[section.path].body == "Cheer loudly for ${subject}."
            )

            # Delete removes it
            store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="latest")
            assert store.resolve(descriptor) is None

    def test_seed_captures_prompt_content(self) -> None:
        with redis_standalone() as client:
            store = RedisPromptOverridesStore(client=client)
            prompt = _build_prompt_with_tool()
            descriptor = PromptDescriptor.from_prompt(prompt)

            override = store.seed(prompt, tag="stable")

            section = descriptor.sections[0]
            assert section.path in override.sections
            assert override.sections[section.path].body == "Greet ${subject} warmly."

            tool_descriptor = descriptor.tools[0]
            assert tool_descriptor.name in override.tool_overrides
            tool_override = override.tool_overrides[tool_descriptor.name]
            assert tool_override.description == "Search stored notes."
            assert tool_override.param_descriptions == {
                "query": "User provided keywords."
            }

            # Resolve returns the seeded override
            resolved = store.resolve(descriptor, tag="stable")
            assert resolved is not None
            assert resolved.sections[section.path].body == "Greet ${subject} warmly."

            # Cleanup
            store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="stable")

    def test_seed_preserves_existing_override(self) -> None:
        with redis_standalone() as client:
            store = RedisPromptOverridesStore(client=client)
            prompt = _build_prompt()
            descriptor = PromptDescriptor.from_prompt(prompt)

            first = store.seed(prompt)
            second = store.seed(prompt)

            assert first.sections == second.sections
            assert first.tool_overrides == second.tool_overrides

            # Cleanup
            store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="latest")

    def test_set_section_override(self) -> None:
        with redis_standalone() as client:
            store = RedisPromptOverridesStore(client=client)
            prompt = _build_prompt()
            descriptor = PromptDescriptor.from_prompt(prompt)
            section = descriptor.sections[0]

            # Seed first
            store.seed(prompt)

            # Update section
            updated = store.set_section_override(
                prompt,
                path=section.path,
                body="Updated greeting for ${subject}.",
            )

            assert (
                updated.sections[section.path].body
                == "Updated greeting for ${subject}."
            )

            # Verify persisted
            resolved = store.resolve(descriptor)
            assert resolved is not None
            assert (
                resolved.sections[section.path].body
                == "Updated greeting for ${subject}."
            )

            # Cleanup
            store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="latest")

    def test_list_overrides(self) -> None:
        with redis_standalone() as client:
            store = RedisPromptOverridesStore(client=client)
            prompt1 = _build_prompt()
            prompt2 = _build_prompt_with_tool()

            # Create overrides
            store.seed(prompt1, tag="v1")
            store.seed(prompt2, tag="v2")

            # List all
            all_overrides = store.list_overrides(ns="integration/tests")
            assert len(all_overrides) >= 2

            # Filter by prompt_key
            filtered = store.list_overrides(
                ns="integration/tests", prompt_key="redis-greeting"
            )
            assert len(filtered) >= 1
            assert all(m.prompt_key == "redis-greeting" for m in filtered)

            # Cleanup
            store.delete(
                ns="integration/tests", prompt_key="redis-greeting", tag="v1"
            )
            store.delete(
                ns="integration/tests", prompt_key="redis-greeting-tools", tag="v2"
            )

    def test_resolve_filters_stale_override(self) -> None:
        with redis_standalone() as client:
            store = RedisPromptOverridesStore(client=client)
            prompt = _build_prompt()
            descriptor = PromptDescriptor.from_prompt(prompt)
            section = descriptor.sections[0]

            # Create override with wrong hash
            override = PromptOverride(
                ns=descriptor.ns,
                prompt_key=descriptor.key,
                tag="stale",
                sections={
                    section.path: SectionOverride(
                        expected_hash=section.content_hash,  # Correct hash for write
                        body="Content.",
                    )
                },
            )
            store.upsert(descriptor, override)

            # Manually corrupt the data by recreating with wrong hash
            # (This simulates prompt template change)
            client.hset(
                "{po:integration/tests}:redis-greeting:stale",
                "sections",
                '{"greeting": {"expected_hash": "'
                + str(OTHER_DIGEST)
                + '", "body": "Stale."}}',
            )

            # Should return None (all sections filtered as stale)
            result = store.resolve(descriptor, tag="stale")
            assert result is None

            # Cleanup
            store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="stale")

    def test_multiple_tags(self) -> None:
        with redis_standalone() as client:
            store = RedisPromptOverridesStore(client=client)
            prompt = _build_prompt()
            descriptor = PromptDescriptor.from_prompt(prompt)
            section = descriptor.sections[0]

            # Create overrides with different tags
            for tag in ["latest", "stable", "v1.0"]:
                override = PromptOverride(
                    ns=descriptor.ns,
                    prompt_key=descriptor.key,
                    tag=tag,
                    sections={
                        section.path: SectionOverride(
                            expected_hash=section.content_hash,
                            body=f"Content for {tag}.",
                        )
                    },
                )
                store.upsert(descriptor, override)

            # Resolve each tag
            for tag in ["latest", "stable", "v1.0"]:
                resolved = store.resolve(descriptor, tag=tag)
                assert resolved is not None
                assert resolved.sections[section.path].body == f"Content for {tag}."

            # Cleanup
            for tag in ["latest", "stable", "v1.0"]:
                store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag=tag)

    def test_custom_key_prefix(self) -> None:
        with redis_standalone() as client:
            store = RedisPromptOverridesStore(
                client=client, key_prefix="myapp_overrides"
            )
            prompt = _build_prompt()
            descriptor = PromptDescriptor.from_prompt(prompt)

            store.seed(prompt)

            # Verify key uses custom prefix
            key = "{myapp_overrides:integration/tests}:redis-greeting:latest"
            assert client.exists(key)

            # Cleanup
            store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="latest")


@pytest.mark.skipif(
    not REDIS_CLUSTER_TESTS_ENABLED,
    reason="Redis Cluster tests disabled (set REDIS_CLUSTER_TESTS=1 to enable)",
)
class TestRedisCluster:
    """Integration tests with Redis Cluster."""

    def test_upsert_resolve_delete_roundtrip(self) -> None:
        with redis_cluster() as client:
            store = RedisPromptOverridesStore(client=client)
            prompt = _build_prompt()
            descriptor = PromptDescriptor.from_prompt(prompt)
            section = descriptor.sections[0]

            # Initially no override
            assert store.resolve(descriptor) is None

            # Create override
            override = PromptOverride(
                ns=descriptor.ns,
                prompt_key=descriptor.key,
                tag="latest",
                sections={
                    section.path: SectionOverride(
                        expected_hash=section.content_hash,
                        body="Cluster greeting for ${subject}.",
                    )
                },
            )

            persisted = store.upsert(descriptor, override)
            assert (
                persisted.sections[section.path].body
                == "Cluster greeting for ${subject}."
            )

            # Resolve returns the override
            resolved = store.resolve(descriptor)
            assert resolved is not None
            assert (
                resolved.sections[section.path].body
                == "Cluster greeting for ${subject}."
            )

            # Delete removes it
            store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="latest")
            assert store.resolve(descriptor) is None

    def test_seed_and_list_overrides(self) -> None:
        with redis_cluster() as client:
            store = RedisPromptOverridesStore(client=client)
            prompt = _build_prompt_with_tool()
            descriptor = PromptDescriptor.from_prompt(prompt)

            override = store.seed(prompt)

            section = descriptor.sections[0]
            assert section.path in override.sections

            # List should find the override
            all_overrides = store.list_overrides(ns="integration/tests")
            assert len(all_overrides) >= 1

            # Cleanup
            store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="latest")

    def test_multiple_namespaces(self) -> None:
        """Verify hash tags route correctly across different namespaces."""
        with redis_cluster() as client:
            store = RedisPromptOverridesStore(client=client)

            prompts = [
                PromptTemplate(
                    ns=f"cluster/ns{i}",
                    key="greeting",
                    sections=[
                        MarkdownSection[_GreetingParams](
                            title="Greeting",
                            template=f"Greet from ns{i}.",
                            key="greeting",
                        )
                    ],
                )
                for i in range(3)
            ]

            # Seed all prompts
            for prompt in prompts:
                store.seed(prompt)

            # Verify each can be resolved
            for i, prompt in enumerate(prompts):
                descriptor = PromptDescriptor.from_prompt(prompt)
                resolved = store.resolve(descriptor)
                assert resolved is not None
                section = descriptor.sections[0]
                assert resolved.sections[section.path].body == f"Greet from ns{i}."

            # Cleanup
            for prompt in prompts:
                descriptor = PromptDescriptor.from_prompt(prompt)
                store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="latest")

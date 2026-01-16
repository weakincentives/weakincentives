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

"""Tests for RedisPromptOverridesStore."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from weakincentives.prompt import MarkdownSection, PromptTemplate, Tool
from weakincentives.prompt.overrides import (
    HexDigest,
    PromptDescriptor,
    PromptOverride,
    PromptOverridesError,
    SectionOverride,
    TaskExampleOverride,
    ToolOverride,
)

if TYPE_CHECKING:
    from redis import Redis


@dataclass
class _GreetingParams:
    subject: str


@dataclass
class _ToolParams:
    query: str = field(metadata={"description": "User provided keywords."})


@dataclass
class _ToolResult:
    result: str


def _build_prompt() -> PromptTemplate[None]:
    return PromptTemplate(
        ns="tests/redis",
        key="greeting",
        sections=[
            MarkdownSection[_GreetingParams](
                title="Greeting",
                template="Greet ${subject} warmly.",
                key="greeting",
            )
        ],
    )


def _build_prompt_with_tool() -> PromptTemplate[None]:
    tool = Tool[_ToolParams, _ToolResult](
        name="search",
        description="Search stored notes.",
        handler=None,
    )
    return PromptTemplate(
        ns="tests/redis",
        key="greeting-tools",
        sections=[
            MarkdownSection[_GreetingParams](
                title="Greeting",
                template="Greet ${subject} warmly.",
                key="greeting",
                tools=[tool],
            )
        ],
    )


VALID_DIGEST = HexDigest("a" * 64)
OTHER_DIGEST = HexDigest("b" * 64)


class TestRedisPromptOverridesStore:
    """Basic CRUD operations for RedisPromptOverridesStore."""

    def test_upsert_resolve_and_delete_roundtrip(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test basic create, read, delete cycle."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        section = descriptor.sections[0]
        assert store.resolve(descriptor) is None

        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag="latest",
            sections={
                section.path: SectionOverride(
                    path=section.path,
                    expected_hash=section.content_hash,
                    body="Cheer loudly for ${subject}.",
                )
            },
        )

        persisted = store.upsert(descriptor, override)
        assert persisted.sections[section.path].body == "Cheer loudly for ${subject}."

        resolved = store.resolve(descriptor)
        assert resolved is not None
        assert resolved.sections[section.path].body == "Cheer loudly for ${subject}."

        store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="latest")
        assert store.resolve(descriptor) is None

        # Deleting again should be a no-op
        store.delete(ns=descriptor.ns, prompt_key=descriptor.key, tag="latest")

    def test_resolve_missing_returns_none(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that resolve returns None for non-existent keys."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        assert store.resolve(descriptor) is None

    def test_seed_captures_prompt_content(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that seed creates an override with current prompt content."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt_with_tool()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        override = store.seed(prompt, tag="stable")
        section = descriptor.sections[0]

        assert section.path in override.sections
        assert override.sections[section.path].body == "Greet ${subject} warmly."

        tool_descriptor = descriptor.tools[0]
        assert tool_descriptor.name in override.tool_overrides
        tool_override = override.tool_overrides[tool_descriptor.name]
        assert tool_override.description == "Search stored notes."
        assert tool_override.param_descriptions == {"query": "User provided keywords."}

        resolved = store.resolve(descriptor, tag="stable")
        assert resolved is not None
        assert resolved.sections[section.path].body == "Greet ${subject} warmly."

    def test_seed_preserves_existing_override(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that seed doesn't overwrite existing overrides."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        store = RedisPromptOverridesStore(client=fake_redis_client)

        first = store.seed(prompt)
        second = store.seed(prompt)

        assert first.sections == second.sections
        assert first.tool_overrides == second.tool_overrides


class TestRedisPromptOverridesStoreValidation:
    """Validation and error handling tests."""

    def test_upsert_rejects_mismatched_metadata(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that upsert rejects overrides with wrong ns/key."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)
        section = descriptor.sections[0]

        override = PromptOverride(
            ns="other",
            prompt_key=descriptor.key,
            tag="latest",
            sections={
                section.path: SectionOverride(
                    path=section.path,
                    expected_hash=section.content_hash,
                    body="Text",
                )
            },
        )

        with pytest.raises(PromptOverridesError):
            store.upsert(descriptor, override)

    def test_upsert_rejects_unknown_section(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that upsert rejects sections not in descriptor."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)
        section = descriptor.sections[0]

        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag="latest",
            sections={
                ("unknown",): SectionOverride(
                    path=("unknown",),
                    expected_hash=section.content_hash,
                    body="Body",
                )
            },
        )

        with pytest.raises(PromptOverridesError):
            store.upsert(descriptor, override)

    def test_upsert_rejects_hash_mismatch(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that upsert rejects sections with wrong hash."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)
        section = descriptor.sections[0]

        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag="latest",
            sections={
                section.path: SectionOverride(
                    path=section.path,
                    expected_hash=OTHER_DIGEST,
                    body="Body",
                )
            },
        )

        with pytest.raises(PromptOverridesError):
            store.upsert(descriptor, override)

    def test_resolve_filters_stale_section(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that resolve filters out sections with wrong hash."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        # Write directly to Redis with a stale hash
        key = store._make_key(descriptor.ns, descriptor.key, "latest")
        section = descriptor.sections[0]
        payload = {
            "version": 2,
            "ns": descriptor.ns,
            "prompt_key": descriptor.key,
            "tag": "latest",
            "sections": {
                "/".join(section.path): {
                    "path": list(section.path),
                    "expected_hash": str(OTHER_DIGEST),  # Wrong hash
                    "body": "Stale content",
                }
            },
            "tools": {},
            "task_example_overrides": [],
        }
        fake_redis_client.set(key, json.dumps(payload))

        # Should return None because stale section is filtered
        assert store.resolve(descriptor) is None

    def test_resolve_invalid_json_raises_error(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that resolve raises error for invalid JSON."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        key = store._make_key(descriptor.ns, descriptor.key, "latest")
        fake_redis_client.set(key, b"{not-valid-json}")

        with pytest.raises(PromptOverridesError):
            store.resolve(descriptor)

    def test_resolve_wrong_version_raises_error(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that resolve raises error for unsupported version."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        key = store._make_key(descriptor.ns, descriptor.key, "latest")
        payload = {
            "version": 99,
            "ns": descriptor.ns,
            "prompt_key": descriptor.key,
            "tag": "latest",
            "sections": {},
            "tools": {},
        }
        fake_redis_client.set(key, json.dumps(payload))

        with pytest.raises(PromptOverridesError):
            store.resolve(descriptor)

    def test_identifier_validation_errors(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that invalid identifiers are rejected."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        store = RedisPromptOverridesStore(client=fake_redis_client)

        with pytest.raises(PromptOverridesError):
            store.delete(ns="   ", prompt_key="key", tag="latest")

        with pytest.raises(PromptOverridesError):
            store.delete(ns="/", prompt_key="key", tag="latest")

        with pytest.raises(PromptOverridesError):
            store.delete(ns="ns", prompt_key="Key", tag="latest")

        with pytest.raises(PromptOverridesError):
            store.delete(ns="ns", prompt_key=" ", tag="latest")

        with pytest.raises(PromptOverridesError):
            store.delete(ns="ns", prompt_key="key", tag="LATEST")


class TestRedisPromptOverridesStoreStore:
    """Tests for the store() method (single override storage)."""

    def test_store_section_override(self, fake_redis_client: Redis[bytes]) -> None:
        """Test storing a single section override."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        section = descriptor.sections[0]
        override = SectionOverride(
            path=section.path,
            expected_hash=section.content_hash,
            body="Updated greeting for ${subject}.",
        )

        result = store.store(descriptor, override)
        assert section.path in result.sections
        assert result.sections[section.path].body == "Updated greeting for ${subject}."

    def test_store_section_override_hash_mismatch(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that store rejects section with wrong hash."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        section = descriptor.sections[0]
        override = SectionOverride(
            path=section.path,
            expected_hash=OTHER_DIGEST,
            body="Bad content",
        )

        with pytest.raises(PromptOverridesError, match="Hash mismatch for section"):
            store.store(descriptor, override)

    def test_store_tool_override(self, fake_redis_client: Redis[bytes]) -> None:
        """Test storing a single tool override."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt_with_tool()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        tool = descriptor.tools[0]
        override = ToolOverride(
            name=tool.name,
            expected_contract_hash=tool.contract_hash,
            description="Updated search description.",
            param_descriptions={"query": "Updated query description"},
        )

        result = store.store(descriptor, override)
        assert tool.name in result.tool_overrides
        assert (
            result.tool_overrides[tool.name].description
            == "Updated search description."
        )

    def test_store_tool_override_hash_mismatch(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that store rejects tool with wrong hash."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt_with_tool()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        tool = descriptor.tools[0]
        override = ToolOverride(
            name=tool.name,
            expected_contract_hash=OTHER_DIGEST,
            description="Bad description",
            param_descriptions={},
        )

        with pytest.raises(PromptOverridesError, match="Hash mismatch for tool"):
            store.store(descriptor, override)

    def test_store_tool_override_unknown_tool(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that store rejects unknown tool name."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt_with_tool()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        override = ToolOverride(
            name="unknown_tool",
            expected_contract_hash=VALID_DIGEST,
            description="Some description",
            param_descriptions={},
        )

        with pytest.raises(PromptOverridesError, match="not registered"):
            store.store(descriptor, override)

    def test_store_task_example_override(self, fake_redis_client: Redis[bytes]) -> None:
        """Test storing a task example override."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        task_override = TaskExampleOverride(
            path=("section", "example"),
            index=0,
            expected_hash=None,
            action="append",
            objective="New objective",
        )

        result = store.store(descriptor, task_override, tag="latest")
        assert len(result.task_example_overrides) == 1
        assert result.task_example_overrides[0] == task_override

    def test_store_task_example_override_updates_existing(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that storing with same path+index updates the existing override."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        task_override1 = TaskExampleOverride(
            path=("section", "example"),
            index=0,
            expected_hash=None,
            action="append",
            objective="First objective",
        )
        task_override2 = TaskExampleOverride(
            path=("section", "example"),
            index=0,
            expected_hash=None,
            action="modify",
            objective="Updated objective",
        )

        store.store(descriptor, task_override1, tag="latest")
        result = store.store(descriptor, task_override2, tag="latest")

        assert len(result.task_example_overrides) == 1
        assert result.task_example_overrides[0].action == "modify"
        assert result.task_example_overrides[0].objective == "Updated objective"

    def test_store_task_example_override_with_different_path(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test storing multiple task examples with different paths."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        task_override1 = TaskExampleOverride(
            path=("section", "example1"),
            index=0,
            expected_hash=None,
            action="append",
            objective="First example",
        )
        task_override2 = TaskExampleOverride(
            path=("section", "example2"),
            index=0,
            expected_hash=None,
            action="append",
            objective="Second example",
        )

        store.store(descriptor, task_override1, tag="latest")
        result = store.store(descriptor, task_override2, tag="latest")

        # Both should be stored since they have different paths
        assert len(result.task_example_overrides) == 2


class TestRedisPromptOverridesStoreFactory:
    """Tests for RedisPromptOverridesStoreFactory."""

    def test_factory_creates_store(self, fake_redis_client: Redis[bytes]) -> None:
        """Test that factory creates working stores."""
        from weakincentives.contrib.prompt_overrides import (
            RedisPromptOverridesStore,
            RedisPromptOverridesStoreFactory,
        )

        factory = RedisPromptOverridesStoreFactory(client=fake_redis_client)
        store = factory.create()

        assert isinstance(store, RedisPromptOverridesStore)
        assert store.client is fake_redis_client

    def test_factory_custom_ttl(self, fake_redis_client: Redis[bytes]) -> None:
        """Test that factory propagates custom TTL."""
        from weakincentives.contrib.prompt_overrides import (
            RedisPromptOverridesStoreFactory,
        )

        custom_ttl = 86400
        factory = RedisPromptOverridesStoreFactory(
            client=fake_redis_client, default_ttl=custom_ttl
        )
        store = factory.create()

        assert store.default_ttl == custom_ttl

    def test_factory_custom_prefix(self, fake_redis_client: Redis[bytes]) -> None:
        """Test that factory propagates custom key prefix."""
        from weakincentives.contrib.prompt_overrides import (
            RedisPromptOverridesStoreFactory,
        )

        factory = RedisPromptOverridesStoreFactory(
            client=fake_redis_client, key_prefix="custom"
        )
        store = factory.create()

        assert store.key_prefix == "custom"


class TestRedisPromptOverridesStoreTTL:
    """Tests for TTL functionality."""

    def test_default_ttl_constant(self) -> None:
        """Test that DEFAULT_TTL_SECONDS is exported and correct."""
        from weakincentives.contrib.prompt_overrides import DEFAULT_TTL_SECONDS

        assert DEFAULT_TTL_SECONDS == 2592000  # 30 days

    def test_store_uses_default_ttl(self, fake_redis_client: Redis[bytes]) -> None:
        """Test that store uses DEFAULT_TTL_SECONDS by default."""
        from weakincentives.contrib.prompt_overrides import (
            DEFAULT_TTL_SECONDS,
            RedisPromptOverridesStore,
        )

        store = RedisPromptOverridesStore(client=fake_redis_client)
        assert store.default_ttl == DEFAULT_TTL_SECONDS

    def test_store_custom_ttl(self, fake_redis_client: Redis[bytes]) -> None:
        """Test that store accepts custom TTL."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        custom_ttl = 86400
        store = RedisPromptOverridesStore(
            client=fake_redis_client, default_ttl=custom_ttl
        )
        assert store.default_ttl == custom_ttl


class TestRedisPromptOverridesStoreEdgeCases:
    """Tests for edge cases and error handling."""

    def test_store_section_override_unknown_section(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that store rejects section not in descriptor."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        override = SectionOverride(
            path=("unknown",),
            expected_hash=VALID_DIGEST,
            body="Content",
        )

        with pytest.raises(PromptOverridesError, match="not registered"):
            store.store(descriptor, override)

    def test_set_with_ttl_zero(self, fake_redis_client: Redis[bytes]) -> None:
        """Test that TTL=0 uses SET without expiration."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client, default_ttl=0)

        section = descriptor.sections[0]
        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag="latest",
            sections={
                section.path: SectionOverride(
                    path=section.path,
                    expected_hash=section.content_hash,
                    body="Content",
                )
            },
        )

        persisted = store.upsert(descriptor, override)
        assert persisted.sections[section.path].body == "Content"

        # Verify no TTL was set
        key = store._make_key(descriptor.ns, descriptor.key, "latest")
        ttl = fake_redis_client.ttl(key)
        assert ttl == -1  # No expiration

    def test_seed_override_cannot_be_resolved(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test seed error when override exists but cannot be resolved."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        # Write directly to Redis with stale data (all entries stale)
        key = store._make_key(descriptor.ns, descriptor.key, "latest")
        payload = {
            "version": 2,
            "ns": descriptor.ns,
            "prompt_key": descriptor.key,
            "tag": "latest",
            "sections": {
                "unknown": {
                    "path": ["unknown"],
                    "expected_hash": str(OTHER_DIGEST),
                    "body": "Stale",
                }
            },
            "tools": {},
            "task_example_overrides": [],
        }
        fake_redis_client.set(key, json.dumps(payload))

        with pytest.raises(
            PromptOverridesError, match="Override exists but could not be resolved"
        ):
            store.seed(prompt, tag="latest")

    def test_resolve_metadata_mismatch(self, fake_redis_client: Redis[bytes]) -> None:
        """Test resolve error when stored metadata doesn't match descriptor."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        # Write directly to Redis with mismatched ns
        key = store._make_key(descriptor.ns, descriptor.key, "latest")
        payload = {
            "version": 2,
            "ns": "wrong/ns",  # Mismatched namespace
            "prompt_key": descriptor.key,
            "tag": "latest",
            "sections": {},
            "tools": {},
            "task_example_overrides": [],
        }
        fake_redis_client.set(key, json.dumps(payload))

        with pytest.raises(PromptOverridesError, match="does not match descriptor"):
            store.resolve(descriptor, tag="latest")


class TestRedisPromptOverridesStoreKeyFormat:
    """Tests for Redis key format."""

    def test_key_format_with_simple_namespace(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test key format with a simple namespace."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        store = RedisPromptOverridesStore(client=fake_redis_client)
        key = store._make_key("agents", "review", "latest")

        assert key == "{prompt:agents:review}:latest"

    def test_key_format_with_nested_namespace(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test key format with a nested namespace."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        store = RedisPromptOverridesStore(client=fake_redis_client)
        key = store._make_key("agents/code-review", "review", "stable")

        assert key == "{prompt:agents/code-review:review}:stable"

    def test_key_format_with_custom_prefix(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test key format with a custom prefix."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        store = RedisPromptOverridesStore(
            client=fake_redis_client, key_prefix="override"
        )
        key = store._make_key("agents", "review", "latest")

        assert key == "{override:agents:review}:latest"


class TestRedisPromptOverridesStoreWithSummaryAndVisibility:
    """Tests for section override with summary and visibility fields."""

    def test_upsert_and_resolve_with_summary_and_visibility(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that summary and visibility fields are preserved."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        section = descriptor.sections[0]
        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag="latest",
            sections={
                section.path: SectionOverride(
                    path=section.path,
                    expected_hash=section.content_hash,
                    body="Full content for ${subject}.",
                    summary="Brief summary for ${subject}.",
                    visibility="summary",
                )
            },
        )

        persisted = store.upsert(descriptor, override)
        assert persisted.sections[section.path].body == "Full content for ${subject}."
        assert (
            persisted.sections[section.path].summary == "Brief summary for ${subject}."
        )
        assert persisted.sections[section.path].visibility == "summary"

        resolved = store.resolve(descriptor)
        assert resolved is not None
        assert resolved.sections[section.path].body == "Full content for ${subject}."
        assert (
            resolved.sections[section.path].summary == "Brief summary for ${subject}."
        )
        assert resolved.sections[section.path].visibility == "summary"

    def test_resolve_without_summary_and_visibility(
        self, fake_redis_client: Redis[bytes]
    ) -> None:
        """Test that missing summary/visibility default to None."""
        from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        store = RedisPromptOverridesStore(client=fake_redis_client)

        section = descriptor.sections[0]
        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag="latest",
            sections={
                section.path: SectionOverride(
                    path=section.path,
                    expected_hash=section.content_hash,
                    body="Body without summary.",
                )
            },
        )

        persisted = store.upsert(descriptor, override)
        assert persisted.sections[section.path].summary is None
        assert persisted.sections[section.path].visibility is None

        resolved = store.resolve(descriptor)
        assert resolved is not None
        assert resolved.sections[section.path].summary is None
        assert resolved.sections[section.path].visibility is None

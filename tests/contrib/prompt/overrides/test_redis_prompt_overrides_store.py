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

"""Unit tests for RedisPromptOverridesStore using mocked Redis client."""

# pyright: reportPrivateUsage=false
# pyright: reportUnusedCallResult=false
# pyright: reportArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportMissingTypeArgument=false
# pyright: reportUnknownParameterType=false

from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

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
        ns="tests/versioning",
        key="versioned-greeting",
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
        ns="tests/versioning",
        key="versioned-greeting-tools",
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


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock Redis client."""
    client = MagicMock()
    client.register_script = MagicMock(return_value=MagicMock())
    return client


@pytest.fixture
def store(mock_client: MagicMock) -> RedisPromptOverridesStore:
    """Create a store with mocked client."""
    return RedisPromptOverridesStore(client=mock_client)


class TestKeyConstruction:
    """Tests for Redis key construction."""

    def test_override_key_format(self, store: RedisPromptOverridesStore) -> None:
        key = store._override_key("demo", "greeting", "latest")
        assert key == "{po:demo}:greeting:latest"

    def test_override_key_with_nested_namespace(
        self, store: RedisPromptOverridesStore
    ) -> None:
        key = store._override_key("webapp/agents", "review", "stable")
        assert key == "{po:webapp/agents}:review:stable"

    def test_override_key_custom_prefix(self, mock_client: MagicMock) -> None:
        store = RedisPromptOverridesStore(client=mock_client, key_prefix="myapp_po")
        key = store._override_key("demo", "greeting", "latest")
        assert key == "{myapp_po:demo}:greeting:latest"


class TestIdentifierValidation:
    """Tests for identifier validation."""

    def test_valid_identifiers(self, store: RedisPromptOverridesStore) -> None:
        assert store._validate_identifier("latest", "tag") == "latest"
        assert store._validate_identifier("my-prompt", "key") == "my-prompt"
        assert store._validate_identifier("my_prompt", "key") == "my_prompt"
        assert store._validate_identifier("my.prompt", "key") == "my.prompt"
        assert store._validate_identifier("v1.0.0", "tag") == "v1.0.0"

    def test_uppercase_normalized(self, store: RedisPromptOverridesStore) -> None:
        assert store._validate_identifier("LATEST", "tag") == "latest"
        assert store._validate_identifier("MyPrompt", "key") == "myprompt"

    def test_invalid_identifier_raises(self, store: RedisPromptOverridesStore) -> None:
        with pytest.raises(RedisPromptOverridesError):
            store._validate_identifier("", "tag")

        with pytest.raises(RedisPromptOverridesError):
            store._validate_identifier(" ", "tag")

        with pytest.raises(RedisPromptOverridesError):
            store._validate_identifier("-invalid", "tag")

        with pytest.raises(RedisPromptOverridesError):
            store._validate_identifier("a" * 65, "tag")

    def test_namespace_validation(self, store: RedisPromptOverridesStore) -> None:
        assert store._validate_namespace("demo") == "demo"
        assert store._validate_namespace("webapp/agents") == "webapp/agents"
        assert store._validate_namespace("a/b/c") == "a/b/c"

    def test_namespace_with_invalid_segment_raises(
        self, store: RedisPromptOverridesStore
    ) -> None:
        with pytest.raises(RedisPromptOverridesError):
            store._validate_namespace("valid/-invalid")


class TestResolve:
    """Tests for resolve operation."""

    def test_resolve_returns_none_when_not_found(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        mock_client.hgetall.return_value = {}

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)

        result = store.resolve(descriptor)
        assert result is None

    def test_resolve_returns_override(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        section = descriptor.sections[0]

        sections_data = {
            "greeting": {
                "expected_hash": str(section.content_hash),
                "body": "Custom greeting for ${subject}.",
            }
        }

        mock_client.hgetall.return_value = {
            b"version": b"1",
            b"ns": b"tests/versioning",
            b"prompt_key": b"versioned-greeting",
            b"tag": b"latest",
            b"sections": json.dumps(sections_data).encode(),
            b"tools": b"{}",
        }

        result = store.resolve(descriptor)
        assert result is not None
        assert result.sections[section.path].body == "Custom greeting for ${subject}."

    def test_resolve_filters_stale_sections(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)

        # Use wrong hash - should be filtered out
        sections_data = {
            "greeting": {
                "expected_hash": str(OTHER_DIGEST),
                "body": "Stale content.",
            }
        }

        mock_client.hgetall.return_value = {
            b"version": b"1",
            b"ns": b"tests/versioning",
            b"prompt_key": b"versioned-greeting",
            b"tag": b"latest",
            b"sections": json.dumps(sections_data).encode(),
            b"tools": b"{}",
        }

        result = store.resolve(descriptor)
        assert result is None

    def test_resolve_connection_error(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        mock_client.hgetall.side_effect = Exception("Connection refused")

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)

        with pytest.raises(RedisPromptOverridesError) as exc_info:
            store.resolve(descriptor)

        assert "Failed to retrieve override" in str(exc_info.value)

    def test_resolve_invalid_version_raises(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)

        mock_client.hgetall.return_value = {
            b"version": b"99",
            b"ns": b"tests/versioning",
            b"prompt_key": b"versioned-greeting",
            b"tag": b"latest",
            b"sections": b"{}",
            b"tools": b"{}",
        }

        with pytest.raises(RedisPromptOverridesError) as exc_info:
            store.resolve(descriptor)

        assert "Unsupported override version" in str(exc_info.value)

    def test_resolve_metadata_mismatch_raises(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)

        mock_client.hgetall.return_value = {
            b"version": b"1",
            b"ns": b"other-namespace",
            b"prompt_key": b"versioned-greeting",
            b"tag": b"latest",
            b"sections": b"{}",
            b"tools": b"{}",
        }

        with pytest.raises(RedisPromptOverridesError):
            store.resolve(descriptor)


class TestUpsert:
    """Tests for upsert operation."""

    def test_upsert_persists_override(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        section = descriptor.sections[0]

        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag="latest",
            sections={
                section.path: SectionOverride(
                    expected_hash=section.content_hash,
                    body="Custom greeting.",
                )
            },
        )

        result = store.upsert(descriptor, override)

        mock_client.hset.assert_called_once()
        call_args = mock_client.hset.call_args
        assert call_args[0][0] == "{po:tests/versioning}:versioned-greeting:latest"

        mapping = call_args[1]["mapping"]
        assert mapping["version"] == "1"
        assert mapping["ns"] == "tests/versioning"
        assert "sections" in mapping

        assert result.sections[section.path].body == "Custom greeting."

    def test_upsert_rejects_mismatched_metadata(
        self,
        store: RedisPromptOverridesStore,
    ) -> None:
        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        section = descriptor.sections[0]

        override = PromptOverride(
            ns="other-namespace",
            prompt_key=descriptor.key,
            tag="latest",
            sections={
                section.path: SectionOverride(
                    expected_hash=section.content_hash,
                    body="Text",
                )
            },
        )

        with pytest.raises(RedisPromptOverridesError):
            store.upsert(descriptor, override)

    def test_upsert_connection_error(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        mock_client.hset.side_effect = Exception("Connection refused")

        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        section = descriptor.sections[0]

        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag="latest",
            sections={
                section.path: SectionOverride(
                    expected_hash=section.content_hash,
                    body="Text",
                )
            },
        )

        with pytest.raises(RedisPromptOverridesError) as exc_info:
            store.upsert(descriptor, override)

        assert "Failed to persist override" in str(exc_info.value)


class TestDelete:
    """Tests for delete operation."""

    def test_delete_removes_key(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        mock_client.delete.return_value = 1

        store.delete(ns="tests/versioning", prompt_key="greeting", tag="latest")

        mock_client.delete.assert_called_once_with(
            "{po:tests/versioning}:greeting:latest"
        )

    def test_delete_missing_key_is_noop(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        mock_client.delete.return_value = 0

        # Should not raise
        store.delete(ns="tests/versioning", prompt_key="greeting", tag="latest")

    def test_delete_connection_error(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        mock_client.delete.side_effect = Exception("Connection refused")

        with pytest.raises(RedisPromptOverridesError) as exc_info:
            store.delete(ns="tests/versioning", prompt_key="greeting", tag="latest")

        assert "Failed to delete override" in str(exc_info.value)


class TestSetSectionOverride:
    """Tests for set_section_override operation."""

    def test_set_section_override_creates_new(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        section = descriptor.sections[0]

        # No existing override
        mock_client.hgetall.return_value = {}

        result = store.set_section_override(
            prompt,
            path=section.path,
            body="New section body.",
        )

        assert result.sections[section.path].body == "New section body."
        mock_client.hset.assert_called_once()

    def test_set_section_override_updates_existing(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        section = descriptor.sections[0]

        # Existing override
        sections_data = {
            "greeting": {
                "expected_hash": str(section.content_hash),
                "body": "Old content.",
            }
        }
        mock_client.hgetall.return_value = {
            b"version": b"1",
            b"ns": b"tests/versioning",
            b"prompt_key": b"versioned-greeting",
            b"tag": b"latest",
            b"sections": json.dumps(sections_data).encode(),
            b"tools": b"{}",
        }

        result = store.set_section_override(
            prompt,
            path=section.path,
            body="Updated content.",
        )

        assert result.sections[section.path].body == "Updated content."

    def test_set_section_override_unknown_section_raises(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        prompt = _build_prompt()

        mock_client.hgetall.return_value = {}

        with pytest.raises(RedisPromptOverridesError) as exc_info:
            store.set_section_override(
                prompt,
                path=("unknown",),
                body="Content.",
            )

        assert "not registered" in str(exc_info.value)


class TestSeed:
    """Tests for seed operation."""

    def test_seed_creates_new_override(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        prompt = _build_prompt_with_tool()
        descriptor = PromptDescriptor.from_prompt(prompt)

        mock_client.exists.return_value = 0

        result = store.seed(prompt, tag="stable")

        section = descriptor.sections[0]
        assert section.path in result.sections
        assert result.sections[section.path].body == "Greet ${subject} warmly."

        tool = descriptor.tools[0]
        assert tool.name in result.tool_overrides
        assert result.tool_overrides[tool.name].description == "Search stored notes."

    def test_seed_returns_existing_override(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        prompt = _build_prompt()
        descriptor = PromptDescriptor.from_prompt(prompt)
        section = descriptor.sections[0]

        mock_client.exists.return_value = 1

        sections_data = {
            "greeting": {
                "expected_hash": str(section.content_hash),
                "body": "Existing content.",
            }
        }
        mock_client.hgetall.return_value = {
            b"version": b"1",
            b"ns": b"tests/versioning",
            b"prompt_key": b"versioned-greeting",
            b"tag": b"stable",
            b"sections": json.dumps(sections_data).encode(),
            b"tools": b"{}",
        }

        result = store.seed(prompt, tag="stable")

        assert result.sections[section.path].body == "Existing content."
        # hset should not be called since we returned existing
        mock_client.hset.assert_not_called()

    def test_seed_error_on_corrupt_existing(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        prompt = _build_prompt()

        mock_client.exists.return_value = 1
        mock_client.hgetall.return_value = {}

        with pytest.raises(RedisPromptOverridesError) as exc_info:
            store.seed(prompt)

        assert "could not be resolved" in str(exc_info.value)


class TestListOverrides:
    """Tests for list_overrides operation."""

    def test_list_overrides_returns_metadata(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        mock_client.scan.return_value = (
            0,
            [b"{po:demo}:greeting:latest"],
        )
        mock_client.hmget.return_value = [
            b"demo",
            b"greeting",
            b"latest",
            b'{"system": {}}',
            b"{}",
        ]

        results = store.list_overrides()

        assert len(results) == 1
        assert results[0].ns == "demo"
        assert results[0].prompt_key == "greeting"
        assert results[0].tag == "latest"
        assert results[0].section_count == 1
        assert results[0].tool_count == 0

    def test_list_overrides_with_namespace_filter(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        mock_client.scan.return_value = (0, [])

        store.list_overrides(ns="demo")

        mock_client.scan.assert_called()
        call_args = mock_client.scan.call_args
        assert "{po:demo}" in call_args[1]["match"]

    def test_list_overrides_with_prompt_key_filter(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        mock_client.scan.return_value = (0, [])

        store.list_overrides(ns="demo", prompt_key="greeting")

        mock_client.scan.assert_called()
        call_args = mock_client.scan.call_args
        assert "{po:demo}:greeting:*" in call_args[1]["match"]

    def test_list_overrides_requires_ns_for_prompt_key(
        self,
        store: RedisPromptOverridesStore,
    ) -> None:
        with pytest.raises(RedisPromptOverridesError) as exc_info:
            store.list_overrides(prompt_key="greeting")

        assert "Cannot filter by prompt_key without" in str(exc_info.value)

    def test_list_overrides_connection_error(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        mock_client.scan.side_effect = Exception("Connection refused")

        with pytest.raises(RedisPromptOverridesError) as exc_info:
            store.list_overrides()

        assert "Failed to list overrides" in str(exc_info.value)


class TestClose:
    """Tests for close operation."""

    def test_close_is_noop(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        store.close()
        mock_client.close.assert_not_called()


class TestParseHashData:
    """Tests for _parse_hash_data helper."""

    def test_parse_hash_data_sections_and_tools(
        self,
        store: RedisPromptOverridesStore,
    ) -> None:
        data = {
            b"version": b"1",
            b"ns": b"demo",
            b"prompt_key": b"greeting",
            b"tag": b"latest",
            b"sections": b'{"key": "value"}',
            b"tools": b'{"tool": "data"}',
        }

        result = store._parse_hash_data(data)

        assert result["version"] == 1
        assert result["ns"] == "demo"
        assert result["sections"] == {"key": "value"}
        assert result["tools"] == {"tool": "data"}

    def test_parse_hash_data_invalid_json_fallback(
        self,
        store: RedisPromptOverridesStore,
    ) -> None:
        data = {
            b"version": b"1",
            b"sections": b"not-json",
            b"tools": b"also-not-json",
        }

        result = store._parse_hash_data(data)

        assert result["sections"] == {}
        assert result["tools"] == {}

    def test_parse_hash_data_invalid_version_string(
        self,
        store: RedisPromptOverridesStore,
    ) -> None:
        data = {
            b"version": b"not-a-number",
        }

        result = store._parse_hash_data(data)

        assert result["version"] == "not-a-number"


class TestToolOverrides:
    """Tests for tool override handling."""

    def test_resolve_with_tool_overrides(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        prompt = _build_prompt_with_tool()
        descriptor = PromptDescriptor.from_prompt(prompt)
        section = descriptor.sections[0]
        tool = descriptor.tools[0]

        sections_data = {
            "greeting": {
                "expected_hash": str(section.content_hash),
                "body": "Custom greeting.",
            }
        }
        tools_data = {
            "search": {
                "expected_contract_hash": str(tool.contract_hash),
                "description": "Custom search description.",
                "param_descriptions": {"query": "Custom query desc."},
            }
        }

        mock_client.hgetall.return_value = {
            b"version": b"1",
            b"ns": b"tests/versioning",
            b"prompt_key": b"versioned-greeting-tools",
            b"tag": b"latest",
            b"sections": json.dumps(sections_data).encode(),
            b"tools": json.dumps(tools_data).encode(),
        }

        result = store.resolve(descriptor)

        assert result is not None
        assert "search" in result.tool_overrides
        assert (
            result.tool_overrides["search"].description == "Custom search description."
        )
        assert result.tool_overrides["search"].param_descriptions == {
            "query": "Custom query desc."
        }

    def test_upsert_with_tool_overrides(
        self,
        store: RedisPromptOverridesStore,
        mock_client: MagicMock,
    ) -> None:
        prompt = _build_prompt_with_tool()
        descriptor = PromptDescriptor.from_prompt(prompt)
        section = descriptor.sections[0]
        tool = descriptor.tools[0]

        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag="latest",
            sections={
                section.path: SectionOverride(
                    expected_hash=section.content_hash,
                    body="Text",
                )
            },
            tool_overrides={
                tool.name: ToolOverride(
                    name=tool.name,
                    expected_contract_hash=tool.contract_hash,
                    description="New description.",
                    param_descriptions={"query": "New param desc."},
                )
            },
        )

        result = store.upsert(descriptor, override)

        assert result.tool_overrides[tool.name].description == "New description."
        mock_client.hset.assert_called_once()

        call_args = mock_client.hset.call_args
        mapping = call_args[1]["mapping"]
        tools_json = json.loads(mapping["tools"])
        assert "search" in tools_json
        assert tools_json["search"]["description"] == "New description."

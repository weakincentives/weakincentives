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

"""Tests for OpenAI web search hosted tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from weakincentives.adapters.openai_hosted import (
    EmptyParams,
    OpenAIUrlCitation,
    OpenAIUserLocation,
    OpenAIWebSearchCodec,
    OpenAIWebSearchConfig,
    OpenAIWebSearchFilters,
    OpenAIWebSearchResult,
    OpenAIWebSearchSection,
    openai_web_search,
)


class TestOpenAIWebSearchConfig:
    """Tests for OpenAI web search configuration types."""

    def test_default_filters(self) -> None:
        """Default filters are empty tuples."""
        filters = OpenAIWebSearchFilters()
        assert filters.allowed_domains == ()
        assert filters.blocked_domains == ()

    def test_filters_with_domains(self) -> None:
        """Filters can be created with domain lists."""
        filters = OpenAIWebSearchFilters(
            allowed_domains=("example.com", "test.org"),
            blocked_domains=("spam.com",),
        )
        assert filters.allowed_domains == ("example.com", "test.org")
        assert filters.blocked_domains == ("spam.com",)

    def test_default_user_location(self) -> None:
        """Default user location has all None values."""
        loc = OpenAIUserLocation()
        assert loc.country is None
        assert loc.city is None
        assert loc.region is None
        assert loc.timezone is None

    def test_user_location_with_values(self) -> None:
        """User location can be created with geographic values."""
        loc = OpenAIUserLocation(
            country="US",
            city="San Francisco",
            region="California",
            timezone="America/Los_Angeles",
        )
        assert loc.country == "US"
        assert loc.city == "San Francisco"
        assert loc.region == "California"
        assert loc.timezone == "America/Los_Angeles"

    def test_default_config(self) -> None:
        """Default config has None for optional fields and True for external_web_access."""
        config = OpenAIWebSearchConfig()
        assert config.filters is None
        assert config.user_location is None
        assert config.search_context_size is None
        assert config.external_web_access is True

    def test_config_with_all_options(self) -> None:
        """Config can be created with all options."""
        config = OpenAIWebSearchConfig(
            filters=OpenAIWebSearchFilters(
                allowed_domains=("example.com",),
            ),
            user_location=OpenAIUserLocation(country="US"),
            search_context_size="high",
            external_web_access=False,
        )
        assert config.filters is not None
        assert config.user_location is not None
        assert config.search_context_size == "high"
        assert config.external_web_access is False


class TestOpenAIWebSearchFactory:
    """Tests for the openai_web_search factory function."""

    def test_default_factory(self) -> None:
        """Factory creates a valid hosted tool with defaults."""
        tool = openai_web_search()
        assert tool.kind == "web_search"
        assert tool.name == "web_search"
        assert isinstance(tool.config, OpenAIWebSearchConfig)

    def test_factory_with_custom_name(self) -> None:
        """Factory respects custom name."""
        tool = openai_web_search(name="my_search")
        assert tool.name == "my_search"

    def test_factory_with_config(self) -> None:
        """Factory uses provided config."""
        config = OpenAIWebSearchConfig(search_context_size="low")
        tool = openai_web_search(config)
        assert tool.config.search_context_size == "low"


class TestOpenAIWebSearchCodec:
    """Tests for the OpenAI web search codec."""

    def test_kind_property(self) -> None:
        """Codec reports correct kind."""
        codec = OpenAIWebSearchCodec()
        assert codec.kind == "web_search"

    def test_serialize_minimal_config(self) -> None:
        """Serialize minimal config produces expected output."""
        codec = OpenAIWebSearchCodec()
        tool = openai_web_search()
        result = codec.serialize(tool)

        assert result == {"type": "web_search_preview"}

    def test_serialize_with_filters(self) -> None:
        """Serialize config with domain filters."""
        codec = OpenAIWebSearchCodec()
        config = OpenAIWebSearchConfig(
            filters=OpenAIWebSearchFilters(
                allowed_domains=("example.com", "test.org"),
                blocked_domains=("spam.com",),
            ),
        )
        tool = openai_web_search(config)
        result = codec.serialize(tool)

        assert result["type"] == "web_search_preview"
        assert result["filters"] == {
            "allowed_domains": ["example.com", "test.org"],
            "blocked_domains": ["spam.com"],
        }

    def test_serialize_with_allowed_domains_only(self) -> None:
        """Serialize config with only allowed domains."""
        codec = OpenAIWebSearchCodec()
        config = OpenAIWebSearchConfig(
            filters=OpenAIWebSearchFilters(
                allowed_domains=("example.com",),
            ),
        )
        tool = openai_web_search(config)
        result = codec.serialize(tool)

        assert result["filters"] == {"allowed_domains": ["example.com"]}

    def test_serialize_with_blocked_domains_only(self) -> None:
        """Serialize config with only blocked domains."""
        codec = OpenAIWebSearchCodec()
        config = OpenAIWebSearchConfig(
            filters=OpenAIWebSearchFilters(
                blocked_domains=("spam.com",),
            ),
        )
        tool = openai_web_search(config)
        result = codec.serialize(tool)

        assert result["filters"] == {"blocked_domains": ["spam.com"]}

    def test_serialize_empty_filters_not_included(self) -> None:
        """Empty filters object doesn't add filters to output."""
        codec = OpenAIWebSearchCodec()
        config = OpenAIWebSearchConfig(
            filters=OpenAIWebSearchFilters(),
        )
        tool = openai_web_search(config)
        result = codec.serialize(tool)

        assert "filters" not in result

    def test_serialize_with_user_location(self) -> None:
        """Serialize config with user location."""
        codec = OpenAIWebSearchCodec()
        config = OpenAIWebSearchConfig(
            user_location=OpenAIUserLocation(
                country="US",
                city="San Francisco",
                region="California",
                timezone="America/Los_Angeles",
            ),
        )
        tool = openai_web_search(config)
        result = codec.serialize(tool)

        assert result["user_location"] == {
            "type": "approximate",
            "country": "US",
            "city": "San Francisco",
            "region": "California",
            "timezone": "America/Los_Angeles",
        }

    def test_serialize_with_partial_user_location(self) -> None:
        """Serialize config with partial user location."""
        codec = OpenAIWebSearchCodec()
        config = OpenAIWebSearchConfig(
            user_location=OpenAIUserLocation(
                country="US",
            ),
        )
        tool = openai_web_search(config)
        result = codec.serialize(tool)

        assert result["user_location"] == {
            "type": "approximate",
            "country": "US",
        }
        assert "city" not in result["user_location"]

    def test_serialize_with_search_context_size(self) -> None:
        """Serialize config with search context size."""
        codec = OpenAIWebSearchCodec()
        config = OpenAIWebSearchConfig(search_context_size="high")
        tool = openai_web_search(config)
        result = codec.serialize(tool)

        assert result["search_context_size"] == "high"

    def test_serialize_with_external_web_access_disabled(self) -> None:
        """Serialize config with external_web_access disabled."""
        codec = OpenAIWebSearchCodec()
        config = OpenAIWebSearchConfig(external_web_access=False)
        tool = openai_web_search(config)
        result = codec.serialize(tool)

        assert result["external_web_access"] is False

    def test_serialize_with_external_web_access_enabled(self) -> None:
        """Serialize config with external_web_access enabled (default)."""
        codec = OpenAIWebSearchCodec()
        config = OpenAIWebSearchConfig(external_web_access=True)
        tool = openai_web_search(config)
        result = codec.serialize(tool)

        # Default True should not add the field
        assert "external_web_access" not in result

    def test_serialize_full_config(self) -> None:
        """Serialize fully configured tool."""
        codec = OpenAIWebSearchCodec()
        config = OpenAIWebSearchConfig(
            filters=OpenAIWebSearchFilters(
                allowed_domains=("example.com",),
            ),
            user_location=OpenAIUserLocation(country="US"),
            search_context_size="medium",
            external_web_access=False,
        )
        tool = openai_web_search(config)
        result = codec.serialize(tool)

        assert result["type"] == "web_search_preview"
        assert result["filters"] == {"allowed_domains": ["example.com"]}
        assert result["user_location"]["country"] == "US"
        assert result["search_context_size"] == "medium"
        assert result["external_web_access"] is False


@dataclass
class MockResponseItem:
    """Mock response item for testing."""

    type: str
    content: list[Any] | None = None


@dataclass
class MockContentPart:
    """Mock content part for testing."""

    type: str
    text: str = ""
    annotations: list[Any] | None = None


@dataclass
class MockAnnotation:
    """Mock annotation for testing."""

    type: str
    url: str = ""
    title: str = ""
    start_index: int = 0
    end_index: int = 0


class TestOpenAIWebSearchCodecParseOutput:
    """Tests for OpenAI web search codec output parsing."""

    def test_parse_empty_response(self) -> None:
        """Parsing empty response returns None."""
        codec = OpenAIWebSearchCodec()
        tool = openai_web_search()
        result = codec.parse_output([], tool)
        assert result is None

    def test_parse_response_without_web_search_call(self) -> None:
        """Parsing response without web_search_call returns None."""
        codec = OpenAIWebSearchCodec()
        tool = openai_web_search()
        items = [MockResponseItem(type="message")]
        result = codec.parse_output(items, tool)
        assert result is None

    def test_parse_response_with_web_search_call(self) -> None:
        """Parsing response with web_search_call returns result."""
        codec = OpenAIWebSearchCodec()
        tool = openai_web_search()

        items = [
            MockResponseItem(type="web_search_call"),
            MockResponseItem(
                type="message",
                content=[
                    MockContentPart(
                        type="output_text",
                        text="Here is the information.",
                        annotations=[],
                    )
                ],
            ),
        ]

        result = codec.parse_output(items, tool)

        assert result is not None
        assert isinstance(result, OpenAIWebSearchResult)
        assert result.text == "Here is the information."
        assert result.citations == ()
        assert result.source_urls == ()

    def test_parse_response_with_citations(self) -> None:
        """Parsing response extracts citations."""
        codec = OpenAIWebSearchCodec()
        tool = openai_web_search()

        items = [
            MockResponseItem(type="web_search_call"),
            MockResponseItem(
                type="message",
                content=[
                    MockContentPart(
                        type="output_text",
                        text="Information from [1].",
                        annotations=[
                            MockAnnotation(
                                type="url_citation",
                                url="https://example.com/article",
                                title="Example Article",
                                start_index=17,
                                end_index=20,
                            ),
                        ],
                    )
                ],
            ),
        ]

        result = codec.parse_output(items, tool)

        assert result is not None
        assert len(result.citations) == 1
        citation = result.citations[0]
        assert citation.url == "https://example.com/article"
        assert citation.title == "Example Article"
        assert citation.start_index == 17
        assert citation.end_index == 20

    def test_parse_response_collects_unique_source_urls(self) -> None:
        """Parsing response collects unique source URLs."""
        codec = OpenAIWebSearchCodec()
        tool = openai_web_search()

        items = [
            MockResponseItem(type="web_search_call"),
            MockResponseItem(
                type="message",
                content=[
                    MockContentPart(
                        type="output_text",
                        text="Info from [1] and [2].",
                        annotations=[
                            MockAnnotation(
                                type="url_citation",
                                url="https://example.com/a",
                                title="Article A",
                                start_index=10,
                                end_index=13,
                            ),
                            MockAnnotation(
                                type="url_citation",
                                url="https://example.com/a",
                                title="Article A Again",
                                start_index=18,
                                end_index=21,
                            ),
                            MockAnnotation(
                                type="url_citation",
                                url="https://example.com/b",
                                title="Article B",
                                start_index=18,
                                end_index=21,
                            ),
                        ],
                    )
                ],
            ),
        ]

        result = codec.parse_output(items, tool)

        assert result is not None
        assert len(result.citations) == 3
        assert len(result.source_urls) == 2
        assert "https://example.com/a" in result.source_urls
        assert "https://example.com/b" in result.source_urls

    def test_parse_response_ignores_non_url_citations(self) -> None:
        """Parsing response ignores non-url_citation annotations."""
        codec = OpenAIWebSearchCodec()
        tool = openai_web_search()

        items = [
            MockResponseItem(type="web_search_call"),
            MockResponseItem(
                type="message",
                content=[
                    MockContentPart(
                        type="output_text",
                        text="Some text.",
                        annotations=[
                            MockAnnotation(
                                type="other_annotation",
                                url="https://example.com",
                            ),
                        ],
                    )
                ],
            ),
        ]

        result = codec.parse_output(items, tool)

        assert result is not None
        assert result.citations == ()

    def test_parse_response_without_message_content(self) -> None:
        """Parsing response without message content returns empty result."""
        codec = OpenAIWebSearchCodec()
        tool = openai_web_search()

        items = [
            MockResponseItem(type="web_search_call"),
        ]

        result = codec.parse_output(items, tool)

        assert result is not None
        assert result.text == ""
        assert result.citations == ()

    def test_parse_response_ignores_non_output_text_parts(self) -> None:
        """Parsing response ignores content parts that are not output_text."""
        codec = OpenAIWebSearchCodec()
        tool = openai_web_search()

        items = [
            MockResponseItem(type="web_search_call"),
            MockResponseItem(
                type="message",
                content=[
                    MockContentPart(
                        type="thinking",
                        text="This is thinking, not output.",
                    ),
                    MockContentPart(
                        type="output_text",
                        text="Actual output text.",
                        annotations=[],
                    ),
                ],
            ),
        ]

        result = codec.parse_output(items, tool)

        assert result is not None
        assert result.text == "Actual output text."


class TestOpenAIWebSearchSection:
    """Tests for the convenience web search section."""

    def test_section_creation(self) -> None:
        """Section can be created."""
        section = OpenAIWebSearchSection()
        assert section.key == "web_search"

    def test_section_with_custom_key(self) -> None:
        """Section respects custom key."""
        section = OpenAIWebSearchSection(key="custom_search")
        assert section.key == "custom_search"

    def test_section_exposes_hosted_tools(self) -> None:
        """Section exposes web search as hosted tool."""
        section = OpenAIWebSearchSection()
        hosted_tools = section.hosted_tools()

        assert len(hosted_tools) == 1
        tool = hosted_tools[0]
        assert tool.kind == "web_search"
        assert isinstance(tool.config, OpenAIWebSearchConfig)

    def test_section_with_config(self) -> None:
        """Section uses provided config."""
        config = OpenAIWebSearchConfig(
            filters=OpenAIWebSearchFilters(
                allowed_domains=("example.com",),
            ),
        )
        section = OpenAIWebSearchSection(config=config)
        hosted_tools = section.hosted_tools()

        assert hosted_tools[0].config.filters is not None
        assert hosted_tools[0].config.filters.allowed_domains == ("example.com",)


class TestOpenAIWebSearchResultTypes:
    """Tests for web search result data types."""

    def test_url_citation_immutable(self) -> None:
        """URL citation is immutable."""
        citation = OpenAIUrlCitation(
            url="https://example.com",
            title="Example",
            start_index=0,
            end_index=10,
        )
        with pytest.raises(AttributeError):
            citation.url = "https://other.com"  # type: ignore[misc]

    def test_web_search_result_immutable(self) -> None:
        """Web search result is immutable."""
        result = OpenAIWebSearchResult(text="Some text")
        with pytest.raises(AttributeError):
            result.text = "Other text"  # type: ignore[misc]


class TestEmptyParams:
    """Tests for EmptyParams type."""

    def test_empty_params_is_dataclass(self) -> None:
        """EmptyParams is a valid dataclass."""
        params = EmptyParams()
        assert params is not None

    def test_empty_params_is_frozen(self) -> None:
        """EmptyParams is immutable."""
        params = EmptyParams()
        # Frozen dataclass can raise AttributeError or TypeError depending on implementation
        with pytest.raises((AttributeError, TypeError)):
            params.foo = "bar"  # type: ignore[attr-defined]


class TestModuleDir:
    """Tests for module __dir__ function."""

    def test_module_dir_includes_exports(self) -> None:
        """Module __dir__ includes all expected exports."""
        from weakincentives.adapters import openai_hosted

        exported = dir(openai_hosted)
        assert "OpenAIWebSearchConfig" in exported
        assert "OpenAIWebSearchSection" in exported
        assert "openai_web_search" in exported

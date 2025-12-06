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

"""OpenAI web search hosted tool implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, override

from ...prompt.hosted_tool import HostedTool
from ...prompt.markdown import MarkdownSection

_MAX_ALLOWED_DOMAINS: Final = 100


@dataclass(slots=True, frozen=True)
class OpenAIWebSearchFilters:
    """Domain filtering for OpenAI web search.

    Attributes:
        allowed_domains: Domains to restrict search results to (max 100).
            When specified, only results from these domains are included.
        blocked_domains: Domains to exclude from search results.
            Results from these domains are never included.
    """

    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class OpenAIUserLocation:
    """Geographic context for OpenAI web search.

    Provides location hints to bias search results toward a specific region.

    Attributes:
        country: ISO 3166-1 alpha-2 country code (e.g., "US", "GB").
        city: City name.
        region: Region or state name.
        timezone: IANA timezone identifier (e.g., "America/New_York").
    """

    country: str | None = None
    city: str | None = None
    region: str | None = None
    timezone: str | None = None


@dataclass(slots=True, frozen=True)
class OpenAIWebSearchConfig:
    """Configuration for OpenAI web_search hosted tool.

    Attributes:
        filters: Domain filtering configuration.
        user_location: Geographic context for search results.
        search_context_size: Amount of context from search results.
            One of "low", "medium", or "high". Defaults to "medium".
    """

    filters: OpenAIWebSearchFilters | None = None
    user_location: OpenAIUserLocation | None = None
    search_context_size: str | None = None


@dataclass(slots=True, frozen=True)
class OpenAIUrlCitation:
    """A citation from OpenAI web search results.

    Attributes:
        url: The URL of the cited source.
        title: The title of the cited page.
        start_index: Character offset where citation begins in response text.
        end_index: Character offset where citation ends in response text.
    """

    url: str
    title: str
    start_index: int
    end_index: int


@dataclass(slots=True, frozen=True)
class OpenAIWebSearchResult:
    """Parsed output from OpenAI web_search invocation.

    Attributes:
        text: The response text that includes information from web search.
        citations: URL citations extracted from the response.
        source_urls: Unique source URLs from the search results.
    """

    text: str
    citations: tuple[OpenAIUrlCitation, ...] = ()
    source_urls: tuple[str, ...] = ()


class OpenAIWebSearchCodec:
    """Codec for OpenAI web_search hosted tool."""

    @property
    def kind(self) -> str:
        """Return the tool kind this codec handles."""
        return "web_search"

    def serialize(
        self,
        tool: HostedTool[OpenAIWebSearchConfig],
    ) -> dict[str, Any]:
        """Convert tool configuration to OpenAI wire format."""
        spec: dict[str, Any] = {"type": "web_search_preview"}
        config = tool.config

        self._add_filters(spec, config)
        self._add_user_location(spec, config)

        if config.search_context_size:
            spec["search_context_size"] = config.search_context_size

        return spec

    @staticmethod
    def _add_filters(spec: dict[str, Any], config: OpenAIWebSearchConfig) -> None:
        """Add domain filters to the spec if configured."""
        if not config.filters:
            return

        filters: dict[str, Any] = {}
        if config.filters.allowed_domains:
            filters["allowed_domains"] = list(config.filters.allowed_domains)
        if config.filters.blocked_domains:
            filters["blocked_domains"] = list(config.filters.blocked_domains)
        if filters:
            spec["filters"] = filters

    @staticmethod
    def _add_user_location(
        spec: dict[str, Any], config: OpenAIWebSearchConfig
    ) -> None:
        """Add user location to the spec if configured."""
        if not config.user_location:
            return

        loc = config.user_location
        location: dict[str, Any] = {"type": "approximate"}
        if loc.country:
            location["country"] = loc.country
        if loc.city:
            location["city"] = loc.city
        if loc.region:
            location["region"] = loc.region
        if loc.timezone:
            location["timezone"] = loc.timezone
        spec["user_location"] = location

    def parse_output(
        self,
        response_items: Sequence[object],
        tool: HostedTool[OpenAIWebSearchConfig],
    ) -> OpenAIWebSearchResult | None:
        """Extract typed output from provider response."""
        web_search_call, message_content = self._find_response_items(response_items)

        if web_search_call is None:
            return None

        text, citations, source_urls = self._extract_content(message_content)

        return OpenAIWebSearchResult(
            text=text,
            citations=tuple(citations),
            source_urls=tuple(sorted(source_urls)),
        )

    @staticmethod
    def _find_response_items(
        response_items: Sequence[object],
    ) -> tuple[object | None, object | None]:
        """Find web_search_call and message items in response."""
        web_search_call = None
        message_content = None

        for item in response_items:
            item_type = getattr(item, "type", None)
            if item_type == "web_search_call":
                web_search_call = item
            elif item_type == "message":
                message_content = item

        return web_search_call, message_content

    @staticmethod
    def _extract_content(
        message_content: object | None,
    ) -> tuple[str, list[OpenAIUrlCitation], set[str]]:
        """Extract text, citations, and source URLs from message content."""
        text = ""
        citations: list[OpenAIUrlCitation] = []
        source_urls: set[str] = set()

        if not message_content:
            return text, citations, source_urls

        content = getattr(message_content, "content", [])
        for part in content:
            if getattr(part, "type", None) != "output_text":
                continue

            text = getattr(part, "text", "")
            for ann in getattr(part, "annotations", []):
                if getattr(ann, "type", None) != "url_citation":
                    continue

                url = getattr(ann, "url", "")
                citations.append(
                    OpenAIUrlCitation(
                        url=url,
                        title=getattr(ann, "title", ""),
                        start_index=getattr(ann, "start_index", 0),
                        end_index=getattr(ann, "end_index", 0),
                    )
                )
                if url:
                    source_urls.add(url)

        return text, citations, source_urls


def openai_web_search(
    config: OpenAIWebSearchConfig | None = None,
    *,
    name: str = "web_search",
) -> HostedTool[OpenAIWebSearchConfig]:
    """Create an OpenAI web search hosted tool.

    Args:
        config: Optional web search configuration. Defaults to an unconfigured
            search with no domain restrictions.
        name: Tool name for the registry. Defaults to "web_search".

    Returns:
        A HostedTool configured for OpenAI web search.
    """
    return HostedTool(
        kind="web_search",
        name=name,
        description="Search the web for current information and cite sources.",
        config=config or OpenAIWebSearchConfig(),
    )


_OPENAI_WEB_SEARCH_TEMPLATE: Final = """\
You have access to web search capabilities. Use web search to find current \
information when needed. Always cite your sources using the provided URL citations."""


@dataclass(slots=True, frozen=True)
class EmptyParams:
    """Empty params type for sections that don't need parameters."""

    pass


class OpenAIWebSearchSection(MarkdownSection[EmptyParams]):
    """Section that enables OpenAI web search.

    This convenience section adds a web search hosted tool with optional
    configuration and includes standard instructions for using web search.

    Example:
        >>> section = OpenAIWebSearchSection(
        ...     config=OpenAIWebSearchConfig(
        ...         filters=OpenAIWebSearchFilters(
        ...             allowed_domains=("example.com",),
        ...         ),
        ...     ),
        ... )
    """

    def __init__(
        self,
        config: OpenAIWebSearchConfig | None = None,
        *,
        key: str = "web_search",
    ) -> None:
        """Initialize the web search section.

        Args:
            config: Optional web search configuration.
            key: Section key for the registry. Defaults to "web_search".
        """
        self._hosted_tool = openai_web_search(config)
        super().__init__(
            title="Web Search",
            key=key,
            template=_OPENAI_WEB_SEARCH_TEMPLATE,
        )

    @override
    def hosted_tools(self) -> tuple[HostedTool[OpenAIWebSearchConfig], ...]:
        """Return the web search hosted tool."""
        return (self._hosted_tool,)


__all__ = [
    "EmptyParams",
    "OpenAIUrlCitation",
    "OpenAIUserLocation",
    "OpenAIWebSearchCodec",
    "OpenAIWebSearchConfig",
    "OpenAIWebSearchFilters",
    "OpenAIWebSearchResult",
    "OpenAIWebSearchSection",
    "openai_web_search",
]

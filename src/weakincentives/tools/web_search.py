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

"""Provider-managed web search tool declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..prompt import NativeTool, ToolExample

WebSearchCallStatus = Literal["in_progress", "searching", "completed", "failed"]
WebSearchActionType = Literal["search", "open_page", "find"]
WebSearchToolType = Literal["web_search", "web_search_2025_08_26"]


@dataclass(slots=True, frozen=True)
class WebSearchSource:
    """Source URL captured in a provider web search call."""

    url: str
    type: Literal["url"] = "url"


@dataclass(slots=True, frozen=True)
class WebSearchAction:
    """Action performed as part of a web search tool call."""

    type: WebSearchActionType
    query: str | None = None
    url: str | None = None
    pattern: str | None = None
    sources: tuple[WebSearchSource, ...] = field(default_factory=tuple)

    def render(self) -> str:
        """Return a concise human-readable summary of the action."""

        if self.type == "search" and self.query:
            return f"search query: {self.query}"
        if self.type == "open_page" and self.url:
            return f"opened page: {self.url}"
        if self.type == "find" and self.pattern and self.url:
            return f"searched for '{self.pattern}' in {self.url}"
        return self.type


@dataclass(slots=True, frozen=True)
class WebSearchCall:
    """Provider-emitted web search call telemetry."""

    id: str
    status: WebSearchCallStatus
    action: WebSearchAction
    type: Literal["web_search_call"] = "web_search_call"

    def render(self) -> str:
        prefix = f"web search ({self.status})"
        action_summary = self.action.render()
        return f"{prefix}: {action_summary}" if action_summary else prefix


@dataclass(slots=True, frozen=True)
class WebSearchFilters:
    """Restrict provider search results to the configured domains."""

    allowed_domains: tuple[str, ...] | None = None


@dataclass(slots=True, frozen=True)
class WebSearchLocation:
    """Approximate user location supplied to the provider."""

    city: str | None = None
    country: str | None = None
    region: str | None = None
    timezone: str | None = None
    type: Literal["approximate"] = "approximate"


@dataclass(slots=True)
class WebSearchTool(NativeTool[WebSearchCall, WebSearchCall]):
    """Native web search tool executed by the provider."""

    provider_type: WebSearchToolType = "web_search"
    search_context_size: Literal["low", "medium", "high"] | None = None
    filters: WebSearchFilters | None = None
    user_location: WebSearchLocation | None = None
    params_type: type[WebSearchCall] = field(
        init=False, repr=False, default=WebSearchCall
    )
    _result_annotation: WebSearchCall = field(
        init=False,
        repr=False,
        default=WebSearchCall(
            id="preview",
            status="completed",
            action=WebSearchAction(type="search", query=""),
        ),
    )


def build_web_search_tool(
    *,
    description: str = "Use the provider's built-in web search.",
    provider_type: WebSearchToolType = "web_search",
    search_context_size: Literal["low", "medium", "high"] | None = None,
    filters: WebSearchFilters | None = None,
    user_location: WebSearchLocation | None = None,
) -> WebSearchTool:
    """Return a configured web search tool declaration."""

    example = ToolExample(
        description="Search the web for fresh information.",
        input=WebSearchCall(
            id="search_123",
            status="searching",
            action=WebSearchAction(type="search", query="latest OpenAI news"),
        ),
        output=WebSearchCall(
            id="search_123",
            status="completed",
            action=WebSearchAction(
                type="search",
                query="latest OpenAI news",
                sources=(WebSearchSource(url="https://openai.com"),),
            ),
        ),
    )

    return WebSearchTool(
        name="web_search",
        description=description,
        examples=(example,),
        provider_type=provider_type,
        search_context_size=search_context_size,
        filters=filters,
        user_location=user_location,
    )


__all__ = [
    "WebSearchAction",
    "WebSearchActionType",
    "WebSearchCall",
    "WebSearchCallStatus",
    "WebSearchFilters",
    "WebSearchLocation",
    "WebSearchTool",
    "WebSearchToolType",
    "build_web_search_tool",
]

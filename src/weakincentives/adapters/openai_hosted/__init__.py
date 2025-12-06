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

"""OpenAI hosted tools for provider-executed capabilities.

This module provides OpenAI-specific hosted tool implementations that run
server-side rather than locally. Unlike user-defined tools with handlers,
hosted tools delegate execution to the provider.

Example usage:

    >>> from weakincentives.adapters.openai_hosted import (
    ...     OpenAIWebSearchSection,
    ...     OpenAIWebSearchConfig,
    ...     OpenAIWebSearchFilters,
    ... )
    >>>
    >>> # Create a domain-restricted web search section
    >>> config = OpenAIWebSearchConfig(
    ...     filters=OpenAIWebSearchFilters(
    ...         allowed_domains=("pubmed.ncbi.nlm.nih.gov", "www.cdc.gov"),
    ...     ),
    ... )
    >>> section = OpenAIWebSearchSection(config=config)
"""

from .web_search import (
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


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})

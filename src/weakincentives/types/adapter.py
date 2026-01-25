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

"""Adapter name constants for LLM provider integrations.

This module defines canonical string identifiers for each supported LLM provider
adapter. Use these constants instead of hardcoded strings when checking or
configuring which adapter is in use.

Constants:
    OPENAI_ADAPTER_NAME: Identifier for the OpenAI API adapter.
    LITELLM_ADAPTER_NAME: Identifier for the LiteLLM proxy adapter.
    CLAUDE_AGENT_SDK_ADAPTER_NAME: Identifier for the Claude Agent SDK adapter.

Type Aliases:
    AdapterName: String type alias for adapter identifiers.

Example:
    >>> from weakincentives.types import AdapterName, OPENAI_ADAPTER_NAME
    >>> def get_client(adapter: AdapterName) -> Client:
    ...     if adapter == OPENAI_ADAPTER_NAME:
    ...         return OpenAIClient()
    ...     raise ValueError(f"Unknown adapter: {adapter}")
"""

from __future__ import annotations

from typing import Final

AdapterName = str
"""String type alias for adapter identifiers.

Use this type in function signatures that accept or return adapter names,
ensuring type checkers recognize the semantic meaning of the string.
"""

OPENAI_ADAPTER_NAME: Final[AdapterName] = "openai"
"""Canonical identifier for the OpenAI adapter.

Use this constant when configuring or detecting the OpenAI API integration
in ``weakincentives.adapters.openai``.
"""

LITELLM_ADAPTER_NAME: Final[AdapterName] = "litellm"
"""Canonical identifier for the LiteLLM adapter.

Use this constant when configuring or detecting the LiteLLM proxy integration
in ``weakincentives.adapters.litellm``, which provides unified access to
multiple LLM providers through a common interface.
"""

CLAUDE_AGENT_SDK_ADAPTER_NAME: Final[AdapterName] = "claude_agent_sdk"
"""Canonical identifier for the Claude Agent SDK adapter.

Use this constant when configuring or detecting the Claude Agent SDK integration
in ``weakincentives.adapters.claude_agent_sdk``.
"""

__all__ = [
    "CLAUDE_AGENT_SDK_ADAPTER_NAME",
    "LITELLM_ADAPTER_NAME",
    "OPENAI_ADAPTER_NAME",
    "AdapterName",
]

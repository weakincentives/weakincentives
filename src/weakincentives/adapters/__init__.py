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

"""LLM provider adapters for evaluating prompts against different backends.

This package provides a unified adapter pattern for executing weakincentives prompts
against various LLM providers. Each adapter implements the :class:`ProviderAdapter`
protocol, which defines a common ``evaluate()`` method for synchronous prompt
evaluation with tool execution, structured output, and budget tracking.

Adapter Pattern
---------------
All adapters share a common interface:

1. Accept a rendered :class:`~weakincentives.prompt.Prompt` with sections and tools
2. Execute an inner loop: call provider -> handle tool calls -> repeat until done
3. Return a :class:`PromptResponse` with text and parsed structured output
4. Support deadline enforcement, budget tracking, and heartbeat signaling

Supported Providers
-------------------
The package includes adapters for three providers:

**OpenAI** (``weakincentives.adapters.openai``)
    Uses the OpenAI Responses API for models like GPT-4o. Requires the ``openai``
    optional dependency: ``pip install weakincentives[openai]``

**LiteLLM** (``weakincentives.adapters.litellm``)
    Multi-provider gateway supporting 100+ models through a unified interface.
    Requires the ``litellm`` optional dependency: ``pip install weakincentives[litellm]``

**Claude Agent SDK** (``weakincentives.adapters.claude_agent_sdk``)
    Full agentic capabilities via Claude Code's SDK with native tool execution,
    workspace isolation, and hook-based state synchronization. Requires the
    ``claude-agent-sdk`` optional dependency: ``pip install weakincentives[claude-agent-sdk]``

Configuration Classes
---------------------
Each adapter has typed configuration dataclasses:

- :class:`LLMConfig`: Base model parameters (temperature, max_tokens, etc.)
- :class:`OpenAIClientConfig`: OpenAI client settings (api_key, base_url, etc.)
- :class:`OpenAIModelConfig`: OpenAI-specific model parameters
- :class:`LiteLLMClientConfig`: LiteLLM completion settings
- :class:`LiteLLMModelConfig`: LiteLLM-specific model parameters

Throttle Handling
-----------------
Adapters detect rate limits, quota exhaustion, and timeouts, raising
:class:`ThrottleError` with retry guidance. Use :class:`ThrottlePolicy` to
configure exponential backoff retry behavior:

- ``max_attempts``: Maximum retry attempts
- ``base_delay``: Initial backoff delay
- ``max_delay``: Maximum backoff delay
- ``max_total_delay``: Maximum total time to spend retrying

Usage Example
-------------
Basic OpenAI adapter usage::

    from weakincentives import Prompt, PromptTemplate, MarkdownSection
    from weakincentives.adapters.openai import OpenAIAdapter, OpenAIModelConfig
    from weakincentives.runtime import Session, InProcessDispatcher
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Summary:
        title: str
        key_points: list[str]

    template = PromptTemplate[Summary](
        ns="example",
        key="summarize",
        sections=(
            MarkdownSection(
                title="Task",
                key="task",
                template="Summarize the following text: {{ text }}",
            ),
        ),
    )

    adapter = OpenAIAdapter(
        model="gpt-4o",
        model_config=OpenAIModelConfig(temperature=0.7, max_tokens=1000),
    )

    session = Session(dispatcher=InProcessDispatcher())
    prompt = Prompt(template).bind(params={"text": "..."})

    response = adapter.evaluate(prompt, session=session)
    print(response.output)  # Summary(title=..., key_points=[...])

Public Exports
--------------
Core Interfaces:
    - :class:`ProviderAdapter`: Abstract base class for all adapters
    - :class:`PromptResponse`: Structured result from adapter evaluation
    - :class:`PromptEvaluationError`: Error during prompt evaluation

Configuration:
    - :class:`LLMConfig`: Base model parameter configuration
    - :class:`OpenAIClientConfig`: OpenAI client instantiation settings
    - :class:`OpenAIModelConfig`: OpenAI-specific model parameters
    - :class:`LiteLLMClientConfig`: LiteLLM completion settings
    - :class:`LiteLLMModelConfig`: LiteLLM-specific model parameters

Throttle Handling:
    - :class:`ThrottleError`: Exception for rate limit/quota/timeout errors
    - :class:`ThrottlePolicy`: Configuration for retry backoff behavior
    - :func:`new_throttle_policy`: Factory function with validation

Adapter Names:
    - :data:`AdapterName`: Type alias for adapter identifiers
    - :data:`OPENAI_ADAPTER_NAME`: Identifier for OpenAI adapter
    - :data:`LITELLM_ADAPTER_NAME`: Identifier for LiteLLM adapter
    - :data:`CLAUDE_AGENT_SDK_ADAPTER_NAME`: Identifier for Claude Agent SDK adapter

See Also
--------
- :mod:`weakincentives.adapters.openai`: OpenAI adapter implementation
- :mod:`weakincentives.adapters.litellm`: LiteLLM adapter implementation
- :mod:`weakincentives.adapters.claude_agent_sdk`: Claude Agent SDK adapter
- :mod:`weakincentives.prompt`: Prompt and template construction
- :mod:`weakincentives.runtime`: Session and event infrastructure
"""

from __future__ import annotations

from ..types import (
    CLAUDE_AGENT_SDK_ADAPTER_NAME,
    LITELLM_ADAPTER_NAME,
    OPENAI_ADAPTER_NAME,
    AdapterName,
)
from .config import (
    LiteLLMClientConfig,
    LiteLLMModelConfig,
    LLMConfig,
    OpenAIClientConfig,
    OpenAIModelConfig,
)
from .core import (
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
)
from .throttle import ThrottleError, ThrottlePolicy, new_throttle_policy

__all__ = [
    "CLAUDE_AGENT_SDK_ADAPTER_NAME",
    "LITELLM_ADAPTER_NAME",
    "OPENAI_ADAPTER_NAME",
    "AdapterName",
    "LLMConfig",
    "LiteLLMClientConfig",
    "LiteLLMModelConfig",
    "OpenAIClientConfig",
    "OpenAIModelConfig",
    "PromptEvaluationError",
    "PromptResponse",
    "ProviderAdapter",
    "ThrottleError",
    "ThrottlePolicy",
    "new_throttle_policy",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *(__all__)})

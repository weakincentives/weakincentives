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
WINK only integrates with agentic harnesses and their SDKs. Native SDK integrations
are too low-level to qualify as an execution harness.

**Claude Agent SDK** (``weakincentives.adapters.claude_agent_sdk``)
    Full agentic capabilities via Claude Code's SDK with native tool execution,
    workspace isolation, and hook-based state synchronization. Requires the
    ``claude-agent-sdk`` optional dependency: ``pip install weakincentives[claude-agent-sdk]``

Configuration Classes
---------------------
- :class:`LLMConfig`: Base model parameters (temperature, max_tokens, etc.)

Throttle Handling
-----------------
Adapters detect rate limits, quota exhaustion, and timeouts, raising
:class:`ThrottleError` with retry guidance. Use :class:`ThrottlePolicy` to
configure exponential backoff retry behavior:

- ``max_attempts``: Maximum retry attempts
- ``base_delay``: Initial backoff delay
- ``max_delay``: Maximum backoff delay
- ``max_total_delay``: Maximum total time to spend retrying

Public Exports
--------------
Core Interfaces:
    - :class:`ProviderAdapter`: Abstract base class for all adapters
    - :class:`PromptResponse`: Structured result from adapter evaluation
    - :class:`PromptEvaluationError`: Error during prompt evaluation

Configuration:
    - :class:`LLMConfig`: Base model parameter configuration

Throttle Handling:
    - :class:`ThrottleError`: Exception for rate limit/quota/timeout errors
    - :class:`ThrottlePolicy`: Configuration for retry backoff behavior
    - :func:`new_throttle_policy`: Factory function with validation

Adapter Names:
    - :data:`AdapterName`: Type alias for adapter identifiers
    - :data:`CLAUDE_AGENT_SDK_ADAPTER_NAME`: Identifier for Claude Agent SDK adapter

See Also
--------
- :mod:`weakincentives.adapters.claude_agent_sdk`: Claude Agent SDK adapter
- :mod:`weakincentives.prompt`: Prompt and template construction
- :mod:`weakincentives.runtime`: Session and event infrastructure
"""

from __future__ import annotations

from ..types import (
    CLAUDE_AGENT_SDK_ADAPTER_NAME,
    CODEX_APP_SERVER_ADAPTER_NAME,
    AdapterName,
)
from .config import (
    LLMConfig,
)
from .core import (
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
)
from .throttle import ThrottleError, ThrottlePolicy, new_throttle_policy

__all__ = [
    "CLAUDE_AGENT_SDK_ADAPTER_NAME",
    "CODEX_APP_SERVER_ADAPTER_NAME",
    "AdapterName",
    "LLMConfig",
    "PromptEvaluationError",
    "PromptResponse",
    "ProviderAdapter",
    "ThrottleError",
    "ThrottlePolicy",
    "new_throttle_policy",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *(__all__)})

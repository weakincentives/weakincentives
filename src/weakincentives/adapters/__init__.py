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

"""Integration adapters for optional third-party providers."""

from __future__ import annotations

from ._names import (
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
from .shared import ThrottleError, ThrottlePolicy, new_throttle_policy
from .tool_runner import ToolRunner

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
    "ToolRunner",
    "new_throttle_policy",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *(__all__)})

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

from importlib import import_module
from typing import TYPE_CHECKING

from .core import (
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
)

__all__ = [
    "ProviderAdapter",
    "PromptResponse",
    "PromptEvaluationError",
    "LiteLLMAdapter",
    "LiteLLMCompletion",
    "create_litellm_completion",
    "OpenAIAdapter",
    "OpenAIProtocol",
]

_OPTIONAL_ATTR_MODULE: dict[str, str] = {
    "LiteLLMAdapter": "litellm",
    "LiteLLMCompletion": "litellm",
    "create_litellm_completion": "litellm",
    "OpenAIAdapter": "openai",
    "OpenAIProtocol": "openai",
}

if TYPE_CHECKING:  # pragma: no cover - imports only for static analysis
    from .litellm import LiteLLMAdapter, LiteLLMCompletion, create_litellm_completion
    from .openai import OpenAIAdapter, OpenAIProtocol


def __getattr__(name: str) -> object:
    """Lazily load optional adapter modules on first attribute access."""

    module_name = _OPTIONAL_ATTR_MODULE.get(name)
    if module_name is None:
        raise AttributeError(  # pragma: no cover - defensive guard
            f"module '{__name__}' has no attribute '{name}'"
        ) from None

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - convenience for REPL
    return sorted({*globals().keys(), *(__all__)})

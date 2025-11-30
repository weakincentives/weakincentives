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

"""Adapter namespace exposing the :mod:`weakincentives.adapters.api` surface."""

from __future__ import annotations

from . import api
from .api import (
    LITELLM_ADAPTER_NAME,
    OPENAI_ADAPTER_NAME,
    AdapterName,
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
    ThrottleError,
    ThrottlePolicy,
    new_throttle_policy,
)

__all__ = [  # noqa: RUF022
    "AdapterName",
    "LITELLM_ADAPTER_NAME",
    "OPENAI_ADAPTER_NAME",
    "PromptEvaluationError",
    "PromptResponse",
    "ProviderAdapter",
    "SessionProtocol",
    "ThrottleError",
    "ThrottlePolicy",
    "api",
    "new_throttle_policy",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *(__all__)})

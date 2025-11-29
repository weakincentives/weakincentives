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

"""Adapter namespace exposing the package's public types."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import *  # noqa: F403

__all__ = [
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

api: object | None = None


def __getattr__(name: str) -> object:
    module = globals().get("api")
    if module is None:
        module = import_module(f"{__name__}.api")
        globals()["api"] = module
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)

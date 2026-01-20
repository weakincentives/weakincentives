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

"""Shared typing helpers for :mod:`weakincentives`."""

from __future__ import annotations

from ._guards import (
    ensure_type,
    is_instance_of,
    narrow_optional,
)
from .adapter import (
    CLAUDE_AGENT_SDK_ADAPTER_NAME,
    LITELLM_ADAPTER_NAME,
    OPENAI_ADAPTER_NAME,
    AdapterName,
)
from .dataclass import (
    DataclassFieldMapping,
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
    is_dataclass_instance,
)
from .json import (
    ContractResult,
    JSONArray,
    JSONArrayT,
    JSONObject,
    JSONObjectT,
    JSONValue,
    ParseableDataclassT,
)

__all__ = [
    "CLAUDE_AGENT_SDK_ADAPTER_NAME",
    "LITELLM_ADAPTER_NAME",
    "OPENAI_ADAPTER_NAME",
    "AdapterName",
    "ContractResult",
    "DataclassFieldMapping",
    "JSONArray",
    "JSONArrayT",
    "JSONObject",
    "JSONObjectT",
    "JSONValue",
    "ParseableDataclassT",
    "SupportsDataclass",
    "SupportsDataclassOrNone",
    "SupportsToolResult",
    "ensure_type",
    "is_dataclass_instance",
    "is_instance_of",
    "narrow_optional",
]

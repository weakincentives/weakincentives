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

"""Token usage extraction utilities for provider adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ..runtime.events import TokenUsage


def _coerce_token_count(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        coerced = int(value)
        return coerced if coerced >= 0 else None
    return None


def token_usage_from_payload(payload: Mapping[str, Any] | None) -> TokenUsage | None:
    """Extract token usage metrics from a provider payload when present."""

    if not isinstance(payload, Mapping):
        return None
    usage_value = payload.get("usage")
    if not isinstance(usage_value, Mapping):
        return None
    usage_payload = cast(Mapping[str, object], usage_value)

    input_tokens = _coerce_token_count(
        usage_payload.get("input_tokens") or usage_payload.get("prompt_tokens")
    )
    output_tokens = _coerce_token_count(
        usage_payload.get("output_tokens") or usage_payload.get("completion_tokens")
    )
    cached_tokens = _coerce_token_count(usage_payload.get("cached_tokens"))

    if all(value is None for value in (input_tokens, output_tokens, cached_tokens)):
        return None

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )


__all__ = [
    "token_usage_from_payload",
]

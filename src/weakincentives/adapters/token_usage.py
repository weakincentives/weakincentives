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
    """Extract token usage metrics from a provider API response payload.

    Parses the ``usage`` field from provider responses, supporting both
    Anthropic-style (``input_tokens``, ``output_tokens``) and OpenAI-style
    (``prompt_tokens``, ``completion_tokens``) naming conventions.

    Args:
        payload: The raw API response mapping from a provider. Expected to
            contain a nested ``usage`` mapping with token count fields.
            May be ``None`` or lack usage data.

    Returns:
        A ``TokenUsage`` instance if any valid token counts were found,
        or ``None`` if the payload is missing, malformed, or contains
        no extractable usage data.

    Example:
        >>> from weakincentives.adapters.token_usage import token_usage_from_payload
        >>> # Anthropic-style response
        >>> payload = {"usage": {"input_tokens": 100, "output_tokens": 50}}
        >>> usage = token_usage_from_payload(payload)
        >>> usage.input_tokens
        100
        >>> # OpenAI-style response
        >>> payload = {"usage": {"prompt_tokens": 80, "completion_tokens": 40}}
        >>> usage = token_usage_from_payload(payload)
        >>> usage.input_tokens
        80

    Note:
        Negative token counts are treated as invalid and converted to ``None``.
        The ``cached_tokens`` field is extracted if present in the usage data.
    """

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

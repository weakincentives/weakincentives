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

"""Shared guardrail utilities used by multiple adapter implementations.

Adapters that use a turn-based protocol (ACP and Codex App Server) share
identical logic for accumulating token usage across continuation rounds.
This module centralises that shared logic to avoid duplication.

Adapter-specific concerns such as feedback content type and log event names
are intentionally kept in each adapter's own ``_guardrails`` module.
"""

from __future__ import annotations

from ...budget import TokenUsage


def accumulate_usage(current: TokenUsage | None, new: TokenUsage) -> TokenUsage:
    """Sum token usage across continuation rounds.

    Returns *new* unchanged when *current* is ``None`` (first round).
    Treats ``None`` token counts as zero when summing.
    """
    if current is None:
        return new
    return TokenUsage(
        input_tokens=(current.input_tokens or 0) + (new.input_tokens or 0),
        output_tokens=(current.output_tokens or 0) + (new.output_tokens or 0),
        cached_tokens=(current.cached_tokens or 0) + (new.cached_tokens or 0),
    )


__all__ = ["accumulate_usage"]

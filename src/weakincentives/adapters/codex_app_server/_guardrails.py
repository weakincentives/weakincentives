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

"""Guardrails support for the Codex App Server adapter.

Provides feedback collection after tool calls.  Task completion is
handled generically by :class:`JsonRpcAdapter._check_task_completion`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...deadlines import Deadline
from ...prompt.feedback import collect_feedback
from ...runtime.session.protocols import SessionProtocol
from .._shared._guardrails import accumulate_usage, resolve_filesystem

if TYPE_CHECKING:
    from ...prompt.protocols import PromptProtocol

__all__ = [
    "accumulate_usage",
    "append_feedback",
    "resolve_filesystem",
]


def append_feedback(
    content_items: list[dict[str, str]],
    *,
    is_error: bool,
    prompt: PromptProtocol[Any] | None,
    session: SessionProtocol | None,
    deadline: Deadline | None,
) -> None:
    """Collect and append feedback after a successful tool call."""
    if is_error or prompt is None or session is None:
        return
    feedback_text = collect_feedback(prompt=prompt, session=session, deadline=deadline)
    if feedback_text:
        content_items.append({"type": "inputText", "text": feedback_text})

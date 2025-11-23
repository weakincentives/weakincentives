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

"""Shared helpers for validating tool execution context."""

from __future__ import annotations

from ..prompt.tools import ToolContext
from ..runtime.session import Session


def ensure_context_uses_session(*, context: ToolContext, session: Session) -> None:
    """Verify ``context`` matches the ``session`` bound to the tool section."""

    if context.session is not session:
        message = (
            "ToolContext session does not match the section session. "
            "Ensure the tool is invoked with the bound session."
        )
        raise RuntimeError(message)
    if context.event_bus is not session.event_bus:
        message = (
            "ToolContext event bus does not match the section session bus. "
            "Ensure the tool is invoked with the bound event bus."
        )
        raise RuntimeError(message)


__all__ = ["ensure_context_uses_session"]

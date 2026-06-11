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

"""Shared sandbox helpers for provider adapters.

Sandbox lifecycle lives on the adapter base class
(:meth:`~weakincentives.adapters.core.ProviderAdapter.open_sandbox` and
the ``evaluate``/``_evaluate`` lease fork). This module holds the
prompt-side glue adapters run per evaluation against the open sandbox.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...prompt.workspace import workspace_preview_params

if TYPE_CHECKING:
    from ...prompt import Prompt
    from ...sandbox import Sandbox

__all__ = ["bind_workspace_preview"]


def bind_workspace_preview(prompt: Prompt[Any], sandbox: Sandbox) -> None:
    """Bind the workspace preview params from the open sandbox.

    Called at the start of every ``_evaluate`` so the rendered prompt
    describes the environment the agent actually acts on. Rebinding is
    idempotent (same-type params replace) and refreshes the listing on
    re-evaluation — a visibility-expansion retry sees files written in
    earlier rounds. No-op when the template declares no sandbox (no
    preview section exists to consume the params).
    """
    if prompt.template.sandbox is not None:
        _ = prompt.bind(workspace_preview_params(sandbox.filesystem))

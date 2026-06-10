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

"""Shared sandbox lifecycle for provider adapters.

Every adapter evaluation runs against exactly one sandbox: the adapter
reads the prompt's :class:`~weakincentives.sandbox.SandboxConfig` (an
empty config when none is declared), asks its provider to materialize it,
threads the opened sandbox into tool execution, points the harness ``cwd``
at ``sandbox.root``, and closes the sandbox when evaluation completes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...prompt.workspace import workspace_preview_params
from ...sandbox import SandboxConfig

if TYPE_CHECKING:
    from ...prompt import Prompt
    from ...sandbox import Sandbox, SandboxProvider

__all__ = ["open_prompt_sandbox"]


def open_prompt_sandbox(prompt: Prompt[Any], provider: SandboxProvider) -> Sandbox:
    """Materialize the prompt's sandbox and bind the workspace preview.

    Opens the prompt template's :class:`SandboxConfig` (an empty config
    when the template declares none) through ``provider``. When the
    template declares environment intent, the workspace preview params are
    bound from the *opened* sandbox so the rendered prompt describes the
    environment the agent actually acts on.

    The caller owns the returned sandbox and must ``close()`` it.
    """
    config = prompt.template.sandbox
    sandbox = provider.open(config if config is not None else SandboxConfig())
    try:
        if config is not None:
            _ = prompt.bind(workspace_preview_params(sandbox.filesystem))
    except Exception:
        sandbox.close()
        raise
    return sandbox

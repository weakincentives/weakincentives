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

"""Prompt optimizers for workspace analysis and digest generation.

This package provides specialized optimizers for prompt optimization workflows.

Available optimizers:

- :class:`WorkspaceDigestOptimizer`: Generates workspace digests using the
  Claude Agent SDK adapter. Creates task-agnostic summaries of workspaces
  that can be cached in sessions for future use.

Example::

    from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer
    from weakincentives.adapters.claude_agent_sdk import HostMount
    from weakincentives.runtime import Session

    session = Session()
    optimizer = WorkspaceDigestOptimizer(
        mounts=[HostMount(host_path="/path/to/project")],
    )
    result = optimizer.optimize(session, section_key="workspace-digest")
"""

from __future__ import annotations

from .workspace_digest import WorkspaceDigestOptimizer, WorkspaceDigestResult

__all__ = [
    "WorkspaceDigestOptimizer",
    "WorkspaceDigestResult",
]

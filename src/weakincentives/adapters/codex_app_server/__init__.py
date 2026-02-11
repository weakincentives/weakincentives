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

"""Codex App Server adapter for weakincentives.

This adapter evaluates WINK prompts by delegating execution to Codex via its
app-server protocol (the same interface powering the Codex VS Code extension).
WINK tools are bridged as Codex dynamic tools over the same stdio channel.

No external Python dependencies beyond WINK and the ``codex`` CLI on PATH.
"""

from __future__ import annotations

from .adapter import CODEX_APP_SERVER_ADAPTER_NAME, CodexAppServerAdapter
from .client import CodexAppServerClient, CodexClientError
from .config import (
    ApiKeyAuth,
    ApprovalPolicy,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
    CodexAuthMode,
    ExternalTokenAuth,
    McpServerConfig,
    Personality,
    ReasoningEffort,
    ReasoningSummary,
    SandboxMode,
)

__all__ = [
    "CODEX_APP_SERVER_ADAPTER_NAME",
    "ApiKeyAuth",
    "ApprovalPolicy",
    "CodexAppServerAdapter",
    "CodexAppServerClient",
    "CodexAppServerClientConfig",
    "CodexAppServerModelConfig",
    "CodexAuthMode",
    "CodexClientError",
    "ExternalTokenAuth",
    "McpServerConfig",
    "Personality",
    "ReasoningEffort",
    "ReasoningSummary",
    "SandboxMode",
]

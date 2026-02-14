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

"""Semantic adapter name definitions shared across provider integrations."""

from __future__ import annotations

from typing import Final

AdapterName = str
"""Adapter identifier for provider integrations."""

CLAUDE_AGENT_SDK_ADAPTER_NAME: Final[AdapterName] = "claude_agent_sdk"
"""Canonical label for the Claude Agent SDK adapter."""

CODEX_APP_SERVER_ADAPTER_NAME: Final[AdapterName] = "codex_app_server"
"""Canonical label for the Codex App Server adapter."""

ACP_ADAPTER_NAME: Final[AdapterName] = "acp"
"""Canonical label for the generic ACP adapter."""

OPENCODE_ACP_ADAPTER_NAME: Final[AdapterName] = "opencode_acp"
"""Canonical label for the OpenCode ACP adapter."""

__all__ = [
    "ACP_ADAPTER_NAME",
    "CLAUDE_AGENT_SDK_ADAPTER_NAME",
    "CODEX_APP_SERVER_ADAPTER_NAME",
    "OPENCODE_ACP_ADAPTER_NAME",
    "AdapterName",
]

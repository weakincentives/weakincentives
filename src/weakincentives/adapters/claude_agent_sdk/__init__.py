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

"""Claude Agent SDK adapter for weakincentives.

This adapter enables weakincentives prompts to leverage Claude's full agentic
capabilities through the official claude-code-sdk Python package. It uses the
SDK's hook system to synchronize state bidirectionally between SDK execution
and the weakincentives Session.

Example:
    >>> from weakincentives import Prompt, MarkdownSection
    >>> from weakincentives.runtime import Session, InProcessDispatcher
    >>> from weakincentives.adapters.claude_agent_sdk import (
    ...     ClaudeAgentSDKAdapter,
    ...     ClaudeAgentSDKClientConfig,
    ...     ClaudeAgentWorkspaceSection,
    ...     HostMount,
    ... )
    >>>
    >>> dispatcher = InProcessDispatcher()
    >>> session = Session(dispatcher=dispatcher)
    >>>
    >>> # Create workspace section with host files
    >>> workspace = ClaudeAgentWorkspaceSection(
    ...     session=session,
    ...     mounts=[HostMount(host_path="src")],
    ...     allowed_host_roots=["/home/user/project"],
    ... )
    >>>
    >>> adapter = ClaudeAgentSDKAdapter(
    ...     model="claude-sonnet-4-5-20250929",
    ...     client_config=ClaudeAgentSDKClientConfig(
    ...         permission_mode="acceptEdits",
    ...         cwd=str(workspace.temp_dir),
    ...     ),
    ... )
    >>>
    >>> from weakincentives import PromptTemplate
    >>> from dataclasses import dataclass
    >>>
    >>> @dataclass(frozen=True)
    ... class ReviewResult:
    ...     summary: str
    ...     issues: list[str]
    ...
    >>> template = PromptTemplate[ReviewResult](
    ...     ns="test",
    ...     key="review",
    ...     sections=(
    ...         MarkdownSection(
    ...             title="Task",
    ...             key="task",
    ...             template="Review the code in src/",
    ...         ),
    ...         workspace,
    ...     ),
    ... )
    >>> prompt = Prompt(template)
    >>>
    >>> response = adapter.evaluate(prompt, session=session)
    >>> workspace.cleanup()
"""

from __future__ import annotations

from ._hooks import create_task_completion_stop_hook
from ._task_completion import (
    CompositeChecker,
    PlanBasedChecker,
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
)
from .adapter import CLAUDE_AGENT_SDK_ADAPTER_NAME, ClaudeAgentSDKAdapter
from .config import (
    ClaudeAgentSDKClientConfig,
    ClaudeAgentSDKModelConfig,
    PermissionMode,
)
from .isolation import (
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_MODEL,
    AuthMode,
    AwsConfigResolution,
    BedrockConfig,
    EphemeralHome,
    IsolationAuthError,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
    get_default_model,
    get_supported_bedrock_models,
    to_anthropic_model_name,
    to_bedrock_model_id,
)
from .workspace import (
    ClaudeAgentWorkspaceSection,
    HostMount,
    HostMountPreview,
    WorkspaceBudgetExceededError,
    WorkspaceSecurityError,
)

__all__ = [
    "CLAUDE_AGENT_SDK_ADAPTER_NAME",
    "DEFAULT_BEDROCK_MODEL",
    "DEFAULT_MODEL",
    "AuthMode",
    "AwsConfigResolution",
    "BedrockConfig",
    "ClaudeAgentSDKAdapter",
    "ClaudeAgentSDKClientConfig",
    "ClaudeAgentSDKModelConfig",
    "ClaudeAgentWorkspaceSection",
    "CompositeChecker",
    "EphemeralHome",
    "HostMount",
    "HostMountPreview",
    "IsolationAuthError",
    "IsolationConfig",
    "NetworkPolicy",
    "PermissionMode",
    "PlanBasedChecker",
    "SandboxConfig",
    "TaskCompletionChecker",
    "TaskCompletionContext",
    "TaskCompletionResult",
    "WorkspaceBudgetExceededError",
    "WorkspaceSecurityError",
    "create_task_completion_stop_hook",
    "get_default_model",
    "get_supported_bedrock_models",
    "to_anthropic_model_name",
    "to_bedrock_model_id",
]

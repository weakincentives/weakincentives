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
capabilities through the official claude-code-sdk Python package. It provides
native tool execution (Read, Write, Edit, Glob, Grep, Bash), workspace isolation,
and hook-based state synchronization between SDK execution and the weakincentives
Session.

Architecture Overview
---------------------
The Claude Agent SDK adapter differs from OpenAI/LiteLLM adapters in several ways:

1. **Native Tools**: Claude Code provides built-in tools for file operations and
   shell commands. These run directly in the SDK sandbox, not through the
   weakincentives tool execution layer.

2. **MCP Bridge**: Custom weakincentives tools are exposed to Claude via an MCP
   (Model Context Protocol) server that bridges between SDK tool calls and
   weakincentives tool handlers.

3. **Hooks**: The SDK's hook system enables interception of tool use, stop events,
   and prompt submission. Hooks synchronize session state and enforce completion
   policies.

4. **Hermetic Isolation**: The adapter creates an ephemeral home directory with
   generated settings, preventing interaction with host ~/.claude configuration.

Key Components
--------------
**Adapter** (:class:`ClaudeAgentSDKAdapter`)
    The main adapter class that evaluates prompts using the Claude Agent SDK.
    Supports structured output, tool execution, and budget tracking.

**Configuration** (:class:`ClaudeAgentSDKClientConfig`, :class:`ClaudeAgentSDKModelConfig`)
    Client-level settings (permission mode, max turns, budget) and model-level
    settings (extended thinking tokens).

**Workspace** (:class:`ClaudeAgentWorkspaceSection`, :class:`HostMount`)
    Prompt section that manages a temporary workspace directory with host file
    mounts. Provides secure file access with glob filtering and byte limits.

**Isolation** (:class:`IsolationConfig`, :class:`EphemeralHome`, :class:`NetworkPolicy`)
    Hermetic isolation configuration that creates ephemeral environments with
    controlled network access and authentication settings.

**Task Completion** (:class:`TaskCompletionChecker`, :class:`PlanBasedChecker`)
    Verification protocols for ensuring tasks are complete before allowing the
    agent to stop. Integrates with the planning tool state.

**Transcript Collection** (:class:`TranscriptCollector`, :class:`TranscriptCollectorConfig`)
    Real-time collection and logging of Claude Agent SDK transcripts from the
    main session and all sub-agent sessions. Emits structured DEBUG-level logs
    for observability.

Authentication Modes
--------------------
The adapter supports multiple authentication configurations:

**Inherit Host Auth** (default)
    Inherits authentication from the host environment. Works with both Anthropic
    API (via ANTHROPIC_API_KEY) and AWS Bedrock (via ~/.aws config)::

        config = IsolationConfig.inherit_host_auth()

**Explicit API Key**
    Uses a provided Anthropic API key directly, disabling Bedrock::

        config = IsolationConfig.with_api_key("sk-ant-...")

**Require Anthropic API**
    Requires ANTHROPIC_API_KEY environment variable, fails fast if not set::

        config = IsolationConfig.for_anthropic_api()

**Require Bedrock**
    Requires AWS Bedrock configuration, fails fast if not available::

        config = IsolationConfig.for_bedrock()

Basic Usage Example
-------------------
Simple adapter usage without workspace mounts::

    from weakincentives import Prompt, PromptTemplate, MarkdownSection
    from weakincentives.adapters.claude_agent_sdk import (
        ClaudeAgentSDKAdapter,
        ClaudeAgentSDKClientConfig,
    )
    from weakincentives.runtime import Session, InProcessDispatcher
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class TaskResult:
        message: str

    template = PromptTemplate[TaskResult](
        ns="example",
        key="hello",
        sections=(
            MarkdownSection(
                title="Task",
                key="task",
                template="Say hello and return a greeting message.",
            ),
        ),
    )

    adapter = ClaudeAgentSDKAdapter(
        model="claude-sonnet-4-5-20250929",
        client_config=ClaudeAgentSDKClientConfig(
            permission_mode="bypassPermissions",
        ),
    )

    session = Session(dispatcher=InProcessDispatcher())
    prompt = Prompt(template)
    response = adapter.evaluate(prompt, session=session)
    print(response.output.message)

Workspace Example
-----------------
Using workspace sections to mount host files::

    from weakincentives import Prompt, PromptTemplate, MarkdownSection
    from weakincentives.adapters.claude_agent_sdk import (
        ClaudeAgentSDKAdapter,
        ClaudeAgentSDKClientConfig,
        ClaudeAgentWorkspaceSection,
        HostMount,
    )
    from weakincentives.runtime import Session, InProcessDispatcher
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ReviewResult:
        summary: str
        issues: list[str]

    session = Session(dispatcher=InProcessDispatcher())

    # Create workspace with host file mounts
    workspace = ClaudeAgentWorkspaceSection(
        session=session,
        mounts=[
            HostMount(
                host_path="src",
                include_glob=("*.py",),
                exclude_glob=("*_test.py",),
                max_bytes=1_000_000,
            ),
        ],
        allowed_host_roots=["/home/user/project"],
    )

    template = PromptTemplate[ReviewResult](
        ns="review",
        key="code",
        sections=(
            MarkdownSection(
                title="Task",
                key="task",
                template="Review the Python code in the workspace.",
            ),
            workspace,
        ),
    )

    adapter = ClaudeAgentSDKAdapter(
        model="claude-sonnet-4-5-20250929",
        client_config=ClaudeAgentSDKClientConfig(
            permission_mode="acceptEdits",
            cwd=str(workspace.temp_dir),
        ),
    )

    try:
        prompt = Prompt(template)
        response = adapter.evaluate(prompt, session=session)
        print(response.output)
    finally:
        workspace.cleanup()

Task Completion Example
-----------------------
Using task completion checking with planning tools::

    from weakincentives.adapters.claude_agent_sdk import (
        ClaudeAgentSDKAdapter,
        ClaudeAgentSDKClientConfig,
        PlanBasedChecker,
    )
    from weakincentives.contrib.tools.planning import Plan

    checker = PlanBasedChecker(plan_type=Plan)

    adapter = ClaudeAgentSDKAdapter(
        model="claude-sonnet-4-5-20250929",
        client_config=ClaudeAgentSDKClientConfig(
            task_completion_checker=checker,
            stop_on_structured_output=True,
        ),
    )

Public Exports
--------------
Adapter:
    - :class:`ClaudeAgentSDKAdapter`: Main adapter for Claude Agent SDK
    - :data:`CLAUDE_AGENT_SDK_ADAPTER_NAME`: Adapter identifier constant

Configuration:
    - :class:`ClaudeAgentSDKClientConfig`: Client-level SDK configuration
    - :class:`ClaudeAgentSDKModelConfig`: Model-level configuration
    - :data:`PermissionMode`: Literal type for permission handling modes

Workspace:
    - :class:`ClaudeAgentWorkspaceSection`: Prompt section for workspace management
    - :class:`HostMount`: Configuration for mounting host files
    - :class:`HostMountPreview`: Summary of materialized mount
    - :class:`WorkspaceBudgetExceededError`: Raised when byte budget exceeded
    - :class:`WorkspaceSecurityError`: Raised when security constraints violated

Isolation:
    - :class:`IsolationConfig`: Hermetic isolation configuration
    - :class:`EphemeralHome`: Manages temporary home directory
    - :class:`NetworkPolicy`: Network access constraints for tools
    - :class:`SandboxConfig`: OS-level sandbox configuration
    - :class:`BedrockConfig`: Detected Bedrock configuration
    - :class:`AuthMode`: Authentication mode enumeration
    - :class:`AwsConfigResolution`: Result of AWS config path resolution
    - :class:`IsolationAuthError`: Raised when authentication unavailable

Task Completion:
    - :class:`TaskCompletionChecker`: Protocol for completion verification
    - :class:`PlanBasedChecker`: Checker based on Plan state
    - :class:`CompositeChecker`: Combines multiple checkers
    - :class:`TaskCompletionContext`: Context for checker evaluation
    - :class:`TaskCompletionResult`: Result of completion check
    - :func:`create_task_completion_stop_hook`: Factory for stop hooks

Transcript Collection:
    - :class:`TranscriptCollector`: Collects and logs SDK transcripts
    - :class:`TranscriptCollectorConfig`: Configuration for transcript collection

Model Utilities:
    - :data:`DEFAULT_MODEL`: Default Anthropic API model ID
    - :data:`DEFAULT_BEDROCK_MODEL`: Default Bedrock model ID
    - :func:`get_default_model`: Get default model for current auth mode
    - :func:`get_supported_bedrock_models`: Get Anthropic-to-Bedrock model mapping
    - :func:`to_anthropic_model_name`: Convert Bedrock ID to Anthropic name
    - :func:`to_bedrock_model_id`: Convert Anthropic name to Bedrock ID

See Also
--------
- :mod:`weakincentives.adapters`: Parent package with adapter pattern overview
- :mod:`weakincentives.prompt`: Prompt and section construction
- :mod:`weakincentives.runtime`: Session and event infrastructure
- :mod:`weakincentives.contrib.tools.planning`: Planning tools for task tracking
"""

from __future__ import annotations

from ..exceptions import ClaudeAgentSDKError
from ._hooks import create_task_completion_stop_hook
from ._task_completion import (
    CompositeChecker,
    PlanBasedChecker,
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
)
from ._transcript_collector import TranscriptCollector, TranscriptCollectorConfig
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
    "ClaudeAgentSDKError",
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
    "TranscriptCollector",
    "TranscriptCollectorConfig",
    "WorkspaceBudgetExceededError",
    "WorkspaceSecurityError",
    "create_task_completion_stop_hook",
    "get_default_model",
    "get_supported_bedrock_models",
    "to_anthropic_model_name",
    "to_bedrock_model_id",
]

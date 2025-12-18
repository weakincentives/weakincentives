# Claude Agent SDK Adapter Specification

> **SDK**: `claude-agent-sdk>=0.1.15`

## Purpose

`ClaudeAgentSDKAdapter` evaluates a `weakincentives.prompt.Prompt` via the Claude
Agent SDK (Claude Code CLI), while keeping orchestration state in a
`weakincentives.runtime.Session`.

It provides:

- Claude Code native tools (Read/Write/Edit/Glob/Grep/Bash)
- weakincentives `Tool` handlers bridged as MCP tools
- Structured output (`PromptTemplate[YourDataclass]`) via JSON Schema
- Optional isolation (`IsolationConfig`) that redirects `HOME` and generates
  `.claude/settings.json` for sandbox/network settings

## Requirements

- Python: `pip install 'weakincentives[claude-agent-sdk]'`
- Claude Code CLI: `npm install -g @anthropic-ai/claude-code`
- Linux sandboxing: bubblewrap (`bwrap`) available on PATH

## Reality Check (What Isolation Means)

When `ClaudeAgentSDKClientConfig.isolation` is set, the adapter:

- creates an ephemeral `HOME` containing `.claude/settings.json`
- passes `env` and `setting_sources=["user"]` to the SDK, so Claude Code reads
  settings from the redirected home

Notes:

- `NetworkPolicy` restricts outbound network for **tools** (for example `Bash`
  running `curl`). It does **not** block the model’s API connection.
- If `IsolationConfig.api_key` is unset, `ANTHROPIC_API_KEY` from the current
  process environment is forwarded to the SDK subprocess.

## Configuration (Fields That Matter)

### `ClaudeAgentSDKClientConfig`

- `permission_mode: PermissionMode = "bypassPermissions"`
- `cwd: str | None = None`
- `max_turns: int | None = None`
- `suppress_stderr: bool = True`
- `stop_on_structured_output: bool = True`
- `isolation: IsolationConfig | None = None`

### `IsolationConfig`

- `network_policy: NetworkPolicy | None = None` (defaults to no tool network)
- `sandbox: SandboxConfig | None = None` (defaults to `SandboxConfig()`)
- `env: Mapping[str, str] | None = None`
- `api_key: str | None = None` (mutually exclusive with `bedrock`)
- `bedrock: BedrockConfig | None = None` (mutually exclusive with `api_key`)
- `include_host_env: bool = False` (copies only non-sensitive vars)

### `BedrockConfig`

AWS Bedrock authentication. Mutually exclusive with `api_key` in
`IsolationConfig`. Supports multiple authentication methods:

**Static credentials:**

- `region: str` (required)
- `access_key_id: str`
- `secret_access_key: str`
- `session_token: str | None = None` (for temporary STS credentials)

**Named profile:**

- `region: str` (required)
- `profile: str`

**IAM role assumption:**

- `region: str` (required)
- `role_arn: str`
- `role_session_name: str | None = None`
- `external_id: str | None = None` (cross-account access)

**Web identity / OIDC:**

- `region: str` (required)
- `role_arn: str`
- `web_identity_token_file: str`
- `role_session_name: str | None = None`

**Default credential chain:**

- `region: str` (required, all other fields None)

**Optional:**

- `endpoint_url: str | None = None` (VPC endpoints, LocalStack, moto)

Factory methods:

- `BedrockConfig.from_static_credentials(region, access_key_id, secret_access_key, ...)`
- `BedrockConfig.from_profile(region, profile, ...)`
- `BedrockConfig.from_role(region, role_arn, ...)`
- `BedrockConfig.from_web_identity(region, role_arn, web_identity_token_file, ...)`
- `BedrockConfig.from_environment(region)`

### `NetworkPolicy`

- `NetworkPolicy.no_network()`
- `NetworkPolicy.with_domains("docs.python.org", "pypi.org")`
- `allowed_domains=("*",)` means unrestricted tool egress (avoid in production)

### `SandboxConfig`

- `enabled: bool = True`
- `writable_paths: tuple[str, ...] = ()`
- `readable_paths: tuple[str, ...] = ()`
- `excluded_commands: tuple[str, ...] = ()`
- `allow_unsandboxed_commands: bool = False`
- `bash_auto_allow: bool = True`

## Tool Bridging (MCP)

Any `Tool` attached to rendered sections (for example
`MarkdownSection(..., tools=(my_tool,))`) is exposed to Claude Code as MCP tools
under the server key `"wink"`. Tool calls publish `ToolInvoked` events.

## Events

- `PromptRendered` (after render)
- `ToolInvoked` (each native tool call + each bridged tool call)
- `PromptExecuted` (completion, includes aggregated `TokenUsage`)

## User Stories

### Story 1: Typed “hello world” (structured output)

As a developer, I want a minimal call that returns a dataclass so I can wire the
agent into a pipeline.

```python
from dataclasses import dataclass

from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime import InProcessEventBus, Session


@dataclass(frozen=True)
class Hello:
    message: str


session = Session(bus=InProcessEventBus())

template = PromptTemplate[Hello](
    ns="demo",
    key="hello",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Say hello. Return JSON with a single field: message.",
        ),
    ],
)

response = ClaudeAgentSDKAdapter().evaluate(Prompt(template), session=session)
print(response.output)
```

### Story 2: Secure code review in a mounted workspace (no tool network)

As a security reviewer, I want the agent to read a repo from a temporary
workspace while tools have no outbound network access, so exfiltration via
`curl`/`wget` isn’t possible.

```python
from dataclasses import dataclass

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
    HostMount,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime import InProcessEventBus, Session


@dataclass(frozen=True)
class Review:
    summary: str
    findings: list[str]


session = Session(bus=InProcessEventBus())

workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(
        HostMount(
            host_path="/abs/path/to/repo",
            mount_path="repo",
            exclude_glob=(".git/*", "*.pyc", "__pycache__/*"),
            max_bytes=5_000_000,
        ),
    ),
    allowed_host_roots=("/abs/path/to",),
)

try:
    adapter = ClaudeAgentSDKAdapter(
        client_config=ClaudeAgentSDKClientConfig(
            permission_mode="bypassPermissions",
            cwd=str(workspace.temp_dir),
            isolation=IsolationConfig(
                network_policy=NetworkPolicy.no_network(),
                sandbox=SandboxConfig(
                    enabled=True,
                    readable_paths=(str(workspace.temp_dir),),
                ),
            ),
        ),
    )

    template = PromptTemplate[Review](
        ns="review",
        key="security",
        sections=[
            MarkdownSection(
                title="Task",
                key="task",
                template=(
                    "Review the code in repo/ for security issues. "
                    "Return JSON: summary, findings."
                ),
            ),
            workspace,
        ],
    )

    response = adapter.evaluate(Prompt(template), session=session)
    print(response.output)
finally:
    workspace.cleanup()
```

### Story 3: Docs assistant with a domain allowlist

As a developer, I want the agent’s tools to access only a small set of docs
hosts so browsing stays on-policy.

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)

adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.with_domains(
                "docs.python.org",
                "pypi.org",
            ),
            sandbox=SandboxConfig(enabled=True),
        ),
    ),
)
```

### Story 4: AWS Bedrock with static credentials

As a developer, I want to use Claude via AWS Bedrock with explicit IAM
credentials so I can run in environments without AWS CLI configuration.

```python
from weakincentives.adapters.claude_agent_sdk import (
    BedrockConfig,
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
)

adapter = ClaudeAgentSDKAdapter(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            bedrock=BedrockConfig.from_static_credentials(
                region="us-east-1",
                access_key_id="AKIAIOSFODNN7EXAMPLE",
                secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            ),
        ),
    ),
)
```

### Story 5: AWS Bedrock with named profile

As a developer, I want to use an AWS profile from `~/.aws/credentials` so I can
reuse existing credential management.

```python
from weakincentives.adapters.claude_agent_sdk import (
    BedrockConfig,
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
)

adapter = ClaudeAgentSDKAdapter(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            bedrock=BedrockConfig.from_profile(
                region="us-west-2",
                profile="bedrock-prod",
            ),
        ),
    ),
)
```

### Story 6: AWS Bedrock with IAM role assumption

As a platform engineer, I want to assume a cross-account IAM role for Bedrock
access so I can maintain security boundaries.

```python
from weakincentives.adapters.claude_agent_sdk import (
    BedrockConfig,
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
)

adapter = ClaudeAgentSDKAdapter(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            bedrock=BedrockConfig.from_role(
                region="eu-west-1",
                role_arn="arn:aws:iam::123456789012:role/BedrockAccessRole",
                role_session_name="weakincentives-session",
                external_id="my-external-id",  # Optional cross-account security
            ),
        ),
    ),
)
```

### Story 7: AWS Bedrock with OIDC/web identity (EKS, GitHub Actions)

As a DevOps engineer, I want to use OIDC federation for Bedrock access so I can
run securely in Kubernetes or CI/CD without static credentials.

```python
from weakincentives.adapters.claude_agent_sdk import (
    BedrockConfig,
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
)

adapter = ClaudeAgentSDKAdapter(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            bedrock=BedrockConfig.from_web_identity(
                region="us-east-1",
                role_arn="arn:aws:iam::123456789012:role/EKSPodRole",
                web_identity_token_file="/var/run/secrets/eks.amazonaws.com/serviceaccount/token",
            ),
        ),
    ),
)
```

### Story 8: AWS Bedrock with default credential chain

As a developer, I want Bedrock to use the default AWS credential resolution
(environment variables, instance profile, etc.) so it works automatically in
EC2/ECS/Lambda.

```python
from weakincentives.adapters.claude_agent_sdk import (
    BedrockConfig,
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
)

# Uses AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY from environment,
# or instance profile, or ECS task role, etc.
adapter = ClaudeAgentSDKAdapter(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            bedrock=BedrockConfig.from_environment(region="us-east-1"),
        ),
    ),
)
```

### Story 9: Expose an internal `Tool` to Claude via MCP

As a platform team, I want the agent to call a typed internal tool so I can keep
side effects and validation in Python, not in prompt text.

```python
from dataclasses import dataclass

from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.runtime import InProcessEventBus, Session


@dataclass(frozen=True)
class SearchParams:
    query: str


@dataclass(frozen=True)
class SearchResult:
    matches: int

    def render(self) -> str:
        return f"Found {self.matches} matches"


def search(params: SearchParams, *, context: ToolContext) -> ToolResult[SearchResult]:
    del context
    return ToolResult(message="ok", value=SearchResult(matches=3))


search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search the internal index",
    handler=search,
)

session = Session(bus=InProcessEventBus())

template = PromptTemplate[None](
    ns="demo",
    key="mcp-tool",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Use the search tool for query: weakincentives. Summarize.",
            tools=(search_tool,),
        ),
    ],
)

response = ClaudeAgentSDKAdapter().evaluate(Prompt(template), session=session)
print(response.text)
```

## Operational Notes

- If you need to observe token usage across multiple evaluations, pass a
  `BudgetTracker` into `evaluate(..., budget_tracker=...)`; passing `budget=...`
  without a tracker creates an internal tracker.
- `stop_on_structured_output=True` (default) stops the agent right after the
  `StructuredOutput` tool runs, so a structured result ends the turn cleanly.
- Windows: sandbox settings may not be enforced; HOME redirection still applies.

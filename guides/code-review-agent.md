# Code Reviewer Example

This document describes the `code_reviewer_example.py` script that ships with
the library. The script assembles a full-featured code review agent
demonstrating prompt composition, progressive disclosure, workspace tools,
planning, and the `AgentLoop` pattern in one place.

## Rationale and Scope

- **Purpose**: Canonical end-to-end walkthrough for a review agent exercising
  prompt templates, custom tool registration, skill composition, workspace
  tooling, and the `AgentLoop` pattern.
- **Scope**: Focused on the bundled `sunfish` repository fixture under
  `test-repositories/`. Mounts are read-only with a 600 KB payload cap.
- **Principles**: Declarative prompt assembly, section-scoped tools and skills,
  reusable workspace tools, and full observability via event subscribers.

## Transactional Tool Execution

The code reviewer benefits from WINK's transactional tool execution. Every tool
call (reading files, updating plans, writing workspace state) is wrapped in a
transaction. If a tool fails:

- Session state (the review plan, visibility overrides) rolls back automatically
- Filesystem changes are undone
- The agent continues from a known-good state

This is particularly valuable when the agent is navigating a large codebase.
A failed `read_file` doesn't corrupt the review session. The agent's state
remains consistent, and the model can retry
or take a different approach without debugging "what state are we actually in?"

No explicit error handling code is required—the framework handles rollback
automatically. See [Sessions](../specs/SESSIONS.md) for details.

## Custom Tool Registration

The example registers a `count_lines` tool that lets the agent gauge file
size and composition during a review. This demonstrates the full tool API:

1. **Typed params/result** — frozen dataclasses with field metadata
1. **Handler** — `(params, *, context: ToolContext) -> ToolResult[R]`
1. **Examples** — `ToolExample` instances documenting expected I/O
1. **Section attachment** — `tools=(count_lines_tool,)` on a section

```python nocheck
@FrozenDataclass()
class CountLinesParams:
    path: str = field(metadata={"description": "Relative file path within the project."})

@FrozenDataclass()
class CountLinesResult:
    path: str
    total: int
    code: int
    blank: int
    comment: int

    def render(self) -> str:
        return f"{self.path}: {self.total} total, {self.code} code, ..."

def count_lines(params: CountLinesParams, *, context: ToolContext) -> ToolResult[CountLinesResult]:
    """Count lines of code, blanks, and comments in a project file."""
    ...
    return ToolResult.ok(result, message=result.render())

tool = Tool[CountLinesParams, CountLinesResult](
    name="count_lines",
    description="Count lines of code, blanks, and comments in a project file.",
    handler=count_lines,
    examples=(
        ToolExample(
            description="Count lines in the main engine file",
            input=CountLinesParams(path="sunfish.py"),
            output=CountLinesResult(path="sunfish.py", total=425, code=298, blank=47, comment=80),
        ),
    ),
)
```

The tool is attached to the **Analysis** section so the agent can call it
while working through the review checklist. See [Tools](../specs/TOOLS.md)
for the full specification.

## Skill Composition

The example mounts two skills from `demo-skills/` onto the **Role** section:

- `code-review` — review checklist (security, error handling, tests, performance)
- `python-style` — PEP 8 / PEP 484 / PEP 257 best-practice guidelines

```python nocheck
from weakincentives.skills import SkillMount

code_review_skill = SkillMount(source=SKILLS_DIR / "code-review")
python_style_skill = SkillMount(source=SKILLS_DIR / "python-style")

MarkdownSection(
    title="Role",
    template="...",
    key="role",
    skills=(code_review_skill, python_style_skill),
)
```

Skills follow the [Agent Skills specification](https://agentskills.io).
Each skill directory contains a `SKILL.md` with YAML frontmatter (`name`,
`description`) and markdown instructions. The adapter injects skill content
into the agent's context at evaluation time, following the same visibility
rules as tools. See [Skills](../specs/SKILLS.md) for the full specification.

## Runtime Architecture

The example uses a two-layer design:

- `CodeReviewLoop` extends `AgentLoop[ReviewTurnParams, ReviewResponse]` for
  request handling with auto-optimization
- `CodeReviewApp` owns the interactive REPL and delegates execution to the loop

### CodeReviewLoop

Implements the `AgentLoop` protocol with these responsibilities:

- **Persistent Session**: Creates a single `Session` at construction time,
  reused across all `execute()` calls
- **Auto-Optimization**: Runs workspace digest optimization automatically on
  first request if no `WorkspaceDigest` exists in the session
- **Deadline Management**: Applies a default 5-minute deadline to each request
- **Prompt Binding**: Creates and binds prompts via `create_prompt()`

```python nocheck
class CodeReviewLoop(AgentLoop[ReviewTurnParams, ReviewResponse]):
    def create_prompt(self, request: ReviewTurnParams) -> Prompt[ReviewResponse]: ...
    def create_session(self) -> Session: ...
    def execute(self, request: ReviewTurnParams, *, budget=None, deadline=None) -> PromptResponse[ReviewResponse]: ...
```

### CodeReviewApp

Owns user interaction:

- Creates an `Dispatcher` with logging subscribers
- Instantiates `CodeReviewLoop` with the adapter and dispatcher
- Runs the interactive REPL loop
- Dumps session state on exit

### Startup Sequence

1. `configure_logging()` sets up logging
1. `build_adapter()` creates the Claude Agent SDK adapter (requires
   `ANTHROPIC_API_KEY`)
1. `CodeReviewApp` creates the event dispatcher and `CodeReviewLoop`
1. `CodeReviewLoop.__init__`:
   - Creates a persistent `Session` via `build_logged_session()`
   - Builds the `PromptTemplate` via `build_task_prompt()`
   - Seeds prompt overrides via `_seed_overrides()`
1. `CodeReviewApp.run()` starts the REPL

### REPL Loop

Each turn:

1. Reads user input
1. Handles exit commands (`exit`, `quit`, empty input)
1. Creates `ReviewTurnParams(request=...)` from user input
1. Calls `loop.execute()` which:
   - Auto-optimizes workspace digest if needed (first request only)
   - Applies default deadline
   - Delegates to `AgentLoop.execute()` for prompt evaluation
1. Renders response via `_render_response_payload`
1. Prints plan snapshot via `render_plan_snapshot`

Debug bundles are created automatically per-request in `debug_bundles/` via
`AgentLoopConfig.debug_bundle`.

## Data Types

### ReviewGuidance

Default guidance parameters for the review brief section.

```python nocheck
@dataclass(slots=True, frozen=True)
class ReviewGuidance:
    focus: str = "Identify potential issues, risks, and follow-up questions..."
```

### ReviewTurnParams

Runtime parameters provided per turn.

```python nocheck
@dataclass(slots=True, frozen=True)
class ReviewTurnParams:
    request: str  # User-provided review request
```

### ReviewResponse

Structured output from the agent.

```python nocheck
@dataclass(slots=True, frozen=True)
class ReviewResponse:
    summary: str
    issues: list[str]
    next_steps: list[str]
```

### ReferenceParams

Parameters for the reference documentation section.

```python nocheck
@dataclass(slots=True, frozen=True)
class ReferenceParams:
    project_name: str = "sunfish"
```

## Prompt Composition

`build_task_prompt` produces `PromptTemplate[ReviewResponse]` with:

- Namespace: `examples/code-review`
- Key: `code-review-session`
- Name: `sunfish_code_review_agent`

### Sections (in order)

1. **Role** (`MarkdownSection`)

   - Expert code-reviewer persona
   - Key: `role`
   - Skills: `code-review`, `python-style`

1. **Review Focus** (`MarkdownSection[ReviewParams]`)

   - Template: `${focus}`
   - Key: `focus`

1. **Workspace Tools** (`WorkspaceSection`)

   - Provides file tools via the adapter
   - Uses sunfish-specific host mounts

1. **Analysis** (`MarkdownSection`)

   - Step-by-step review instructions
   - Key: `analysis`
   - Tools: `count_lines`

1. **Output Format** (`MarkdownSection`)

   - Structured JSON output specification
   - Key: `output-format`

### Mount Configuration

```python nocheck
SUNFISH_MOUNT_INCLUDE_GLOBS = (
    "*.md", "*.py", "*.txt", "*.yml", "*.yaml", "*.toml",
    ".gitignore", "*.json", "*.cfg", "*.ini", "*.sh", "*.6",
)
SUNFISH_MOUNT_EXCLUDE_GLOBS = (
    "**/__pycache__/**", "**/.git/**", "**/.venv/**",
    "**/node_modules/**", "**/*.pyc",
    "**/*.pickle", "**/*.png", "**/*.bmp",
)
SUNFISH_MOUNT_MAX_BYTES = 600_000
```

## Progressive Disclosure

The Reference Documentation section demonstrates progressive disclosure:

1. Section starts with `visibility=SectionVisibility.SUMMARY`
1. Model sees only the summary text initially
1. Model can call `open_sections` tool to expand
1. `AgentLoop` handles `VisibilityExpansionRequired` internally
1. Full section content becomes visible on retry

## Auto-Optimization

The example automatically optimizes the workspace digest on first use:

1. `CodeReviewLoop.execute()` checks if `WorkspaceDigest` exists in session
1. If missing, calls `_run_optimization()` before processing the request
1. Creates a child session via `build_logged_session(parent=...)`
1. Runs `WorkspaceDigestOptimizer` which stores the digest in the session
1. Digest is available for subsequent requests via `WorkspaceDigestSection`

This eliminates the need for manual optimization commands.

## Overrides

### Override Storage

- Store: `LocalPromptOverridesStore` (default: `~/.weakincentives/prompts`)
- Scoped by: namespace + prompt key + tag
- Seeded once at startup via `store.seed(prompt, tag=...)`

### Tag Resolution

Override tag is resolved in order:

1. Explicit `override_tag` argument
1. `CODE_REVIEW_PROMPT_TAG` environment variable
1. Default: `"latest"`

## Deadlines

Each request has a deadline to prevent runaway execution:

```python nocheck
DEFAULT_DEADLINE_MINUTES = 5

def _default_deadline() -> Deadline:
    return Deadline(
        expires_at=datetime.now(UTC) + timedelta(minutes=DEFAULT_DEADLINE_MINUTES)
    )
```

Custom deadlines can be passed to `execute()` if needed.

## Observability

### Event Subscribers

Attached via `attach_logging_subscribers(dispatcher)` from `examples.logging`:

| Event | Output |
| ---------------- | ------------------------------------ |
| `PromptRendered` | Full prompt text with label |
| `ToolInvoked` | Params, result, payload, token usage |
| `PromptExecuted` | Token usage summary |

### Plan Snapshots

`render_plan_snapshot` formats the current plan:

```
Objective: <objective> (status: <status>)
- <step_id> [<status>] <title> — details: <details>; notes: <notes>
```

## Running the Example

```bash
ANTHROPIC_API_KEY=sk-ant-... uv run python code_reviewer_example.py
```

### Environment Variables

| Variable | Required | Default | Description |
| ------------------------ | -------- | -------------- | ------------------- |
| `ANTHROPIC_API_KEY` | Yes | - | Anthropic API key |
| `CLAUDE_MODEL` | No | `claude-opus-4-6` | Model to use |
| `CODE_REVIEW_PROMPT_TAG` | No | `latest` | Overrides tag |

### REPL Commands

| Input | Action |
| ----------------------- | ------------------------ |
| Non-empty text | Submit as review request |
| `exit` / `quit` / empty | Terminate REPL |

### Output

- Intro banner with configuration summary
- `[prompt]` blocks showing rendered prompts (via logging)
- `[tool]` blocks showing tool invocations (via logging)
- `--- Agent Response ---` with formatted output
- `--- Plan Snapshot ---` with current plan state

## Shared Utilities

The example imports several helpers from the `examples` package:

| Function | Purpose |
| ---------------------------- | -------------------------------- |
| `build_logged_session` | Create session with logging tags |
| `configure_logging` | Set up console logging |
| `render_plan_snapshot` | Format plan state for display |
| `resolve_override_tag` | Resolve tag from arg/env/default |
| `attach_logging_subscribers` | Attach event logging to dispatcher |

## Key Files

| File | Purpose |
| --------------------------------- | --------------------------------------- |
| `code_reviewer_example.py` | Main script |
| `demo-skills/code-review/` | Code-review skill (mounted on Role) |
| `demo-skills/python-style/` | Python-style skill (mounted on Role) |
| `test-repositories/sunfish/` | Mounted repository fixture |
| `debug_bundles/` | Per-request debug bundle output |

## Next Steps

- [Claude Agent SDK](claude-agent-sdk.md): Deep dive into workspace tools
- [Progressive Disclosure](progressive-disclosure.md): Understand visibility
  management
- [Prompt Overrides](prompt-overrides.md): Customize prompts without code
  changes
- [Orchestration](orchestration.md): Scale with AgentLoop

# Code Reviewer Example

This document describes the `code_reviewer_example.py` script that ships with
the library. The script assembles a full-featured code review agent
demonstrating prompt composition, progressive disclosure, workspace tools,
planning, and the `MainLoop` pattern in one place.

## Rationale and Scope

- **Purpose**: Canonical end-to-end walkthrough for a review agent exercising
  prompt templates, overrides, workspace tooling, and auto-optimization.
- **Scope**: Focused on the bundled `sunfish` repository fixture under
  `test-repositories/`. Mounts are read-only with a 600 KB payload cap.
- **Principles**: Declarative prompt assembly, ergonomic overrides (tagged by
  namespace/key), reusable planning/workspace tools, and full observability via
  logging.

## Transactional Tool Execution

The code reviewer benefits from WINK's transactional tool execution. Every tool
call (reading files, updating plans, writing workspace state) is wrapped in a
transaction. If a tool fails:

- Session state (the review plan, visibility overrides) rolls back automatically
- Filesystem changes are undone
- The agent continues from a known-good state

This is particularly valuable when the agent is navigating a large codebase.
A failed `read_file` or interrupted `planning_update_step` doesn't corrupt the
review session. The agent's plan remains consistent, and the model can retry
or take a different approach without debugging "what state are we actually in?"

No explicit error handling code is required—the framework handles rollback
automatically. See [Sessions](../specs/SESSIONS.md) for details.

## Runtime Architecture

The example uses a two-layer design with mailbox-driven orchestration:

- `CodeReviewLoop` extends `MainLoop[ReviewTurnParams, ReviewResponse]` and
  runs as a background worker processing requests from a mailbox
- `CodeReviewApp` owns the interactive REPL and sends requests via the mailbox

### CodeReviewLoop

Implements the `MainLoop` protocol with these responsibilities:

- **Persistent Session**: Creates a single `Session` at construction time,
  reused across all requests
- **Auto-Optimization**: Runs workspace digest optimization automatically on
  first request if no `WorkspaceDigest` exists in the session
- **Prompt Binding**: Builds prompts via `prepare()` which returns the bound
  prompt and persistent session

```python
class CodeReviewLoop(MainLoop[ReviewTurnParams, ReviewResponse]):
    def prepare(self, request: ReviewTurnParams) -> tuple[Prompt[ReviewResponse], Session]:
        """Prepare prompt and session for the given request."""
        ...
```

The loop runs via `loop.run()` which polls the request mailbox and processes
each `MainLoopRequest`. Results are routed back via the `reply_to` address.

### CodeReviewApp

Owns user interaction and mailbox setup:

- Creates request and response `InMemoryMailbox` instances with reply routing
- Instantiates `CodeReviewLoop` with the adapter and request mailbox
- Runs the loop in a background thread via `loop.run()`
- Sends `MainLoopRequest` messages with `reply_to` routing
- Receives `MainLoopResult` from the response mailbox
- Dumps session state on exit

### Startup Sequence

1. `configure_logging()` sets up logging
1. `build_adapter()` creates the OpenAI adapter (requires `OPENAI_API_KEY`)
1. `CodeReviewApp.__init__`:
   - Creates response mailbox (`InMemoryMailbox`)
   - Creates request mailbox with `RegistryResolver` for reply routing
   - Instantiates `CodeReviewLoop` with adapter and request mailbox
1. `CodeReviewLoop.__init__`:
   - Creates a persistent `Session` via `build_logged_session()`
   - Builds the `PromptTemplate` via `build_task_prompt()`
   - Seeds prompt overrides via `_seed_overrides()`
1. `CodeReviewApp.run()`:
   - Starts the worker thread running `loop.run()`
   - Enters the interactive REPL

### REPL Loop

Each turn:

1. Reads user input
1. Handles exit commands (`exit`, `quit`, empty input)
1. Creates `ReviewTurnParams(request=...)` from user input
1. Wraps it in `MainLoopRequest` with default deadline
1. Sends to request mailbox with `reply_to="code-review-responses"`
1. Polls response mailbox until matching `MainLoopResult` arrives
1. Worker thread (running `loop.run()`) processes the request:
   - Auto-optimizes workspace digest if needed (first request only)
   - Calls `prepare()` to build prompt and get session
   - Evaluates prompt and sends `MainLoopResult` to reply address
1. Main thread renders response via `_render_response`
1. Prints plan snapshot via `render_plan_snapshot`

On exit, dumps session state to `snapshots/` via `dump_session_tree`.

## Data Types

### ReviewGuidance

Default guidance parameters for the review brief section.

```python
@dataclass(slots=True, frozen=True)
class ReviewGuidance:
    focus: str = "Identify potential issues, risks, and follow-up questions..."
```

### ReviewTurnParams

Runtime parameters provided per turn.

```python
@dataclass(slots=True, frozen=True)
class ReviewTurnParams:
    request: str  # User-provided review request
```

### ReviewResponse

Structured output from the agent.

```python
@dataclass(slots=True, frozen=True)
class ReviewResponse:
    summary: str
    issues: list[str]
    next_steps: list[str]
```

### ReferenceParams

Parameters for the reference documentation section.

```python
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

1. **Code Review Brief** (`MarkdownSection[ReviewGuidance]`)

   - Template describing tooling and output format
   - Key: `code-review-brief`

1. **Workspace Digest** (`WorkspaceDigestSection`)

   - Cached workspace notes from session
   - Auto-populated on first request via optimization

1. **Reference Documentation** (`MarkdownSection[ReferenceParams]`)

   - Progressive disclosure section (starts summarized)
   - Key: `reference-docs`
   - Visibility: `SectionVisibility.SUMMARY`
   - Summary: "Documentation for ${project_name} is available..."

1. **Planning Tools** (`PlanningToolsSection`)

   - Strategy: `PlanningStrategy.PLAN_ACT_REFLECT`
   - `accepts_overrides=True`

1. **Workspace Tools** (conditional)

   - `PodmanSandboxSection` if Podman connection available
   - `VfsToolsSection` fallback otherwise
   - Both use `_sunfish_mounts()` configuration

1. **Review Request** (`MarkdownSection[ReviewTurnParams]`)

   - Template: `${request}`
   - Key: `review-request`

### Mount Configuration

```python
SUNFISH_MOUNT_INCLUDE_GLOBS = (
    "*.md", "*.py", "*.txt", "*.yml", "*.yaml", "*.toml",
    "*.gitignore", "*.json", "*.cfg", "*.ini", "*.sh", "*.6",
)
SUNFISH_MOUNT_EXCLUDE_GLOBS = ("**/*.pickle", "**/*.png", "**/*.bmp")
SUNFISH_MOUNT_MAX_BYTES = 600_000
```

## Progressive Disclosure

The Reference Documentation section demonstrates progressive disclosure:

1. Section starts with `visibility=SectionVisibility.SUMMARY`
1. Model sees only the summary text initially
1. Model can call `open_sections` tool to expand
1. `MainLoop` handles `VisibilityExpansionRequired` internally
1. Full section content becomes visible on retry

## Auto-Optimization

The example automatically optimizes the workspace digest on first use:

1. `CodeReviewLoop.prepare()` checks if `WorkspaceDigest` exists in session
1. If missing, calls `_run_optimization()` before building the prompt
1. Creates a child session via `build_logged_session(parent=...)`
1. Builds `OptimizationContext` with adapter, dispatcher, store, tag
1. Runs `WorkspaceDigestOptimizer` with `PersistenceScope.SESSION`
1. Digest is stored in the session for subsequent requests

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

```python
DEFAULT_DEADLINE_MINUTES = 5

def _default_deadline() -> Deadline:
    return Deadline(
        expires_at=datetime.now(UTC) + timedelta(minutes=DEFAULT_DEADLINE_MINUTES)
    )
```

Custom deadlines can be passed in `MainLoopRequest` if needed.

## Observability

### Logging

The example uses Python's standard `logging` module configured via
`configure_logging()`. Key logging points:

- Workspace optimization progress (`_LOGGER.info`)
- Worker thread lifecycle events
- Request/response flow in the mailbox-based architecture

### Plan Snapshots

`render_plan_snapshot` formats the current plan:

```
Objective: <objective> (status: <status>)
- <step_id> [<status>] <title> — details: <details>; notes: <notes>
```

## Running the Example

```bash
OPENAI_API_KEY=sk-... uv run python code_reviewer_example.py
```

### Environment Variables

| Variable | Required | Default | Description |
| ------------------------ | -------- | --------- | -------------- |
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-5.1` | Model to use |
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
| ----------------------- | -------------------------------- |
| `build_logged_session` | Create session with logging tags |
| `configure_logging` | Set up console logging |
| `render_plan_snapshot` | Format plan state for display |
| `resolve_override_tag` | Resolve tag from arg/env/default |

## Key Files

| File | Purpose |
| ---------------------------- | ----------------------------- |
| `code_reviewer_example.py` | Main script |
| `examples/` | Shared example utilities |
| `test-repositories/sunfish/` | Mounted repository fixture |
| `snapshots/` | Session dump output directory |
| `~/.weakincentives/prompts/` | Default overrides storage |

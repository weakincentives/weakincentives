# Code Reviewer Example

This document describes the `code_reviewer_example.py` script that ships with
the library. The script assembles a full-featured code review agent
demonstrating prompt composition, progressive disclosure, workspace tools,
planning, MainLoop, EvalLoop, and a Textual-based TUI interface.

## Rationale and Scope

- **Purpose**: Canonical end-to-end walkthrough for a review agent exercising
  prompt templates, overrides, workspace tooling, auto-optimization, and
  evaluation workflows.
- **Scope**: Focused on the bundled `sunfish` repository fixture under
  `test-repositories/`. Mounts are read-only with a 600 KB payload cap.
- **Principles**: Declarative prompt assembly, ergonomic overrides (tagged by
  namespace/key), reusable planning/workspace tools, and full observability via
  event subscribers.

## Running the Example

The example supports three modes of operation:

```bash
# Interactive TUI mode (requires textual)
OPENAI_API_KEY=sk-... uv run python code_reviewer_example.py

# Start in evaluation mode
OPENAI_API_KEY=sk-... uv run python code_reviewer_example.py --eval

# Convert existing snapshots to a dataset
uv run python code_reviewer_example.py --convert --snapshot-dir snapshots/ --output datasets/reviews.jsonl
```

### Environment Variables

| Variable | Required | Default | Description |
| ------------------------ | -------- | --------- | --------------- |
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4.1` | Model to use |
| `CODE_REVIEW_PROMPT_TAG` | No | `latest` | Overrides tag |

### CLI Options

| Option | Description |
| ------------------- | ------------------------------------------- |
| `--eval` | Start in evaluation mode |
| `--dataset PATH` | Path to evaluation dataset (JSONL format) |
| `--convert` | Convert snapshots to a dataset and exit |
| `--snapshot-dir DIR` | Directory containing snapshot JSONL files |
| `--output PATH` | Output path for converted dataset |

## Textual TUI Interface

The example uses [Textual](https://textual.textualize.io/) to provide a rich
terminal user interface with:

- **Review Tab**: Split view showing review response and plan snapshot
- **Evaluation Tab**: Progress bar, results list, and summary statistics
- **History Tab**: Browse previous review sessions

### Keyboard Shortcuts

| Key | Action |
| --------- | -------------------------------- |
| `Ctrl+Q` | Quit the application |
| `Ctrl+E` | Switch between Review/Eval tabs |
| `Ctrl+R` | Run evaluation |

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
automatically. See [Execution State](../specs/EXECUTION_STATE.md) for details.

## Runtime Architecture

The example uses a layered design:

- `CodeReviewLoop` extends `MainLoop[ReviewTurnParams, ReviewResponse]` for
  request handling with auto-optimization
- `CodeReviewApp` is a Textual app that owns the TUI and delegates execution to
  the loop
- `EvalLoop` wraps the MainLoop for dataset-driven evaluation

### CodeReviewLoop

Implements the `MainLoop` protocol with these responsibilities:

- **Persistent Session**: Creates a single `Session` at construction time,
  reused across all `execute()` calls
- **Auto-Optimization**: Runs workspace digest optimization automatically on
  first request if no `WorkspaceDigest` exists in the session
- **Deadline Management**: Applies a default 5-minute deadline to each request
- **Prompt Binding**: Creates and binds prompts via `prepare()`

```python
class CodeReviewLoop(MainLoop[ReviewTurnParams, ReviewResponse]):
    def prepare(self, request: ReviewTurnParams) -> tuple[Prompt[ReviewResponse], Session]:
        ...
```

### CodeReviewApp (Textual TUI)

The `CodeReviewApp` class is a Textual application that:

- Creates `InMemoryMailbox` instances for requests and responses
- Instantiates `CodeReviewLoop` with the adapter and mailboxes
- Runs a background worker thread for the MainLoop
- Handles user input and displays results in the TUI
- Supports both interactive review and batch evaluation modes
- Dumps session state on exit

### Startup Sequence

1. `configure_logging()` sets up logging
1. `build_adapter()` creates the OpenAI adapter (requires `OPENAI_API_KEY`)
1. `CodeReviewApp` creates mailboxes and `CodeReviewLoop`
1. `CodeReviewLoop.__init__`:
   - Creates a persistent `Session` via `build_logged_session()`
   - Builds the `PromptTemplate` via `build_task_prompt()`
   - Seeds prompt overrides via `_seed_overrides()`
1. `CodeReviewApp.run()` starts the TUI

### MainLoop Integration

Each review request flows through the MainLoop:

1. User enters review request in TUI
1. `MainLoopRequest` is created with request and deadline
1. Request is sent to the request mailbox
1. Background worker calls `loop.run()` which:
   - Auto-optimizes workspace digest if needed (first request only)
   - Calls `prepare()` to get prompt and session
   - Evaluates prompt via adapter
1. Result is sent to response mailbox
1. TUI displays formatted response and plan snapshot

On exit, dumps session state to `snapshots/` via `dump_session_tree`.

## EvalLoop Integration

The example includes full EvalLoop support for dataset-driven evaluation:

### Running Evaluations

1. Click "Run Evaluation" button or press `Ctrl+R`
1. The app creates a fresh `CodeReviewLoop` for isolation
1. `EvalLoop` wraps the loop with an evaluator function
1. Dataset is submitted via `submit_dataset()`
1. Progress is shown in real-time with pass/fail indicators
1. Summary report displays aggregate statistics

### Snapshot to Dataset Conversion

Past review sessions can be converted to evaluation datasets:

```bash
uv run python code_reviewer_example.py --convert \
  --snapshot-dir snapshots/ \
  --output datasets/reviews.jsonl
```

This:

1. Loads all `.jsonl` files from the snapshot directory
1. Extracts `ReviewTurnParams` and `ReviewResponse` from each snapshot
1. Creates evaluation samples with expected criteria
1. Writes JSONL dataset for future evaluation runs

### Evaluator Function

The `review_response_evaluator` scores responses against expected criteria:

```python
@dataclass(slots=True, frozen=True)
class ExpectedReviewResponse:
    keywords: tuple[str, ...] = ()
    min_issues: int = 0
    min_next_steps: int = 0

def review_response_evaluator(
    output: ReviewResponse, expected: ExpectedReviewResponse
) -> Score:
    # Check issues count, next steps count, and keyword presence
    # Returns Score(value=0.0-1.0, passed=bool, reason=str)
```

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

1. **Workspace Tools** (`VfsToolsSection` + `AstevalSection`)

   - Virtual filesystem for file operations
   - Asteval for safe Python expression evaluation

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
1. If missing, calls `_run_optimization()` before processing the request
1. Creates a child session via `build_logged_session(parent=...)`
1. Builds `OptimizationContext` with adapter, bus, store, tag
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

Custom deadlines can be passed to `execute()` if needed.

## Observability

### Event Subscribers

Attached via `attach_logging_subscribers(bus)` from `examples.logging`:

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

## Key Files

| File | Purpose |
| ---------------------------- | -------------------------------- |
| `code_reviewer_example.py` | Main script with TUI |
| `examples/` | Shared example utilities |
| `test-repositories/sunfish/` | Mounted repository fixture |
| `snapshots/` | Session dump output directory |
| `datasets/` | Evaluation dataset storage |
| `~/.weakincentives/prompts/` | Default overrides storage |

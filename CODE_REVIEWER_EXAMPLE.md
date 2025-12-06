# Code Reviewer Example

This document describes the `code_reviewer_example.py` script that ships with
the library. The script assembles a full-featured code review agent
demonstrating prompt composition, progressive disclosure, workspace tools,
planning, and adapter optimization in one place.

## Rationale and Scope

- **Purpose**: Canonical end-to-end walkthrough for a review agent exercising
  prompt templates, overrides, workspace tooling, and optimization hooks.
- **Scope**: Focused on the bundled `sunfish` repository fixture under
  `test-repositories/`. Mounts are read-only with a 600 KB payload cap.
- **Principles**: Declarative prompt assembly, ergonomic overrides (tagged by
  namespace/key), reusable planning/workspace tools, and full observability via
  event subscribers.

## Runtime Architecture

`CodeReviewApp` wires together:

- `ProviderAdapter[ReviewResponse]` for LLM communication
- `Prompt[ReviewResponse]` wrapping a `PromptTemplate` with overrides
- `Session` for state management
- `LocalPromptOverridesStore` for persistent overrides
- `EventBus` for telemetry

### Startup Sequence

1. `_ensure_test_repository_available` verifies `test-repositories/` exists
1. `build_task_prompt(session=...)` composes the prompt template
1. `LocalPromptOverridesStore` initializes and seeds overrides
1. `_resolve_override_tag` selects tag: explicit arg → `CODE_REVIEW_PROMPT_TAG`
   env → `"latest"`
1. `CodeReviewApp.run()` starts the REPL loop

### REPL Loop

Each turn:

1. Reads user input
1. Handles special commands (`optimize`, `exit`, `quit`)
1. Binds `ReviewTurnParams(request=...)` to the prompt
1. Calls `adapter.evaluate()` with visibility overrides
1. Catches `VisibilityExpansionRequired` for progressive disclosure retries
1. Renders response via `_render_response_payload`
1. Prints plan snapshot via `_render_plan_snapshot`

On exit, dumps session state to `snapshots/` via `dump_session`.

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

### RuntimeContext

Holds all runtime handles for the REPL.

```python
@dataclass(slots=True)
class RuntimeContext:
    prompt: Prompt[ReviewResponse]
    session: Session
    bus: EventBus
    overrides_store: LocalPromptOverridesStore
    override_tag: str
    visibility_overrides: dict[SectionPath, SectionVisibility]
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

   - Cached workspace notes from session or overrides
   - Populated via `optimize` command

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
1. `_evaluate_turn` catches `VisibilityExpansionRequired`
1. Updates `visibility_overrides` and retries evaluation
1. Full section content becomes visible

## Overrides and Optimization

### Override Storage

- Store: `LocalPromptOverridesStore` (default: `~/.weakincentives/prompts`)
- Scoped by: namespace + prompt key + tag
- Seeded once at startup via `store.seed(prompt, tag=...)`

### Optimize Command

The `optimize` REPL command:

1. Creates a child session via `_create_optimization_session`
1. Builds `OptimizationContext` with adapter, bus, store, tag
1. Runs `WorkspaceDigestOptimizer` with `PersistenceScope.SESSION`
1. Prints the resulting workspace digest

## Observability

### Event Subscribers

Attached via `_attach_logging_subscribers(bus)`:

| Event | Handler | Output |
|-------|---------|--------|
| `PromptRendered` | `_print_rendered_prompt` | Full prompt text with label |
| `ToolInvoked` | `_log_tool_invocation` | Params, result, payload, token usage |
| `PromptExecuted` | `_log_prompt_executed` | Token usage summary |

### Token Usage Formatting

`_format_usage_for_log` renders `TokenUsage` as:
`token usage: input=N, output=N, cached=N, total=N`

### Plan Snapshots

`_render_plan_snapshot` formats the current plan:

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
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-5.1` | Model to use |
| `CODE_REVIEW_PROMPT_TAG` | No | `latest` | Overrides tag |

### REPL Commands

| Input | Action |
|-------|--------|
| Non-empty text | Submit as review request |
| `optimize` | Refresh workspace digest |
| `exit` / `quit` / empty | Terminate REPL |

### Output

- Intro banner with configuration summary
- `[prompt]` blocks showing rendered prompts
- `[tool]` blocks showing tool invocations
- `--- Agent Response ---` with formatted output
- `--- Plan Snapshot ---` with current plan state

## Testing

Tests in `tests/test_code_reviewer_example.py` cover:

- Prompt rendering and logging
- Default workspace digest behavior
- Overrides precedence
- Optimize command with stub adapter
- Structured response formatting
- Progressive disclosure flow

The `_create_runtime_context` helper enables test reuse without booting the
full REPL.

## Key Files

| File | Purpose |
|------|---------|
| `code_reviewer_example.py` | Main script |
| `test-repositories/sunfish/` | Mounted repository fixture |
| `snapshots/` | Session dump output directory |
| `~/.weakincentives/prompts/` | Default overrides storage |

# Weak Incentives (WINK)

WINK is a Python toolkit for building background coding agents that translate
end-user instructions into reliable application updates. The public API is
centered on typed prompts, tool contracts, replayable sessions, and
provider-agnostic adapters so you can keep determinism, observability, and
safety front and center while iterating on agent behaviors.

Audience: the guide is written for autonomous coding agents such as OpenAI
Codex (web and CLI), Claude Code, and Cursor Background Agents. It focuses on
calls and data structures these agents wire up—no internal implementation
knowledge required.

This document is the package README published to PyPI. It focuses on the
supported, public API surface you can depend on when wiring WINK into your own
agent or orchestration system.

## Installation

WINK targets Python 3.12+. Install the core library:

```bash
pip install weakincentives
```

Optional extras enable specific providers or tooling:

- `pip install "weakincentives[openai]"` for the OpenAI adapter.
- `pip install "weakincentives[litellm]"` for the LiteLLM adapter.
- `pip install "weakincentives[asteval]"` to enable the sandboxed Python eval
  tool.
- `pip install "weakincentives[podman]"` for Podman-based sandboxes.
- `pip install "weakincentives[wink]"` for the demo CLI (`wink`).

## Key Concepts

- **Prompts** (`weakincentives.prompt.Prompt`): Composable, dataclass-driven
  blueprints that render deterministic model inputs while exposing tool
  contracts.
- **Tools** (`weakincentives.prompt.Tool`): Declarative descriptions of
  capabilities the model can invoke. Tools surface type-checked handlers and
  renderable schemas.
- **Sessions** (`weakincentives.runtime.Session`): Redux-style ledgers that
  record every prompt render and tool invocation as immutable events. Reducers
  keep state deterministic and replayable.
- **Adapters** (`weakincentives.adapters.ProviderAdapter`): Bridges to model
  providers that negotiate tool calls and structured outputs without locking you
  into a single vendor.
- **Structured Output** (`parse_structured_output`): JSON-schema-backed parsing
  that turns model responses into typed dataclass instances.
- **Overrides** (`PromptOverride`, `LocalPromptOverridesStore`): Hash-based
  prompt overrides that let you refine prompt text safely in version control.

## Quickstart: Build a Coding Agent

The example below shows how to build a coding agent that takes user
instructions, renders a prompt with tool contracts, and executes tool calls in a
replayable session.

```python
from dataclasses import dataclass
from weakincentives import (
    Prompt,
    MarkdownSection,
    Tool,
    ToolContext,
    ToolResult,
    PromptResponse,
    parse_structured_output,
)
from weakincentives.adapters import ProviderAdapter, PromptEvaluationError
from weakincentives.adapters.api import OPENAI_ADAPTER_NAME
from weakincentives.runtime import (
    InProcessEventBus,
    Session,
    append,
    select_latest,
    configure_logging,
)

# 1) Define the structured plan you expect the model to return.
@dataclass
class Plan:
    summary: str
    steps: list[str]

# 2) Describe a tool the model can call to edit files.
@dataclass
class ApplyPatchArgs:
    path: str
    diff: str


def apply_patch(context: ToolContext[ApplyPatchArgs]) -> ToolResult:
    patch = context.args.diff
    # ...apply the patch to the local filesystem...
    return context.success(result={"applied": True})

apply_patch_tool = Tool(
    name="apply_patch",
    description="Apply a unified diff to the workspace",
    args=ApplyPatchArgs,
    handler=apply_patch,
)

# 3) Compose a prompt with Markdown sections and the tool.
user_goal = "Add a health-check endpoint to the FastAPI app"
prompt = Prompt(
    name="coding-agent",
    sections=[
        MarkdownSection(
            heading="Goal",
            content=user_goal,
        ),
        MarkdownSection(
            heading="Instructions",
            content=(
                "Propose a plan, then call `apply_patch` with a diff that satisfies"
                " the goal."
            ),
        ),
    ],
    tools=[apply_patch_tool],
    structured_output=Plan,  # Optional: request a typed plan before tools run.
)

# 4) Run the prompt through an adapter and capture the response.
adapter: ProviderAdapter = ProviderAdapter.for_name(
    OPENAI_ADAPTER_NAME,
    model="gpt-4o-mini",
)

configure_logging()
event_bus = InProcessEventBus()
session = Session(event_bus=event_bus)

try:
    rendered = prompt.render()
    response: PromptResponse = adapter.evaluate(rendered, session=session)
    plan = parse_structured_output(Plan, response.message)
    latest_patch_call = select_latest(session, event_type="tool_invocation")
    print("Plan:", plan)
    print("Last tool call:", latest_patch_call)
except PromptEvaluationError as exc:
    # Inspect exc.response for partial data or retry cues.
    raise
```

### What happens at runtime?

- The prompt renders Markdown plus a JSON-schema tool contract for
  `apply_patch`.
- `ProviderAdapter.evaluate` negotiates tool calls with the provider, emitting
  `PromptExecuted` and `ToolInvoked` events into the session ledger.
- `parse_structured_output` parses the model reply into the `Plan` dataclass.
- The session keeps every event immutable so you can snapshot, replay, or
  inspect them later.

## Prompt Authoring (`weakincentives.prompt`)

- **`Prompt`**: Construct with `name`, ordered `sections` (typically
  `MarkdownSection`), optional `tools`, and optional `structured_output` (a
  dataclass or `StructuredOutputConfig`). Call `render()` to obtain a
  `RenderedPromptProtocol` instance the adapters accept.
- **`MarkdownSection`**: Simple Markdown content with an optional `heading`. Use
  multiple sections to keep prompt text modular and override-friendly.
- **`Tool`**: Declarative tool description. Key fields:
  - `name`: Identifier exposed to the model.
  - `description`: Natural-language guidance.
  - `args`: Dataclass defining argument schema.
  - `handler`: Callable accepting `ToolContext[TArgs]` and returning
    `ToolResult`.
  - `renderable_result`: Optional `ToolRenderableResult` to predeclare the shape
    of results.
- **`ToolContext`**: Passed to tool handlers. Provides `args`, `session`, and
  helpers like `success(result=...)` or `failure(error=...)` to build a
  `ToolResult` with consistent metadata.
- **`ToolResult`**: Immutable record of a handler outcome (`result` or
  `error`).
- **`StructuredOutputConfig` / `parse_structured_output`**: Attach a
  dataclass-driven schema to a prompt or parse the final model message into a
  concrete instance. `OutputParseError` surfaces validation issues.
- **`Chapter` and `Section`**: Lower-level composition primitives for advanced
  prompt layouts. Use `ChaptersExpansionPolicy` to control how nested chapters
  expand.
- **Delegation helpers**: `DelegationPrompt`, `DelegationSummarySection`,
  `ParentPromptSection`, and related params let you compose supervising agents
  that delegate subtasks while maintaining typed contracts.
- **Overrides**: Use `PromptDescriptor`/`SectionDescriptor` and
  `LocalPromptOverridesStore` to persist hash-scoped overrides alongside your
  repo. Overrides are validated against the descriptor hashes to prevent drift.

## Adapter Layer (`weakincentives.adapters`)

- **`ProviderAdapter`**: Primary entrypoint for sending rendered prompts to a
  model provider. Construct via `ProviderAdapter.for_name(name, **kwargs)` using
  `OPENAI_ADAPTER_NAME` or `LITELLM_ADAPTER_NAME` constants. The adapter handles
  tool negotiation, streaming, and structured-output contracts.
- **`PromptResponse`**: Carries the final model message, any emitted tool calls,
  and provider metadata. Available even when `PromptEvaluationError` is raised
  so you can inspect partial results.
- **`SessionProtocol`**: Minimal interface adapters expect when you pass
  `session=` to `evaluate`. `runtime.Session` implements it out of the box.
- **Throttling**: Use `new_throttle_policy` or `ThrottlePolicy` to limit
  provider calls (e.g., rate limits). `ThrottleError` signals enforced delays.

## Runtime & Events (`weakincentives.runtime`)

- **`Session`**: Immutable event ledger with Redux-like reducers. Feed events in
  with `append(session, event)` or convenience selectors like `replace_latest`
  and `upsert_by`. `Snapshot`/`SnapshotProtocol` provide persistence helpers.
- **Reducers and selectors**: Use `TypedReducer` with `ReducerContext` to manage
  typed state slices. `select_all`, `select_latest`, and `select_where` help read
  accumulated data for analytics, auditing, or downstream actions.
- **Events**: `PromptExecuted` and `ToolInvoked` events capture every model
  exchange. `EventBus`/`InProcessEventBus` publish events to reducers and custom
  observers. `HandlerFailure` and `PublishResult` offer backpressure and error
  reporting controls.
- **Logging**: `configure_logging()` wires a structured logger; `get_logger`
  retrieves a module-level logger. `StructuredLogger` is a protocol you can
  implement for custom sinks.

## Session Snapshots for Debugging

Use `Session.snapshot()` to capture an immutable JSON bundle of the session's
state for offline debugging, replay, or handoff to another agent. Snapshots can
be stored, shared, or restored with `Snapshot.from_json(...)` and
`Session.rollback(...)` when you need to rehydrate session slices.

Snapshot payloads follow a stable, schema-versioned layout:

- **Envelope**: `version` (currently `"1"`), `created_at` (ISO timestamp), and
  optional `parent_id` plus `children_ids` when sessions are nested.
- **Slices**: Each entry captures one session slice. Fields include
  `slice_type` (module-qualified dataclass for the slice key), `item_type`
  (dataclass type stored within the slice), and `items` (a list of serialized
  dataclass mappings produced by `weakincentives.serde.dump`).
- **Deterministic ordering**: Slices and items are sorted during serialization
  so diffs stay stable in version control.

You can export snapshots as files, feed them into analysis tools, or inspect the
raw JSON to understand what prompts, tool invocations, and reducer-managed data
existed at a point in time—without needing live provider access.

## Deadlines (`weakincentives.Deadline`)

`Deadline` instances communicate time budgets across prompts, tools, and
provider calls. Pass them into prompts or adapters to ensure long-running tasks
fail fast and emit consistent timeout metadata.

## Data Types (`weakincentives.types`)

`JSONValue` represents JSON-compatible payloads frequently used in tool results
and structured outputs.

## Prompt Overrides in CI/CD

Ship prompts and overrides together:

1. Render a `PromptDescriptor` for your prompt and commit it.
2. Store edits in a `LocalPromptOverridesStore` scoped to your repo root.
3. Gate deployments by validating override hashes so prompts only change when
   descriptors do.

## CLI (`wink` extra)

Install the `wink` extra to experiment locally:

```bash
pip install "weakincentives[wink]"
wink --help
```

The CLI demonstrates prompt rendering, override validation, and adapter wiring
without writing additional code.

## Versioning & Stability

- Public APIs are the objects exported from `weakincentives.api` and the
  submodules documented above.
- Adapters are optional; include only the extras you need.
- Keep `StructuredOutputConfig`, tool schemas, and overrides in version control
  so your agents remain deterministic and auditable.

## License

Apache License 2.0. See `LICENSE` for details.

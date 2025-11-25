# Weak Incentives (WINK)

WINK is an open source toolkit for developing and optimizing side-effect-free
background agents (e.g., research, review, and coding agents) that translate
end-user instructions into deterministic actions. The library is designed for
human developers, but this README is written so automated coding agents (OpenAI
Codex web/CLI, Claude Code, Cursor background agents, etc.) can be productive
consumers of the API surface without depending on internals.

The public API centers on typed prompts, declarative tool contracts, replayable
sessions, and provider-agnostic adapters so you can keep determinism,
observability, and safety front and center while iterating on agent behaviors.

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

## Agent-facing operational notes

- WINK does not run unattended background agents by itself. It provides
  deterministic primitives that research/review/coding agents (or humans) drive
  explicitly via prompts, tool handlers, and adapters.
- Rendering is side-effect-free: `Prompt.render()` produces a typed
  `RenderedPrompt` containing message content, declared tools, and any
  structured-output schema, but does not contact providers until you pass it to
  an adapter.
- Tool handlers are synchronous callables; use them to gate filesystem or
  network access and to enforce policy before applying patches. Handlers accept
  the typed params plus a keyword-only `context` and return `ToolResult`
  instances (`ToolResult(message=..., value=..., success=True/False)`) to keep
  session logs consistent.
- `PromptResponse` carries the prompt name, rendered text, and parsed output
  (when structured output is requested) so you can safely resume after partial
  failures or retries.
- Sessions are immutable ledgers: reducers consume `PromptRendered`,
  `PromptExecuted`, and `ToolInvoked` events that include `event_id`,
  `session_id`, timestamps, and provider metadata so you can join prompt and
  tool flows deterministically.

## Quickstart: Build a Coding Agent

The example below shows how to build a coding agent that takes user
instructions, renders a prompt with tool contracts, and executes tool calls in a
replayable session.

```python
from dataclasses import dataclass
from weakincentives import (
    MarkdownSection,
    Prompt,
    Tool,
    ToolContext,
    ToolResult,
    parse_structured_output,
)
from weakincentives.adapters import PromptEvaluationError
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.runtime import InProcessEventBus, Session, configure_logging

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


@dataclass
class ApplyPatchResult:
    applied: bool


def apply_patch(params: ApplyPatchArgs, *, context: ToolContext) -> ToolResult[ApplyPatchResult]:
    patch = params.diff
    # ...apply the patch to the local filesystem...
    return ToolResult(message="applied", value=ApplyPatchResult(applied=True))


apply_patch_tool = Tool[ApplyPatchArgs, ApplyPatchResult](
    name="apply_patch",
    description="Apply a unified diff to the workspace.",
    handler=apply_patch,
)

# 3) Compose a prompt with Markdown sections and the tool.
user_goal = "Add a health-check endpoint to the FastAPI app"
prompt = Prompt[
    Plan
](
    ns="agent",  # Namespace for grouping related prompts.
    key="coding-agent",  # Unique identifier for overrides and session events.
    name="coding-agent",
    sections=[
        MarkdownSection(
            title="Goal",
            template=user_goal,
            key="goal",
        ),
        MarkdownSection(
            title="Instructions",
            template=(
                "Propose a plan, then call `apply_patch` with a diff that satisfies"
                " the goal."
            ),
            key="instructions",
        ),
    ],
    tools=(apply_patch_tool,),
)

# 4) Run the prompt through an adapter and capture the response.
adapter = OpenAIAdapter(model="gpt-4o-mini")

configure_logging()
event_bus = InProcessEventBus()
session = Session(bus=event_bus)

try:
    response = adapter.evaluate(
        prompt,
        bus=event_bus,
        session=session,
    )
    plan = response.output or parse_structured_output(Plan, response.text or "")
    print("Plan:", plan)
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

- **`Prompt`**: Construct with `ns`, `key`, optional `name`, ordered `sections`
  (typically `MarkdownSection`), optional `tools`, and an optional
  structured-output specialization (e.g., `Prompt[Plan]`). Call `render()` to
  obtain a `RenderedPrompt` if you need to inspect the provider payload;
  adapters render internally when you call `evaluate`.
- **`MarkdownSection`**: Simple Markdown content built from a `title`,
  `template`, and unique `key`. Use multiple sections to keep prompt text modular
  and override-friendly.
- **`Tool`**: Declarative tool description parameterized as `Tool[Params, Result]` where `Params` **must** be a dataclass type and `Result` is a dataclass (or sequence of dataclasses). Key fields:
  - `name`: Identifier exposed to the model.
  - `description`: Natural-language guidance (1-200 ASCII characters).
  - `handler`: Callable accepting `(params: Params, *, context: ToolContext)`
    and returning `ToolResult[Result]`.
- **`ToolContext`**: Passed to tool handlers. Provides `prompt`, optional
  `rendered_prompt`, `adapter`, `session`, `event_bus`, and optional
  `deadline`—use these to enforce policy and publish custom events.
- **`ToolResult`**: Structured handler response with `message`, `value`, and a
  `success` flag.
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

- **`OpenAIAdapter` / `LiteLLMAdapter`**: Primary entrypoints for sending prompts
  to their respective providers. They handle tool negotiation, streaming, and
  structured-output contracts.
- **`PromptResponse`**: Carries the prompt name, rendered text, and parsed output
  (when you request structured output). Available even when
  `PromptEvaluationError` is raised so you can inspect partial results.
- **`SessionProtocol`**: Minimal interface adapters expect when you pass
  `session=` to `evaluate`. `runtime.Session` implements it out of the box.
- **Throttling**: Use `new_throttle_policy` or `ThrottlePolicy` to limit
  provider calls (e.g., rate limits). `ThrottleError` signals enforced delays.

### Usage & cost instrumentation

- Adapters return parsed `PromptResponse` objects; provider-specific token usage
  is not attached. To budget spend, instrument the provider client you pass into
  an adapter. For OpenAI, you can wrap the `responses.create` method exposed by
  the `OpenAI` client and propagate usage metadata into your own ledger or
  `EventBus` subscriber without changing the response shape that the adapter
  expects.

```python
from openai import OpenAI
from weakincentives.adapters.openai import OpenAIAdapter


class MeteredClient:
    def __init__(self, **kwargs: object) -> None:
        base = OpenAI(**kwargs)
        usage_log: list[dict[str, int]] = []

        class _Responses:
            def create(_, **payload: object) -> object:
                response = base.responses.create(**payload)
                usage = getattr(response, "usage", {}) or {}
                usage_log.append(dict(usage))
                return response

        self.responses = _Responses()
        self.usage_log = usage_log


metered_client = MeteredClient()
adapter = OpenAIAdapter(model="gpt-4o-mini", client=metered_client)
result = adapter.evaluate(prompt, bus=event_bus, session=session)
usage = metered_client.usage_log[-1] if metered_client.usage_log else {}
tokens_in = usage.get("input_tokens", 0)
tokens_out = usage.get("output_tokens", 0)
# gpt-4o-mini (Nov 2024): $0.15 / 1M input tokens, $0.60 / 1M output tokens.
cost = (tokens_in * 0.00000015) + (tokens_out * 0.0000006)
```

- Example: if the OpenAI Responses API returns `input_tokens=800` and
  `output_tokens=200` for a `gpt-4o-mini` call, the total cost is
  `800 * 0.00000015 + 200 * 0.0000006 = $0.00024`. Adjust the rates when you use
  other models; multiply the returned token counts by the provider's posted
  per-token prices to keep your budget tracking accurate.

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
1. Store edits in a `LocalPromptOverridesStore` scoped to your repo root.
1. Gate deployments by validating override hashes so prompts only change when
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

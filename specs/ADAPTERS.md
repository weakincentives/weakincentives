# Adapter Evaluation Specification

## Introduction

Adapters bridge a rendered `Prompt` to a specific model provider using a synchronous, blocking API. Each adapter takes a
prompt plus its parameter dataclasses, calls the provider until a final assistant reply is produced, executes any
requested tools locally, and returns a typed `PromptResponse`. The surface stays intentionally narrow: a single prompt
render per request, no streaming transport, and no concurrency requirements.

## Goals

- **Prompt-Centric**: Accept a `Prompt` instance and the dataclass params needed to render it without duplicating
  templating logic in adapters.
- **Deterministic Tooling**: Execute declared `Tool` handlers synchronously with structured params, collecting the
  resulting `ToolResult` objects for downstream consumers.
- **Typed Outputs**: Reuse the structured output metadata exposed by `Prompt` to parse the final assistant message into
  a dataclass when available, falling back to raw text otherwise.
- **Provider-Agnostic Core**: Keep evaluation semantics the same for every model API so swapping providers only affects
  the adapter implementation, not the calling code.

## Guiding Principles

- **Blocking API**: `evaluate` runs to completion on the calling thread; no async contract or streaming callbacks.
- **Single Evaluation Scope**: One adapter call equals one rendered prompt and one terminal assistant message.
- **Tool Safety**: Tool execution happens locally via registered handlers; adapters never proxy tool calls back to the
  provider.
- **Structured Diagnostics**: Failures raise purpose-built exceptions with enough context (prompt name, tool, provider
  payload) to debug without scraping logs.

## Core Interfaces

### `ProviderAdapter`

```python
class ProviderAdapter(Protocol):
    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *params: object,
        goal_section_key: str,
        chapters_expansion_policy: ChaptersExpansionPolicy = ChaptersExpansionPolicy.INTENT_CLASSIFIER,
        parse_output: bool = True,
        bus: EventBus,
    ) -> PromptResponse[OutputT]:
        ...
```

Implementations own the provider client and any serialization glue needed for that API. The call must not mutate the
provided `Prompt` instance.

- `params`: Positional dataclass instances forwarded to `prompt.render(*params)`; adapters must preserve type matching.
- `goal_section_key`: Keyword-only pointer to the `Section.key` representing the
  user's immediate goal or intent. Chapter gating MUST anchor on this section as
  described in `specs/CHAPTERS.md`.
- `chapters_expansion_policy`: Keyword-only selector describing how aggressively
  the adapter may open chapters for this evaluation. Implementations MUST
  support at least the three core states defined in `specs/CHAPTERS.md` and MAY
  introduce provider-specific extensions.
- `parse_output`: When `True`, adapters call `parse_structured_output` on the final message if the prompt declares structured
  output; disable to keep only the raw text.
- `bus`: Evaluation-scoped event dispatcher supplied by the caller. Pass `NullEventBus()` to discard telemetry or reuse a
  shared bus when coordinating multiple adapters within the same request.

### `PromptResponse`

```python
@dataclass(slots=True)
class PromptResponse(Generic[OutputT]):
    prompt_name: str
    text: str | None
    output: OutputT | None
    tool_results: tuple[ToolInvoked, ...]
    provider_payload: dict[str, Any] | None = None
```

- `prompt_name`: Mirrors `prompt.name` for logging.
- `text`: The final assistant message (plain string) when structured output is absent or parsing is disabled.
- `output`: Parsed dataclass or list when available; `None` if the prompt did not declare structured output.
- `tool_results`: Ordered `ToolInvoked` events describing each executed tool call.
- `provider_payload`: Optional raw response fragment returned by the SDK for auditing.

Adapters emit the same `ToolInvoked` instances through the evaluation-scoped event bus provided by the caller. Consumers
that subscribe to `ToolInvoked` will receive the identical objects stored on the `PromptResponse`.

## Evaluation Flow

1. **Resolve Chapters** – Derive the subset of chapters to open based on the
   provided `goal_section_key`, the selected
   `chapters_expansion_policy`, and the heuristics defined in
   `specs/CHAPTERS.md`. Render against the resulting chapter snapshot.
1. **Render** – Call `rendered = prompt.render(*params)` once per evaluation using
   the chapter-filtered prompt snapshot. This yields the markdown
   (`rendered.text`), tool registry (`rendered.tools`), and optional output contract metadata.
1. **Prepare Payload** – Construct the provider-specific request body using `rendered.text` as the system prompt (or
   equivalent) and translate each `Tool` into the provider's tool schema.
1. **Call Provider** – Issue a blocking completion/chat request. When the provider emits a tool call, decode the name
   and arguments, materialize the params dataclass (e.g., via `serde.parse`), run the `Tool.handler`, publish a
   `ToolInvoked` event through the supplied bus (capturing the params/result/call ID), append it to
   `PromptResponse.tool_results`, and feed the handler's `ToolResult.message` back to the provider as the tool response.
1. **Repeat** – Continue the call-tool-respond loop until the provider returns a final assistant message with no further
   tool invocations.
1. **Assemble Response** – Populate `PromptResponse` with the final text. If `rendered.output_type` is present and
   `parse_output` is `True`, call `parse_structured_output(final_text, rendered)` to produce `output`; otherwise leave it as
   `None`.
1. **Return** – Hand the fully populated `PromptResponse` back to the caller.

## Tool Execution

- Match tool calls by exact `Tool.name`. Missing handlers raise `PromptEvaluationError` immediately.
- Arguments must deserialize into the declared params dataclass. Validation failures bubble as
  `PromptEvaluationError` with the offending payload attached.
- Handlers run synchronously and must return `ToolResult[...]`, setting `success=False` and `value=None` (or a structured
  error payload) when they cannot fulfill the request.
- Adapters wrap handler exceptions and convert them into `ToolResult(success=False, value=None, message="…")` instances,
  publish the `ToolInvoked` event, and continue the evaluation instead of surfacing a `PromptEvaluationError`.
- The `ToolResult.message` is the only content echoed back to the provider; the structured payload stays local and is
  captured in the `ToolInvoked` event/response entry.

## Error Handling

Raise `PromptEvaluationError` (new exception type in the adapters package) for:

- Provider failures (non-2xx responses, SDK errors).
- Unknown tool names or missing handlers when the model requests a tool.
- Parameter deserialization errors.
- Structured output parsing failures when `parse_output=True`.

Adapters should log handler exceptions but treat them as tool failures, returning `ToolResult(success=False, value=None, message="…")` to the provider so the LLM can react without aborting evaluation.

The exception should expose:

- `prompt_name`
- `phase` (`"request"`, `"tool"`, `"response"`, etc.)
- Provider-specific diagnostics (status code, request id)
- The original exception or payload when relevant

## Non-Goals

- Streaming or incremental token delivery.
- Multi-turn conversation management; higher-level orchestration should build on top of `ProviderAdapter` if needed.
- Automatic retries or backoff logic; callers decide whether to retry failed evaluations.
- Background execution or async contracts.

## Usage Sketch

```python
from dataclasses import dataclass

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.events import InProcessEventBus
from weakincentives.prompt import Prompt, MarkdownSection


@dataclass
class MessageParams:
    sender: str
    topic: str


adapter: ProviderAdapter  # e.g., OpenAIAdapter
bus = InProcessEventBus()
prompt = Prompt(
    name="draft_reply",
    sections=[
        MarkdownSection[MessageParams](
            key="task",
            title="Task",
            template="Please draft a reply to ${sender} about ${topic}.",
        )
    ],
)
response = adapter.evaluate(
    prompt,
    MessageParams(sender="Jordan", topic="launch plan"),
    goal_section_key="task",
    chapters_expansion_policy=ChaptersExpansionPolicy.INTENT_CLASSIFIER,
    bus=bus,
)
print(response.text or response.output)
for call in response.tool_results:
    print(call.name, call.result.value)
```

The caller supplies the prompt and dataclass params, the adapter handles rendering, provider interaction, tool execution,
and structured output parsing, and the returned `PromptResponse` aggregates every artifact from the single evaluation.

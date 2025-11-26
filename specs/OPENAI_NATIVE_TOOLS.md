# OpenAI Native Tools via Responses

## Overview

This document describes how to integrate OpenAI-managed native tools (such as
code interpreter or search) into the existing Responses-only OpenAI adapter
without breaking the current prompt, tool, and event contracts implemented in
`src/weakincentives`. Native tools are **provider-executed**: the adapter must
opt a request into a built-in capability and surface the provider's results,
while keeping the runtime's tool wiring, DbC validation, and event publishing
unchanged.

## Goals

- Preserve the adapter's current request/response pipeline (Responses API,
  blocking calls, no streaming) while adding support for native tool entries.
- Keep prompt structure, section composition, and tool registration compatible
  with the existing `Section` and `Tool` abstractions so prompts remain
  deterministic.
- Emit the same runtime events that existing tools produce so downstream
  automation continues to consume `ToolInvoked` with no schema changes.

## Non-Goals

- Re-introducing Chat Completions or streaming (Responses-only remains the
  contract).
- Changing `Tool` validation rules (names, descriptions, handler signatures) for
  function tools.
- Defining a brand-new event surface for native tools; existing `ToolInvoked`
  stays the shared event type.

## Section Abstractions

Each native tool should be expressed as a **dedicated section class** exported by
an adapter-owned module (for example,
`src/weakincentives/adapters/openai/native_tools/file_search_section.py`). These
sections must align with the real `Section` API in `src/weakincentives/prompt`:

- Subclass `Section[ParamsT]` with a concrete dataclass parameter type that
  captures the provider configuration knobs for a single native tool. Do **not**
  rely on class attributes like `sections = (...)`; the `Prompt` constructor
  expects `Section` instances passed via its `sections=` argument at runtime.
- Implement `render` and `clone` per the base-class contract. Native tool
  sections typically render no visible text but must still participate in the
  section tree so placeholder validation and overrides remain deterministic.
- Register per-tool metadata on the section instance so the adapter can discover
  which native tools to request. Because the runtime collects tools from
  `Section.tools()`, attach **no `Tool` instances** for native tools (they are
  provider-managed), but expose the native tool settings on the section so the
  adapter can translate them into provider payloads.
- Run validation in the section constructor: confirm the model advertises the
  native tool, assert required config (corpus IDs, resource limits), and forbid
  mixing incompatible knobs (for example, disabling native tools when structured
  output forbids tool use).

### Example skeleton

```python
from dataclasses import dataclass
from weakincentives.prompt.section import Section
from weakincentives.prompt._types import SupportsDataclass


@dataclass(slots=True)
class FileSearchParams:
    corpus_ids: tuple[str, ...] = ()
    max_results: int = 5


class OpenAIFileSearchSection(Section[FileSearchParams]):
    def __init__(self, *, default_params: FileSearchParams | None = None) -> None:
        super().__init__(
            title="File search",
            key="openai_file_search",
            default_params=default_params or FileSearchParams(),
            tools=(),  # provider executes the tool; no local handler
            accepts_overrides=False,
        )

    def render(self, params: SupportsDataclass | None, depth: int, number: str) -> str:
        _ = params  # native tool sections usually render nothing
        return ""

    def clone(self, **kwargs: object) -> "OpenAIFileSearchSection":
        default = kwargs.get("default_params", self.default_params)
        return OpenAIFileSearchSection(default_params=default)
```

## Prompt Execution Flow

The current OpenAI adapter builds Responses payloads by:

1. Rendering sections into a single markdown string and seeding
   `initial_messages=[{"role": "system", "content": rendered.text}]`.
1. Translating `Tool` instances from the rendered prompt into function tool
   specs via `tool_to_spec` in `src/weakincentives/adapters/shared.py` and
   validating them in `_responses_tool_spec` inside
   `src/weakincentives/adapters/openai.py`. Only `{"type": "function"}` entries
   are accepted today.
1. Normalizing request messages through `_normalize_input_messages` so
   `function_call` and `function_call_output` payloads align with Responses
   expectations.
1. Executing tool calls locally via `ToolExecutor` and publishing `ToolInvoked`
   events with the tool name, params, result, and `call_id` (no native-tool flag
   exists in the event schema).

Native tool support must extend this flow without regressing existing behavior:

- Introduce a provider-native tool envelope that bypasses `tool_to_spec` for the
  native entries while keeping function tools untouched.
- Map adapter-level `tool_choice` into the provider format using the existing
  `_responses_tool_choice` helper, and reject unsupported shapes with
  `PromptEvaluationError` to preserve DbC guards.
- Preserve the `function_call_output` stitching and `ToolInvoked` publishing
  paths so downstream consumers continue to observe tool results in the same
  event shape. Provider-returned native tool outputs should be inserted into the
  transcript before the next assistant turn, exactly like function tool outputs
  are handled today.

## Tool Registration Flow

- **Capability discovery**: gate native tool sections behind a per-model allow
  list (for example, adapter-level metadata such as
  `openai_native_tools={"gpt-4.1": {"file_search"}}`). Construction should fail
  fast when the model does not advertise the requested tool.
- **Schema construction**: native tool specs differ from the function schemas
  produced by `tool_to_spec`. Extend `_responses_tool_spec` to accept minimal
  `{"type": "built_in", "name": ...}` entries while keeping existing validation
  for function tools intact. Reject attempts to attach `parameters`/`strict`
  payloads the Responses API will ignore for native tools.
- **Prompt metadata wiring**: attach the native tool envelope to the rendered
  prompt (for example, by extending `RenderedPrompt` to carry a
  `native_tools: tuple[Mapping[str, object], ...]` collection) so the adapter can
  merge them into the `tools` list alongside the function specs it already
  produces.
- **DbC enforcement**: maintain the current early failures for malformed tool
  specs by raising `PromptEvaluationError` during request construction. Tool
  choice must reference a known native tool name when present.

## Integration Points in the Current OpenAI Adapter

The following code paths require updates to recognize native tools while keeping
existing behavior unchanged:

- `_responses_tool_spec` (in `src/weakincentives/adapters/openai.py`) currently
  rejects anything except `{"type": "function"}`. Add a branch that passes
  through `{"type": "built_in", "name": ...}` after validating the name.
- `_responses_tool_choice` should map native tool choices into
  `{"type": "built_in", "name": <tool>}` envelopes while continuing to raise
  `PromptEvaluationError` for unsupported shapes or missing names.
- `_normalize_input_messages` rewrites assistant tool calls into
  `function_call` parts and tool outputs into `function_call_output` messages.
  Extend it to preserve provider-emitted native tool calls without forcing them
  into the function-call schema so the runtime can route them unchanged.
- `OpenAIAdapter.evaluate` constructs the request payload via `_call_provider`.
  Merge native tool specs into the `tools` array without altering existing
  deadline handling, response formatting, or error normalization logic.

## Usage Examples

### Building a prompt with a native tool section

Native tool sections are regular `Section` instances passed to `Prompt` at
construction time:

```python
from dataclasses import dataclass
from weakincentives.adapters.openai.native_tools.file_search_section import (
    OpenAIFileSearchSection,
)
from weakincentives.prompt import MarkdownSection, Prompt


@dataclass(slots=True)
class ContentParams:
    topic: str


prompt = Prompt(
    ns="research",
    key="native-file-search",
    name="native_file_search",
    sections=(
        OpenAIFileSearchSection(),
        MarkdownSection[ContentParams](
            title="Task",
            key="task",
            template="Summarize the latest findings about ${topic} using the file search tool if needed.",
        ),
    ),
)
```

### Mixing native and function tools when calling the adapter

Adapters still receive function tools via the `Tool` abstraction; native tools
flow through the new section metadata. A minimal invocation mirrors existing
integration tests in `integration-tests/test_openai_adapter_integration.py`:

```python
from dataclasses import dataclass
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.adapters.openai.native_tools.file_search_section import (
    OpenAIFileSearchSection,
)
from weakincentives.prompt import MarkdownSection, Prompt, Tool, ToolContext, ToolResult
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session


@dataclass(slots=True)
class ContentParams:
    topic: str


@dataclass(slots=True)
class EchoParams:
    text: str


def echo(params: EchoParams, *, context: ToolContext) -> ToolResult[EchoParams]:
    _ = context
    return ToolResult.success(params)


echo_tool = Tool[EchoParams, EchoParams](
    name="echo",
    description="Echo the provided text.",
    handler=echo,
)

prompt = Prompt(
    ns="research",
    key="native-file-search",
    name="native_file_search",
    sections=(
        OpenAIFileSearchSection(),
        MarkdownSection[ContentParams](
            title="Task",
            key="task",
            template="Summarize the latest findings about ${topic}.",
            tools=(echo_tool,),
        ),
    ),
)

adapter = OpenAIAdapter(model="gpt-4.1")
session = Session()
bus = InProcessEventBus()

response = adapter.evaluate(
    prompt,
    ContentParams(topic="LLM safety"),
    EchoParams(text="hello"),
    parse_output=True,
    bus=bus,
    session=session,
)
```

- The prompt supplies native tool metadata via `OpenAIFileSearchSection`, while
  the function tool continues to flow through `Tool` → `tool_to_spec` →
  `_responses_tool_spec`.
- Tool invocations (native or function) still publish `ToolInvoked` events using
  the existing serializer (`serialize_tool_message`) and argument parser
  (`parse_tool_arguments`).

## Integration Tests

Add a dedicated module under `integration-tests/` (for example,
`integration-tests/test_openai_native_tools_integration.py`) and gate it behind
`OPENAI_API_KEY` using the existing `openai` marker. Follow the current patterns
in `integration-tests/test_openai_adapter_integration.py`:

- Construct prompts with concrete `Section` instances (no class attributes) and
  pass dataclass params to `Prompt.render`/`OpenAIAdapter.evaluate`.
- Cover at least one end-to-end round trip per native tool, asserting that the
  returned transcript contains the provider-emitted tool call and that
  `ToolInvoked` was published with the expected `name` and `call_id`.
- Verify DbC enforcement by asserting that unsupported native tool names or
  malformed configurations raise `PromptEvaluationError` before the provider
  request is issued.
- Keep tests behind the `openai` marker and reuse the `Session`/`NullEventBus`
  scaffolding from existing integration tests to maintain consistent harnesses.

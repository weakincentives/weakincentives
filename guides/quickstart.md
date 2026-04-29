# Quickstart

This guide builds an end-to-end agent in fewer than 80 lines of Python:
a typed prompt with one tool, a session that records progress, and an
agent loop driven by the deterministic noop adapter.

```python
from dataclasses import dataclass, replace

from weakincentives.adapters import NoopAdapter, ScriptedResponse
from weakincentives.core import (
    MarkdownSection,
    Prompt,
    Replace,
    Session,
    Tool,
    ToolCall,
    ToolContext,
    ToolResult,
    reducer,
)
from weakincentives.runtime import AgentLoop


# 1. State slice — the agent's memory of what it has done.
@dataclass(frozen=True)
class AddNote:
    text: str


@dataclass(frozen=True)
class Notes:
    items: tuple[str, ...] = ()

    @reducer(on=AddNote)
    def add(self, event: AddNote) -> Replace["Notes"]:
        return Replace((replace(self, items=(*self.items, event.text)),))


# 2. Section parameters — typed inputs to the prompt.
@dataclass(frozen=True)
class Greeting:
    name: str


# 3. A tool the model can call.
def remember(params: object, context: ToolContext) -> ToolResult[None]:
    if not isinstance(params, dict):
        return ToolResult.error("expected {'text': str}")
    text = params.get("text")
    if not isinstance(text, str):
        return ToolResult.error("'text' must be a string")
    context.session.dispatch(AddNote(text=text))
    return ToolResult.ok(None, message=f"noted: {text}")


note_tool = Tool(name="note", description="Record a note", handler=remember)


# 4. Compose the prompt.
prompt = Prompt(
    ns="demo",
    key="notes",
    sections=(
        MarkdownSection[Greeting](
            title="Greeting",
            key="greet",
            template="Hello $name. You can call `note(...)`.",
            params_type=Greeting,
            default_params=Greeting(name="world"),
            tools=(note_tool,),
        ),
    ),
)


# 5. Drive the agent loop.
session = Session()
session.install(Notes, initial=Notes)

adapter = NoopAdapter(
    responses=(
        ScriptedResponse(
            text="thinking",
            tool_calls=(
                ToolCall(name="note", arguments={"text": "remember the milk"}),
            ),
        ),
        ScriptedResponse(text="done", finish_reason="stop"),
    )
)

result = AgentLoop(adapter=adapter, session=session).run(prompt)

assert result.iterations == 2
assert session[Notes].latest() == Notes(items=("remember the milk",))
```

## What just happened?

- `Session` started empty. `session.install(Notes, initial=Notes)`
  registered the `@reducer` method and seeded an initial value.
- `Prompt.render()` produced a markdown document with the section's
  body and surfaced `note_tool` via `RenderedPrompt.tools`.
- `NoopAdapter` replayed two scripted responses. The first emitted a
  `ToolCall`; the second was a terminal text response.
- `AgentLoop` dispatched the tool call through `execute_tool`, which
  ran the handler inside a transaction. The handler dispatched
  `AddNote(...)`; the reducer turned it into the new `Notes` state.

To swap in a real provider, replace `NoopAdapter` with any object
satisfying `weakincentives.core.ProviderAdapter` — for example,
`OpenAICompatibleAdapter` from `weakincentives.adapters` against a
real OpenAI client.

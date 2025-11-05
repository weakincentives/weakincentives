<!-- mdformat off -->

<!--
marp: true
paginate: true
class: lead
-->

# Weak Incentives

## Typed building blocks for side-effect-free background agents

```python
from weakincentives import Prompt, MarkdownSection

print(Prompt.__mro__[0].__name__)
```

______________________________________________________________________

## Quickstart

1. **Install** the package and optional extras with [`uv`](https://github.com/astral-sh/uv):
   ```bash
   uv add weakincentives
   uv add "weakincentives[asteval]"
   uv add "weakincentives[openai]"
   uv add "weakincentives[litellm]"
   ```
1. **Sync the repo** (if developing locally):
   ```bash
   uv sync --extra asteval --extra openai --extra litellm
   ```
1. **Explore the examples** in `code_reviewer_example.py` and the prompts under `specs/` to understand the runtime patterns.

```python
from weakincentives.logging import configure_logging

configure_logging(level="INFO")
```

______________________________________________________________________

## Observable session state

- Redux-inspired session ledger captures every prompt and tool interaction.
- Reducers attach domain-specific validation while keeping runs deterministic.
- In-process event bus emits `ToolInvoked` and `PromptExecuted` events for telemetry.
- Built-in planning, virtual filesystem, and Python evaluation sections register reducers automatically.

```python
from weakincentives.events import InProcessEventBus, ToolInvoked
from weakincentives.session import Session, ToolData, append

bus = InProcessEventBus()
session = Session(bus=bus)
session.register_reducer(ToolData, append())

bus.publish(ToolInvoked(tool_name="search", arguments={"query": "mdformat"}))
print(session.select_all(ToolData))
```

______________________________________________________________________

## Composable prompt blueprints

- Prompt sections are typed dataclasses with validated placeholders.
- Sections assemble into reusable prompt trees that enforce strict contracts.
- Markdown renders stay predictable and version-control-friendly.
- Tool contracts surface alongside prompts to keep structured replies consistent.

```python
from dataclasses import dataclass

from weakincentives.prompt import MarkdownSection, Prompt

@dataclass(slots=True)
class ReviewParams:
    repository: str
    target_branch: str

review_body = MarkdownSection[
    ReviewParams
](
    title="Code Review",
    key="review",
    template="""Repository: ${repository}\nBranch: ${target_branch}""",
)

prompt = Prompt[str](ns="demo", key="review", sections=[review_body])
rendered = prompt.render(ReviewParams(repository="weakincentives", target_branch="main"))
print(rendered.text)
```

______________________________________________________________________

## Override-friendly workflows

- Prompt overrides enable experimentation without changing source-controlled defaults.
- Hash-based descriptors keep overrides aligned with prompt schema changes.
- On-disk overrides are validated and resolved relative to the Git root.
- Optimization loops plug into the same override surface as manual tweaks.

```python
from weakincentives.prompt.local_prompt_overrides_store import LocalPromptOverridesStore
from weakincentives.prompt.versioning import PromptDescriptor

store = LocalPromptOverridesStore()
descriptor = PromptDescriptor(ns="demo", key="review")
override = store.resolve(descriptor)
print(override)
```

______________________________________________________________________

## Provider adapters

- Conversation loop negotiates tool calls across model providers.
- JSON Schema-enforced response formats normalize structured payloads.
- Runtime stays model-agnostic while adapters share the same negotiation contract.

```python
from dataclasses import dataclass

from weakincentives.adapters.core import PromptResponse
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.events import InProcessEventBus
from weakincentives.prompt import MarkdownSection, Prompt
from weakincentives.session import Session

@dataclass(slots=True)
class ReviewParams:
    repository: str
    target_branch: str

review_body = MarkdownSection[ReviewParams](
    title="Code Review",
    key="review",
    template="Repository: ${repository}\nBranch: ${target_branch}",
)
prompt = Prompt[str](ns="demo", key="review", sections=[review_body])

adapter = OpenAIAdapter(model="gpt-4.1-mini")
bus = InProcessEventBus()
session = Session(bus=bus)

response: PromptResponse[str] = adapter.evaluate(
    prompt,
    ReviewParams(repository="weakincentives", target_branch="main"),
    bus=bus,
    session=session,
)
print(response.output)
```

______________________________________________________________________

## Local-first, deterministic execution

- No mandatory hosted services—everything runs locally by default.
- Reproducible renders keep diffs meaningful and easy to review.
- Code review example combines overrides, session telemetry, and replayable tooling.

```python
from weakincentives.events import NullEventBus
from weakincentives.session import Session, ToolData

bus = NullEventBus()
session = Session(bus=bus, session_id="local-demo")
print({"session_id": session.session_id, "events": session.select_all(ToolData)})
```

______________________________________________________________________

## Next steps

- Read the specs in `specs/` for deep dives into sessions, prompts, tooling, and overrides.
- Extend the library with typed tools and prompts tailored to your workflow.
- Wire the Marp workflow to publish these slides via GitHub Pages.

```python
from weakincentives import parse_structured_output

print(parse_structured_output.__doc__.splitlines()[0])
```

<!-- mdformat on -->

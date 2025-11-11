<!-- mdformat off -->

<!--
marp: true
paginate: true
class: lead
-->

# Weak Incentives

## Typed building blocks for side-effect-free background agents

```python
from weakincentives import Prompt

print(Prompt.__mro__[0].__name__)
```

______________________________________________________________________

## Quickstart

1. Install the core package and extras when needed: `uv add weakincentives 'weakincentives[openai]'`
1. Mirror CI locally: `uv sync --all-extras`
1. Run `code_reviewer_example.py` to see prompts, sessions, and tools in action.

```python
from weakincentives.logging import configure_logging

configure_logging("INFO")
```

______________________________________________________________________

## Observable session state

- Append-only session ledger captures prompts and tool calls.
- Reducers validate records while keeping the run deterministic.
- Event bus emits `ToolInvoked` and `PromptExecuted` telemetry out of the box.

```python
from weakincentives.events import InProcessEventBus, ToolInvoked
from weakincentives.session import Session, ToolData, append

bus = InProcessEventBus()
Session(bus=bus).register_reducer(ToolData, append())
bus.publish(ToolInvoked(tool_name="search", arguments={"query": "mdformat"}))
```

______________________________________________________________________

## Composable prompt blueprints

- Sections are typed dataclasses with validated placeholders.
- Prompts reuse those sections so contracts stay consistent.
- Markdown output is deterministic and diff-friendly.

```python
from dataclasses import dataclass

from weakincentives.prompt import MarkdownSection, Prompt

@dataclass
class ReviewParams:
    repository: str

Prompt[str](
    ns="demo",
    key="review",
    sections=[
        MarkdownSection[ReviewParams](
            title="Review",
            key="review",
            template="Repo: ${repository}",
        )
    ],
)
```

______________________________________________________________________

## Built-in sections

- Planning, virtual filesystem, and evaluation ship as ready-made sections.
- Each registers reducers so session state mirrors plans, mounts, and evals.
- Optional `AstevalSection` runs guarded calculations inline.

```python
from pathlib import Path

from weakincentives.tools import PlanningToolsSection, VfsToolsSection

planning = PlanningToolsSection()
vfs = VfsToolsSection(allowed_host_roots=(Path("/srv"),))

print(planning.key, vfs.key)
```

______________________________________________________________________

## Tool suites and helpers

- Tools expose typed payloads for plans, VFS access, and evals.
- `PlanningToolsSection` wraps create/update/complete lifecycle calls.
- `VfsToolsSection` enforces allowlists for host paths and patches.

```python
section = PlanningToolsSection()
print([tool.name for tool in section.tools])
```

______________________________________________________________________

## Override-friendly workflows

- Overrides let you test changes without touching the default prompt set.
- Descriptors hash the schema so stale overrides are rejected early.
- Stores resolve relative to the repo root for reproducible runs.

```python
from weakincentives.prompt.local_prompt_overrides_store import LocalPromptOverridesStore
from weakincentives.prompt.versioning import PromptDescriptor

descriptor = PromptDescriptor(ns="demo", key="review")
override = LocalPromptOverridesStore().resolve(descriptor)
```

______________________________________________________________________

## Provider adapters

- Negotiation loop handles tool calls for each provider integration.
- JSON Schema validation keeps structured responses aligned.
- Swapping adapters preserves the same runtime contract.

```python
from weakincentives.adapters.openai import OpenAIAdapter

# prompt, params, bus, and session come from the prior setup
adapter = OpenAIAdapter(model="gpt-4.1-mini")
response = adapter.evaluate(prompt, params, bus=bus, session=session)
print(response.output[:60])
```

______________________________________________________________________

## Local-first, deterministic execution

- Local defaults avoid relying on hosted services.
- Deterministic renders keep diffs small and reviewable.
- Code review example shows overrides, telemetry, and tooling together.

```python
from weakincentives.events import NullEventBus
from weakincentives.session import Session, ToolData

bus = NullEventBus()
session = Session(bus=bus, session_id="local-demo")
print({"session_id": session.session_id, "events": session.select_all(ToolData)})
```

______________________________________________________________________

## Next steps

- Dive into `specs/` for detailed behavior and extension points.
- Add domain-specific prompts and tools on top of the typed primitives.
- Enable the Pages workflow to publish or export the deck when ready.

```python
from weakincentives import parse_structured_output

print(parse_structured_output.__doc__.splitlines()[0])
```

<!-- mdformat on -->

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

1. Install the core package, then add extras when you need adapters:
   ```bash
   uv add weakincentives "weakincentives[openai]"
   ```
1. Sync the repo with all extras to mirror CI:
   ```bash
   uv sync --all-extras
   ```
1. Run `code_reviewer_example.py` to see sessions, prompts, and tools working together.

```python
from weakincentives.logging import configure_logging

configure_logging(level="INFO")
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
session = Session(bus=bus)
session.register_reducer(ToolData, append())

bus.publish(ToolInvoked(tool_name="search", arguments={"query": "mdformat"}))
print(session.select_all(ToolData))
```

______________________________________________________________________

## Composable prompt blueprints

- Sections are typed dataclasses with validated placeholders.
- Prompts reuse those sections so contracts stay consistent.
- Markdown output is deterministic and diff-friendly.

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

## Built-in sections

- Planning, virtual filesystem, and evaluation ship as ready-made sections.
- Each registers reducers so session state mirrors plans, mounts, and evals.
- Optional `AstevalSection` runs guarded calculations inline.

```python
from pathlib import Path

from weakincentives.tools import (
    HostMount,
    PlanningToolsSection,
    VfsPath,
    VfsToolsSection,
)

diff_root = Path("/srv/agent-mounts")
vfs_section = VfsToolsSection(
    allowed_host_roots=(diff_root,),
    mounts=(
        HostMount(
            host_path="octo_widgets/cache-layer.diff",
            mount_path=VfsPath(("diffs", "cache-layer.diff")),
        ),
    ),
)
planning_section = PlanningToolsSection()

print((planning_section.key, vfs_section.key))
```

______________________________________________________________________

## Tool suites and helpers

- Tools expose typed payloads for plans, VFS access, and evals.
- `PlanningToolsSection` wraps create/update/complete lifecycle calls.
- `VfsToolsSection` enforces allowlists for host paths and patches.

```python
from weakincentives.tools import PlanningToolsSection

section = PlanningToolsSection()
tool_names = tuple(tool.name for tool in section.tools)

print(tool_names[:3])
```

______________________________________________________________________

## Override-friendly workflows

- Overrides let you test changes without touching the default prompt set.
- Descriptors hash the schema so stale overrides are rejected early.
- Stores resolve relative to the repo root for reproducible runs.

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

- Negotiation loop handles tool calls for each provider integration.
- JSON Schema validation keeps structured responses aligned.
- Swapping adapters preserves the same runtime contract.

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

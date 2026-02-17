# Prompts

*Canonical spec: [specs/PROMPTS.md](../specs/PROMPTS.md)*

The prompt system is the heart of WINK. This guide explains how to think about
prompts as structured, typed objects rather than strings.

## Why Prompts Are Objects, Not Strings

Most systems treat prompts as strings and hope conventions keep everything
aligned. Prompt text in one place, tool definitions in another, schema
expectations in another, memory in another. Teams add layers—prompt registries,
tool catalogs, schema validators—each separately maintained. They drift. When
something breaks, you're hunting across files to understand what was actually
sent to the model.

WINK's design goal is **predictability**:

- Prompt text is deterministic—same inputs, same outputs
- Placeholder names are validated at construction time
- Composition is explicit (no magic string concatenation)
- Overrides are safe (hash-validated)

## PromptTemplate: The Immutable Definition

`PromptTemplate[OutputT]` is the immutable definition of an agent prompt. Think
of it as a blueprint that never changes once defined.

```python nocheck
from dataclasses import dataclass
from weakincentives.prompt import PromptTemplate, MarkdownSection

@dataclass(slots=True, frozen=True)
class Params:
    question: str

template = PromptTemplate(
    ns="support",
    key="faq",
    sections=(
        MarkdownSection(
            title="Instruction",
            key="instruction",
            template="Answer clearly and briefly.",
        ),
        MarkdownSection(
            title="Question",
            key="question",
            template="Question: ${question}",
        ),
    ),
)
```

**Key properties:**

- `ns` (namespace) and `key` uniquely identify a prompt family — both are
  validated against `^[a-z0-9][a-z0-9._-]{0,63}$` and normalized to lowercase
- `name` is for human readability and logging
- `sections` is a tree (technically a forest) of `Section` objects
- Optional structured output is declared by the type parameter `OutputT`
- Optional `task_completion_checker` for verifying goals before the agent stops

**Validation rules** (you'll hit these early, and that's the point):

- Namespace and key values must match `^[a-z0-9][a-z0-9._-]{0,63}$` (no
  slashes, spaces, or uppercase — uppercase is silently normalized)
- Section keys must be stable identifiers (lowercase alphanumerics + `-`)
- Placeholders must match fields on a dataclass bound at render time
- Binding is by dataclass type: you can't bind two instances of the same
  dataclass type

These constraints exist to catch errors early. A typo in a placeholder name
fails at construction, not when the model is mid-response.

## Prompt: The Runtime Binding

`Prompt[OutputT]` is the runtime binding that connects a template to specific
parameters:

```python nocheck
from weakincentives.prompt import Prompt

prompt = Prompt(template).bind(Params(question="What is WINK?"))
```

You can also configure overrides:

```python nocheck
prompt = Prompt(template, overrides_store=store, overrides_tag="stable")
prompt = prompt.bind(Params(question="What is WINK?"))
```

Rendering returns a `RenderedPrompt[OutputT]` which contains:

- `text`: final markdown
- `tools`: tools contributed by enabled sections (in traversal order)
- `structured_output`: schema config when declared
- `descriptor`: a hash-based descriptor used by the overrides system

The distinction between `PromptTemplate` and `Prompt` matters: templates are
reusable and immutable; prompts are bound to specific parameters and can carry
override configuration.

## Sections: The Building Blocks

A `Section[ParamsT]` is a node in the prompt tree. Every section has:

- `title`: used to render markdown headings
- `key`: stable ID within the prompt tree
- Optional `children`: subsections
- Optional `tools`: tool contracts to expose
- Optional `enabled`: a predicate that can disable the section at render time
- Optional `summary` + `visibility`: for progressive disclosure
- `accepts_overrides`: whether the override system may replace its body

A section must implement `render_body(...)`. The most common type is
`MarkdownSection`, but contributed tool suites are also sections (workspace
digest, Claude Agent SDK workspace, etc.).

## MarkdownSection: The Workhorse

`MarkdownSection` renders a `string.Template` with `${name}` placeholders:

```python nocheck
from dataclasses import dataclass
from weakincentives.prompt import MarkdownSection

@dataclass(slots=True, frozen=True)
class User:
    name: str
    plan: str

profile = MarkdownSection(
    title="User",
    key="user",
    template="Name: ${name}\nPlan: ${plan}",
)
```

**Why `string.Template`?** It's deliberately simple:

- No expressions
- No loops
- No conditionals

Complex formatting belongs in your Python code (where it can be tested). If you
find yourself wanting loops in your template, that's a sign you should compute
the string in Python and pass it as a param.

## Structured Output

Structured output is declared by parameterizing the prompt template:

- `PromptTemplate[OutputDataclass]` → output is a dataclass instance
- `PromptTemplate[list[OutputDataclass]]` → output is a list of dataclass
  instances

Adapters will:

1. Instruct the model to return JSON for the schema
1. Parse the response into your dataclass type
1. Return it as `PromptResponse.output`

If you need to parse output yourself (or you're using a custom adapter), use
`parse_structured_output(...)`:

```python nocheck
from weakincentives.prompt import parse_structured_output

rendered = prompt.render(session=session)
output = parse_structured_output(response_text, rendered)
```

When `OutputT` is a list, the parser accepts either:

- A JSON array, or
- An object wrapper of the form `{"items": [...]}` (ARRAY_WRAPPER_KEY is
  `"items"`)

## Dynamic Scoping with enabled()

Sections can be turned on/off at render time using `enabled`. This is one of
WINK's most powerful context tools: you can build a large prompt template, then
render only what's relevant.

Supported signatures:

- `() -> bool`
- `(*, session) -> bool`
- `(params) -> bool`
- `(params, *, session) -> bool`

**Example**: include "deep debugging instructions" only when a session flag is
set:

```python nocheck
from dataclasses import dataclass
from weakincentives.prompt import MarkdownSection
from weakincentives.runtime import Session

@dataclass(slots=True, frozen=True)
class DebugFlag:
    enabled: bool

def debug_enabled(flag: DebugFlag, *, session: Session) -> bool:
    del session
    return flag.enabled

debug_section = MarkdownSection(
    title="Debug",
    key="debug",
    template="If something fails, include stack traces and hypotheses.",
    enabled=debug_enabled,
)
```

Disabled sections don't just hide their text—their tools also disappear from the
prompt. This lets you build a comprehensive template and enable only the
capabilities relevant to the current context.

## Session-Bound Sections and Cloning

Some sections are **pure**: they depend only on params and render the same text
every time. You can safely store those in a module-level `PromptTemplate`.

Other sections are **session-bound**: they capture runtime resources (a session,
workspace connection, etc.). Examples:

- `WorkspaceSection(session=..., mounts=...)`
- `WorkspaceDigestSection(session=...)`

For those, prefer one of these patterns:

**Pattern A: Build the template per session**

```python nocheck
from typing import Any
from weakincentives.prompt import WorkspaceSection, HostMount
from weakincentives.contrib.tools import WorkspaceDigestSection
from weakincentives.prompt import PromptTemplate, MarkdownSection
from weakincentives.runtime import Session


def build_prompt_template(*, session: Session) -> PromptTemplate[Any]:
    mounts = [HostMount(host_path="/path/to/project")]
    return PromptTemplate(
        ns="example",
        key="session-bound",
        sections=(
            MarkdownSection(title="Instructions", key="instructions"),
            WorkspaceDigestSection(session=session),
            WorkspaceSection(session=session, mounts=mounts),
        ),
    )
```

**Pattern B: Clone session-bound sections**

Sections support `clone(**overrides)` to create a new instance with updated
fields. This lets you reuse "static" pieces but still pass a fresh session each
run.

This sounds minor, but it prevents a common bug: accidentally sharing a tool
section (and its internal state) across multiple sessions. Each session should
get its own tool sections.

## Prompt Cleanup

After evaluation completes, call `prompt.cleanup()` to release resources held
by sections (temporary directories, file handles, etc.). `AgentLoop` does this
automatically after debug bundle artifacts have been captured.

Sections that manage external resources should override
`Section.cleanup()`:

```python nocheck
class MyWorkspaceSection(Section):
    def cleanup(self) -> None:
        # Release temporary workspace directory
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
```

If you're using `AgentLoop`, you don't need to call `cleanup()` yourself. For
manual evaluation (calling `adapter.evaluate()` directly), call it after
you're done with the prompt:

```python nocheck
try:
    response = adapter.evaluate(prompt, session=session)
finally:
    prompt.cleanup()
```

## Few-Shot Traces with TaskExamplesSection

WINK supports few-shot examples as first-class sections via
`TaskExamplesSection`.

**Why this matters**: examples are often more effective than "more
instructions", and keeping them as typed objects makes them easier to maintain
and override.

A `TaskExample` can include:

- Input params (dataclasses)
- Expected output (structured)
- Optional tool call traces

This is especially useful for tools: you can show correct tool usage once, and
many models generalize better from examples than from abstract instructions.

See `weakincentives.prompt.task_examples` for details.

## Next Steps

- [Tools](tools.md): Learn how tool contracts and handlers work
- [Sessions](sessions.md): Add state management with reducers
- [Progressive Disclosure](progressive-disclosure.md): Control context size
- [Prompt Overrides](prompt-overrides.md): Iterate without code changes

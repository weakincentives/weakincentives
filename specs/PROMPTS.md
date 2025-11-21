# Prompt Class Specification

## Introduction

The `Prompt` abstraction centralizes every string template that flows to an LLM so the codebase has a single,
inspectable source for system prompts and per-turn instructions. Each prompt is a markdown document assembled from
typed `Section` objects whose templates rely on Python's `string.Template` engine. By forcing prompt authors to
declare explicit data carriers we reduce the chance of mismatched placeholders, make it easy to audit prompt usage,
and give downstream callers a predictable, debuggable API.

## Goals

Keep authoring friction low while providing strong validation. Make prompt structure obvious in logs and docs without
forcing developers to learn complex templating rules. Encourage reuse by composing hierarchical sections, yet ensure
that every placeholder a template references is captured in a dataclass so rendering bugs surface before a request
reaches an LLM. The design should be simple enough to maintain but strict enough to prevent silent failures.

## Guiding Principles

- **Type-Safety First**: Every placeholder maps to a dataclass field so template issues surface early in development
  rather than during an LLM call.
- **Strict, Predictable Failures**: Validation and render errors fail loudly with actionable context instead of
  silently dropping sections or placeholders.
- **Composable Markdown Structure**: Hierarchical sections with deterministic heading levels keep long prompts
  readable and easy to audit.
- **Minimal Templating Surface**: Limiting features to `Template.substitute` plus boolean selectors prevents
  complex control flow while still allowing dynamic content.
- **Declarative over Imperative**: Prompts describe structured content (sections + params) instead of embedding logic,
  which keeps diffs clear and tooling feasible.

## Design Overview

A `Prompt` owns a **namespace** (`ns`), a required machine-readable `key`, an optional human-readable `name`, and
an ordered tree of `Section` instances. Rendering walks this tree depth first and produces markdown where the
heading level is `##` for roots and adds one `#` per level of depth (so depth one becomes `###`, depth two `####`,
and so on). Each heading is also prefixed with a hierarchical number derived from its position in the traversal:
top-level sections start at `1`, their first child becomes `1.1`, its child becomes `1.1.1`, and so on.
Chapters participate in this same numbering scheme so the rendered document exposes a stable breadcrumb for every
visible fragment. The namespace
groups prompts by logical domain (for example `webapp/agents`, `backoffice/cron`, `infra/sweeper`) and participates
in versioning plus override resolution to prevent collisions across applications. The implementation keeps
`Section` as the abstract base class that defines the shared contract—metadata, parameter typing, optional defaults,
`children`, and two core methods: `is_enabled(params)` to determine visibility and `render(params, depth)` to emit
the markdown fragment (including the heading). Future variants can plug in alternative templating engines or emit
structured output (markdown tables, CSV, JSON) without rewriting prompt logic, so long as they honor the heading
pipeline. The default concrete subclass, `MarkdownSection`, relies on `Template.substitute` to render its `body` string
(missing placeholders raise immediately), applies `textwrap.dedent` and stripping before substitution, and emits
normalized markdown. Concrete sections are instantiated by specializing the generic `Section[ParamsT]` base class
(for example `MarkdownSection[GuidanceParams](...)`). This pins the dataclass type to the section before any instance is
created, and the base class rejects attempts to construct an unspecialized section or provide multiple type arguments.
Each specialized section exposes the `params_type` metadata, accepts an optional `default_params` instance that
pre-populates values, stores the raw `body` string interpreted by the concrete section class, wires optional child
sections through the `children` collection, optionally contributes prompt tools through the `tools` sequence, and
supports an optional boolean `enabled` callable. The callable receives the effective dataclass instance (either the
override passed to `render` or the fallback defaults) and lets authors skip entire subtrees dynamically while still
staying inside the strict `Template` feature set. Sections also declare an `accepts_overrides: bool` flag that defaults
to `True`. When false, the section is excluded from prompt descriptors and the override system ignores any supplied
replacement bodies. Built-in sections provided by the framework (including the generated response format section)
default this flag to `False` so automatic optimization infrastructure leaves them untouched, but their constructors
expose an `accepts_overrides` argument so callers can opt in when a specific deployment is ready for tuning. When a built-in section opts in, the tools it contributes inherit the same `accepts_overrides` value so the entire suite toggles together.

## Construction Rules

When a `Prompt` is instantiated it registers every section by the type of its parameter dataclass, storing the default
instance if provided. Parameter types can repeat across sections—callers supply overrides by type and the prompt fans
that instance out to every matching section. Each section may still define its own default instance; when no override is
provided, the prompt uses the section-specific default, falling back to the first default declared for that type, and
finally the dataclass' zero-argument constructor. The constructor also parses each section's template, extracts
every placeholder token, and verifies that each token corresponds to an attribute on the declared dataclass. Extra
dataclass attributes are acceptable, but missing placeholders trigger `PromptValidationError` with enough context
(section title, placeholder name) for developers to resolve the issue quickly. Default instances are optional; when
absent we rely on the dataclass' own default field values by instantiating it with no arguments during rendering.

## Cloning and reuse

Sections and chapters MUST expose a `clone(**kwargs)` method that returns a deep copy of the
component tree suitable for insertion into a new prompt. Clones behave as if they were
constructed from scratch: numbering restarts from the destination prompt, default parameter
dataclasses are duplicated (not reused), and the cloned objects share no references to the
original prompt, `Session`, or `EventBus` instances. Implementations must recursively clone
children so the entire subtree remains decoupled.

Tool-backed sections MAY accept runtime objects (for example `session` or `bus`) as keyword
arguments to `clone`. When supplied, the clone MUST wire reducers and tool handlers against the
provided instances instead of the originals. This is critical for reusable sections such as the
VFS and Podman tool suites, which need to register their reducers on the target session and bind
tool telemetry to the new event bus when transplanted into another prompt.

### Prompt namespace (`ns`) — REQUIRED

Prompts MUST declare a non-empty `ns: str`. The `(ns, key)` pair identifies a prompt
family during versioning/overrides and avoids collisions across complex apps.

### Prompt key (`key`) — REQUIRED

Prompts MUST declare a non-empty `key: str`. Keys scope prompt instances within a namespace and participate in
hashing, override lookup, and tool descriptor construction.

### Section key (`key`) — REQUIRED

All `Section` instances MUST declare a non-empty `key: str`. Keys MUST match:

```
^[a-z0-9][a-z0-9._-]{0,63}$
```

Rationale: stable machine identifiers that are safe for file names, JSON keys, and CLIs.
The `key` becomes part of the **SectionPath** used by the versioning system.
Relying on title slugs is no longer permitted.

### Built-in Response Format section key

When structured output is enabled, the framework appends a built-in
`Response Format` section. Its **key is fixed** to `response-format` so
SectionPaths are deterministic.

## Rendering Flow

`Prompt.render` accepts dataclass instances as positional arguments. Ordering is irrelevant because rendering matches
instances by their concrete dataclass type. For each section we compute the effective params by starting with its
configured default when present, otherwise instantiating the dataclass with no arguments (trusting field defaults and
factories). If instantiating the dataclass fails because required fields lack values and no override was supplied we
surface `PromptRenderError`. Once parameters are in place we call
`section.is_enabled(params)`; disabled sections short-circuit the traversal, meaning their children do not render and
their defaults are ignored. Active sections invoke `section.render(params, depth)`, which always emits a markdown
heading at the appropriate depth followed by the body content. Heading titles must include the numbering prefix
generated from the section's location in the tree (`1`, `1.2`, `1.2.1`, etc.) so rendered prompts expose the same
structure optimization tooling inspects. Text bodies are dedented, stripped, and separated by a
single blank line so the final document is readable and deterministic. Enabled sections contribute any registered
`Tool` instances to the rendered prompt; tools from disabled sections never appear in the result.

Rendering returns a `RenderedPrompt` dataclass. Besides the markdown string (`.text`) and structured output metadata,
the object surfaces:

- `.tools` – ordered tuple of tools contributed by enabled sections.
- `.tool_param_descriptions` – optional mapping of tool name → field description overrides supplied by the
  override system.
- Prompt descriptors MUST capture the numbering string assigned to each section (and chapter) so callers can map
  section keys or `SectionPath`s back to the human-facing headings rendered in `.text`.

## Validation and Error Handling

All validation errors throw `PromptValidationError`, while issues discovered during rendering—missing dataclass
instances, failed selector callables, `Template` substitution failures—raise `PromptRenderError`. Both exceptions
should carry structured data describing the section path, the offending placeholder, and the dataclass type so calling
code can log or surface actionable diagnostics. The library intentionally exposes no configuration switches for
silently dropping sections or coercing mismatched data; strict failure modes keep bugs visible and avoid confusing LLM
transcripts.

## Non-Goals

We deliberately exclude templating features that go beyond `Template.substitute`: no conditionals, loops, or
arbitrary expression evaluation. Prompt composition also stops at sections; we do not embed one prompt inside another,
favoring explicit `children` for reuse. Telemetry, logging sinks, and additional metadata such as channel tags or
custom heading levels remain out of scope until real-world usage demonstrates a need. The only validation we perform
concerns placeholder presence and dataclass coverage; naming conventions are unchecked by design.

## Usage Sketch

```python
from dataclasses import dataclass
from weakincentives.prompt import Prompt, MarkdownSection

@dataclass
class MessageRoutingParams:
    recipient: str
    subject: str | None = None

@dataclass
class ToneParams:
    tone: str = "friendly"

@dataclass
class ContentParams:
    summary: str


@dataclass
class InstructionParams:
    pass

tone_section = MarkdownSection[ToneParams](
    title="Tone",
    template="""
    Target tone: ${tone}
    """,
    key="tone",
)

content_section = MarkdownSection[ContentParams](
    title="Content Guidance",
    template="""
    Include the following summary:
    ${summary}
    """,
    key="content-guidance",
    enabled=lambda params: bool(params.summary.strip()),
)

compose_email = Prompt(
    ns="demo",
    key="compose-email",
    name="compose_email",
    sections=[
        MarkdownSection[MessageRoutingParams](
            title="Message Routing",
            template="""
            To: ${recipient}
            Subject: ${subject}
            """,
            key="routing",
            default_params=MessageRoutingParams(subject="(optional subject)"),
        ),
        MarkdownSection[InstructionParams](
            title="Instruction",
            template="""
            Please craft the email below.
            """,
            key="instruction",
            children=[tone_section, content_section],
        ),
    ],
)

rendered = compose_email.render(
    MessageRoutingParams(recipient="Jordan", subject="Q2 sync"),
    ToneParams(tone="warm"),
    ContentParams(summary="Top takeaways from yesterday's meeting."),
)
# Example SectionPaths now:
# ("routing",), ("instruction",), ("instruction","tone"),
# ("instruction","content-guidance"), ("response-format",)
```

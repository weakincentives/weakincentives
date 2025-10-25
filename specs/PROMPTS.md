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
- **Minimal Templating Surface**: Limiting features to `Template.safe_substitute` plus boolean selectors prevents
  complex control flow while still allowing dynamic content.
- **Declarative over Imperative**: Prompts describe structured content (sections + params) instead of embedding logic,
  which keeps diffs clear and tooling feasible.

## Design Overview
A `Prompt` owns a name and an ordered tree of `Section` instances. Rendering walks this tree depth first and produces
markdown where the heading level is `##` for roots and adds one `#` per level of depth (so depth one becomes `###`,
depth two `####`, and so on). The implementation keeps `Section` as the abstract base class that defines the shared
contract—metadata, parameter typing, optional defaults, `children`, and two core methods: `is_enabled(params)` to
determine visibility and `render(params, depth)` to emit the markdown fragment (including the heading). Future
variants can plug in alternative templating engines or emit structured output (markdown tables, CSV, JSON) without
rewriting prompt logic, so long as they honor the heading pipeline. The default concrete subclass, `TextSection`, relies
on `Template.safe_substitute` to render its `body` string, applies `textwrap.dedent` and stripping before substitution,
and emits normalized markdown. Each section declares a `params` dataclass type that lists the variables it requires, an
optional `defaults` instance that pre-populates values, a raw `body` string interpreted by the concrete section class,
optional child sections exposed through the `children` collection, and an optional boolean `enabled` callable. The
callable receives the effective dataclass instance (defaults merged with any override supplied at render time) and lets
authors skip entire subtrees dynamically while still staying inside the strict `Template` feature set.

## Construction Rules
When a `Prompt` is instantiated it registers every section by the type of its parameter dataclass, storing the default
instance if provided. Types must be unique across the entire tree so type based lookup remains deterministic; a
duplicate immediately raises `PromptValidationError`. The constructor also parses each section's template, extracts
every placeholder token, and verifies that each token corresponds to an attribute on the declared dataclass. Extra
dataclass attributes are acceptable, but missing placeholders trigger `PromptValidationError` with enough context
(section title, placeholder name) for developers to resolve the issue quickly. Default instances are optional; when
absent we rely on the dataclass' own default field values by instantiating it with no arguments during rendering.

## Rendering Flow
`Prompt.render` accepts an iterable of dataclass instances. Ordering is irrelevant because rendering matches instances
by their concrete dataclass type. For each section we compute the effective params by starting with its configured
default when present, otherwise instantiating the dataclass with no arguments (trusting field defaults and factories),
and then replacing that baseline with any provided override. If instantiating the dataclass fails because required
fields lack values and no override was supplied we surface `PromptRenderError`. Once parameters are in place we call
`section.is_enabled(params)`; disabled sections short-circuit the traversal, meaning their children do not render and
their defaults are ignored. Active sections invoke `section.render(params, depth)`, which always emits a markdown
heading at the appropriate depth followed by the body content. Text bodies are dedented, stripped, and separated by a
single blank line so the final document is readable and deterministic.

## Validation and Error Handling
All validation errors throw `PromptValidationError`, while issues discovered during rendering—missing dataclass
instances, failed selector callables, `Template` substitution failures—raise `PromptRenderError`. Both exceptions
should carry structured data describing the section path, the offending placeholder, and the dataclass type so calling
code can log or surface actionable diagnostics. The library intentionally exposes no configuration switches for
silently dropping sections or coercing mismatched data; strict failure modes keep bugs visible and avoid confusing LLM
transcripts.

## Non-Goals
We deliberately exclude templating features that go beyond `Template.safe_substitute`: no conditionals, loops, or
arbitrary expression evaluation. Prompt composition also stops at sections; we do not embed one prompt inside another,
favoring explicit `children` for reuse. Telemetry, logging sinks, and additional metadata such as channel tags or
custom heading levels remain out of scope until real-world usage demonstrates a need. The only validation we perform
concerns placeholder presence and dataclass coverage; naming conventions are unchecked by design.

## Usage Sketch
```python
from dataclasses import dataclass
from weakincentives.prompts import Prompt, TextSection

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

tone_section = TextSection(
    title="Tone",
    body="""
    Target tone: ${tone}
    """,
    params=ToneParams,
)

content_section = TextSection(
    title="Content Guidance",
    body="""
    Include the following summary:
    ${summary}
    """,
    params=ContentParams,
    enabled=lambda params: bool(params.summary.strip()),
)

compose_email = Prompt(
    name="compose_email",
    sections=[
        TextSection(
            title="Message Routing",
            body="""
            To: ${recipient}
            Subject: ${subject}
            """,
            params=MessageRoutingParams,
            defaults=MessageRoutingParams(subject="(optional subject)"),
        ),
        TextSection(
            title="Instruction",
            body="""
            Please craft the email below.
            """,
            children=[tone_section, content_section],
        ),
    ],
)

rendered = compose_email.render(
    [
        MessageRoutingParams(recipient="Jordan", subject="Q2 sync"),
        ToneParams(tone="warm"),
        ContentParams(summary="Top takeaways from yesterday's meeting."),
    ]
)
```

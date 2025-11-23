# Chapters Specification

## Introduction

Chapters extend the prompt system with a coarse-grained switch that governs which
parts of a prompt tree enter the model's context window. Each chapter groups a
set of prompt sections that should either render together or remain completely
hidden from the large language model. Adapters inspect chapters during evaluation
to decide which content becomes visible for a given turn and which content stays
dark. Conceptually a chapter is a specialized `Section` subclass whose value
comes from aggregating other sections; it never renders standalone content or
tools.

## Design Rationale and Scope

Chapters exist to express **prompt-wide visibility decisions** that cannot be
captured by section-level `enabled` predicates alone.

- Use chapters when multiple sections must open or stay hidden as a unit (e.g.,
  PII-handling prompts, experimental personas, or provider-specific rendering
  tracks).
- Chapters **do not** replace fine-grained section gating. Section `enabled`
  predicates continue to run inside an open chapter to skip individual pieces of
  content without reopening the visibility boundary.
- Current implementations keep chapters **flat** and **root-scoped**; nesting is
  discouraged to avoid recursive visibility evaluation.
- Chapter definitions belong alongside prompt definitions—they are not tooling
  overlays or runtime mutations. Overrides continue to operate at the section
  and tool layers.

Chapters should remain a coarse feature toggle rather than a rendering strategy
or telemetry mechanism; those concerns live in adapters and runtime plumbing.

## Goals

- **Explicit visibility boundaries**: Treat a chapter as the authoritative
  toggle for large prompt regions, ensuring sensitive or off-topic material does
  not leak into the context window.
- **Predictable defaults**: Closed chapters are the default so new content
  remains opt-in until adapters explicitly open it.
- **Composable structure**: Chapters wrap existing `Section` trees without
  changing section authoring practices.
- **Deterministic evaluation**: Decisions about open or closed state happen
  inside adapter `evaluate()` before rendering, producing a new `Prompt`
  instance with only the active sections.

## Guiding Principles

- **Stable identifiers**: Keys are normalized up front and must be unique per
  prompt. `Chapter.__post_init__` normalizes keys and validates default params,
  while `Prompt.__init__` rejects duplicate chapter keys before registration.
- **Closed until proven open**: Base prompts only contain their root sections;
  chapter content is added via `Prompt.expand_chapters`, keeping new chapters
  inert until an adapter opts in.
- **Deterministic clones**: `Chapter.clone` deep-copies sections and default
  params so expanded prompts cannot leak runtime state across sessions.
- **Strict parameter handling**: Chapters with parameterized `enabled`
  predicates require matching dataclass instances; mismatches raise
  `PromptValidationError` instead of silently coercing types.
- **Single-shot expansion**: Expanded prompts disable further chapter expansion
  to prevent double-counting or divergent visibility across render calls.

## Chapter Template

Authors define chapters with the following fields. The template is expressed as
Python-style pseudocode to mirror the surrounding prompt abstractions.

```python
@dataclass
class Chapter[ParamsT]:
    key: str                     # machine-safe identifier, stable across releases
    title: str                   # human-readable heading used in documentation
    description: str | None = None  # optional rationale for auditing tools
    sections: tuple[Section[SupportsDataclass], ...] = ()  # ordered section payload
    enabled: Callable[[ParamsT], bool] | None = None  # optional fine-grained gate
```

- `key` MUST be unique within a prompt namespace and follow the same identifier
  rules as sections (`^[a-z0-9][a-z0-9._-]{0,63}$`).
- `title` SHOULD mirror the chapter's user-facing headline.
- `description` captures intent, data sensitivity, or logging hints.
- `sections` include the existing section instances that render when the chapter
  is open.
- `enabled` receives the effective chapter parameters and returns whether the
  chapter may open. When absent the chapter remains eligible to open.
- Chapters remain section derivatives—they expose the same metadata surface but
  do not declare their own tools. Tool declarations stay scoped to the sections
  that the chapter aggregates.

Chapter parameter types follow the same rules as section parameters: they are
structured dataclasses specialized at definition time. A chapter may expose
defaults that adapters can override during adapter evaluation.

## Expansion Policy

Adapters interpret a `ChaptersExpansionPolicy` enum to decide how aggressively
they may reveal chapter content during evaluation.

```python
class ChaptersExpansionPolicy(StrEnum):
    ALL_INCLUDED = "all_included"         # open every chapter before rendering
    INTENT_CLASSIFIER = "intent_classifier"  # use classifier heuristics tied to goal section content
```

- `ALL_INCLUDED` trades privacy for completeness by rendering every chapter.
- `INTENT_CLASSIFIER` runs a deterministic gate (rules or secondary model)
  before rendering to open only chapters aligned with the goal section context.

Implementations MAY extend the enum with provider-specific strategies, but they
MUST document the additional states alongside the adapter.

### Current Behavior

- `Prompt.expand_chapters` implements `ALL_INCLUDED` and raises
  `NotImplementedError` for other policies. Callers must supply additional
  expansion strategies at the adapter layer before relying on
  `INTENT_CLASSIFIER`.
- Chapter parameters are optional but validated: provided parameters must match
  the declared dataclass type and unknown chapter keys raise
  `PromptValidationError` before any section is opened.

## Section Composition

Chapters do not replace section-level `enabled` callables. When a chapter opens,
individual sections still consult their own selectors. This allows a chapter to
provide the coarse-grained boundary while sections inside continue to perform
fine-grained feature gating. Chapters also never own tools directly; only the
sections they contain register tool handlers.

## Cloning

Chapters MUST implement `clone(**kwargs)` so orchestrators can transplant chapter
trees into new prompts. Clones are deep copies: parameter defaults, description
metadata, and every nested section instance are duplicated rather than shared.
Runtime bindings (such as `Session` or `EventBus` objects accepted by tool-backed
sections) MUST be forwarded when supplied so the cloned chapter re-registers any
reducers or tools against the new runtime. Open/closed state never carries over;
the destination prompt or adapter reevaluates visibility after cloning.

## Implementation Notes

- **Enabled predicates**: Zero-argument `enabled` callables are automatically
  wrapped to ignore chapter parameters; predicates that accept parameters expect
  the declared dataclass type and propagate exceptions as validation errors.
- **Default params**: Parameterized chapters may ship `default_params`; these are
  cloned during expansion to keep per-evaluation state isolated.
- **Descriptor inclusion**: `PromptDescriptor.from_prompt` records declared
  chapters (even before expansion) so override tooling and telemetry can reason
  about the intended visibility boundaries.
- **Tool awareness**: Because chapters never own tools, adapter tool discovery
  continues to walk section instances only; opening a chapter simply appends
  more sections to the registry snapshot used by the renderer.

## Evaluation Responsibilities

Chapter gating happens inside `ProviderAdapter.evaluate()`. Implementations MUST
accept keyword arguments that (a) point at the section expressing the user's
goal or intent so gating heuristics can anchor on a consistent signal and (b)
describe how the adapter may expand chapters beyond the default closed state.

```python
class ProviderAdapter(Protocol):
    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *params: object,
        goal_section_key: str,
        chapters_expansion_policy: ChaptersExpansionPolicy = ChaptersExpansionPolicy.ALL_INCLUDED,
        parse_output: bool = True,
        bus: EventBus,
    ) -> PromptResponse[OutputT]:
        ...
```

1. Start from the base prompt definition, where all chapters default to the
   **closed** state.
1. Honor the selected `chapters_expansion_policy` to decide how chapters may
   open:
   - `ALL_INCLUDED`: every chapter opens without additional adjudication and is
     the default when callers omit the argument.
   - `INTENT_CLASSIFIER`: the adapter uses a deterministic classifier (rule-
     based or model-driven) anchored on the `goal_section_key` content to decide
     which chapters to open before rendering.
1. Locate the section identified by `goal_section_key`. The key MUST refer to a
   concrete `Section.key` inside the prompt definition; adapters SHOULD raise a
   configuration error if the key is missing or ambiguous.
1. Inspect each chapter in declaration order and decide whether to open it for
   the current evaluation. Decisions MAY consider
   - the goal/intent section content,
   - the user message, accumulated thread history, or tool state,
   - chapter metadata (`key`, `title`, `description`), and
   - optional chapter-level parameters.
1. For each chapter resolved to the open state, include its sections in the new
   prompt tree. Closed chapters contribute no sections and remain hidden.
1. When a chapter is open, still respect section-level `enabled` callables to
   skip individual sections as needed.
1. Produce a fresh `Prompt` instance whose section tree contains only the open
   chapter content and render that snapshot. This ensures subsequent render calls
   operate on a deterministic decision.

Adapters MUST thread chapter metadata into the prompt descriptor so optimization
tooling can inspect the declared visibility boundaries. Dedicated telemetry or
logging streams for chapter decisions remain out of scope; rely on existing
descriptor metadata and redaction guidance when sharing prompt snapshots.

## Lifecycle Considerations

- **Default safety**: Treat the closed state as the default whenever new
  chapters ship. Adapters must explicitly opt in to opening them.
- **Overrides**: Prompt override tooling interacts with chapters the same way it
  interacts with sections. Overrides for closed chapters remain inert until the
  chapter opens, and adapters do not need bespoke hooks to mutate chapter
  defaults at runtime.
- **Per-evaluation decisions**: Adapters MUST recompute chapter visibility for
  every evaluation instead of caching the outcome for a session so shifts in
  user intent immediately influence chapter state.
- **Nested chapters**: Chapters SHOULD remain flat to avoid nested visibility
  calculations. When nesting is unavoidable, the outer chapter's state gates all
  inner content.
- **Versioning**: Include chapter metadata in prompt descriptors so versioned
  prompts capture the visibility structure.

## Code References and Behaviors

- **Chapter normalization and gating**: `src/weakincentives/prompt/sections/chapter.py`
  normalizes keys, coerces `enabled` callables, clones default params, and
  exposes `is_enabled` for adapter use.
- **Prompt-level expansion**: `src/weakincentives/prompt/prompt.py` enforces
  unique chapter keys, keeps base sections separate from chapter content, and
  expands chapters via `ALL_INCLUDED` while guarding against unknown parameter
  keys and repeated expansion.
- **Descriptor export**: `src/weakincentives/prompt/overrides/versioning.py`
  adds declared chapters to `PromptDescriptor` so overrides and versioned
  snapshots preserve visibility intent even before expansion.

## Descriptor Metadata

`PromptDescriptor` exposes chapter information via a dedicated dataclass so
adapters and tooling can inspect which visibility boundaries were declared.

```python
@dataclass(slots=True)
class ChapterDescriptor:
    key: str
    title: str
    description: str | None
    parent_path: tuple[str, ...]
```

- `key`, `title`, and `description` mirror the chapter declaration.
- `parent_path` records the section path anchoring the chapter. Chapters are
  currently root-scoped so the value is `()`, but nested structures can reuse
  the field.
  `PromptDescriptor.chapters` preserves the declaration order and is populated
  even when a prompt snapshot has already expanded its chapters. This keeps
  override tooling deterministic while surfacing the declared visibility
  structure to adapters and telemetry pipelines.

# Structured Output via `Prompt[OutputT]`

## Introduction

Some prompts must yield machine-parseable responses (e.g., a JSON object or array) rather than free-form text. To keep authoring friction low and avoid new template concepts, structured output is declared by specializing the existing `Prompt` type with an output dataclass. Authors write `Prompt[OutputT]` (or `Prompt[list[ItemT]]`) and the framework handles the rest: injects concise JSON-only return instructions, exposes the declared type on the rendered artifact, and provides a small helper to parse and validate the final assistant message back into typed Python objects.

## Goals

Keep the surface area tiny while delivering strong guarantees:

- Enable structured responses with **one declaration**: `Prompt[OutputT]`.
- Avoid requiring authors to manage new `Section` subclasses or extra config files—the framework appends a builtin
  `ResponseFormatSection` when needed.
- Keep prompts readable: inject a small, deterministic "Response Format" block.
- Validate outputs strictly into typed dataclasses with actionable failures.
- Remain fully backward-compatible when no output type is declared.

## Guiding Principles

- **Type-Safety First**: The output contract is a Python dataclass; parsing produces that type (or a list of it).
- **Strict, Predictable Failures**: Construction and parsing fail loudly with `PromptValidationError` and `OutputParseError` instead of guessing.
- **Minimal Surface**: No new templating features, no schema DSLs, no logging/tracing knobs.
- **Provider-Agnostic Core**: The prompt layer describes the contract; adapters may optionally request JSON mode, but prompts remain portable.

## Design Overview

`Prompt` becomes *optionally generic*. When specialized, `Prompt[OutputT]` declares that the final assistant message must be JSON matching the dataclass `OutputT`. If the specialization is `Prompt[list[ItemT]]`, the final message must be a top-level JSON array of objects matching `ItemT`.

At render time the prompt assembles markdown from its sections (unchanged). When an output type is present and
injection is enabled the prompt appends a builtin `ResponseFormatSection` to the end of the tree. The section renders a
deterministic "Response Format" block with a root-level heading (`## Response Format`) that instructs the model to
return **only** a single fenced JSON block with the required top-level container (object or array) and no extra prose.

`RenderedPrompt` carries the declared output metadata so downstream code can call a small parser that turns the model's final message into an instance of `OutputT` (or a `list[ItemT]`) with strict validation and conservative type coercions.

## Response Format Section

The framework exposes an internal `ResponseFormatSection`, a `TextSection[ResponseFormatParams]`, that renders the JSON
return instructions. Authors never instantiate it directly; the prompt inserts the section automatically whenever
structured output is requested and `inject_output_instructions=True`.

```python
@dataclass(slots=True)
class ResponseFormatParams:
    article: Literal["a", "an"]
    container: Literal["object", "array"]
    extra_clause: str  # either ". Do not add extra keys." or "."
```

The dataclass lives inside the prompts package and is typed with `typing.Literal` so downstream tooling can rely on the
exact strings emitted.

The section title is always `"Response Format"`, ensuring a root-level `## Response Format` heading. Its body template
remains fixed:

```
Return ONLY a single fenced JSON code block. Do not include any text
before or after the block.

The top-level JSON value MUST be ${article} ${container} that matches the fields
of the expected schema${extra_clause}
```

During construction the prompt resolves `container` (`"object"` for scalar outputs, `"array"` for `list[...]`), derives
`article` from the container, and sets `extra_clause` to `. Do not add extra keys.` unless `allow_extra_keys=True`.
Those params feed the section at render time so the output instructions stay consistent with the declared contract.

## Construction Rules

- `Prompt` may be specialized as `Prompt[T]` or `Prompt[list[T]]` where `T` is a **dataclass**.

- Any other specialization raises `PromptValidationError` at construction with `dataclass_type=T`.

- Two rendering options are stored on the instance:

  - `inject_output_instructions: bool = True` -- append the "Response Format" block automatically.
  - `allow_extra_keys: bool = False` -- unknown JSON keys are rejected by default (strict mode).

- If `Prompt` is **not** specialized, behavior is unchanged; no output contract is implied.

**Normalization of the specialization**

- `T` -> **object** container, output type = `T`.
- `list[T]` -> **array** container, item type = `T`.
- Nested lists or unions are unsupported in v1.

## Rendering Flow

Rendering of sections is identical to the existing `Prompt` flow: depth-first traversal, deterministic headings, `TextSection` substitution, and skipping disabled subtrees. After the standard render:

1. Join rendered sections with a single blank line between fragments.

1. If the prompt is specialized and `inject_output_instructions=True`, append a `ResponseFormatSection` populated with the
   derived params above. Rendering the section produces the canonical instructions template with `${container}` and
   `${extra_clause}` resolved for the current prompt.

1. Return a `RenderedPrompt` that also exposes:

   - `output_type: type[Any] | None`
   - `output_container: Literal["object","array"] | None`
   - `allow_extra_keys: bool | None`

This keeps the final prompt readable and the output contract easy to discover programmatically.

## Validation and Error Handling

- **Construction**

  - Non-dataclass output types (or non-list specializations) raise `PromptValidationError` with the offending `dataclass_type`.

- **Rendering**

  - Unchanged for sections; existing `PromptRenderError` behavior applies.

- **Parsing**

  - `parse_output(text, rendered)` raises `OutputParseError` when:

    - the prompt was not specialized,
    - the top-level JSON container is wrong,
    - required dataclass fields are missing,
    - unknown keys are present while `allow_extra_keys=False`,
    - values cannot be safely coerced to field types.

### Parsing Rules (minimal, strict)

- **Extraction cascade**:

  1. Prefer a single fenced `json ... ` block.
  1. Else parse the entire message as JSON.
  1. Else scan for the first plausible top-level `{...}` or `[...]` and parse.

- **Containers**:

  - `object` -> payload must be a JSON object.
  - `array` -> payload must be a JSON array of objects.

- **Dataclass validation**:

  - Required fields = dataclass fields without defaults.
  - Unknown keys are rejected unless `allow_extra_keys=True` (ignored if allowed).
  - Conservative coercions only: `"123"`->`int`, `"3.14"`->`float`, case-insensitive `"true"/"false"`->`bool`, `"null"/"none"`->`None`. Nested dataclasses and lists recurse.
  - No lossy or complex conversions.

## Non-Goals

- No `OutputSpec` type, no schema blobs in the prompt, no KV/text output mode in v1.
- No additional `Section` types or templating features for authors to manage; the builtin `ResponseFormatSection` is injected automatically when required.
- No streaming decode, repair loops, or retry orchestration.
- No logging/tracing/telemetry concerns.

## Usage Sketch

```python
from dataclasses import dataclass
from weakincentives.prompts import Prompt, TextSection
from weakincentives.prompts.structured import parse_output  # small helper

# 1) Declare the output type as a dataclass
@dataclass
class Summary:
    title: str
    gist: str
    url: str | None = None

# 2) Standard input params for your sections
@dataclass
class Guidance:
    topic: str

# 3) Build a prompt specialized with the output type
summarize = Prompt[Summary](
    name="summarize_entity",
    sections=[
        TextSection[Guidance](
            title="Task",
            body="""
            Write a brief summary of ${topic}. Include a canonical URL if known.
            """,
        ),
    ],
    # Defaults shown explicitly; both are optional
    inject_output_instructions=True,
    allow_extra_keys=False,
)

# 4) Render as usual
rendered = summarize.render(Guidance(topic="Ada Lovelace"))
# rendered.text now ends with a "## Response Format" JSON-only block
# rendered.output_type is Summary, rendered.output_container is "object"

# 5) After the model replies, parse into the typed dataclass
final_text = "... assistant message here ..."
typed_summary: Summary = parse_output(final_text, rendered)

# ---- Array output example ----

@dataclass
class SearchResult:
    title: str
    url: str
    score: float

search_prompt = Prompt[list[SearchResult]](
    name="search",
    sections=[
        TextSection[Guidance](
            title="Task",
            body="Return the top results for ${topic} with relevance scores.",
        )
    ],
)

search_rendered = search_prompt.render(Guidance(topic="discrete math"))
# ...get assistant message...
results: list[SearchResult] = parse_output("...model reply...", search_rendered)
```

______________________________________________________________________

**Summary**
Declaring `Prompt[OutputT]` (or `Prompt[list[OutputT]]`) is all that's needed to require JSON-structured answers. The framework appends concise return instructions, exposes the output contract on `RenderedPrompt`, and supplies a single `parse_output` helper to validate and materialize typed results. Everything else in the prompt system--sections, tooling, error semantics--stays the same.

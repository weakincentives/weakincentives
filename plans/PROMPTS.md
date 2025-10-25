# PROMPTS TDD Plan

Small, test-first iterations to deliver the prompt abstraction described in `specs/PROMPTS.md`.

## Iteration 1 – Exception Types and Module Surface
- Tests: `tests/prompts/test_exceptions.py` ensures `PromptValidationError` and `PromptRenderError` inherit `Exception`, capture `message`, and store structured context (`section_path`, `dataclass`, `placeholder`).
- Code: introduce `src/weakincentives/prompts/__init__.py` exporting the two exception classes with the expected attributes.

## Iteration 2 – Section Base Contract
- Tests: `tests/prompts/test_section_base.py` defines a lightweight concrete subclass for the test and asserts default `is_enabled` returns `True`, `children` defaults to empty tuple, and parameter dataclass metadata is preserved.
- Code: implement abstract `Section` base class with `title`, `params`, optional `defaults`, `children`, and `enabled` callable, plus abstract `render` method and default `is_enabled` logic.

## Iteration 3 – TextSection Rendering Basics
- Tests: `tests/prompts/test_text_section.py` uses `TextSection` with a simple dataclass to verify `render(params, depth=0)` emits `## {title}` followed by dedented, stripped body. Include a failing test for `${placeholder}` substitution using `safe_substitute`.
- Code: implement `TextSection` rendering via `textwrap.dedent` + `Template.safe_substitute`, ensure depth-based heading prefixing, and enforce string body normalization.

## Iteration 4 – Prompt Construction Skeleton
- Tests: `tests/prompts/test_prompt_init.py` creates a two-section tree and asserts instantiating `Prompt` collects sections depth-first and registers parameter types. Expect `PromptValidationError` when two sections share the same params dataclass.
- Code: implement `Prompt` constructor that flattens the tree, enforces unique dataclass types, stores defaults, and prepares placeholder metadata storage.

## Iteration 5 – Placeholder Validation
- Tests: extend `test_prompt_init` to include a `TextSection` whose template references an unknown placeholder and assert `PromptValidationError` with the offending placeholder name.
- Code: parse each section template, introspect `Template` placeholders, compare with dataclass fields, and raise the structured validation error on mismatch.

## Iteration 6 – Rendering Parameter Resolution
- Tests: `tests/prompts/test_prompt_render.py` verifies `Prompt.render([...])` merges defaults with overrides, instantiates dataclasses when no override supplied, and raises `PromptRenderError` when required values are missing. Assert ordering of supplied instances is irrelevant.
- Code: implement render flow: build a lookup by dataclass type, merge defaults with overrides, call `is_enabled`, and collect rendered markdown with appropriate blank lines between sections.

## Iteration 7 – Conditional Sections and Children Depth
- Tests: extend render tests to include sections with `enabled` predicates and nested children, ensuring disabled parents short-circuit children and headings increase per depth (`##`, `###`, `####`).
- Code: ensure traversal respects `enabled` flag, stops descending when False, and calculates depth-based heading levels correctly.

## Iteration 8 – Robust Error Reporting
- Tests: simulate `safe_substitute` failures and selector exceptions to assert `PromptRenderError` wraps the original exception and surfaces context (section path, dataclass type).
- Code: centralize error handling around render loop, enriching raised exceptions with context payloads for diagnostics.

## Iteration 9 – Integration & Public API
- Tests: author high-level scenario mirroring the spec's usage sketch to confirm end-to-end rendering and exception safety, plus ensure package exports `Prompt`, `Section`, and `TextSection`.
- Code: finalize module exports, add `__all__`, and document `py.typed` inclusion if new package files require updates.

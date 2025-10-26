# TOOLS TDD Plan

Incremental test-first iterations for delivering the tool registration features defined in `specs/TOOLS.md`.

## Completed Iterations
- **Iteration 1 – Tool Dataclass Validation**: Implemented the generic `Tool[ParamsT, ResultT]` with name/description validation and dataclass-bound generics; accompanying tests exercise successful construction and validation failures.
- **Iteration 2 – Handler Signature Enforcement**: Added synchronous handler validation (single positional param annotated as `ParamsT`, returns `ToolResult[ResultT]`) and the corresponding test coverage.
- **Iteration 3 – ToolResult Contract & Handler Returns**: Introduced `ToolResult`, enforced handler return annotations, and ensured bad generic arguments are rejected with tests.

## Remaining Iterations

### Iteration 4 – ToolsSection Rendering Basics
- Tests: new `tests/prompts/test_tools_section.py` verifies `ToolsSection` inherits `Section`, renders only a heading plus optional description template via `safe_substitute`, exposes its tools in order, and raises `PromptRenderError` when placeholders are missing.
- Code: implement `ToolsSection` in `tool.py` – store a tuple of `Tool` instances, dedent/strip the optional description template in `render`, override `placeholder_names()` to expose template placeholders, and return tools via an override.

### Iteration 5 – Section Tool Surface
- Tests: extend `test_tools_section.py` to assert the base `Section.tools()` returns an empty tuple while `ToolsSection` returns its declared tools unchanged.
- Code: add a `tools()` method to `Section` defaulting to `()`, and ensure `ToolsSection` overrides it to expose its members without mutating them.

### Iteration 6 – Prompt Tool Aggregation
- Tests: `tests/prompts/test_prompt_tools.py` builds prompts with nested sections to assert `Prompt.tools(*params)` returns tools in depth-first order, respects defaults/overrides, and excludes tools from sections whose `is_enabled` resolves False.
- Code: factor shared parameter resolution into a private helper reused by `render` and the new `Prompt.tools()` method, iterate registered sections mirroring render traversal, honoring enablement before collecting tools.

### Iteration 7 – Duplicate Contract Validation
- Tests: extend `test_prompt_tools.py` to confirm duplicate tool names or shared `params` dataclasses across `ToolsSection`s trigger `PromptValidationError` with the offending section path and dataclass.
- Code: during prompt registration, track seen tool names and dataclass types, raising structured `PromptValidationError` when collisions occur.

### Iteration 8 – Public API & Integration Scenario
- Tests: `tests/prompts/test_prompt_tools_integration.py` mirrors the spec example, asserting markdown output stays minimal while `Prompt.tools()` exposes the declared tool and handler, and checks the module exports `Tool`, `ToolResult`, and `ToolsSection` via `weakincentives.prompts`.
- Code: wire the new classes into `src/weakincentives/prompts/__init__.py`, ensure packaging metadata (`py.typed`) stays accurate, and address any small refactors uncovered by the integration test.

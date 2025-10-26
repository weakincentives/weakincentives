# TOOLS TDD Plan

Incremental test-first iterations for delivering the tool registration features defined in `specs/TOOLS.md`.

## Iteration 1 – Tool Dataclass Validation
- Tests: `tests/prompts/test_tool_dataclass.py` exercises successful construction, ASCII name pattern enforcement, description trimming/length limits (1–200 chars), and rejects non-dataclass `params`, raising `PromptValidationError` with context.
- Code: add `Tool` dataclass in `src/weakincentives/prompts/tool.py` with `__post_init__` validations for name, description, and params dataclass type, normalizing stored values.

## Iteration 2 – Handler Signature Enforcement
- Tests: extend `test_tool_dataclass.py` to cover sync and async handlers accepting exactly one argument of the declared params type, and to assert signature/annotation mismatches raise `PromptValidationError` identifying the offending tool.
- Code: enhance `Tool.__post_init__` to inspect handler call signatures, allowing optional handlers, and attach failure metadata via `PromptValidationError`.

## Iteration 3 – ToolsSection Rendering Basics
- Tests: new `tests/prompts/test_tools_section.py` verifies `ToolsSection` inherits `Section`, renders only a heading plus optional description template via `safe_substitute`, exposes its tools in order, and raises `PromptRenderError` when placeholders are missing.
- Code: implement `ToolsSection` in `tool.py` – store a tuple of `Tool` instances, dedent/strip the optional description template in `render`, override `placeholder_names()` to expose template placeholders, and return tools via an override.

## Iteration 4 – Section Tool Surface
- Tests: extend `test_tools_section.py` to assert the base `Section.tools()` returns an empty tuple while `ToolsSection` returns its declared tools unchanged.
- Code: add a `tools()` method to `Section` defaulting to `()`, and ensure `ToolsSection` overrides it to expose its members without mutating them.

## Iteration 5 – Prompt Tool Aggregation
- Tests: `tests/prompts/test_prompt_tools.py` builds prompts with nested sections to assert `Prompt.tools(*params)` returns tools in depth-first order, respects defaults/overrides, and excludes tools from sections whose `is_enabled` resolves False.
- Code: factor shared parameter resolution into a private helper reused by `render` and the new `Prompt.tools()` method, iterate registered sections mirroring render traversal, honoring enablement before collecting tools.

## Iteration 6 – Duplicate Contract Validation
- Tests: extend `test_prompt_tools.py` to confirm duplicate tool names or shared `params` dataclasses across `ToolsSection`s trigger `PromptValidationError` with the offending section path and dataclass.
- Code: during prompt registration, track seen tool names and dataclass types, raising structured `PromptValidationError` when collisions occur.

## Iteration 7 – Public API & Integration Scenario
- Tests: `tests/prompts/test_prompt_tools_integration.py` mirrors the spec example, asserting markdown output stays minimal while `Prompt.tools()` exposes the declared tool and handler, and checks the module exports `Tool` and `ToolsSection` via `weakincentives.prompts`.
- Code: wire the new classes into `src/weakincentives/prompts/__init__.py`, ensure packaging metadata (`py.typed`) stays accurate, and address any small refactors uncovered by the integration test.

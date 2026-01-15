# Task Examples Specification

Multi-step trajectory examples for few-shot learning.

**Source:** `src/weakincentives/prompt/examples.py`

## Core Types

### TaskStep

```python
TaskStep[ParamsT, ResultT](
    tool_name: str,
    example: ToolExample[ParamsT, ResultT],
)
```

### TaskExample

Section capturing complete trajectory:

```python
TaskExample(
    key="security-review",
    objective="Review auth module for vulnerabilities",
    outcome="Identified SQL injection issue",
    steps=[TaskStep(tool_name="read_file", example=...)],
)
```

### TaskExamplesSection

Container for `TaskExample` children:

```python
TaskExamplesSection(
    key="examples",
    title="Workflow Examples",
    examples=[review_example, perf_example],
)
```

## Validation

| Check | Error |
|-------|-------|
| Empty objective | `"objective must not be empty"` |
| Empty steps | `"steps must not be empty"` |
| Unknown tool | `"Unknown tool \"X\" in task example"` |
| Type mismatch | `"input type mismatch for tool \"X\""` |

## Rendering

```markdown
## Task Examples

### 1. Review authentication module

**Objective:** Review auth module for security issues.

**Steps:**

1. **read_file** - Read the auth source
   - input: `{"path": "src/auth.py"}`
   - output: `{"content": "..."}`

**Outcome:** Identified SQL injection vulnerability.
```

## Progressive Disclosure

```python
TaskExample(
    ...,
    visibility=SectionVisibility.SUMMARY,
    summary="Security review workflow example.",
)
```

## Limitations

- Linear steps only (no branching)
- No intermediate state capture
- Static tool references
- Step outputs not usable as later inputs

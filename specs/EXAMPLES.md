# Examples Specification

## Purpose

`TaskExamplesSection` provides trajectory-based examples demonstrating multi-step
agent behavior. Unlike single-tool `ToolExample`, `TaskExample` captures complete
sequences: objective, ordered tool calls, expected outcome.

**Implementation:** `src/weakincentives/prompt/task_examples.py`

## Core Types

### TaskStep

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | Tool being invoked |
| `example` | `ToolExample` | Invocation details |

### TaskExample (Section)

| Property | Type | Description |
|----------|------|-------------|
| `objective` | `str` | Task goal (1-500 ASCII chars) |
| `outcome` | `str \| dataclass` | Expected result |
| `steps` | `tuple[TaskStep, ...]` | Ordered tool invocations |

### TaskExamplesSection

Container section for `TaskExample` children.

## Validation Rules

- Objective: 1-500 printable ASCII
- Outcome: `str` (no structured output) or instance of output type
- Steps: Non-empty, sequential
- Tool names: Must exist in prompt's tool registry
- Types: Example input/output must match tool's parameter/result types

## Rendering

```markdown
## Task Examples

### 1. Review authentication module

**Objective:** Review src/auth.py for security issues.

**Steps:**

1. **read_file** - Read the target file
   - input: {"path": "src/auth.py"}
   - output: {"content": "def authenticate(): ..."}

**Outcome:** Identified 2 vulnerabilities.
```

## Section Hierarchy

`TaskExample` instances register as children of `TaskExamplesSection`:
- Per-example visibility/enabled predicates
- Section paths: `"task-examples.example-auth-review"`
- Cloning support

## Integration

```python
template = PromptTemplate(
    ns="agents/reviewer",
    key="code-review",
    sections=[
        MarkdownSection(tools=[read_tool, search_tool], ...),
        TaskExamplesSection(
            examples=[
                TaskExample(
                    key="security-review",
                    objective="Review src/auth.py for security issues",
                    outcome="Identified vulnerabilities",
                    steps=[TaskStep(tool_name="read_file", example=...)],
                ),
            ],
        ),
    ],
)
```

## Limitations

- No branching (linear sequence only)
- No intermediate state capture
- Static tool references (resolved at construction)
- No step dependencies

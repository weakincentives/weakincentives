# Examples Specification

## Purpose

`TaskExamplesSection` provides trajectory-based examples demonstrating multi-step
agent behavior. Unlike single-tool `ToolExample`, `TaskExample` captures complete
sequences: objective, ordered tool calls, expected outcome.

**Implementation:** `src/weakincentives/prompt/task_examples.py`

**Use TaskExample when:** Demonstrating multi-step workflows, showing tool
chaining patterns, teaching investigation strategies.

**Use ToolExample when:** Showing individual tool usage, parameter formats,
expected outputs for single operations.

## Core Types

### TaskStep

| Field | Type | Description |
| ----------- | --------------------------- | --------------------- |
| `tool_name` | `str` | Tool being invoked |
| `example` | `ToolExample[Params, Result]` | Invocation details |

### TaskExample (Section)

| Property | Type | Description |
| ----------- | ------------------------- | -------------------------------- |
| `objective` | `str` | Task goal (1-500 ASCII chars) |
| `outcome` | `str \| dataclass` | Expected result |
| `steps` | `tuple[TaskStep, ...]` | Ordered tool invocations |
| `title` | `str \| None` | Display title (auto-derived) |
| `enabled` | `EnabledPredicate \| None`| Conditional inclusion |
| `visibility`| `VisibilitySelector` | FULL, SUMMARY, or HIDDEN |
| `summary` | `str \| None` | Abbreviated version for SUMMARY |

### TaskExamplesSection

Container section for `TaskExample` children. Requires at least one example.

## Validation Rules

- **Objective:** 1-500 printable ASCII characters
- **Outcome type matching:**
  - Prompts with `PromptTemplate[OutputType]`: outcome must be `OutputType` instance
  - Prompts without structured output: outcome must be `str`
- **Steps:** Non-empty sequence of `TaskStep` instances
- **Tool names:** Must match pattern `^[a-z0-9_-]{1,64}$`
- **Tool existence:** Tool names validated against prompt's tool registry at bind time

## Rendered Output

````markdown
## Task Examples

### 1. Review authentication module

**Objective:** Review src/auth.py for security issues.

**Steps:**

1. **read_file** - Read the target file
   - input:
     ```json
     {"path": "src/auth.py"}
     ```
   - output:
     ```
     {"content": "def authenticate(): ..."}
     ```

**Outcome:** Identified 2 vulnerabilities in password handling.
````

## Usage Patterns

### Basic Task Example

```python
from weakincentives.prompt import TaskExample, TaskExamplesSection, TaskStep, ToolExample

# Define steps with their tool examples
read_step = TaskStep(
    tool_name="read_file",
    example=ToolExample(
        description="Read the target file",
        input=ReadFileParams(path="src/auth.py"),
        output=ReadFileResult(content="def authenticate(): ..."),
    ),
)

search_step = TaskStep(
    tool_name="search_code",
    example=ToolExample(
        description="Search for password handling",
        input=SearchParams(pattern="password", path="src/"),
        output=SearchResult(matches=[...]),
    ),
)

# Create the example
example = TaskExample(
    key="security-review",
    objective="Review src/auth.py for security issues",
    outcome="Identified 2 vulnerabilities in password handling",
    steps=[read_step, search_step],
)

# Add to prompt template
template = PromptTemplate(
    sections=[
        MarkdownSection(tools=[read_tool, search_tool], ...),
        TaskExamplesSection(examples=[example]),
    ],
)
```

### Structured Outcome (with typed output)

```python
@dataclass(frozen=True)
class ReviewResult:
    vulnerabilities: tuple[str, ...]
    severity: str

# Outcome must match PromptTemplate's output type
example = TaskExample(
    key="typed-review",
    objective="Review authentication module",
    outcome=ReviewResult(
        vulnerabilities=("SQL injection", "Weak hashing"),
        severity="high",
    ),
    steps=[...],
)
```

### Conditional Examples

```python
# Show example only for certain contexts
example = TaskExample(
    key="advanced-review",
    objective="Deep security audit with static analysis",
    outcome="...",
    steps=[...],
    enabled=lambda params: params.expertise_level == "advanced",
    visibility=SectionVisibility.FULL,
)
```

### Summary Mode

```python
# Provide abbreviated version for context-constrained scenarios
example = TaskExample(
    key="long-workflow",
    objective="Complete end-to-end integration test",
    outcome="All tests passing",
    steps=[...],  # Many steps
    summary="Run integration tests: setup → execute → validate → cleanup",
    visibility=SectionVisibility.SUMMARY,  # Use summary by default
)
```

## Section Hierarchy

`TaskExample` instances register as children of `TaskExamplesSection`:

- Section paths: `("task-examples", "security-review")`
- Override targeting via `PromptOverridesStore` using section path tuples
- Per-example visibility predicates
- Cloning preserves all properties

## Integration with Prompt System

TaskExamplesSection integrates with the standard section lifecycle:

1. **Construction:** Validates examples, tool names
1. **Registration:** Examples become addressable children
1. **Binding:** Tool references validated against prompt's tool registry
1. **Rendering:** Each example renders with numbered heading
1. **Overrides:** Individual examples can be disabled/modified

## Limitations

- **Linear sequences only:** No branching or conditional steps
- **No intermediate state:** Cannot capture session state between steps
- **Static tool references:** Tool names resolved at construction, not runtime
- **No step dependencies:** Cannot express "step 2 uses output of step 1"
- **Single outcome:** Cannot show multiple possible outcomes

## Related Specifications

- `specs/PROMPTS.md` - Section system and rendering
- `specs/TOOLS.md` - ToolExample for single-tool demonstrations

# Task Examples Specification

## Purpose

The `TaskExamplesSection` provides trajectory-based examples that demonstrate
multi-step agent behavior. While `ToolExample` shows single-tool invocations,
`TaskExample` captures complete sequences: an objective, an ordered list of tool
calls, and an expected outcome. These examples serve as few-shot demonstrations
for complex agent workflows.

## Guiding Principles

- **Trajectory-first design**: Examples model complete task executions, not
  isolated tool calls.
- **Type-safe at construction**: All validation happens when the prompt template
  is built, not at render time.
- **Tool coherence**: Each step references a tool by name and must match that
  tool's parameter and result types.
- **Deterministic ordering**: Steps execute in declaration order; the sequence
  matters for trajectory learning.

## Core Schemas

### TaskStep

`TaskStep[ParamsT, ResultT]` represents a single tool invocation within a
trajectory:

```python
@dataclass(slots=True, frozen=True)
class TaskStep(Generic[ParamsT, ResultT]):
    """Single tool invocation in a task trajectory."""

    tool_name: str
    example: ToolExample[ParamsT, ResultT]
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | Name of the tool being invoked (must match `^[a-z0-9_-]{1,64}$`) |
| `example` | `ToolExample[ParamsT, ResultT]` | The invocation details (description, input, output) |

The `example.description` field describes the reasoning or purpose for this
specific step within the trajectory (e.g., "Fetch user profile to check
permissions").

### TaskExample

`TaskExample` captures a complete trajectory from objective to outcome:

```python
@dataclass(slots=True, frozen=True)
class TaskExample:
    """Complete task trajectory demonstrating multi-step agent behavior."""

    objective: str
    outcome: str
    steps: tuple[TaskStep[Any, Any], ...]
```

**Fields:**

| Field | Type | Constraints |
|-------|------|-------------|
| `objective` | `str` | 1-500 ASCII characters; the task goal |
| `outcome` | `str` | 1-500 ASCII characters; the expected result |
| `steps` | `tuple[TaskStep, ...]` | Non-empty ordered sequence of tool invocations |

### TaskExamplesSection

`TaskExamplesSection` renders task examples with their tool trajectories:

```python
class TaskExamplesSection(Section[TaskExamplesParamsT]):
    """Section that renders multi-step task examples."""

    def __init__(
        self,
        *,
        key: str = "task-examples",
        title: str = "Task Examples",
        examples: Sequence[TaskExample],
        tools: Sequence[Tool[Any, Any]],
        **kwargs: object,
    ) -> None: ...
```

**Constructor Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `key` | No | Section identifier (default: `"task-examples"`) |
| `title` | No | Display title (default: `"Task Examples"`) |
| `examples` | Yes | One or more `TaskExample` instances |
| `tools` | Yes | Tools referenced by the examples |

## Validation Rules

All validation occurs at `PromptTemplate` construction time. Invalid examples
raise `PromptValidationError` immediately.

### Objective and Outcome Validation

- Must be 1-500 ASCII characters (same rules as `ToolExample.description` but
  with extended limit for richer context)
- Must not be blank after stripping whitespace
- Must contain only printable ASCII characters

```python
# Valid
TaskExample(
    objective="Review the authentication module for security issues",
    outcome="Identified 3 vulnerabilities with remediation steps",
    steps=(...),
)

# Invalid: empty objective
TaskExample(
    objective="",  # PromptValidationError
    outcome="Done",
    steps=(...),
)
```

### Steps Validation

- `steps` must be non-empty (at least one step required)
- Steps are validated in declaration order

```python
# Invalid: empty steps
TaskExample(
    objective="Do something",
    outcome="Done",
    steps=(),  # PromptValidationError: "steps must not be empty"
)
```

### Tool Name Resolution

Each `TaskStep.tool_name` must reference a tool provided to the section:

```python
lookup_tool = Tool[LookupParams, LookupResult](name="lookup", ...)
search_tool = Tool[SearchParams, SearchResult](name="search", ...)

# Valid: tool names match provided tools
TaskExamplesSection(
    examples=[
        TaskExample(
            objective="Find information",
            outcome="Found results",
            steps=(
                TaskStep(tool_name="lookup", example=...),  # OK
                TaskStep(tool_name="search", example=...),  # OK
            ),
        ),
    ],
    tools=[lookup_tool, search_tool],
)

# Invalid: unknown tool name
TaskExamplesSection(
    examples=[
        TaskExample(
            objective="Find information",
            outcome="Found results",
            steps=(
                TaskStep(tool_name="unknown", example=...),  # PromptValidationError
            ),
        ),
    ],
    tools=[lookup_tool],
)
```

Error message format:
```
PromptValidationError: Unknown tool "unknown" in task example step 0.
Available tools: lookup, search.
Section path: ("task-examples",)
```

### Type Coherence

Each step's `ToolExample` must match the referenced tool's parameter and result
types. Validation reuses the existing `Tool._validate_examples` logic:

```python
@dataclass
class LookupParams:
    entity_id: str

@dataclass
class LookupResult:
    url: str

lookup_tool = Tool[LookupParams, LookupResult](name="lookup", ...)

# Valid: types match
TaskStep(
    tool_name="lookup",
    example=ToolExample(
        description="Fetch entity",
        input=LookupParams(entity_id="abc"),
        output=LookupResult(url="https://..."),
    ),
)

# Invalid: wrong input type
TaskStep(
    tool_name="lookup",
    example=ToolExample(
        description="Fetch entity",
        input=SearchParams(query="abc"),  # PromptValidationError: type mismatch
        output=LookupResult(url="https://..."),
    ),
)
```

Error message format:
```
PromptValidationError: Task example step 1 input type mismatch for tool "lookup".
Expected: LookupParams, got: SearchParams.
Section path: ("task-examples",)
```

### Duplicate Tool Names

Tools passed to `TaskExamplesSection` must have unique names:

```python
# Invalid: duplicate tool names
TaskExamplesSection(
    examples=[...],
    tools=[
        Tool(name="lookup", ...),
        Tool(name="lookup", ...),  # PromptValidationError
    ],
)
```

## Rendering

### Markdown Structure

Task examples render as numbered examples with nested tool call sequences:

```markdown
## 3. Task Examples

### Example 1: Review authentication module

**Objective:** Review the authentication module for security vulnerabilities.

**Steps:**

1. **read_file** - Read the auth module source
   - input:
     ```json
     {"path": "src/auth.py"}
     ```
   - output:
     ```
     {"content": "def authenticate(user, password): ..."}
     ```

2. **search_patterns** - Search for common vulnerability patterns
   - input:
     ```json
     {"pattern": "sql.*\\+.*user", "path": "src/"}
     ```
   - output:
     ```
     {"matches": [{"file": "src/auth.py", "line": 42}]}
     ```

3. **report_issue** - Report the SQL injection vulnerability
   - input:
     ```json
     {"severity": "high", "title": "SQL injection in auth module"}
     ```
   - output:
     ```
     {"issue_id": "SEC-123"}
     ```

**Outcome:** Identified SQL injection vulnerability and created issue SEC-123.

---

### Example 2: ...
```

### Rendering Rules

- Example titles derive from objective (truncated to 60 characters if needed)
- Steps render in declaration order with 1-based numbering
- Input/output use fenced code blocks (JSON for input, plain for output)
- Horizontal rules separate multiple examples
- Step descriptions appear after the tool name in bold

## Integration

### With PromptTemplate

```python
from weakincentives.prompt import (
    PromptTemplate,
    MarkdownSection,
    TaskExamplesSection,
    TaskExample,
    TaskStep,
    Tool,
    ToolExample,
)

# Define tools
read_tool = Tool[ReadParams, ReadResult](
    name="read_file",
    description="Read a file from the workspace.",
    handler=read_handler,
)

search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search for patterns in files.",
    handler=search_handler,
)

# Define task examples
review_example = TaskExample(
    objective="Review src/auth.py for security issues",
    outcome="Identified 2 vulnerabilities with severity ratings",
    steps=(
        TaskStep(
            tool_name="read_file",
            example=ToolExample(
                description="Read the target file",
                input=ReadParams(path="src/auth.py"),
                output=ReadResult(content="def authenticate(): ..."),
            ),
        ),
        TaskStep(
            tool_name="search",
            example=ToolExample(
                description="Search for SQL patterns",
                input=SearchParams(pattern="SELECT.*%s"),
                output=SearchResult(matches=[Match(line=42)]),
            ),
        ),
    ),
)

# Build prompt
template = PromptTemplate(
    ns="agents/reviewer",
    key="code-review",
    sections=[
        MarkdownSection(
            title="Instructions",
            key="instructions",
            template="Review code for security issues.",
            tools=[read_tool, search_tool],
        ),
        TaskExamplesSection(
            title="Example Workflows",
            key="example-workflows",
            examples=[review_example],
            tools=[read_tool, search_tool],
        ),
    ],
)
```

### Tool Registration

Tools in `TaskExamplesSection` are registered with the prompt alongside tools
from other sections. The section participates in duplicate name detection:

```python
# This raises PromptValidationError due to duplicate "read_file"
template = PromptTemplate(
    ns="test",
    key="duplicate",
    sections=[
        MarkdownSection(
            key="tools",
            title="Tools",
            template="...",
            tools=[Tool(name="read_file", ...)],
        ),
        TaskExamplesSection(
            key="examples",
            examples=[...],
            tools=[Tool(name="read_file", ...)],  # Duplicate!
        ),
    ],
)
```

### Progressive Disclosure

`TaskExamplesSection` supports summary visibility:

```python
TaskExamplesSection(
    examples=[...],
    tools=[...],
    visibility=SectionVisibility.SUMMARY,
    summary="Example workflows available for review tasks.",
)
```

When summarized, the section renders only the summary text with the standard
expansion hint.

## Error Handling

### Exception Types

All validation errors raise `PromptValidationError` with:

- `message`: Human-readable description
- `section_path`: Path to the section (e.g., `("task-examples",)`)
- `placeholder`: Field name when applicable (e.g., `"objective"`, `"steps"`)

### Error Scenarios

| Scenario | Error Message |
|----------|---------------|
| Empty objective | `"objective must not be empty"` |
| Objective too long | `"objective must be <= 500 characters"` |
| Empty outcome | `"outcome must not be empty"` |
| Outcome too long | `"outcome must be <= 500 characters"` |
| Empty steps | `"steps must not be empty"` |
| Unknown tool name | `"Unknown tool \"X\" in task example step N"` |
| Input type mismatch | `"Task example step N input type mismatch for tool \"X\""` |
| Output type mismatch | `"Task example step N output type mismatch for tool \"X\""` |
| Duplicate tool name | `"Duplicate tool name: X"` |

## Usage Example

Complete example demonstrating a multi-tool workflow:

```python
from dataclasses import dataclass, field
from weakincentives.prompt import (
    PromptTemplate,
    MarkdownSection,
    TaskExamplesSection,
    TaskExample,
    TaskStep,
    Tool,
    ToolExample,
    ToolContext,
    ToolResult,
)

# Parameter and result types
@dataclass(slots=True, frozen=True)
class FetchParams:
    url: str = field(metadata={"description": "URL to fetch"})

@dataclass(slots=True, frozen=True)
class FetchResult:
    status: int
    body: str

@dataclass(slots=True, frozen=True)
class ParseParams:
    content: str = field(metadata={"description": "Content to parse"})
    format: str = field(metadata={"description": "Expected format (json, xml)"})

@dataclass(slots=True, frozen=True)
class ParseResult:
    data: dict[str, str]

@dataclass(slots=True, frozen=True)
class StoreParams:
    key: str = field(metadata={"description": "Storage key"})
    value: str = field(metadata={"description": "Value to store"})

@dataclass(slots=True, frozen=True)
class StoreResult:
    stored: bool

# Tool definitions
fetch_tool = Tool[FetchParams, FetchResult](
    name="fetch_url",
    description="Fetch content from a URL.",
    handler=fetch_handler,
)

parse_tool = Tool[ParseParams, ParseResult](
    name="parse_content",
    description="Parse structured content.",
    handler=parse_handler,
)

store_tool = Tool[StoreParams, StoreResult](
    name="store_data",
    description="Store data in the session.",
    handler=store_handler,
)

# Task example: fetch, parse, and store workflow
etl_example = TaskExample(
    objective="Fetch API data, parse the JSON response, and store the result",
    outcome="Successfully fetched, parsed, and stored user data",
    steps=(
        TaskStep(
            tool_name="fetch_url",
            example=ToolExample(
                description="Fetch the user API endpoint",
                input=FetchParams(url="https://api.example.com/users/123"),
                output=FetchResult(status=200, body='{"name": "Alice"}'),
            ),
        ),
        TaskStep(
            tool_name="parse_content",
            example=ToolExample(
                description="Parse the JSON response",
                input=ParseParams(content='{"name": "Alice"}', format="json"),
                output=ParseResult(data={"name": "Alice"}),
            ),
        ),
        TaskStep(
            tool_name="store_data",
            example=ToolExample(
                description="Store the parsed user data",
                input=StoreParams(key="user:123", value="Alice"),
                output=StoreResult(stored=True),
            ),
        ),
    ),
)

# Build prompt with task examples
template = PromptTemplate(
    ns="agents/etl",
    key="data-pipeline",
    sections=[
        MarkdownSection(
            title="Instructions",
            key="instructions",
            template="Process data using the available tools.",
            tools=[fetch_tool, parse_tool, store_tool],
        ),
        TaskExamplesSection(
            title="Workflow Examples",
            key="workflow-examples",
            examples=[etl_example],
            tools=[fetch_tool, parse_tool, store_tool],
        ),
    ],
)
```

## Implementation Checklist

- [ ] `TaskStep` frozen dataclass with `tool_name` and `example` fields
- [ ] `TaskExample` frozen dataclass with `objective`, `outcome`, and `steps`
- [ ] `TaskExamplesSection` extending `Section` with validation in `__init__`
- [ ] Objective/outcome validation (1-500 ASCII chars, non-blank)
- [ ] Steps non-empty validation
- [ ] Tool name resolution against provided tools
- [ ] Type coherence validation reusing `Tool._validate_examples` logic
- [ ] Duplicate tool name detection
- [ ] Markdown rendering with numbered steps and fenced blocks
- [ ] Progressive disclosure support (summary visibility)
- [ ] Integration tests for validation error messages
- [ ] Unit tests for rendering output format

## Limitations

- **No branching**: Steps form a linear sequence; conditional paths are not
  supported.
- **No intermediate state**: Examples don't capture session state between steps.
- **Static tools**: Tool references are resolved at construction; dynamic tool
  registration is not supported.
- **No step dependencies**: Steps cannot reference outputs from previous steps
  in their inputs (the example is illustrative, not executable).

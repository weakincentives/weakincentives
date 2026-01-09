# Chapter 3: Prompts

> **Canonical Reference**: See [specs/PROMPTS.md](/specs/PROMPTS.md) for the complete specification.

## Introduction

The prompt system is the heart of WINK. It's not a generic templating engine—it's a **type-safe, composable, deterministic framework for building LLM prompts as first-class programs**. Every string template that flows to an LLM is centralized, inspectable, and versioned.

The design prioritizes predictability over flexibility:

- **Deterministic rendering**: Same inputs always produce identical output
- **Early validation**: Placeholder errors surface at construction time, not runtime
- **Explicit composition**: No magic string concatenation
- **Safe overrides**: Hash-validated version management
- **Resource co-location**: Prompts own their runtime dependencies

If you've been frustrated by prompt "spaghetti" in other frameworks—where templates are scattered across files, placeholders fail silently, and runtime behavior is unpredictable—WINK's prompt system will feel like a breath of fresh air.

## Overview: The Prompt Lifecycle

```mermaid
flowchart TB
    subgraph Construction["1. Construction"]
        Template["PromptTemplate&lt;OutputT&gt;<br/>(sections, resources)"]
        Prompt["Prompt(template)"]
        Bind["bind(params)"]
    end

    subgraph Lifecycle["2. Resource Lifecycle"]
        Enter["__enter__()"]
        Start["resources.start()"]
        Use["Prompt evaluation"]
        Exit["__exit__()"]
        Close["resources.close()"]
    end

    subgraph Rendering["3. Render Pipeline"]
        CheckEnabled["Check enabled()"]
        CheckVisibility["Resolve visibility"]
        Substitute["Template.substitute()"]
        ApplyOverrides["Apply overrides"]
        BuildMarkdown["Build markdown tree"]
    end

    subgraph Output["4. Rendered Output"]
        RenderedPrompt["RenderedPrompt&lt;OutputT&gt;"]
        Text["Markdown text"]
        Tools["Tool schemas"]
        OutputSchema["Structured output schema"]
    end

    Template --> Prompt --> Bind
    Bind --> Enter --> Start --> Use
    Use --> CheckEnabled --> CheckVisibility --> Substitute
    Substitute --> ApplyOverrides --> BuildMarkdown
    BuildMarkdown --> RenderedPrompt
    RenderedPrompt --> Text
    RenderedPrompt --> Tools
    RenderedPrompt --> OutputSchema
    Use --> Exit --> Close
```

The lifecycle has four distinct phases:

1. **Construction**: Define your prompt template with sections, tools, and resources
2. **Binding**: Attach runtime parameters (dataclass instances) to the prompt
3. **Rendering**: Walk the section tree, apply overrides, and build markdown
4. **Resource Management**: Initialize and cleanup resources via context manager

This separation ensures that templates are reusable, parameters are validated early, and resources are properly managed.

## PromptTemplate: The Blueprint

`PromptTemplate[OutputT]` is the immutable blueprint for your prompt. Think of it as the "class" definition—it describes structure, not runtime state.

### Core Properties

```python
from dataclasses import dataclass
from weakincentives.prompt import PromptTemplate, MarkdownSection

@dataclass(slots=True, frozen=True)
class SupportParams:
    question: str
    context: str

template = PromptTemplate(
    ns="support",           # Namespace for grouping related prompts
    key="faq",              # Unique key within namespace
    name="faq_assistant",   # Human-readable name for logging
    sections=(              # Tree of Section objects
        MarkdownSection(
            title="Instructions",
            key="instructions",
            template="Answer customer questions clearly and concisely.",
        ),
        MarkdownSection(
            title="Context",
            key="context",
            template="Background: ${context}",
        ),
        MarkdownSection(
            title="Question",
            key="question",
            template="Customer question: ${question}",
        ),
    ),
)
```

### Structured Output Declaration

Declare typed outputs using generic specialization:

```python
@dataclass(slots=True, frozen=True)
class Answer:
    summary: str
    details: list[str]
    confidence: float

# Object output
template = PromptTemplate[Answer](
    ns="support",
    key="faq",
    sections=(...),
)

# Array output
template = PromptTemplate[list[Answer]](
    ns="support",
    key="faq-batch",
    sections=(...),
)
```

When you declare structured output:

- The adapter instructs the model to return JSON matching your schema
- `parse_structured_output()` validates and deserializes the response
- Non-dataclass types raise `PromptValidationError` at construction time

### Validation Rules

WINK validates your template **at construction time** to catch errors early:

- **Namespace and key**: Required, non-empty strings
- **Section keys**: Must match `^[a-z0-9][a-z0-9._-]{0,63}$`
- **Placeholders**: Must correspond to fields on bound dataclasses
- **Output types**: Must be dataclasses (not primitives, dicts, or arbitrary objects)

A typo in a placeholder name fails **immediately**, not when the model is mid-response.

### Namespaces and Keys

The `(ns, key)` pair uniquely identifies a prompt family for:

- **Versioning**: Track prompt evolution over time
- **Overrides**: Apply tested optimizations ([Chapter 11](11-prompt-optimization.md))
- **Debugging**: Inspect which prompt generated which response ([Chapter 13](13-debugging.md))

Choose namespaces that reflect your agent's domain (e.g., `"code-review"`, `"research"`, `"planning"`), and keys that describe the specific task (e.g., `"analyze-diff"`, `"summarize-papers"`, `"create-plan"`).

## Prompt: The Runtime Binding

`Prompt[OutputT]` is the runtime instance—it binds parameters, manages resources, and renders output.

### Basic Usage

```python
from weakincentives.prompt import Prompt

# Create and bind parameters
prompt = Prompt(template).bind(
    SupportParams(
        question="How do I reset my password?",
        context="User account created 2023-01-15",
    )
)

# Render within resource context
with prompt.resources:
    rendered = prompt.render()
    print(rendered.text)
    print([t.name for t in rendered.tools])
```

### Binding Multiple Parameter Types

You can bind multiple dataclass types to a single prompt:

```python
@dataclass(slots=True, frozen=True)
class UserInfo:
    name: str
    plan: str

@dataclass(slots=True, frozen=True)
class QuestionInfo:
    text: str
    category: str

prompt = Prompt(template).bind(
    UserInfo(name="Alice", plan="Enterprise"),
    QuestionInfo(text="How do I export data?", category="data"),
)
```

**Key rule**: You can bind **at most one instance per dataclass type**. Binding the same type twice replaces the previous value. Providing duplicate types in a single `bind()` call raises `PromptValidationError`.

### RenderedPrompt

Rendering returns a `RenderedPrompt[OutputT]` with:

```python
@dataclass(frozen=True)
class RenderedPrompt:
    text: str                                    # Final markdown
    tools: tuple[Tool, ...]                      # Tools from enabled sections
    structured_output: StructuredOutputConfig    # Schema config (if declared)
    descriptor: PromptDescriptor                 # Hash for overrides system
```

The rendered prompt is immutable and serializable—perfect for logging, debugging, and version control.

## Sections: Composable Prompt Components

A `Section[ParamsT]` is a node in your prompt's tree structure. Sections compose hierarchically to build complex prompts from reusable pieces.

```mermaid
graph TB
    Root["Prompt Template"]
    Instructions["Instructions Section"]
    Context["Context Section"]
    ContextSub1["Context.Background"]
    ContextSub2["Context.Constraints"]
    Question["Question Section"]
    Tools["Tools Section"]
    ToolsSub1["Tools.Planning"]
    ToolsSub2["Tools.VFS"]

    Root --> Instructions
    Root --> Context
    Context --> ContextSub1
    Context --> ContextSub2
    Root --> Question
    Root --> Tools
    Tools --> ToolsSub1
    Tools --> ToolsSub2

    style Root fill:#e1f5ff
    style Tools fill:#fff4e1
    style ToolsSub1 fill:#fff4e1
    style ToolsSub2 fill:#fff4e1
```

### Section Anatomy

Every section has:

```python
class Section(ABC, Generic[ParamsT]):
    title: str                              # Renders as markdown heading
    key: str                                # Stable identifier
    children: tuple[Section, ...] = ()      # Nested subsections
    tools: tuple[Tool, ...] = ()            # Tools exposed by this section
    enabled: Callable[..., bool] | None     # Dynamic enable/disable
    summary: str | None = None              # For progressive disclosure
    visibility: SectionVisibility = FULL    # FULL or SUMMARY
    accepts_overrides: bool = True          # Allow override system
    default_params: ParamsT | None = None   # Fallback parameters

    @abstractmethod
    def render_body(self, params: ParamsT, session: Session) -> str:
        """Render the section's body content."""
        ...

    def resources(self) -> ResourceRegistry:
        """Resources required by this section."""
        return ResourceRegistry()
```

### Heading Levels and Numbering

WINK generates markdown headings with deterministic numbering:

- Root sections: `## 1. Title`, `## 2. Title`
- Depth 1 children: `### 1.1. Subtitle`, `### 1.2. Subtitle`
- Depth 2 children: `#### 1.1.1. Sub-subtitle`

This ensures consistent formatting and makes prompts readable in markdown viewers.

### Tools and Sections

Sections can contribute tools to the prompt. When a section is disabled, **its tools disappear too**. This is powerful: you can build a comprehensive toolset and enable only what's relevant for the current context.

```python
from weakincentives.prompt import Tool

search_tool = Tool(
    name="search_docs",
    description="Search documentation",
    parameters_type=SearchParams,
    handler=search_handler,
)

docs_section = MarkdownSection(
    title="Documentation",
    key="docs",
    template="You have access to the documentation search tool.",
    tools=(search_tool,),
    enabled=lambda params: params.has_docs_access,
)
```

When `has_docs_access` is `False`, both the section text **and** the `search_docs` tool are excluded from the rendered prompt.

## MarkdownSection: The Workhorse

`MarkdownSection` is the concrete implementation you'll use most often. It renders a `string.Template` with `${name}` placeholders.

### Basic Usage

```python
from dataclasses import dataclass
from weakincentives.prompt import MarkdownSection

@dataclass(slots=True, frozen=True)
class UserProfile:
    name: str
    plan: str
    joined: str

profile_section = MarkdownSection(
    title="User Profile",
    key="profile",
    template="""
    Name: ${name}
    Plan: ${plan}
    Member since: ${joined}
    """,
)
```

### Why string.Template?

WINK deliberately uses Python's `string.Template`, which supports **only simple placeholder substitution**:

- ✅ `${name}` - Simple substitution
- ❌ `${name.upper()}` - No expressions
- ❌ `${for x in items}` - No loops
- ❌ `${if condition}` - No conditionals

**Why this limitation?** Complex formatting belongs in your Python code, where it can be tested, type-checked, and debugged. If you need loops or conditionals, compute the string in Python and pass it as a parameter:

```python
@dataclass(slots=True, frozen=True)
class ReportParams:
    summary: str
    items_formatted: str  # Pre-formatted in Python

# In your code:
items_text = "\n".join(f"- {item.name}: {item.value}" for item in items)
params = ReportParams(
    summary="Analysis complete",
    items_formatted=items_text,
)
```

This keeps templates simple, predictable, and easy to override.

## Structured Output: Type-Safe Responses

WINK prompts can declare structured output, turning raw LLM responses into typed Python objects.

```mermaid
flowchart LR
    subgraph Declaration["Declaration"]
        Template["PromptTemplate&lt;OutputT&gt;"]
        Schema["JSON Schema Generation"]
    end

    subgraph Execution["Execution"]
        Adapter["Adapter.evaluate()"]
        Model["LLM (JSON mode)"]
        Response["Raw JSON response"]
    end

    subgraph Parsing["Parsing"]
        Extract["Extract JSON block"]
        Validate["Validate schema"]
        Deserialize["Deserialize to OutputT"]
        Result["Typed result: OutputT"]
    end

    Template --> Schema
    Schema --> Adapter
    Adapter --> Model
    Model --> Response
    Response --> Extract
    Extract --> Validate
    Validate --> Deserialize
    Deserialize --> Result

    style Template fill:#e1f5ff
    style Result fill:#e1ffe1
```

### Declaring Output Types

```python
from dataclasses import dataclass
from weakincentives.prompt import PromptTemplate

@dataclass(slots=True, frozen=True)
class CodeReview:
    summary: str
    issues: list[str]
    suggestions: list[str]
    approved: bool

template = PromptTemplate[CodeReview](
    ns="code-review",
    key="analyze-pr",
    sections=(...),
)
```

### Adapter Integration

Adapters automatically handle structured output:

```python
from weakincentives.adapters.openai import OpenAIAdapter

adapter = OpenAIAdapter(model="gpt-4")
prompt = Prompt(template).bind(params)

with prompt.resources:
    response = adapter.evaluate(prompt, session=session)
    result: CodeReview = response.output  # Typed and validated
```

The adapter:

1. Generates a JSON schema from your dataclass
2. Instructs the model to use JSON mode
3. Parses and validates the response
4. Returns a typed instance

### Manual Parsing

If you need to parse output yourself (e.g., when using a custom adapter):

```python
from weakincentives.prompt import parse_structured_output

rendered = prompt.render()
raw_response = custom_adapter.generate(rendered.text)
output: CodeReview = parse_structured_output(raw_response, rendered)
```

### Array Outputs

For array outputs, the parser accepts two formats:

```python
template = PromptTemplate[list[CodeReview]](...)

# Format 1: Direct array
# [{"summary": "...", "issues": [...]}, {"summary": "...", "issues": [...]}]

# Format 2: Object wrapper
# {"items": [{"summary": "...", "issues": [...]}, {"summary": "...", "issues": [...]}]}
```

The wrapper key is `"items"` (see `ARRAY_WRAPPER_KEY` in the source).

### Validation Rules

`parse_structured_output()` performs strict validation:

- **Container type**: Object vs. array must match declaration
- **Required fields**: All non-optional dataclass fields must be present
- **Type coercion**: Conservative (string to int, but not arbitrary conversions)
- **Extra keys**: Rejected by default (unless `allow_extra_keys=True`)

Failures raise `OutputParseError` with the raw response attached for debugging.

## Dynamic Scoping with enabled()

One of WINK's most powerful features is **dynamic section enabling**. You can build a large, comprehensive prompt template, then render only the sections relevant to the current context.

```mermaid
flowchart TB
    subgraph RenderTime["Render Time"]
        Start["Start rendering"]
        Section["For each section"]
        CheckEnabled{"enabled() callable?"}
        CallEnabled["Call enabled()"]
        IsEnabled{"Returns True?"}
        RenderSection["Render section + tools"]
        SkipSection["Skip section + tools"]
        Continue["Continue to next section"]
    end

    Start --> Section
    Section --> CheckEnabled
    CheckEnabled -->|Yes| CallEnabled
    CheckEnabled -->|No| RenderSection
    CallEnabled --> IsEnabled
    IsEnabled -->|True| RenderSection
    IsEnabled -->|False| SkipSection
    RenderSection --> Continue
    SkipSection --> Continue
    Continue --> Section

    style RenderSection fill:#e1ffe1
    style SkipSection fill:#ffe1e1
```

### Supported Signatures

The `enabled` callable supports multiple signatures:

```python
# 1. No arguments
enabled=lambda: should_enable()

# 2. Session only
enabled=lambda *, session: session[DebugMode].enabled

# 3. Params only
enabled=lambda params: params.debug_enabled

# 4. Params and session
enabled=lambda params, *, session: params.level == "expert" and session[State].ready
```

WINK inspects the signature and provides the appropriate arguments at render time.

### Example: Debug Instructions

```python
from dataclasses import dataclass
from weakincentives.prompt import MarkdownSection
from weakincentives.runtime import Session

@dataclass(slots=True, frozen=True)
class DebugFlag:
    enabled: bool

def is_debug_enabled(flag: DebugFlag, *, session: Session) -> bool:
    # Could also check session state here
    del session
    return flag.enabled

debug_section = MarkdownSection(
    title="Debug Instructions",
    key="debug",
    template="""
    When debugging:
    - Include full stack traces
    - Show intermediate values
    - Explain your reasoning step-by-step
    """,
    enabled=is_debug_enabled,
)

template = PromptTemplate(
    ns="agent",
    key="task",
    sections=(
        MarkdownSection(title="Instructions", key="instructions", template="..."),
        debug_section,  # Only included when DebugFlag.enabled=True
    ),
)

# Render with debug enabled
prompt = Prompt(template).bind(DebugFlag(enabled=True))
rendered = prompt.render()  # Includes debug section

# Render with debug disabled
prompt = Prompt(template).bind(DebugFlag(enabled=False))
rendered = prompt.render()  # Excludes debug section
```

### Tools and Dynamic Enabling

**Critical**: When a section is disabled, its tools are also excluded. This lets you build comprehensive toolsets and expose only what's needed:

```python
admin_tools_section = MarkdownSection(
    title="Admin Tools",
    key="admin-tools",
    template="You have access to administrative tools.",
    tools=(delete_user_tool, modify_permissions_tool),
    enabled=lambda user: user.is_admin,
)
```

If `user.is_admin` is `False`, the section text **and** both tools disappear from the rendered prompt. The model never knows they exist.

## Session-Bound Sections and Cloning

Some sections are **pure**: they depend only on parameters and render the same text every time. These can safely live in module-level `PromptTemplate` instances.

Other sections are **session-bound**: they capture runtime resources like sessions, filesystems, or sandbox connections. Examples:

- `PlanningToolsSection(session=...)` - Planning tools ([Chapter 4](04-tools.md))
- `VfsToolsSection(session=...)` - Virtual filesystem ([Chapter 12](12-workspace-tools.md))
- `WorkspaceDigestSection(session=...)` - Workspace summaries ([Chapter 10](10-progressive-disclosure.md))
- `PodmanSandboxSection(session=...)` - Sandboxed execution ([Chapter 12](12-workspace-tools.md))

### Pattern A: Build Template Per Session

The safest approach is to build your template fresh for each session:

```python
from typing import Any
from weakincentives.contrib.tools import PlanningToolsSection, VfsToolsSection
from weakincentives.prompt import PromptTemplate, MarkdownSection
from weakincentives.runtime import Session


def build_prompt_template(*, session: Session) -> PromptTemplate[Any]:
    return PromptTemplate(
        ns="agent",
        key="task",
        sections=(
            MarkdownSection(
                title="Instructions",
                key="instructions",
                template="Complete the task step by step.",
            ),
            PlanningToolsSection(session=session),  # Fresh per session
            VfsToolsSection(session=session),        # Fresh per session
        ),
    )


# Usage:
session = Session(bus=dispatcher)
template = build_prompt_template(session=session)
prompt = Prompt(template).bind(params)
```

### Pattern B: Clone Session-Bound Sections

For templates with mostly static content, you can clone just the session-bound parts:

```python
from weakincentives.prompt import PromptTemplate, MarkdownSection
from weakincentives.contrib.tools import PlanningToolsSection

# Base template with placeholder section
base_planning_section = PlanningToolsSection(session=None)  # Placeholder

BASE_TEMPLATE = PromptTemplate(
    ns="agent",
    key="task",
    sections=(
        MarkdownSection(title="Instructions", key="instructions", template="..."),
        base_planning_section,
    ),
)

# Clone for each session
def prepare_prompt(session: Session) -> Prompt:
    # Clone the planning section with the real session
    planning_section = base_planning_section.clone(session=session)

    # Build template with cloned section
    template = PromptTemplate(
        ns=BASE_TEMPLATE.ns,
        key=BASE_TEMPLATE.key,
        sections=(
            BASE_TEMPLATE.sections[0],  # Static section
            planning_section,            # Cloned section
        ),
    )

    return Prompt(template)
```

Sections support `clone(**overrides)` to create new instances with updated fields. This is especially useful for tool-backed sections that need to rewire reducers and handlers to new session instances.

### Why This Matters

Accidentally sharing a tool section (and its internal state) across multiple sessions is a common bug. Each session should get its own tool sections to ensure isolation and avoid cross-session contamination.

## Few-Shot Examples with TaskExamplesSection

WINK supports few-shot examples as first-class sections via `TaskExamplesSection`. Examples are often more effective than "more instructions," and keeping them as typed objects makes them easier to maintain and override.

### Why Few-Shot Examples?

Models often generalize better from concrete examples than from abstract instructions. For complex tasks (especially tool usage), showing the model **one correct execution** can be more valuable than paragraphs of documentation.

### TaskExample Structure

A `TaskExample` can include:

```python
from dataclasses import dataclass
from weakincentives.prompt import TaskExample

@dataclass(frozen=True)
class SearchParams:
    query: str

@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str

example = TaskExample(
    input_params=(SearchParams(query="Python async patterns"),),
    output=SearchResult(
        title="Async/Await in Python",
        url="https://docs.python.org/3/library/asyncio.html",
    ),
    tool_calls=[
        ToolCallExample(
            tool_name="search_docs",
            params=SearchParams(query="Python async patterns"),
            result="Found 3 results...",
        ),
    ],
)
```

### Using TaskExamplesSection

```python
from weakincentives.prompt import TaskExamplesSection

examples_section = TaskExamplesSection(
    title="Examples",
    key="examples",
    examples=(example1, example2),
)

template = PromptTemplate(
    ns="search",
    key="query",
    sections=(
        MarkdownSection(title="Instructions", key="instructions", template="..."),
        examples_section,
    ),
)
```

The section renders examples in a structured format that models can learn from. For details on the format and available options, see `weakincentives.prompt.task_examples`.

## Resource Lifecycle Management

Prompts own their resource dependencies and manage lifecycle via the context manager protocol. This ensures resources are properly initialized and cleaned up.

### Resource Sources

When you enter a prompt's context (`with prompt.resources:`), it collects resources from three sources:

1. **Template resources** - Declared on `PromptTemplate.resources`
2. **Section resources** - Collected from all sections via `section.resources()`
3. **Bind-time resources** - Passed to `bind(resources=...)`

Resources merge in order; later sources override earlier ones on conflict.

```python
from weakincentives.resources import ResourceRegistry, Binding
from weakincentives.filesystem import Filesystem
from weakincentives.contrib.tools.vfs import VirtualFilesystem

# Template resources
template = PromptTemplate(
    ns="agent",
    key="task",
    sections=(...),
    resources=ResourceRegistry.of(
        Binding(HTTPClient, lambda r: HTTPClient(timeout=30)),
    ),
)

# Bind-time resources (override template)
prompt = Prompt(template).bind(
    params,
    resources=ResourceRegistry.of(
        Binding(Filesystem, lambda r: VirtualFilesystem()),
    ),
)

# Access within context
with prompt.resources:
    fs = prompt.resources.get(Filesystem)       # From bind-time
    http = prompt.resources.get(HTTPClient)     # From template
```

### Context Manager Protocol

```python
# Resource lifecycle
prompt = Prompt(template).bind(params)

with prompt.resources:
    # Resources are initialized and available
    fs = prompt.resources.get(Filesystem)
    clock = prompt.resources.get(Clock)

    # Render and evaluate
    rendered = prompt.render()
    response = adapter.evaluate(prompt, session=session)

# Resources are automatically cleaned up
```

**Key rule**: Resources are **only available within the context**. Accessing `prompt.resources.get()` outside the `with` block raises `RuntimeError`.

### Section Resource Contribution

Sections can contribute resources by overriding `resources()`:

```python
from weakincentives.prompt import Section
from weakincentives.resources import ResourceRegistry, Binding
from weakincentives.filesystem import Filesystem

class WorkspaceSection(Section):
    def __init__(self, filesystem: Filesystem):
        self._filesystem = filesystem
        super().__init__(title="Workspace", key="workspace")

    def resources(self) -> ResourceRegistry:
        # Contribute filesystem to prompt resources
        return ResourceRegistry.of(
            Binding(Filesystem, lambda r: self._filesystem),
        )

    def render_body(self, params, session):
        return f"Workspace available with {len(self._filesystem.list('/'))} files."
```

The prompt automatically collects resources from all sections, including children. See [Chapter 5](05-sessions.md) for more on the resource registry and dependency injection.

## Progressive Disclosure

Sections support **progressive disclosure** to reduce token usage by initially hiding detailed content. The model can request expansion when needed.

### Section Visibility

```python
from weakincentives.prompt import MarkdownSection, SectionVisibility

context_section = MarkdownSection(
    title="Project Context",
    key="context",
    template="""
    # Project structure
    - 42 files
    - 12,345 lines of code
    - Last updated: 2024-01-15
    [... detailed file listings ...]
    """,
    summary="Project context available (42 files, 12K lines).",
    visibility=SectionVisibility.SUMMARY,
)
```

When rendered with `SUMMARY` visibility:

```markdown
## 1. Project Context

Project context available (42 files, 12K lines).

---
[This section is summarized. To view full content, call `read_section` with key "context".]
```

### Disclosure Tools

WINK automatically injects disclosure tools based on whether summarized sections have tools:

- **Sections WITHOUT tools**: `read_section` - Returns content without changing state
- **Sections WITH tools**: `open_sections` - Permanently expands sections and enables tools

```python
# Without tools: read_section is injected
context_section = MarkdownSection(
    title="Context",
    key="context",
    template="...",
    summary="Context available.",
    visibility=SUMMARY,
    # No tools
)

# With tools: open_sections is injected
docs_section = MarkdownSection(
    title="Documentation",
    key="docs",
    template="...",
    summary="Documentation search available.",
    visibility=SUMMARY,
    tools=(search_docs_tool,),  # Has tools
)
```

### Handling open_sections

The `open_sections` tool raises `VisibilityExpansionRequired` rather than returning a result:

```python
from weakincentives.prompt import VisibilityExpansionRequired

prompt = Prompt(template).bind(params)

with prompt.resources:
    while True:
        try:
            response = adapter.evaluate(prompt, session=session)
            break
        except VisibilityExpansionRequired as e:
            # Permanently expand requested sections
            for path, visibility in e.requested_overrides.items():
                session[VisibilityOverrides].apply(
                    SetVisibilityOverride(path=path, visibility=visibility)
                )
            # Loop back to re-render with expanded sections
```

See [Chapter 10](10-progressive-disclosure.md) for complete details on progressive disclosure patterns and optimization strategies.

## Error Handling

WINK uses a clear exception hierarchy for prompt errors:

### Exception Types

```python
from weakincentives.prompt import (
    PromptValidationError,  # Construction failures
    PromptRenderError,      # Rendering failures
    OutputParseError,       # Structured output validation failures
    VisibilityExpansionRequired,  # Progressive disclosure expansion request
)
```

### Validation Errors (Construction Time)

Raised when building `PromptTemplate` or `Prompt`:

```python nocheck
# Empty namespace
PromptValidationError: "Namespace cannot be empty"

# Invalid section key
PromptValidationError: "Section key 'My Section!' is invalid (must match ^[a-z0-9][a-z0-9._-]{0,63}$)"

# Non-dataclass output type
PromptValidationError: "Output type must be a dataclass, got <class 'dict'>"

# Duplicate parameter types in single bind()
PromptValidationError: "Duplicate parameter type UserParams provided to bind()"
```

### Render Errors (Render Time)

Raised when calling `prompt.render()`:

```python nocheck
# Missing placeholder in dataclass
PromptRenderError: "Template references ${unknown}, but UserParams has no such field"

# Missing required field at render
PromptRenderError: "Required field 'name' missing on UserParams"

# Template substitution failure
PromptRenderError: "Failed to substitute template: ..."
```

### Output Parse Errors (Post-Response)

Raised when calling `parse_structured_output()`:

```python nocheck
# Wrong container type
OutputParseError: "Expected object output, got array"

# Missing required fields
OutputParseError: "Required field 'summary' missing in output"

# Extra keys (when not allowed)
OutputParseError: "Unexpected key 'extra_field' in output"

# Type mismatch
OutputParseError: "Field 'count' expected int, got str"
```

### Best Practices

1. **Catch validation errors early**: Run `make check` before committing
2. **Add fallback handling**: Wrap `parse_structured_output()` in try/except for production
3. **Log raw responses**: On `OutputParseError`, log the raw response for debugging
4. **Use strict types**: Leverage pyright to catch type errors before runtime

## Best Practices

### 1. Keep Templates Pure

Prefer stateless, parameter-only sections when possible:

```python
# Good: Pure section
greeting = MarkdownSection(
    title="Greeting",
    key="greeting",
    template="Hello, ${name}!",
)

# Acceptable: Session-bound section (when necessary)
planning = PlanningToolsSection(session=session)
```

### 2. Validate Placeholders Early

Run your tests with complete parameter coverage to catch missing placeholders:

```python
# Test that all placeholders resolve
def test_prompt_renders():
    prompt = Prompt(template).bind(complete_params)
    with prompt.resources:
        rendered = prompt.render()
        assert rendered.text  # Should not raise
```

### 3. Use Hierarchical Keys

Organize section keys hierarchically for clarity:

```python
sections=(
    MarkdownSection(title="Instructions", key="instructions"),
    MarkdownSection(title="Context", key="context"),
    MarkdownSection(title="Background", key="context.background"),  # Nested
    MarkdownSection(title="Constraints", key="context.constraints"),  # Nested
)
```

This makes override paths and debugging much clearer.

### 4. Leverage enabled() for Context

Use dynamic enabling to build one comprehensive template that adapts to context:

```python
template = PromptTemplate(
    ns="agent",
    key="task",
    sections=(
        MarkdownSection(title="Instructions", key="instructions", template="..."),
        MarkdownSection(
            title="Expert Mode",
            key="expert",
            template="...",
            enabled=lambda level: level.value >= ExpertLevel.ADVANCED,
        ),
        MarkdownSection(
            title="Safety Guidelines",
            key="safety",
            template="...",
            enabled=lambda level: level.value <= ExpertLevel.BEGINNER,
        ),
    ),
)
```

### 5. Format Complex Content in Python

Don't try to do loops or conditionals in templates. Compute the string in Python:

```python
# Bad: Trying to loop in template
template = "Tasks: ${tasks}"  # Expects pre-formatted string

# Good: Format in Python
tasks_text = "\n".join(f"{i+1}. {task.name}" for i, task in enumerate(tasks))
params = TaskParams(tasks=tasks_text)
```

### 6. Use Structured Output for Reliability

When you need structured data, declare it explicitly:

```python
# Good: Declared structured output
template = PromptTemplate[TaskResult](...)
response = adapter.evaluate(prompt, session=session)
result: TaskResult = response.output  # Typed and validated

# Bad: Parsing output yourself
template = PromptTemplate(...)  # No output type
response = adapter.evaluate(prompt, session=session)
result = json.loads(response.output)  # Untyped, error-prone
```

### 7. Test with Progressive Disclosure

If you use progressive disclosure, test both the summary and expanded states:

```python
def test_summary_rendering():
    prompt = Prompt(template).bind(params)
    with prompt.resources:
        rendered = prompt.render()
        assert "This section is summarized" in rendered.text

def test_expanded_rendering():
    session[VisibilityOverrides].apply(SetVisibilityOverride(("context",), FULL))
    prompt = Prompt(template).bind(params)
    with prompt.resources:
        rendered = prompt.render()
        assert "This section is summarized" not in rendered.text
```

## Summary

WINK's prompt system provides:

- **Type-safe construction**: Placeholders validated at construction time
- **Composable structure**: Hierarchical sections with deterministic rendering
- **Dynamic scoping**: Enable/disable sections based on runtime context
- **Structured output**: Type-safe parsing with schema validation
- **Resource management**: Co-located dependencies with automatic lifecycle
- **Progressive disclosure**: Minimize tokens while preserving access to detail
- **Override support**: Version and optimize prompts safely ([Chapter 11](11-prompt-optimization.md))

The prompt system is intentionally constrained—simple templating, explicit composition, strict validation. These constraints catch errors early and make prompts predictable, debuggable, and maintainable.

## Next Steps

- **[Chapter 4: Tools](04-tools.md)** - Learn about WINK's sandboxed, deterministic tool execution
- **[Chapter 5: Sessions](05-sessions.md)** - Understand event-driven state management
- **[Chapter 10: Progressive Disclosure](10-progressive-disclosure.md)** - Master cost optimization techniques
- **[Chapter 11: Prompt Optimization](11-prompt-optimization.md)** - Version and A/B test your prompts
- **[Chapter 13: Debugging](13-debugging.md)** - Inspect and troubleshoot prompt rendering

---

**Canonical Reference**: See [specs/PROMPTS.md](/specs/PROMPTS.md) for the complete specification, including resource lifecycle details, transactional tool execution, and advanced patterns.

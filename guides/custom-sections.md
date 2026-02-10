# Creating Custom Section Types

*Canonical spec: [specs/PROMPTS.md](../specs/PROMPTS.md)*

Sections are the building blocks of prompts in WINK. While `MarkdownSection`
handles most cases, you'll sometimes need custom behavior: dynamic content,
session-bound state, tool suites, or specialized rendering logic.

This guide explains when and how to create custom section types.

## When to Create Custom Sections

Use `MarkdownSection` directly when:

- Your content is static or uses simple `${placeholder}` substitution
- Tools can be defined inline without complex initialization
- Progressive disclosure (summary/full) with templates is sufficient
- Resources don't require session-bound lifecycle management

Create a custom section when:

- Content must be computed dynamically from session state at render time
- Tools require session access or complex initialization logic
- You need a custom `clone()` to manage session-bound state
- You want to encapsulate reusable tool suites with their own setup

## The Section Hierarchy

WINK provides a clear inheritance path:

```
Section[ParamsT]          # Abstract base - full control
    ↓
MarkdownSection[ParamsT]  # Template-based - most common
    ↓
Your Tool Suite           # Tools + resources + instructions
```

Choose your base class based on how much default behavior you want.

## Pattern 1: Simple MarkdownSection Subclass

When you need a reusable section with pre-configured settings but no custom
logic, subclass `MarkdownSection`:

```python nocheck
from weakincentives.prompt import MarkdownSection


class CodeReviewInstructionsSection(MarkdownSection[None]):
    """Reusable instructions for code review agents."""

    def __init__(self) -> None:
        super().__init__(
            title="Code Review Guidelines",
            key="code-review-guidelines",
            template="""
            Review code for:
            - Correctness: Does it do what it claims?
            - Clarity: Is the intent obvious?
            - Security: Are there injection or validation issues?

            Be specific. Cite line numbers. Suggest fixes.
            """,
        )
```

This is useful for standardizing sections across multiple prompts without
repeating configuration.

## Pattern 2: Tool Suite Section

Tool suites bundle related tools with their documentation. They're the most
common custom section type.

```python nocheck
from dataclasses import field
from typing import override

from weakincentives.dataclasses import FrozenDataclass
from weakincentives.prompt import MarkdownSection, Tool, ToolContext, ToolResult
from weakincentives.resources import ResourceRegistry
from weakincentives.runtime import Session


# 1. Define parameter and result types
@FrozenDataclass()
class CalculateParams:
    expression: str = field(metadata={"description": "Math expression to evaluate."})


@FrozenDataclass()
class CalculateResult:
    value: float = field(metadata={"description": "Computed result."})
    expression: str = field(metadata={"description": "Original expression."})


# 2. Define the tool handler
def calculate_handler(
    params: CalculateParams, *, context: ToolContext
) -> ToolResult[CalculateResult]:
    """Evaluate a simple math expression."""
    try:
        # In production, use a safe evaluator like asteval
        result = eval(params.expression)  # noqa: S307
        return ToolResult.ok(
            CalculateResult(value=float(result), expression=params.expression),
            message=f"Result: {result}",
        )
    except Exception as e:
        return ToolResult.error(f"Failed to evaluate: {e}")


# 3. Create the tool suite section
class CalculatorSection(MarkdownSection[None]):
    """Calculator tool suite with evaluation capabilities."""

    def __init__(self, *, session: Session) -> None:
        self._session = session

        # Build tools
        calculate_tool = Tool[CalculateParams, CalculateResult](
            name="calculate",
            description="Evaluate a mathematical expression.",
            handler=calculate_handler,
        )

        super().__init__(
            title="Calculator",
            key="calculator",
            template="Use the calculate tool for math operations.",
            tools=(calculate_tool,),
        )

    @property
    def session(self) -> Session:
        return self._session

    @override
    def clone(self, **kwargs: object) -> "CalculatorSection":
        session = kwargs.get("session")
        if not isinstance(session, Session):
            raise TypeError("session keyword argument required")
        return CalculatorSection(session=session)
```

**Key points:**

- Store `session` as a private field for access during tool execution
- Build tools in `__init__` and pass them to the parent via `tools=`
- Override `clone()` to accept a new session when the section is reused

## Pattern 3: Resource-Contributing Section

Sections can contribute resources that tools access via `ToolContext`:

```python nocheck
from typing import Protocol, override

from weakincentives.prompt import MarkdownSection, Tool, ToolContext, ToolResult
from weakincentives.resources import ResourceRegistry
from weakincentives.runtime import Session


# 1. Define a protocol for the resource
class DatabaseConnection(Protocol):
    def query(self, sql: str) -> list[dict]: ...


# 2. Section that contributes the resource
class DatabaseSection(MarkdownSection[None]):
    """Database access with query tool."""

    def __init__(self, *, session: Session, connection: DatabaseConnection) -> None:
        self._session = session
        self._connection = connection

        query_tool = Tool[QueryParams, QueryResult](
            name="query_db",
            description="Execute a read-only SQL query.",
            handler=query_handler,
        )

        super().__init__(
            title="Database",
            key="database",
            template="Query the database using the query_db tool.",
            tools=(query_tool,),
        )

    @override
    def resources(self) -> ResourceRegistry:
        """Contribute the database connection to the resource registry."""
        return ResourceRegistry.build({DatabaseConnection: self._connection})

    @override
    def clone(self, **kwargs: object) -> "DatabaseSection":
        session = kwargs.get("session")
        if not isinstance(session, Session):
            raise TypeError("session keyword argument required")
        return DatabaseSection(session=session, connection=self._connection)


# 3. Tool handler accesses the resource via context
def query_handler(
    params: QueryParams, *, context: ToolContext
) -> ToolResult[QueryResult]:
    db = context.resources.get(DatabaseConnection)
    results = db.query(params.sql)
    return ToolResult.ok(QueryResult(rows=results))
```

The `resources()` method is called during prompt rendering to collect all
resources contributed by sections. Tools then access these via
`context.resources.get(Protocol)`.

## Pattern 4: Dynamic Content Section

When content depends on session state or must be computed at render time,
override `render_body()`:

```python nocheck
from typing import override

from weakincentives.prompt import MarkdownSection, SectionVisibility
from weakincentives.runtime import Session


class StatusSection(MarkdownSection[None]):
    """Display current session status dynamically."""

    def __init__(self, *, session: Session) -> None:
        self._session = session
        super().__init__(
            title="Current Status",
            key="status",
            template="",  # Not used - we override render_body
        )

    @override
    def render_body(
        self,
        params: object,
        *,
        visibility: SectionVisibility | None = None,
        path: tuple[str, ...] = (),
        session: object = None,
    ) -> str:
        # Pull state from session and format dynamically
        plan = self._session[Plan].latest()
        if plan is None:
            return "No active plan."

        lines = [f"**Objective:** {plan.objective}"]
        lines.append(f"**Status:** {plan.status}")
        if plan.steps:
            lines.append("\n**Steps:**")
            for step in plan.steps:
                lines.append(f"- [{step.status}] {step.title}")
        return "\n".join(lines)

    @override
    def clone(self, **kwargs: object) -> "StatusSection":
        session = kwargs.get("session")
        if not isinstance(session, Session):
            raise TypeError("session keyword argument required")
        return StatusSection(session=session)
```

## Pattern 5: Extending Section Directly

For complete control over rendering, extend `Section` instead of
`MarkdownSection`. This is rare but useful when template substitution isn't
appropriate:

```python nocheck
from typing import Self, override

from weakincentives.prompt import Section, SectionVisibility
from weakincentives.runtime import Session


class DigestSection(Section[None]):
    """Render cached digest content from session state."""

    def __init__(self, *, session: Session) -> None:
        self._session = session
        super().__init__(
            title="Workspace Digest",
            key="workspace-digest",
            summary="Cached workspace content available.",
            visibility=SectionVisibility.SUMMARY,
        )

    @override
    def render_body(
        self,
        params: object,
        *,
        visibility: SectionVisibility | None = None,
        path: tuple[str, ...] = (),
        session: object = None,
    ) -> str:
        if visibility == SectionVisibility.SUMMARY:
            return self.summary or ""

        digest = self._session[WorkspaceDigest].latest()
        if digest is None:
            return "No workspace digest cached."
        return digest.content

    @override
    def clone(self, **kwargs: object) -> Self:
        session = kwargs.get("session")
        if not isinstance(session, Session):
            raise TypeError("session keyword argument required")
        return type(self)(session=session)
```

## The Clone Contract

Every section must implement `clone(**kwargs)`. This is critical because
sections are reused across sessions, and session-bound state must not leak.

**Rules for clone():**

1. Accept `**kwargs` and extract `session` when your section is session-bound
1. Recursively clone children: `child.clone(**kwargs)`
1. Clone `default_params` if set: `clone_dataclass(self.default_params)`
1. Preserve configuration that should persist (database connections, configs)
1. Return a fresh instance bound to the new session

```python nocheck
@override
def clone(self, **kwargs: object) -> "MyToolSection":
    session = kwargs.get("session")
    if not isinstance(session, Session):
        raise TypeError("session keyword argument required")

    # Clone children
    cloned_children = [child.clone(**kwargs) for child in self.children]

    return MyToolSection(
        session=session,
        config=self._config,  # Preserve immutable config
        children=cloned_children,
    )
```

## Progressive Disclosure

Sections support showing summaries until expanded. Configure this with
`summary` and `visibility`:

```python nocheck
MarkdownSection(
    title="API Reference",
    key="api-reference",
    template="""
    ## Endpoints

    GET /users - List all users
    POST /users - Create a user
    GET /users/{id} - Get user by ID
    ...extensive documentation...
    """,
    summary="API documentation available. Use read_section to view.",
    visibility=SectionVisibility.SUMMARY,
)
```

When `visibility=SUMMARY`, the section renders its `summary` instead of the
full template. The model can expand it using `open_sections` or `read_section`
tools (provided by WINK's progressive disclosure system).

For dynamic visibility based on session state:

```python nocheck
def visibility_selector(*, session: Session) -> SectionVisibility:
    """Show full content if digest exists, otherwise summary."""
    digest = session[WorkspaceDigest].latest()
    return SectionVisibility.FULL if digest else SectionVisibility.SUMMARY


MarkdownSection(
    title="Digest",
    key="digest",
    template="...",
    summary="...",
    visibility=visibility_selector,
)
```

## Conditional Rendering

Disable sections at render time with `enabled`:

```python nocheck
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class FeatureFlags:
    debug_mode: bool


def debug_enabled(flags: FeatureFlags, *, session: Session) -> bool:
    del session
    return flags.debug_mode


MarkdownSection(
    title="Debug Tools",
    key="debug-tools",
    template="Verbose debugging instructions...",
    tools=(debug_tool,),
    enabled=debug_enabled,
)
```

When disabled:

- The section doesn't render
- Its tools are unavailable to the model
- Children are also skipped

## Tool Policies

Attach policies to enforce constraints on tool usage:

```python nocheck
from weakincentives.contrib.tools import ReadBeforeWritePolicy


class FilesystemSection(MarkdownSection[None]):
    def __init__(self, *, session: Session) -> None:
        super().__init__(
            title="Filesystem",
            key="filesystem",
            template="...",
            tools=(read_file, write_file, edit_file),
            policies=(ReadBeforeWritePolicy(),),
        )
```

Policies are evaluated before tool execution and can reject calls that violate
constraints.

## Testing Custom Sections

Test sections without a model by calling methods directly:

```python nocheck
def test_status_section_renders_plan():
    session = Session()
    session.install(Plan, initial=lambda: Plan(objective="Test", status="active"))
    session.dispatch(SetupPlan(objective="Build feature", initial_steps=("Step 1",)))

    section = StatusSection(session=session)
    body = section.render_body(None)

    assert "Build feature" in body
    assert "Step 1" in body


def test_section_clone_preserves_config():
    session1 = Session()
    section1 = MyToolSection(session=session1, config=Config(timeout=30))

    session2 = Session()
    section2 = section1.clone(session=session2)

    assert section2.session is session2
    assert section2._config.timeout == 30
```

## Summary

| Pattern | Base Class | Use Case |
| --- | --- | --- |
| Reusable config | `MarkdownSection` | Standardized sections across prompts |
| Tool suite | `MarkdownSection` | Bundle tools + docs + resources |
| Resource provider | `MarkdownSection` | Inject dependencies for tools |
| Dynamic content | `MarkdownSection` | Render from session state |
| Full control | `Section` | Non-template rendering logic |

Start with `MarkdownSection` and only drop down to `Section` when you need
complete rendering control. Always implement `clone()` correctly for
session-bound sections.

## Next Steps

- [Prompts](prompts.md): Understand the prompt system sections live in
- [Tools](tools.md): Define tool contracts and handlers
- [Sessions](sessions.md): Manage state with reducers
- [Progressive Disclosure](progressive-disclosure.md): Control context size
- [Workspace Tools](workspace-tools.md): See VFS and planning tool suites

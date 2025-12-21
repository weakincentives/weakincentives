# Progressive Examples

This directory contains a series of examples that build on each other,
introducing WINK concepts incrementally. Start with `01_minimal_prompt.py`
and work your way up.

## Overview

| Example | Lines | Concepts Introduced |
|---------|-------|---------------------|
| `01_minimal_prompt.py` | ~110 | PromptTemplate, MarkdownSection, structured output, OpenAIAdapter |
| `02_with_tools.py` | ~180 | Tool, ToolHandler, ToolResult, tool parameters |
| `03_with_session.py` | ~170 | Session, PlanningToolsSection, multi-turn state |
| `04_with_workspace.py` | ~220 | VfsToolsSection, HostMount, WorkspaceDigestSection |
| `05_full_agent.py` | ~340 | MainLoop, EventBus, progressive disclosure, REPL |

## Running the Examples

Each example requires an OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...

# Run any example directly
uv run python examples/progressive/01_minimal_prompt.py
uv run python examples/progressive/02_with_tools.py
# ... etc
```

Optionally set a specific model:

```bash
export OPENAI_MODEL=gpt-4o  # Default is gpt-4o-mini
```

## Example Progression

### 01: Minimal Prompt

The simplest possible WINK example. Shows:
- Defining structured output as a frozen dataclass
- Creating a PromptTemplate with sections
- Binding parameters and evaluating against an adapter
- Receiving typed, validated output

```python
# Define output schema
@dataclass(slots=True, frozen=True)
class PRDescription:
    title: str
    summary: str
    changes: list[str]

# Create template
template = PromptTemplate[PRDescription](
    ns="examples", key="pr", sections=[...]
)

# Evaluate
response = adapter.evaluate(Prompt(template).bind(params), session=session)
print(response.output.title)  # Typed access
```

### 02: With Tools

Adds callable tools that the LLM can invoke. Shows:
- Defining tool parameters and results as dataclasses
- Creating tool handlers with the `(params, *, context)` signature
- Attaching tools to sections
- The LLM calling tools and receiving results

```python
def word_count_handler(
    params: WordCountParams, *, context: ToolContext
) -> ToolResult[WordCountResult]:
    return ToolResult(message="...", value=result, success=True)

word_count_tool = Tool[WordCountParams, WordCountResult](
    name="word_count",
    description="Count words in text.",
    handler=word_count_handler,
)
```

### 03: With Session

Introduces Redux-style session state that persists across turns. Shows:
- Creating a Session that lives across multiple evaluations
- PlanningToolsSection for structured task planning
- The LLM creating and updating plans
- Querying session state with `session[Plan].latest()`

```python
session = Session()
template = build_template(session)  # Pass session to sections

# Multiple turns, same session
for message in conversation:
    prompt = Prompt(template).bind(Params(message=message))
    adapter.evaluate(prompt, session=session)

# State persists
plan = session[Plan].latest()
```

### 04: With Workspace

Adds a virtual filesystem for safe file exploration. Shows:
- VfsToolsSection exposing ls, read, write, glob, grep tools
- HostMount to mirror local directories (read-only)
- WorkspaceDigestSection for auto-generated summaries
- Combining planning and file exploration

```python
VfsToolsSection(
    session=session,
    mounts=(
        HostMount(
            host_path=str(project_dir),
            mount_path=VfsPath(("project",)),
            include_glob=("*.py", "*.md"),
        ),
    ),
    allowed_host_roots=(str(project_dir.parent),),
)
```

### 05: Full Agent

The complete pattern with MainLoop orchestration. Shows:
- MainLoop for standardized request/response handling
- EventBus for observability (logging renders, tools, tokens)
- Progressive disclosure (sections start summarized)
- Deadline enforcement
- Interactive REPL

```python
class AssistantLoop(MainLoop[UserRequest, AssistantResponse]):
    def create_prompt(self, request: UserRequest) -> Prompt[AssistantResponse]:
        return Prompt(self._template).bind(request)

    def create_session(self) -> Session:
        return self._session  # Persistent across turns

# Execute
loop.execute(UserRequest(request="..."), deadline=deadline)
```

## Next Steps

After completing these examples, see:
- `code_reviewer_example.py` - Full production-style agent
- `guides/code-review-agent.md` - Detailed walkthrough
- `specs/` - Design specifications for each component

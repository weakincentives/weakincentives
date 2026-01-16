# Quickstart

This guide gets you from zero to a working agent as quickly as possible. You'll
learn the essential patterns without getting bogged down in details.

## Install

The core package has no mandatory third-party dependencies. This is intentional:
you should be able to use WINK's prompt and session primitives without pulling
in OpenAI or any other provider SDK.

```bash
pip install weakincentives
```

Provider and tool extras (pick what you need):

```bash
pip install "weakincentives[openai]"           # OpenAI adapter
pip install "weakincentives[litellm]"          # LiteLLM adapter
pip install "weakincentives[claude-agent-sdk]" # Claude Agent SDK adapter

pip install "weakincentives[asteval]"          # Safe Python expression eval tool
pip install "weakincentives[podman]"           # Podman sandbox tools
pip install "weakincentives[wink]"             # Debug UI (wink CLI)
```

**Python requirement**: 3.12+. We use modern Python features liberally and don't
maintain compatibility with older versions.

## Your First Structured Agent

This example is intentionally small but complete. It demonstrates the four
essential concepts:

- **Typed params**: Input data as frozen dataclasses
- **Structured output**: Return type as a frozen dataclass
- **Prompt template**: Deterministic prompt structure
- **Session**: Recording of what happened

```python
from dataclasses import dataclass

from weakincentives.prompt import Prompt, PromptTemplate, MarkdownSection
from weakincentives.runtime import InProcessDispatcher, Session

@dataclass(slots=True, frozen=True)
class SummarizeRequest:
    text: str

@dataclass(slots=True, frozen=True)
class Summary:
    title: str
    bullets: tuple[str, ...]

template = PromptTemplate[Summary](
    ns="docs",
    key="summarize",
    name="Doc summarizer",
    sections=(
        MarkdownSection(
            title="Task",
            key="task",
            template=(
                "Summarize the input.\n\n"
                "Return JSON with:\n"
                "- title: short title\n"
                "- bullets: 3-7 bullet points\n"
            ),
        ),
        MarkdownSection(
            title="Input",
            key="input",
            template="${text}",
        ),
    ),
)

prompt = Prompt(template).bind(SummarizeRequest(
    text="WINK is a Python library for building agents. It treats prompts as "
         "typed programs. Tools are explicit. State is inspectable."
))

dispatcher = InProcessDispatcher()
session = Session(dispatcher=dispatcher)

# To actually run this, you need an adapter and API key:
#
# from weakincentives.adapters.openai import OpenAIAdapter
# adapter = OpenAIAdapter(model="gpt-4o-mini")
# response = adapter.evaluate(prompt, session=session)
# print(response.output)
# # -> Summary(title='WINK Overview', bullets=('Python library...', ...))
```

**Key ideas:**

- Binding is by dataclass type. `bind(SummarizeRequest(...))` sets `${text}`.
- `PromptTemplate[Summary]` declares structured output. Adapters parse the
  model's JSON response into your dataclass automatically.
- The `slots=True, frozen=True` pattern is used everywhere. Immutable
  dataclasses prevent accidental mutation and work well with the session's
  event-driven model.

## Adding a Tool

Tools are registered by sections. Handlers receive immutable `ToolContext`
(including session and resources) and return a `ToolResult`.

Here's a toy tool that returns the current time. In a real agent this would be a
filesystem read, an API call, or some other operation with side effects.

```python
from dataclasses import dataclass
from datetime import UTC, datetime

from weakincentives.prompt import Tool, ToolContext, ToolResult, MarkdownSection, ToolExample

@dataclass(slots=True, frozen=True)
class NowParams:
    tz: str = "UTC"

@dataclass(slots=True, frozen=True)
class NowResult:
    iso: str

    def render(self) -> str:
        return self.iso

def now_handler(params: NowParams, *, context: ToolContext) -> ToolResult[NowResult]:
    del context  # not used here
    if params.tz != "UTC":
        return ToolResult.error("Only UTC supported in this demo.")
    return ToolResult.ok(
        NowResult(iso=datetime.now(UTC).isoformat()),
        message="Current time.",
    )

now_tool = Tool[NowParams, NowResult](
    name="now",
    description="Return the current time (UTC).",
    handler=now_handler,
    examples=(
        ToolExample(
            description="Get UTC time",
            input=NowParams(tz="UTC"),
            output=NowResult(iso="2025-01-01T00:00:00+00:00"),
        ),
    ),
)

tools_section = MarkdownSection(
    title="Tools",
    key="tools",
    template="You may call tools when needed.",
    tools=(now_tool,),
)
```

The pattern to notice: **tools and their documentation live together**. The
section says "You may call tools when needed" and provides the `now` tool. The
model sees both in the same context.

## A Complete, Copy-Paste Ready Agent

Here's a minimal but complete agent you can run. It answers questions about a
topic using a search tool.

```python
"""Minimal WINK agent: a topic Q&A bot with a mock search tool."""
import os
from dataclasses import dataclass
from weakincentives.prompt import (
    Prompt, PromptTemplate, MarkdownSection, Tool, ToolContext, ToolResult
)
from weakincentives.runtime import Session
from weakincentives.adapters.openai import OpenAIAdapter

# 1. Define structured output
@dataclass(slots=True, frozen=True)
class Answer:
    summary: str
    sources: tuple[str, ...]

# 2. Define a tool
@dataclass(slots=True, frozen=True)
class SearchParams:
    query: str

@dataclass(slots=True, frozen=True)
class SearchResult:
    snippets: tuple[str, ...]
    def render(self) -> str:
        return "\n".join(f"- {s}" for s in self.snippets)

def search_handler(params: SearchParams, *, context: ToolContext) -> ToolResult[SearchResult]:
    # In a real agent, this would call a search API
    del context
    return ToolResult.ok(
        SearchResult(snippets=(
            f"Result 1 about {params.query}",
            f"Result 2 about {params.query}",
        )),
        message=f"Found results for: {params.query}",
    )

search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search for information about a topic.",
    handler=search_handler,
)

# 3. Define params
@dataclass(slots=True, frozen=True)
class QuestionParams:
    question: str

# 4. Build the prompt template
template = PromptTemplate[Answer](
    ns="demo",
    key="qa-agent",
    sections=(
        MarkdownSection(
            title="Instructions",
            key="instructions",
            template=(
                "You are a helpful research assistant.\n\n"
                "Use the search tool to find information, then answer the "
                "question with a summary and list of sources."
            ),
            tools=(search_tool,),
        ),
        MarkdownSection(
            title="Question",
            key="question",
            template="${question}",
        ),
    ),
)

# 5. Run the agent
def main():
    session = Session()
    adapter = OpenAIAdapter(model="gpt-4o-mini")

    prompt = Prompt(template).bind(QuestionParams(
        question="What is the capital of France?"
    ))

    response = adapter.evaluate(prompt, session=session)

    if response.output is not None:
        print(f"Summary: {response.output.summary}")
        print(f"Sources: {response.output.sources}")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example")
    else:
        main()
```

**What's happening:**

1. `PromptTemplate[Answer]` declares that the model must return JSON matching
   the `Answer` dataclass
1. The `search` tool is registered on the Instructions section—the model sees
   the tool alongside the instructions for using it
1. `adapter.evaluate()` sends the prompt to OpenAI, executes any tool calls, and
   parses the structured response
1. You get back `response.output` as a typed `Answer` instance

This is the core loop. Everything else in WINK builds on this: sessions for
state, sections for organization, progressive disclosure for token management.

## What's Next

Now that you have a working agent, you can:

- [Prompts](prompts.md): Learn how prompt templates and sections work in depth
- [Tools](tools.md): Understand tool contracts, context, and policies
- [Sessions](sessions.md): Add state management with reducers
- [Code Review Agent](code-review-agent.md): See a full-featured example

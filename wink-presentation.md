# WINK

## Weak Incentives for Agent Systems

*A Python library for building typed, testable agents*

---

# The Problem with Agent Systems

- Prompts as strings drift from tools and schemas
- State management is ad-hoc "magic dicts"
- Side effects happen everywhere
- Debugging is archaeology
- Provider lock-in

---

# What "Weak Incentives" Means

Shape the prompt, tools, and context so the model's **easiest path is the correct one**

- Clear instructions co-located with tools
- Typed contracts guide valid outputs
- Progressive disclosure keeps models focused
- Explicit state provides needed context

*Not about constraining—about encouraging correct behavior through structure*

---

# The Core Bet

**Prompts are first-class, typed programs—not strings**

```python
template = PromptTemplate(
    ns="support",
    key="faq",
    sections=(
        MarkdownSection(
            title="Question",
            key="question",
            template="Question: ${question}",
            tools=(search_tool,),
        ),
    ),
)
```

---

# Why Typed Prompts Matter

1. **Deterministic inspection** — same inputs, same outputs
2. **Safe iteration** — hash-validated overrides prevent drift
3. **Validation at construction** — typos caught early
4. **Tools co-located with instructions** — they can't drift apart

---

# Three Core Abstractions

| Abstraction | Purpose |
|-------------|---------|
| **PromptTemplate** | Immutable definition (sections, tools) |
| **Session** | Event-driven state (reducers, snapshots) |
| **Tool** | Typed contract + handler |

---

# PromptTemplate

The immutable blueprint

```python
@dataclass(slots=True, frozen=True)
class Params:
    question: str

template = PromptTemplate[Answer](
    ns="qa", key="basic",
    sections=(
        MarkdownSection(
            title="Instruction",
            key="instruction",
            template="Answer: ${question}",
        ),
    ),
)
```

---

# Sessions

**Event-driven, reducer-managed state**

```python
# Queries are read-only
facts = session[Fact].all()
latest = session[Fact].latest()
filtered = session[Fact].where(lambda f: f.key == "x")

# Mutations flow through dispatch
session.dispatch(AddStep(step="read README"))
```

*Everything is recorded. Snapshots are trivial.*

---

# Sessions: Reducers

Pure functions: state + event → new state

```python
@dataclass(slots=True, frozen=True)
class AgentPlan:
    steps: tuple[str, ...]

    @reducer(on=AddStep)
    def add_step(self, event: AddStep) -> Replace["AgentPlan"]:
        return Replace((
            replace(self, steps=(*self.steps, event.step)),
        ))
```

---

# Tools

**The only place where side effects happen**

```python
def handler(
    params: MyParams,
    *,
    context: ToolContext
) -> ToolResult[MyResult]:
    # Side effects here
    return ToolResult.ok(
        MyResult(answer="42"),
        message="ok"
    )
```

---

# Tool Policies

Declarative constraints on tool usage

```python
# Require 'test' and 'build' before 'deploy'
deploy_policy = SequentialDependencyPolicy(
    dependencies={
        "deploy": frozenset({"test", "build"}),
    }
)

# Require reading before overwriting
read_first = ReadBeforeWritePolicy()
```

---

# Transactional Tool Execution

1. **Snapshot** — capture state before tool runs
2. **Execute** — run the handler
3. **Commit or Rollback** — restore on failure

*Failed tools never leave inconsistent state*

---

# Provider Adapters

Same agent, different providers

| Adapter | Provider |
|---------|----------|
| `OpenAIAdapter` | OpenAI API |
| `LiteLLMAdapter` | 100+ providers |
| `ClaudeAgentSdkAdapter` | Claude Code native |

```python
adapter = OpenAIAdapter(model="gpt-4o")
response = adapter.evaluate(prompt, session=session)
```

---

# Technical Strategy

1. **Don't compete at the model layer** — models commoditize
2. **Differentiate with your system of record** — domain knowledge
3. **Keep product semantics out of prompts** — encode in tools
4. **Use provider runtimes; own the tools**
5. **Build evaluation as your control plane**

---

# Policies Over Workflows

| Aspect | Workflow | Policy |
|--------|----------|--------|
| Specifies | Steps to execute | Constraints to satisfy |
| On unexpected | Fails or branches | Agent reasons |
| Composability | Sequential coupling | Independent conjunction |
| Agent role | Executor | Reasoner |

*Prefer declarative policies over prescriptive workflows*

---

# Progressive Disclosure

Control context size with summaries

```python
rules_section = MarkdownSection(
    title="Game Rules",
    key="rules",
    template=FULL_RULES,
    disclosure=collapsed(
        summary="Rules available on request."
    ),
)
```

*Let the model request what it needs*

---

# Debugging

Sessions record everything

```bash
uv run wink query debug_bundles/*.zip
```

- Prompts sent
- Tool calls made
- Responses received
- State at every point

---

# What WINK Is

- A Python library for prompts-as-agents
- A runtime for state (Session) and orchestration (AgentLoop)
- Adapters (OpenAI, LiteLLM, Claude Agent SDK)
- Contributed tool suites (VFS, Podman, planning)

---

# What WINK Is Not

- A distributed workflow engine
- A framework that "owns" your architecture
- A multi-agent coordination system
- An async-first streaming framework

*WINK plays well with others*

---

# Getting Started

```bash
git clone https://github.com/weakincentives/starter.git
cd starter
make install
make redis
export ANTHROPIC_API_KEY=your-key
make agent
```

```bash
make dispatch QUESTION="What is the secret number?"
# Returns: 42
```

---

# Learn More

| Resource | Description |
|----------|-------------|
| **guides/** | How-to guides |
| **specs/** | Design specifications |
| **llms.md** | Full API reference |
| **starter** | Working example agent |

---

# Thank You

**WINK — Weak Incentives**

*Shape the structure. Trust the model.*

`pip install weakincentives`

https://github.com/weakincentives/weakincentives

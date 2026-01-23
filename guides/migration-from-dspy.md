# Coming from DSPy?

If you've built LLM programs with DSPy, here's how WINK compares.

## Different Bets on Where Value Lives

**DSPy** centers on **automatic optimization**: you declare input/output
signatures, compose modules, and let optimizers compile better prompts and
few-shot examples. The framework treats prompts as implementation details that
should be generated, not written.

**WINK** centers on **explicit, inspectable prompts**: you write prompts as
typed section trees, control exactly what the model sees, and iterate via
version-controlled overrides. The framework treats prompts as first-class
artifacts that should be readable, testable, and auditable.

Both approaches have merit. DSPy shines when you have good metrics and want to
automate prompt tuning. WINK shines when you need to understand exactly what's
being sent to the model and why.

## Concept Mapping

| DSPy | WINK |
| --- | --- |
| Signature | Structured output dataclass + `PromptTemplate` |
| Module (`Predict`, `ChainOfThought`) | `Section` (instructions + tools) |
| Program (composed modules) | `PromptTemplate` (tree of sections) |
| Optimizer / Teleprompter | Prompt overrides + manual iteration |
| Compilation | No equivalent (prompts are explicit) |
| `dspy.ReAct` | `PlanningToolsSection` + tool sections |
| Metric | Evaluation framework |
| Trace | Session events + debug UI |

## What's Familiar

**Typed inputs and outputs.** DSPy signatures declare input/output fields; WINK
uses frozen dataclasses for the same purpose. Both catch type mismatches early.

**Composition.** DSPy composes modules into programs; WINK composes sections
into prompt templates. Both encourage modular, reusable components.

**Tool use.** DSPy modules like `ReAct` handle tool calling; WINK sections
register tools alongside their instructions.

## What's Different

**Prompts are visible.** In DSPy, prompts are generated artifacts—you don't
typically read or edit them directly. In WINK, `prompt.render()` returns the
exact markdown sent to the model. You can inspect, test, and version it.

**No automatic optimization.** DSPy's optimizers (BootstrapFewShot, MIPROv2,
etc.) generate prompts automatically. WINK uses hash-validated overrides for
manual iteration. You can build optimization workflows on top, but the framework
doesn't assume you want automated prompt generation.

**State is explicit.** DSPy traces execution but doesn't expose a structured
state model. WINK sessions are typed, reducer-managed state containers. Every
state change is an event you can query, snapshot, and restore.

**Tools and instructions are co-located.** In DSPy, tool definitions are
separate from module logic. In WINK, the section that explains "use this tool
for X" is the same section that registers the tool.

**Deterministic by default.** WINK prompt rendering is pure—same inputs produce
same outputs. You can write tests that assert on exact prompt text. DSPy's
compiled prompts depend on optimizer state and training data.

## WINK Strengths in Depth

Beyond the philosophical differences, WINK offers several concrete capabilities
that address common pain points in agent development:

### Transactional Tool Execution

Every tool call is wrapped in a transaction. If a tool fails, WINK automatically
rolls back session state and filesystem changes to their pre-call state. No
corrupted state, no defensive rollback code in every handler.

```python nocheck
# Tool fails halfway through? State is automatically restored.
# No need for try/except blocks with manual cleanup.
def risky_handler(params: Params, *, context: ToolContext) -> ToolResult[R]:
    # If this raises, all state changes are rolled back
    return ToolResult.ok(result)
```

### Visual Debug UI

When something goes wrong, `wink debug` opens a local server that renders the
full prompt/tool timeline. See exactly what was sent to the model, what tools
were called, what they returned, and how state evolved.

```bash
wink debug debug_bundles/session.jsonl --port 8000
```

This is how you answer "why did the agent do X?"—not by adding print statements,
but by inspecting the full execution trace.

### Tool Policies

Declarative constraints that govern when tools can be invoked. Instead of
embedding validation logic in handlers, express constraints compositionally:

```python nocheck
from weakincentives.prompt import SequentialDependencyPolicy, ReadBeforeWritePolicy

# Require 'test' before 'deploy'
deploy_policy = SequentialDependencyPolicy(
    dependencies={"deploy": frozenset({"test"})}
)

# Require reading a file before overwriting it
read_first = ReadBeforeWritePolicy()
```

When a tool call violates a policy, WINK returns an error without executing the
handler. The model learns to follow the constraints.

### Budget and Deadline Controls

Prevent runaway agents with explicit resource limits:

```python nocheck
from weakincentives import Budget, Deadline
from datetime import timedelta

response = adapter.evaluate(
    prompt,
    session=session,
    deadline=Deadline.from_timeout(timedelta(minutes=5)),
    budget=Budget(max_total_tokens=20_000),
)
```

Token budgets and wall-clock deadlines are checked at key points throughout
execution. When limits are exceeded, you get a clear exception.

### Built-in Evaluation Framework

EvalLoop wraps your MainLoop—same prompt templates, same tools, same adapters.
Run evaluations against production code, not a separate test harness:

```python nocheck
from weakincentives.evals import EvalLoop, exact_match, tool_called, all_of

# Combine output evaluation with behavioral assertions
evaluator = all_of(
    exact_match,                      # Output must match expected
    tool_called("search"),            # Agent must have used search
    token_usage_under(max_tokens=5000),  # Stay within budget
)

eval_loop = EvalLoop(loop=main_loop, evaluator=evaluator, requests=mailbox)
```

Session evaluators let you assert not just *what* the agent produced, but *how*
it got there—tool usage patterns, token consumption, state invariants.

### Progressive Disclosure

Sections can default to summaries and expand on demand. Instead of stuffing
everything into the prompt upfront, let the model request what it needs:

```python nocheck
section = MarkdownSection(
    title="Reference Documentation",
    key="docs",
    template=full_documentation,
    summary="Reference docs available. Use read_docs tool for details.",
)
```

This keeps initial token counts low while giving the agent access to full
context when needed.

### Sandboxed Execution

Out-of-the-box support for safe file operations:

- **VFS tools**: In-memory virtual filesystem tracked as session state
- **Podman sandbox**: OS-level isolation with network policies
- **Claude Agent SDK adapter**: Native sandboxing via bubblewrap/seatbelt

Mount host directories read-only; the sandbox prevents accidental writes.

### Provider Portability

Adapters make switching between providers straightforward. The same prompt
template works with OpenAI, LiteLLM (for Bedrock, Vertex, etc.), or Claude:

```python nocheck
# Same prompt, different providers
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.adapters.litellm import LiteLLMAdapter
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter

# Pick one—your agent definition stays the same
adapter = OpenAIAdapter(model="gpt-4o")
adapter = LiteLLMAdapter(model="bedrock/anthropic.claude-3-sonnet")
adapter = ClaudeAgentSDKAdapter(model="claude-sonnet-4-5-20250929")
```

### Type Safety Everywhere

Pyright strict mode is enforced. Params, tool calls, tool results, structured
outputs, session state—all typed with dataclasses. Type mismatches surface at
construction time, not when the model is mid-response.

## When to Use WINK Instead of DSPy

**Inspectability matters:**

- You need to see exactly what prompts are being sent
- You want a visual debug UI to trace agent behavior
- Auditability is required (compliance, debugging, post-mortems)

**Reliability matters:**

- You need transactional tool execution with automatic rollback
- You want declarative policies (read-before-write, sequential dependencies)
- You need budget and deadline controls to prevent runaway agents

**Production concerns matter:**

- You're building tool-heavy agents where prompt/tool co-location prevents drift
- You need sandboxed execution (VFS, Podman, or OS-level isolation)
- You want evaluation to run against production code, not a separate harness
- Provider portability is important (OpenAI today, Bedrock tomorrow)

**Developer experience matters:**

- You value determinism and testability
- You want strict type safety (Pyright strict mode enforced)
- You prefer explicit iteration over black-box optimization

## When to Stick with DSPy

- You have good metrics and want automated prompt optimization
- You're doing research where prompt generation is part of the experiment
- You want to bootstrap few-shot examples automatically
- You prefer declaring intent (signatures) over writing prompts

## Migration Path

If you're moving from DSPy to WINK:

**1. Convert signatures to dataclasses:**

```python nocheck
# DSPy
class QA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

# WINK
@dataclass(slots=True, frozen=True)
class QuestionParams:
    question: str

@dataclass(slots=True, frozen=True)
class Answer:
    answer: str
```

**2. Convert modules to sections:**

```python nocheck
# DSPy
qa = dspy.ChainOfThought(QA)

# WINK
qa_section = MarkdownSection(
    title="Question Answering",
    key="qa",
    template="Think step by step, then answer the question.\n\nQuestion: ${question}",
)
```

**3. Convert programs to templates:**

```python nocheck
template = PromptTemplate[Answer](
    ns="qa",
    key="chain-of-thought",
    sections=(qa_section,),
)
```

**4. Replace optimizers with overrides:**

```python nocheck
# Instead of compiled prompts, use the override system
prompt = Prompt(template, overrides_store=store, overrides_tag="v2")
```

**5. Add tools explicitly:**

DSPy's `ReAct` handles tool use implicitly; WINK requires explicit tool
registration on sections. This is more verbose but makes tool availability
obvious from the prompt structure.

## The Key Mindset Shift

DSPy optimizes prompts for you; WINK gives you tools to write and iterate on
prompts yourself—with guardrails that make production deployment safer.

If you've been frustrated by not knowing what DSPy is actually sending to the
model, WINK's explicit approach will feel liberating. Every prompt is
inspectable. Every tool call is traceable. Every state change is auditable.

If you've struggled with partial failures corrupting agent state, WINK's
transactional tool execution eliminates that class of bugs entirely.

If you've deployed agents that ran up token bills or timed out ungracefully,
WINK's budget and deadline controls give you explicit resource boundaries.

If you've relied heavily on DSPy's optimizers, you'll need to build or adopt
optimization workflows separately. But you'll gain the ability to understand
exactly what your agent is doing—and why—at every step.

## Next Steps

- [Philosophy](philosophy.md): Understand the WINK approach
- [Prompts](prompts.md): Learn how prompt templates work
- [Evaluation](evaluation.md): Build your own evaluation pipeline

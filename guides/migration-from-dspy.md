# Coming from DSPy?

If you've built LLM programs with DSPy, here's how WINK compares—and why the
differences matter for production agent systems.

## The Core Philosophical Difference

**DSPy** bets on **compilation**: declare signatures, compose modules, let
optimizers generate prompts. The framework treats prompts as implementation
details that should be automatically derived from metrics and training data.

**WINK** bets on **policies over workflows**: give the agent maximum degrees of
freedom, then add minimal constraints where they matter. The framework treats
prompts as explicit, inspectable artifacts and uses declarative policies—not
rigid workflows—to guide behavior.

This isn't a surface-level distinction. It reflects fundamentally different
views on how to build reliable agent systems.

## Policies, Not Workflows

Most agent frameworks assume you need to choreograph behavior: define states,
transitions, routers, and branching logic. They spend their complexity budget on
orchestration—deciding which prompts to run when.

This approach has a problem: **workflows fight the model's natural reasoning.**
When you force an agent through a rigid state machine, you're constraining
exactly what makes LLMs powerful—their ability to reason flexibly about novel
situations.

WINK takes the opposite stance: **err on the side of giving the agent more
freedom.** Let it reason, plan, and adapt. Only constrain where constraints
actually matter for safety or correctness.

Policies are how you express those constraints:

```python nocheck
from weakincentives.prompt import SequentialDependencyPolicy, ReadBeforeWritePolicy

# The agent can call tools freely, but:
# - Must run tests before deploying
# - Must read a file before overwriting it

policies = (
    SequentialDependencyPolicy(dependencies={"deploy": frozenset({"test"})}),
    ReadBeforeWritePolicy(),
)
```

When the agent tries to violate a policy, WINK returns an error—the agent learns
the constraint exists and adapts. But within those bounds, it has full autonomy.
No state machine telling it what step comes next. No router deciding which
"agent" handles the request.

**This is the key insight:** modern LLMs don't need elaborate orchestration.
They need good context, clear tools, and minimal guardrails. Policies give you
the guardrails without the choreography.

Compare this to DSPy's module composition. In DSPy, you explicitly wire modules
together: `ChainOfThought` feeds into `Retrieve` feeds into `Predict`. The
structure is baked into code. In WINK, you define tools, write instructions, set
policies—and let the model figure out how to sequence operations.

## Concept Mapping

| DSPy | WINK |
| --- | --- |
| Signature | Structured output dataclass + `PromptTemplate` |
| Module (`Predict`, `ChainOfThought`) | `Section` (instructions + tools) |
| Program (composed modules) | `PromptTemplate` (tree of sections) |
| Module composition / wiring | Policies (declarative constraints) |
| Optimizer / Teleprompter | Prompt overrides + manual iteration |
| Compilation | No equivalent (prompts are explicit) |
| `dspy.ReAct` | Tools + policies (agent chooses its own path) |
| Metric | Evaluation framework |
| Trace | Session events + debug UI |

## What You Gain by Switching

### Full Prompt Visibility

In DSPy, prompts are generated artifacts—you don't typically read them. When
something goes wrong, you're reverse-engineering what the optimizer produced.

In WINK, `prompt.render()` returns the exact markdown sent to the model. You can
inspect it, test it, diff it, version it. No surprises.

```python nocheck
rendered = prompt.render(session=session)
print(rendered.text)  # Exact prompt markdown
print([t.name for t in rendered.tools])  # Available tools
```

This is how you debug agents: not by adding print statements, but by inspecting
what the model actually saw.

### Transactional Tool Execution

Every tool call in WINK is wrapped in a transaction. If a tool fails partway
through, session state and filesystem changes are automatically rolled back. No
corrupted state. No defensive cleanup code in every handler.

```python nocheck
def risky_handler(params: Params, *, context: ToolContext) -> ToolResult[R]:
    context.session.dispatch(SomeStateChange(...))
    # If anything below raises, the state change is rolled back
    do_something_dangerous()
    return ToolResult.ok(result)
```

DSPy doesn't have this concept—tool failures can leave your program in
inconsistent states that you have to handle manually.

### Visual Debug UI

When something goes wrong, run `wink debug` to open a local server that renders
the full execution trace:

```bash
wink debug debug_bundles/session.jsonl
```

See exactly what was sent to the model at each turn, what tools were called,
what they returned, how state evolved. This is orders of magnitude faster than
print-statement debugging.

### Declarative Constraints Without Rigid Workflows

We covered this above, but it's worth emphasizing: policies let you express
"what must be true" without specifying "what step comes next."

```python nocheck
# Wrong mental model (workflow thinking):
# 1. First read the file
# 2. Then edit the file
# 3. Then run tests

# Right mental model (policy thinking):
# The agent can do whatever it wants, but:
# - It must read before writing (ReadBeforeWritePolicy)
# - It should run tests (instructions, not hard constraint)
```

The agent might read, edit, read again, edit again, then test. Or it might
research first, then read, then edit in one shot. The policy ensures safety; the
agent chooses the path.

### Resource Boundaries

Prevent runaway agents with explicit budgets and deadlines:

```python nocheck
from datetime import UTC, datetime, timedelta

from weakincentives import Budget, Deadline

response = adapter.evaluate(
    prompt,
    session=session,
    deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5)),
    budget=Budget(max_total_tokens=20_000),
)
```

When limits are exceeded, you get a clear exception—not a surprise bill or an
agent that runs forever.

### Evaluation That Tests Production Code

WINK's `EvalLoop` wraps your `AgentLoop`. Same prompts, same tools, same
adapters. You're evaluating the actual system, not a test harness that
approximates it.

```python nocheck
from weakincentives.evals import EvalLoop, exact_match, tool_called, all_of

evaluator = all_of(
    exact_match,                         # Output correctness
    tool_called("search"),               # Behavioral assertion
    token_usage_under(max_tokens=5000),  # Resource constraint
)

eval_loop = EvalLoop(loop=agent_loop, evaluator=evaluator, requests=mailbox)
```

Session evaluators let you assert on *how* the agent solved the problem—tool
usage patterns, token consumption, state invariants—not just whether it got the
right answer.

### Provider Portability

WINK separates your agent definition (prompts, tools, state) from the execution
runtime. Adapters implement a common `ProviderAdapter` protocol, so the same
prompt template works regardless of which adapter drives execution:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from weakincentives.adapters.codex_app_server import CodexAppServerAdapter
from weakincentives.adapters.acp import ACPAdapter

# Pick one—your agent definition stays the same
adapter = ClaudeAgentSDKAdapter()
adapter = CodexAppServerAdapter(config=codex_config)
adapter = ACPAdapter(config=acp_config)
```

Adapters handle provider-specific details (message formats, tool call
conventions, sandboxing). Your prompts, tools, and session state don't change
when you switch adapters.

### Sandboxed Execution

Out-of-the-box support for safe file operations:

- **VFS tools**: In-memory virtual filesystem tracked as session state
- **Podman sandbox**: OS-level isolation with network policies
- **Claude Agent SDK adapter**: Native sandboxing via bubblewrap/seatbelt

Mount host directories read-only. The sandbox prevents accidental (or
malicious) writes to places they shouldn't go.

### Type Safety Everywhere

Pyright strict mode is enforced throughout WINK. Params, tool calls, tool
results, structured outputs, session state—all typed with dataclasses. Type
mismatches surface at construction time, not when the model is mid-response.

DSPy signatures provide some type information, but it's not enforced at the same
level. WINK catches more bugs at development time.

## What You Give Up

**Automatic prompt optimization.** DSPy's optimizers (BootstrapFewShot, MIPROv2,
etc.) can automatically improve prompts given good metrics. WINK requires manual
iteration via the override system. You can build optimization workflows on top,
but the framework doesn't assume you want automated prompt generation.

**Declarative signatures.** DSPy lets you declare "I want input X and output Y"
and generates the prompt. WINK requires you to write the prompt yourself. This
is more work upfront but gives you complete control.

**Few-shot bootstrapping.** DSPy can automatically generate few-shot examples
from training data. In WINK, you write examples manually or build your own
bootstrapping pipeline.

## When to Use WINK Instead

**You need to understand what's happening.** WINK's explicit prompts, visual
debug UI, and session events mean you can always trace exactly what the agent
did and why. If auditability matters—compliance, debugging, post-mortems—WINK
makes it straightforward.

**You want agent autonomy with safety constraints.** Policies let you give the
agent freedom while enforcing invariants. If you've been fighting rigid workflow
frameworks that don't match how LLMs actually reason, WINK's approach will feel
natural.

**You're going to production.** Transactional tool execution, resource budgets,
sandboxing, and provider portability are production concerns. WINK addresses
them directly.

**You value testability.** Deterministic prompt rendering means you can write
tests that assert on exact prompt text. Session snapshots mean you can reproduce
failures. Evaluation that wraps production code means you're testing what you're
actually deploying.

## When to Stick with DSPy

**You have good metrics and want automated optimization.** If you can define
clear success criteria and have training data, DSPy's optimizers can
systematically improve prompts in ways that would take humans much longer.

**You're doing research.** If prompt generation is part of your experiment—if
you're studying what kinds of prompts work best—DSPy's compilation model is
designed for that.

**You prefer declaring intent over writing prompts.** Some developers prefer
"give me X from Y" over writing the instructions that produce X from Y. DSPy's
signature model fits that preference.

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

**2. Convert modules to sections with explicit instructions:**

```python nocheck
# DSPy
qa = dspy.ChainOfThought(QA)

# WINK - you write the instructions the model sees
qa_section = MarkdownSection(
    title="Question Answering",
    key="qa",
    template="""Think step by step about the question, then provide your answer.

Question: ${question}

Reason through this carefully before answering.""",
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

**4. Replace module composition with policies:**

```python nocheck
# DSPy - explicit module wiring
class MyProgram(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought(QA)

    def forward(self, question):
        context = self.retrieve(question)
        return self.generate(question=question, context=context)

# WINK - tools + policies, agent chooses the path
search_section = MarkdownSection(
    title="Research",
    key="research",
    template="Use the search tool to find relevant information before answering.",
    tools=(search_tool,),
)

# Policy ensures search happens before answer, but doesn't dictate workflow
policies = (SequentialDependencyPolicy(dependencies={"answer": frozenset({"search"})}),)
```

**5. Replace optimizers with overrides:**

```python nocheck
# Instead of compiled prompts, use the override system for iteration
prompt = Prompt(template, overrides_store=store, overrides_tag="v2")
```

## The Mindset Shift

DSPy optimizes prompts for you. WINK gives you tools to write prompts yourself—
with policies that keep the agent safe while preserving its autonomy.

If you've been frustrated by not knowing what DSPy is actually sending to the
model, WINK's explicit approach will feel liberating.

If you've been fighting workflow frameworks that force agents through rigid
state machines, WINK's policy model will make more sense. Define what must be
true. Let the agent figure out how to get there.

If you've struggled with partial failures corrupting agent state, transactional
tool execution eliminates that entire class of bugs.

If you've deployed agents that ran up bills or timed out ungracefully, explicit
budgets and deadlines give you control.

The tradeoff is real: you write more upfront. But you understand what you've
built, you can debug it when it breaks, and you can deploy it with confidence.

## Next Steps

- [Philosophy](philosophy.md): Understand the "weak incentives" approach
- [Prompts](prompts.md): Learn how prompt templates work
- [Tools](tools.md): Build tool handlers with policies
- [Evaluation](evaluation.md): Test your agent systematically

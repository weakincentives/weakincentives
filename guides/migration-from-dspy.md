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

## When to Use WINK Instead of DSPy

- You need to inspect and understand exactly what prompts are being sent
- You're building systems where auditability matters (compliance, debugging)
- You want to iterate on prompts manually with version control
- You value determinism and testability over automatic optimization
- You're building tool-heavy agents where prompt/tool co-location helps

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
prompts yourself.

If you've been frustrated by not knowing what DSPy is actually sending to the
model, WINK's explicit approach may feel liberating.

If you've relied heavily on DSPy's optimizers, you'll need to build or adopt
optimization workflows separately.

## Next Steps

- [Philosophy](philosophy.md): Understand the WINK approach
- [Prompts](prompts.md): Learn how prompt templates work
- [Evaluation](evaluation.md): Build your own evaluation pipeline

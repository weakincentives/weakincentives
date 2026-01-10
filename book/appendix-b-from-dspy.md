# Appendix B: Coming from DSPy?

If you've built LLM programs with DSPy, here's how WINK compares.

## Different bets on where value lives

DSPy centers on **automatic optimization**: you declare input/output signatures, compose modules, and let optimizers compile better prompts and few-shot examples. The framework treats prompts as implementation details that should be generated, not written.

WINK centers on **explicit, inspectable prompts**: you write prompts as typed section trees, control exactly what the model sees, and iterate via version-controlled overrides. The framework treats prompts as first-class artifacts that should be readable, testable, and auditable.

Both approaches have merit. DSPy shines when you have good metrics and want to automate prompt tuning. WINK shines when you need to understand exactly what's being sent to the model and why.

## Concept mapping

Here's how core DSPy concepts translate to WINK:

| DSPy | WINK | Notes |
| --- | --- | --- |
| **Signature** | Structured output dataclass + `PromptTemplate` | Both define typed I/O |
| **Module** (`Predict`, `ChainOfThought`) | `Section` (instructions + tools) | Reusable components |
| **Program** (composed modules) | `PromptTemplate` (tree of sections) | Composition pattern |
| **Optimizer / Teleprompter** | Prompt overrides + manual iteration | Manual vs. automatic |
| **Compilation** | No equivalent | Prompts are explicit |
| **`dspy.ReAct`** | `PlanningToolsSection` + tool sections | Tool-using agents |
| **Metric** | Evaluation framework | See [Chapter 8: Evaluation](08-evaluation.md) |
| **Trace** | Session events + debug UI | Execution telemetry |

## What's familiar

### Typed inputs and outputs

DSPy signatures declare input/output fields; WINK uses frozen dataclasses for the same purpose. Both catch type mismatches early.

**DSPy:**

```python
class QA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

**WINK:**

```python
@dataclass(frozen=True)
class QuestionParams:
    question: str

@dataclass(frozen=True)
class Answer:
    answer: str
```

### Composition

DSPy composes modules into programs; WINK composes sections into prompt templates. Both encourage modular, reusable components.

**DSPy:**

```python
class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve()
        self.generate = dspy.ChainOfThought("context, question -> answer")
```

**WINK:**

```python
template = PromptTemplate[Answer](
    ns="rag",
    key="qa",
    sections=[
        retrieval_section,
        generation_section,
    ],
)
```

### Tool use

DSPy modules like `ReAct` handle tool calling; WINK sections register tools alongside their instructions.

**Both support:**

- Typed tool parameters
- Multiple tools per agent
- Tool result handling

## What's different

### Prompts are visible

In DSPy, prompts are generated artifacts—you don't typically read or edit them directly. In WINK, `prompt.render()` returns the exact markdown sent to the model. You can inspect, test, and version it.

**DSPy:**

```python
# Generated prompt is opaque
predictor = dspy.ChainOfThought("question -> answer")
result = predictor(question="What is Python?")
# You don't see the actual prompt text
```

**WINK:**

```python
# Prompt is explicit and inspectable
prompt = Prompt(template).bind(QuestionParams(question="What is Python?"))
rendered = prompt.render()
print(rendered.text)  # See exact markdown sent to model
```

### No automatic optimization

DSPy's optimizers (BootstrapFewShot, MIPROv2, etc.) generate prompts automatically. WINK uses hash-validated overrides for manual iteration. You can build optimization workflows on top, but the framework doesn't assume you want automated prompt generation.

**DSPy workflow:**

```python
# Define metric
def validate_answer(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# Optimize automatically
optimizer = BootstrapFewShot(metric=validate_answer)
compiled_qa = optimizer.compile(QA(), trainset=train_examples)
```

**WINK workflow:**

```python
# Manual iteration with overrides
store = PromptOverridesStore()
store.set_override(
    ns="qa",
    key="instructions",
    tag="v2",
    override=Override.text("Improved instructions..."),
    base_hash="abc123",
)
prompt = Prompt(template, overrides_store=store, overrides_tag="v2")
```

### State is explicit

DSPy traces execution but doesn't expose a structured state model. WINK sessions are typed, reducer-managed state containers. Every state change is an event you can query, snapshot, and restore.

**WINK state management:**

```python
# Explicit state slices
session = Session(bus=InProcessDispatcher())
session[Plan].seed(initial_plan)
session.dispatch(AddStep(step="Research topic"))

# Query state
current_plan = session[Plan].latest()

# Snapshot and restore
snapshot = session.snapshot()
session.restore(snapshot)
```

### Tools and instructions are co-located

In DSPy, tool definitions are separate from module logic. In WINK, the section that explains "use this tool for X" is the same section that registers the tool. They can't drift apart.

**DSPy:**

```python
@dspy.tool
def search(query: str) -> str:
    """Search the web."""
    return web_search(query)

# Tools registered separately from instructions
react = dspy.ReAct("question -> answer", tools=[search])
```

**WINK:**

```python
# Tool and instructions in same section
search_section = MarkdownSection(
    title="Search",
    key="search",
    template="Use search when you need current information about: ${topic}",
    tools=(search_tool,),
)
```

### Deterministic by default

WINK prompt rendering is pure—same inputs produce same outputs. You can write tests that assert on exact prompt text. DSPy's compiled prompts depend on optimizer state and training data.

**Implications:**

- WINK prompts are reproducible across environments
- Version control diffs show exact prompt changes
- No hidden optimizer state affecting behavior
- Easier to debug unexpected outputs

## When to use WINK instead of DSPy

Choose WINK when:

- **You need to inspect and understand exactly what prompts are being sent**
- **You're building systems where auditability matters** (compliance, debugging)
- **You want to iterate on prompts manually with version control**
- **You value determinism and testability over automatic optimization**
- **You're building tool-heavy agents** where prompt/tool co-location helps
- **You need to explain prompt behavior to non-technical stakeholders**

## When to stick with DSPy

Stick with DSPy when:

- **You have good metrics and want automated prompt optimization**
- **You're doing research** where prompt generation is part of the experiment
- **You want to bootstrap few-shot examples automatically**
- **You prefer declaring intent (signatures) over writing prompts**
- **You have large training datasets** to leverage for optimization

## Migration path

If you're moving from DSPy to WINK:

### 1. Convert signatures to dataclasses

**DSPy:**

```python
class QA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

**WINK:**

```python
@dataclass(slots=True, frozen=True)
class QuestionParams:
    question: str

@dataclass(slots=True, frozen=True)
class Answer:
    """Short factoid answer, often between 1 and 5 words."""
    answer: str
```

### 2. Convert modules to sections

**DSPy:**

```python
qa = dspy.ChainOfThought(QA)
```

**WINK:**

```python
qa_section = MarkdownSection(
    title="Question Answering",
    key="qa",
    template="Think step by step, then answer the question.\n\nQuestion: ${question}",
)
```

### 3. Convert programs to templates

**DSPy:**

```python
class MultiHop(dspy.Module):
    def __init__(self):
        self.generate_query = dspy.ChainOfThought("question -> search_query")
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        query = self.generate_query(question=question).search_query
        passages = self.retrieve(query).passages
        return self.generate_answer(context=passages, question=question)
```

**WINK:**

```python
template = PromptTemplate[Answer](
    ns="multihop",
    key="qa",
    sections=[
        MarkdownSection(
            title="Query Generation",
            key="query-gen",
            template="Generate a search query for: ${question}",
        ),
        MarkdownSection(
            title="Answer Generation",
            key="answer-gen",
            template="Context: ${context}\n\nQuestion: ${question}\n\nThink step by step, then answer.",
        ),
    ],
)
```

### 4. Replace optimizers with overrides

**Instead of compiled prompts:**

```python
# DSPy: optimizer generates prompts
compiled_qa = optimizer.compile(QA(), trainset=examples)
```

**Use WINK's override system:**

```python
# WINK: manual iteration with version control
store = PromptOverridesStore()
store.set_override(
    ns="qa",
    key="instructions",
    tag="v2",
    override=Override.text("Improved instructions..."),
    base_hash="abc123",  # Hash of original template
)
prompt = Prompt(template, overrides_store=store, overrides_tag="v2")
```

### 5. Add tools explicitly

**DSPy:**

```python
# Tools implicit in ReAct
react = dspy.ReAct("question -> answer", tools=[search, calculator])
```

**WINK:**

```python nocheck
# Tools explicit on sections
template = PromptTemplate[Answer](
    ns="agent",
    key="tools",
    sections=[
        instructions_section,
        MarkdownSection(
            title="Tools",
            key="tools",
            template="Use these tools as needed:",
            tools=(search_tool, calculator_tool),
        ),
    ],
)
```

## Key mindset shift

**DSPy optimizes prompts for you; WINK gives you tools to write and iterate on prompts yourself.**

If you've been frustrated by not knowing what DSPy is actually sending to the model, WINK's explicit approach may feel liberating. If you've relied heavily on DSPy's optimizers, you'll need to build or adopt optimization workflows separately.

## Comparison table

| Aspect | DSPy | WINK |
| --- | --- | --- |
| **Prompt visibility** | Generated (opaque) | Explicit (inspectable) |
| **Optimization** | Automatic via optimizers | Manual via overrides |
| **State model** | Traces only | Typed slices + reducers |
| **Determinism** | Depends on optimizer state | Pure by default |
| **Tool management** | Separate from prompts | Co-located in sections |
| **Best for** | Metric-driven optimization | Auditable, debuggable systems |
| **Learning curve** | Signatures → optimize | Write prompts → iterate |

## Can you use both?

Yes! You could:

- Use DSPy to bootstrap initial prompts via optimization
- Export compiled prompts as WINK templates
- Use WINK for production serving with explicit prompts
- Use DSPy's metrics framework with WINK's evaluation system

Example integration:

```python nocheck
# Use DSPy to optimize
compiled = optimizer.compile(MyModule(), trainset=examples)

# Extract generated prompt text
generated_prompt = compiled.get_prompt_text()  # hypothetical

# Convert to WINK template for production
production_section = MarkdownSection(
    title="Generated",
    key="optimized",
    template=generated_prompt,
)
```

Both frameworks are valuable. Choose DSPy if you want automated optimization; choose WINK if you want explicit, inspectable prompts with full control.

# GEPA Optimizer Specification

## Purpose

Enable automated prompt optimization through **Generalist-to-Expert Prompt
Adaptation (GEPA)**, a population-based evolutionary algorithm that evolves
prompt instructions across multiple target sections. This specification covers
the integration of GEPA into Weak Incentives' existing optimizer framework,
the required core store extensions, and the algorithm implementation.

## Guiding Principles

- **Native integration**: GEPA modules map to prompt section paths, leveraging
  existing override stores and session primitives.
- **Minimal new abstractions**: Extend core stores only where broadly useful;
  algorithm-specific logic lives in contrib.
- **Observable execution**: Rollouts emit standard session events for tracing.
- **Deterministic reproducibility**: Seeded RNG ensures repeatable optimization
  runs.
- **Fail-safe mutations**: Invalid prompt mutations are detected and rejected
  before acceptance.

```mermaid
flowchart TB
    subgraph Population["Candidate Pool"]
        C0["Candidate 0<br/>(baseline)"]
        C1["Candidate 1"]
        C2["Candidate 2"]
        Cn["..."]
    end

    subgraph Selection["Selection (Algorithm 2)"]
        Pareto["D_pareto<br/>validation set"]
        Scores["scores_matrix"]
        ParetoSelect["Pareto coverage<br/>selection"]
    end

    subgraph Mutation["Mutation Step"]
        Minibatch["Sample minibatch<br/>from D_feedback"]
        Rollouts["Run rollouts"]
        Meta["Meta-prompt<br/>reflection"]
        Accept["Accept if<br/>improved"]
    end

    subgraph Result["Result"]
        Best["Best candidate<br/>overrides"]
        Persist["Persist to<br/>overrides store"]
    end

    Population --> ParetoSelect
    Pareto --> Scores
    Scores --> ParetoSelect
    ParetoSelect --> Minibatch
    Minibatch --> Rollouts
    Rollouts --> Meta
    Meta --> Accept
    Accept -->|yes| Population
    Accept -->|budget exhausted| Best
    Best --> Persist
```

## Background: The GEPA Algorithm

GEPA optimizes compound LLM systems by maintaining a pool of prompt candidates
and evolving them through reflection-based mutation. The key insight is that
different candidates may specialize on different subsets of the validation data,
so the algorithm maintains **Pareto-optimal specialists** rather than converging
to a single best candidate.

### Algorithm Overview

1. **Initialize** a candidate pool with the baseline prompt configuration
2. **Evaluate** candidates on a small validation subset (D_pareto)
3. **Select** a candidate using Pareto coverage (diverse specialists)
4. **Mutate** a module (section) via LLM reflection on feedback from D_feedback
5. **Accept** mutation if minibatch score improves
6. **Repeat** until budget exhausted
7. **Return** best overall candidate

### Optional: System-Aware Merge

The merge extension (Algorithms 3/4 in the paper) recombines improvements from
different candidates by tracking ancestry and selectively merging module
overrides. This is optional and disabled by default.

## Mapping GEPA to Weak Incentives

### Modules as Section Paths

GEPA's "modules" map directly to Weak Incentives **section paths**:

| GEPA Concept | Weak Incentives Primitive |
|--------------|---------------------------|
| Module | Section path `tuple[str, ...]` |
| Module instruction | Section body (via override) |
| Module examples | `ToolExample` / `TaskExample` instances |
| Candidate | Mapping of overrides (section bodies + examples) |
| System trajectory | Session events (`ToolInvoked`, `PromptExecuted`) |

This approach:

- Works with existing override infrastructure
- Allows optimizing multiple sections in one prompt (round-robin)
- Uses section content hashes for staleness detection
- **Extends to example optimization** via new override types

### Examples as First-Class Optimization Targets

Examples are a powerful optimization lever. GEPA treats examples as optimizable
modules alongside instruction text:

| Example Type | Description | Optimization Target |
|--------------|-------------|---------------------|
| `ToolExample` | Single tool invocation demo | description, input, output |
| `TaskExample` | Multi-step trajectory | objective, steps, outcome |
| `TaskExamplesSection` | Container with summary | summary text, child selection |

Multiple `TaskExamplesSection` instances with different summaries enable
**specialist examples** that GEPA can activate/deactivate based on task type.

### Alternative: Multi-Prompt Systems

For compound systems with multiple distinct prompts, a future extension could
treat each prompt template as a module. This requires a `System` abstraction
and is deferred to a later phase.

## Core Extensions

### InMemoryPromptOverridesStore

GEPA maintains many candidate overrides in memory. A new store implementation
avoids filesystem overhead:

```python
class InMemoryPromptOverridesStore(PromptOverridesStore):
    """In-memory override storage for transient candidate populations."""

    def __init__(self) -> None: ...

    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None: ...

    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride: ...

    def delete(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> None: ...

    def set_section_override(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
        path: tuple[str, ...],
        body: str,
    ) -> PromptOverride: ...

    def seed(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride: ...
```

**Implementation notes:**

- Store keyed by `(ns, prompt_key, tag)` tuple
- Thread-safe via lock (candidates evaluated in parallel)
- Same validation as `LocalPromptOverridesStore`

### OverlayPromptOverridesStore

Layer candidate overrides on top of a base store:

```python
class OverlayPromptOverridesStore(PromptOverridesStore):
    """Composite store layering overlay overrides on top of a base store."""

    def __init__(
        self,
        *,
        base: PromptOverridesStore | None,
        overlay: PromptOverridesStore,
    ) -> None: ...
```

**Resolution behavior:**

- `resolve()` merges base and overlay overrides
- Overlay wins for conflicting section paths
- `set_section_override()` writes only to overlay

This enables "candidate = base project overrides + candidate-specific mutations".

### File Locations

```
src/weakincentives/prompt/overrides/
  memory_store.py      # InMemoryPromptOverridesStore
  overlay_store.py     # OverlayPromptOverridesStore
  __init__.py          # Export new stores
```

## Example Override System

GEPA extends the override system to support example mutations. This enables
optimizing the few-shot demonstrations that guide model behavior.

### Example Override Types

```python
@dataclass(slots=True, frozen=True)
class ToolExampleOverride:
    """Override for a single ToolExample attached to a tool."""

    tool_name: str
    example_index: int              # Position in tool.examples tuple
    expected_hash: HexDigest        # Hash of original example
    description: str | None = None  # Override description
    input_json: str | None = None   # Serialized input dataclass
    output_json: str | None = None  # Serialized output dataclass
```

```python
@dataclass(slots=True, frozen=True)
class TaskStepOverride:
    """Override for a single step within a TaskExample."""

    step_index: int
    tool_name: str | None = None           # Override tool reference
    description: str | None = None         # Override step description
    input_json: str | None = None          # Serialized input
    output_json: str | None = None         # Serialized output
```

```python
@dataclass(slots=True, frozen=True)
class TaskExampleOverride:
    """Override for a TaskExample section."""

    expected_hash: HexDigest           # Hash of original TaskExample
    objective: str | None = None       # Override objective text
    outcome_json: str | None = None    # Serialized outcome (str or dataclass)
    step_overrides: dict[int, TaskStepOverride] = field(default_factory=dict)
    # Key is step_index
```

### Extended PromptOverride

The `PromptOverride` dataclass gains two new fields:

```python
@dataclass(slots=True, frozen=True)
class PromptOverride:
    """Runtime replacements for prompt sections validated by an overrides store."""

    ns: str
    prompt_key: str
    tag: str
    sections: dict[tuple[str, ...], SectionOverride] = field(default_factory=dict)
    tool_overrides: dict[str, ToolOverride] = field(default_factory=dict)

    # NEW: Example overrides
    tool_example_overrides: dict[tuple[str, int], ToolExampleOverride] = field(
        default_factory=dict
    )  # Key: (tool_name, example_index)

    task_example_overrides: dict[tuple[str, ...], TaskExampleOverride] = field(
        default_factory=dict
    )  # Key: section_path to TaskExample
```

### Extended Store Protocol

New methods for example manipulation:

```python
class PromptOverridesStore(Protocol):
    # ... existing methods ...

    def set_tool_example_override(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
        tool_name: str,
        example_index: int,
        description: str | None = None,
        input_json: str | None = None,
        output_json: str | None = None,
    ) -> PromptOverride: ...

    def set_task_example_override(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
        path: tuple[str, ...],  # Path to TaskExample section
        objective: str | None = None,
        outcome_json: str | None = None,
        step_overrides: dict[int, TaskStepOverride] | None = None,
    ) -> PromptOverride: ...

    def add_task_example(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
        container_path: tuple[str, ...],  # Path to TaskExamplesSection
        key: str,
        objective: str,
        outcome_json: str,
        steps_json: str,  # Serialized list of TaskStep
    ) -> PromptOverride: ...

    def remove_task_example(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
        path: tuple[str, ...],  # Path to TaskExample to remove
    ) -> PromptOverride: ...
```

### Example Hash Computation

For staleness detection:

```python
def hash_tool_example(example: ToolExample) -> HexDigest:
    """Hash a ToolExample for override validation."""
    canonical = json.dumps({
        "description": example.description,
        "input": dump(example.input),
        "output": dump(example.output),
    }, sort_keys=True)
    return hash_text(canonical)


def hash_task_example(example: TaskExample) -> HexDigest:
    """Hash a TaskExample for override validation."""
    steps_data = [
        {
            "tool_name": step.tool_name,
            "description": step.example.description,
            "input": dump(step.example.input),
            "output": dump(step.example.output),
        }
        for step in example.steps
    ]
    canonical = json.dumps({
        "objective": example.objective,
        "outcome": dump(example.outcome) if not isinstance(example.outcome, str) else example.outcome,
        "steps": steps_data,
    }, sort_keys=True)
    return hash_text(canonical)
```

### Override Application

When rendering a prompt with example overrides:

1. **ToolExample overrides** - Applied during tool schema generation; the
   overridden example replaces the original in the rendered examples list
2. **TaskExample overrides** - Applied during `TaskExample.render()`;
   individual fields merged with originals
3. **Added TaskExamples** - Appended to the container's children during render
4. **Removed TaskExamples** - Filtered out during section traversal

### Validation Rules

- `input_json` and `output_json` must parse to valid dataclass instances
  matching the tool's type parameters
- `outcome_json` must match the prompt's output type (string or dataclass)
- Step indices must be valid (0 to len(steps)-1)
- Hash mismatches log warnings and skip the override (same as section overrides)

## GEPA Data Types

### Example and Feedback

```python
@dataclass(slots=True, frozen=True)
class GepaExample:
    """A single training/validation example."""

    id: str
    params: tuple[SupportsDataclass, ...]  # Arguments to prompt.bind()
    label: object                           # Ground truth for scoring
```

```python
@dataclass(slots=True, frozen=True)
class RolloutFeedback:
    """Evaluation result for a single rollout."""

    score: float        # Numeric quality signal
    feedback: str       # Textual explanation for meta-prompt
    eval_trace: str | None = None  # Optional test/eval logs
```

```python
@dataclass(slots=True, frozen=True)
class RolloutRecord:
    """Complete record of a single evaluation rollout."""

    example_id: str
    inputs_repr: str      # Formatted input params
    output_repr: str      # Formatted model output
    tool_trace_repr: str  # Formatted tool invocation trace
    feedback: RolloutFeedback
```

### Feedback Function Protocol

```python
class FeedbackFn(Protocol):
    """User-provided scoring and feedback generation."""

    def __call__(
        self,
        example: GepaExample,
        response: PromptResponse[object],
        session: SessionProtocol,
    ) -> RolloutFeedback: ...
```

### Formatting Hooks

```python
class ExampleFormatter(Protocol):
    """Format example inputs/outputs for meta-prompt inclusion."""

    def __call__(
        self,
        example: GepaExample,
        response: PromptResponse[object],
    ) -> tuple[str, str]:  # (inputs_repr, output_repr)
        ...


class TraceFormatter(Protocol):
    """Format tool traces for meta-prompt inclusion."""

    def __call__(
        self,
        session: SessionProtocol,
    ) -> str: ...
```

Default implementations:

- `ExampleFormatter`: JSON dump of params + response text/output
- `TraceFormatter`: Iterate `ToolInvoked` events, render name/params/result

## GepaConfig

```python
@dataclass(slots=True, frozen=True)
class GepaConfig:
    """Configuration for GEPA optimization."""

    # Budget and batching
    rollout_budget: int              # Max total rollouts across all iterations
    minibatch_size: int              # Examples per mutation evaluation
    pareto_set_size: int             # Size of D_pareto validation subset

    # Target sections (instruction modules)
    target_section_paths: tuple[tuple[str, ...], ...]

    # Target examples (example modules) - NEW
    target_tool_examples: tuple[ToolExampleTarget, ...] = ()
    target_task_examples: tuple[tuple[str, ...], ...] = ()  # Paths to TaskExample sections

    # Dataset
    training_examples: Sequence[GepaExample]

    # Scoring
    feedback_fn: FeedbackFn

    # Optional formatting hooks
    example_formatter: ExampleFormatter | None = None
    trace_formatter: TraceFormatter | None = None

    # Hyperparameters
    seed: int = 0
    enable_merge: bool = False  # Enable GEPA+Merge (Algorithm 3/4)

    # Example optimization settings - NEW
    enable_example_generation: bool = False  # Allow generating new TaskExamples
    max_task_examples_per_section: int = 5   # Cap on examples per TaskExamplesSection
    example_mutation_rate: float = 0.3       # Probability of mutating examples vs instructions

    # Robustness
    max_example_chars: int = 6_000
    max_trace_chars: int = 6_000
    escape_dollar_signs: bool = True  # Escape $ in MarkdownSection bodies

    # Persistence
    persist: bool = True
    persist_scope: PersistenceScope = PersistenceScope.GLOBAL
```

### Example Target Types

```python
@dataclass(slots=True, frozen=True)
class ToolExampleTarget:
    """Identifies a ToolExample for optimization."""

    tool_name: str
    example_index: int  # Index in tool.examples tuple


@dataclass(slots=True, frozen=True)
class ExampleModule:
    """Union type for GEPA example modules."""

    kind: Literal["tool_example", "task_example"]
    tool_target: ToolExampleTarget | None = None
    task_example_path: tuple[str, ...] | None = None
```

## GepaResult

```python
@dataclass(slots=True, frozen=True)
class GepaResult:
    """Result of GEPA optimization."""

    # Section overrides
    best_section_overrides: dict[tuple[str, ...], str]  # path -> body

    # Example overrides - NEW
    best_tool_example_overrides: dict[tuple[str, int], ToolExampleOverride]
    best_task_example_overrides: dict[tuple[str, ...], TaskExampleOverride]
    generated_task_examples: list[GeneratedTaskExample]  # Newly created examples

    # Metrics
    best_pareto_mean: float
    scores_matrix: list[list[float]]  # candidates x D_pareto
    ancestry: list[int | None]        # Parent index per candidate
    accepted_mutations: int
    accepted_example_mutations: int   # NEW
    total_rollouts: int
```

```python
@dataclass(slots=True, frozen=True)
class GeneratedTaskExample:
    """A TaskExample generated during optimization."""

    container_path: tuple[str, ...]  # Path to parent TaskExamplesSection
    key: str
    objective: str
    outcome_json: str
    steps_json: str
```

## GepaOptimizer

```python
class GepaOptimizer(BasePromptOptimizer[object, GepaResult]):
    """GEPA-based prompt section optimization."""

    def __init__(
        self,
        context: OptimizationContext,
        config: GepaConfig,
        *,
        optimizer_config: OptimizerConfig | None = None,
    ) -> None: ...

    def optimize(
        self,
        prompt: Prompt[object],
        *,
        session: SessionProtocol,
    ) -> GepaResult: ...
```

**File location:**

```
src/weakincentives/contrib/optimizers/
  gepa/
    __init__.py
    config.py        # GepaConfig, GepaExample
    types.py         # RolloutFeedback, RolloutRecord, protocols
    trace_format.py  # Default formatters
    selection.py     # Algorithm 2 (Pareto selection)
    mutation.py      # Reflection meta-prompt
    merge.py         # Optional Algorithm 3/4
    prompts.py       # Meta-prompt templates
    optimizer.py     # GepaOptimizer
```

## Algorithm Implementation

### Candidate Representation

```python
@dataclass(slots=True, frozen=True)
class Candidate:
    """Internal representation of a GEPA candidate."""

    # Section body overrides
    section_overrides: dict[tuple[str, ...], str]  # section_path -> body

    # Example overrides - NEW
    tool_example_overrides: dict[tuple[str, int], ToolExampleOverride] = field(
        default_factory=dict
    )
    task_example_overrides: dict[tuple[str, ...], TaskExampleOverride] = field(
        default_factory=dict
    )
    generated_examples: list[GeneratedTaskExample] = field(default_factory=list)

    parent_index: int | None = None
```

### Dataset Splitting

At initialization, split `training_examples` into:

- **D_pareto**: First `pareto_set_size` examples (validation for selection)
- **D_feedback**: Remaining examples (minibatch source for mutation)

### Main Loop (Algorithm 1)

```python
def optimize(self, prompt, *, session):
    # 1. Initialize
    rng = random.Random(self.config.seed)
    D_pareto, D_feedback = self._split_dataset()
    candidates = [Candidate(overrides={}, parent_index=None)]
    scores_matrix = [self._evaluate_on_pareto(candidates[0], D_pareto)]
    module_idx = 0
    rollouts_used = 0

    # 2. Main loop
    while rollouts_used < self.config.rollout_budget:
        # Select candidate via Pareto coverage
        k = self._select_candidate(scores_matrix, rng)

        # Round-robin module selection
        module_path = self.config.target_section_paths[module_idx]
        module_idx = (module_idx + 1) % len(self.config.target_section_paths)

        # Sample minibatch
        minibatch = rng.sample(D_feedback, min(self.config.minibatch_size, len(D_feedback)))

        # Evaluate parent on minibatch
        parent_rollouts = [self._run_rollout(candidates[k], ex) for ex in minibatch]
        rollouts_used += len(parent_rollouts)
        sigma = mean(r.feedback.score for r in parent_rollouts)

        # Generate mutation via meta-prompt
        current_body = self._get_effective_body(candidates[k], module_path, prompt)
        new_body = self._propose_mutation(current_body, parent_rollouts, module_path)

        # Build child candidate
        child_overrides = dict(candidates[k].overrides)
        child_overrides[module_path] = new_body
        child = Candidate(overrides=child_overrides, parent_index=k)

        # Evaluate child on same minibatch
        child_rollouts = [self._run_rollout(child, ex) for ex in minibatch]
        rollouts_used += len(child_rollouts)
        sigma_prime = mean(r.feedback.score for r in child_rollouts)

        # Accept if improved
        if sigma_prime > sigma:
            candidates.append(child)
            scores_matrix.append(self._evaluate_on_pareto(child, D_pareto))
            rollouts_used += len(D_pareto)

        # Optional: merge step
        if self.config.enable_merge and len(candidates) >= 2:
            self._attempt_merge(candidates, scores_matrix, rng)

    # 3. Return best
    best_idx = self._find_best_candidate(scores_matrix)
    return self._build_result(candidates, scores_matrix, best_idx, rollouts_used)
```

### Pareto Selection (Algorithm 2)

The selection algorithm maintains diversity by sampling candidates that are
Pareto-optimal on different validation examples:

```python
def _select_candidate(
    self,
    scores_matrix: list[list[float]],
    rng: random.Random,
) -> int:
    n_candidates = len(scores_matrix)
    n_examples = len(scores_matrix[0])

    # 1. Find best score per example
    best_per_example = [
        max(scores_matrix[k][i] for k in range(n_candidates))
        for i in range(n_examples)
    ]

    # 2. Build P_star sets (candidates achieving best on each example)
    P_star = [
        {k for k in range(n_candidates) if scores_matrix[k][i] == best_per_example[i]}
        for i in range(n_examples)
    ]

    # 3. Union of all P_star sets
    C = set().union(*P_star)

    # 4. Remove dominated candidates
    C = self._remove_dominated(C, scores_matrix)

    # 5. Compute coverage frequency
    f = {k: sum(1 for p in P_star if k in p) for k in C}

    # 6. Sample proportional to frequency
    total = sum(f.values())
    r = rng.random() * total
    cumulative = 0.0
    for k in sorted(C):  # Sorted for determinism
        cumulative += f[k]
        if cumulative >= r:
            return k
    return max(C)  # Fallback
```

**Dominance check:**

Candidate `a` dominates `b` if:
- `scores[a][i] >= scores[b][i]` for all i
- `scores[a][i] > scores[b][i]` for at least one i

### Mutation via Meta-Prompt

The reflection prompt asks the LLM to improve an instruction based on feedback:

```python
def _propose_mutation(
    self,
    current_body: str,
    rollouts: list[RolloutRecord],
    module_path: tuple[str, ...],
) -> str:
    # Build meta-prompt
    meta_prompt = self._build_meta_prompt(current_body, rollouts)

    # Evaluate via adapter
    response = self._context.adapter.evaluate(
        meta_prompt,
        session=self._create_optimization_session(prompt),
        deadline=self._context.deadline,
    )

    # Parse new instruction from response
    new_body = self._parse_instruction_block(response.text)

    # Sanitize for MarkdownSection compatibility
    if self.config.escape_dollar_signs:
        new_body = self._escape_dollars(new_body, module_path)

    return new_body
```

**Meta-prompt structure:**

````markdown
## Current Instruction

The following instruction is used in a prompt section:

```
{current_body}
```

## Training Examples with Feedback

{for each rollout in rollouts}
### Example {rollout.example_id}

**Input:** {rollout.inputs_repr}

**Output:** {rollout.output_repr}

**Tool Trace:** {rollout.tool_trace_repr}

**Score:** {rollout.feedback.score}

**Feedback:** {rollout.feedback.feedback}

{end for}

## Task

Based on the feedback above, write an improved instruction that addresses the
identified issues. Return ONLY the updated instruction in a fenced block:

```instruction
<your improved instruction here>
```
````

**Parsing:**

1. Find first `` ```instruction `` block
2. Fallback: first `` ``` `` block
3. Fallback: entire response stripped

### Example Mutation via Meta-Prompt

When the module selection picks an example target, GEPA uses specialized
meta-prompts for example mutation.

#### ToolExample Mutation

````markdown
## Current Tool Example

Tool: {tool_name}
Description: {tool_description}

Example {example_index}:
- Description: {example.description}
- Input: {json(example.input)}
- Output: {json(example.output)}

## Training Examples with Feedback

{rollouts with scores and feedback}

## Task

Based on the feedback above, improve this tool example to better demonstrate
correct usage. The example should help the model understand when and how to
use this tool effectively.

Return the improved example in this format:

```tool_example
description: <improved description>
input: <JSON input matching tool params>
output: <JSON output matching tool result>
```
````

#### TaskExample Mutation

````markdown
## Current Task Example

Objective: {task_example.objective}

Steps:
{for step in task_example.steps}
{step_index}. {step.tool_name}: {step.example.description}
   Input: {json(step.example.input)}
   Output: {json(step.example.output)}
{end for}

Outcome: {task_example.outcome}

## Training Examples with Feedback

{rollouts with scores and feedback}

## Task

Based on the feedback above, improve this task example to better demonstrate
the workflow. Consider:
- Is the objective clear and achievable?
- Are the steps in logical order?
- Do the intermediate results flow correctly?
- Does the outcome match what the steps produce?

Return the improved example:

```task_example
objective: <improved objective>
steps:
  - tool: <tool_name>
    description: <step description>
    input: <JSON>
    output: <JSON>
  - ...
outcome: <improved outcome>
```
````

#### TaskExample Generation

When `enable_example_generation=True` and feedback indicates missing coverage:

````markdown
## Existing Task Examples

{summaries of current TaskExamples in the section}

## Training Examples with Feedback

{rollouts showing gaps in coverage}

## Task

The feedback indicates scenarios not covered by existing examples. Generate
a new task example that demonstrates the missing workflow.

Requirements:
- Use only tools available in the prompt: {available_tools}
- The objective should be specific and achievable
- Steps should use realistic input/output values
- The outcome should reflect successful completion

```new_task_example
key: <unique_key>
objective: <specific objective>
steps:
  - tool: <tool_name>
    description: <why this step>
    input: <JSON>
    output: <JSON>
  - ...
outcome: <expected result>
```
````

### Module Selection with Examples

The main loop selects between instruction modules and example modules:

```python
def _select_module(self, rng: random.Random) -> Module:
    """Select next module to mutate (instruction or example)."""
    all_modules = self._build_module_list()

    if rng.random() < self.config.example_mutation_rate:
        # Prefer example modules when they exist
        example_modules = [m for m in all_modules if m.is_example]
        if example_modules:
            return rng.choice(example_modules)

    # Round-robin through all modules
    self._module_idx = (self._module_idx + 1) % len(all_modules)
    return all_modules[self._module_idx]
```

Where `Module` is:

```python
@dataclass(slots=True, frozen=True)
class Module:
    """A GEPA optimization target."""

    kind: Literal["section", "tool_example", "task_example"]
    path: tuple[str, ...] | None = None        # For sections and task_examples
    tool_target: ToolExampleTarget | None = None  # For tool_examples

    @property
    def is_example(self) -> bool:
        return self.kind in ("tool_example", "task_example")
```

### System-Aware Merge (Optional)

When `enable_merge=True`, attempt to combine improvements from different
candidates:

```python
def _attempt_merge(
    self,
    candidates: list[Candidate],
    scores_matrix: list[list[float]],
    rng: random.Random,
) -> None:
    # Select two candidates with good performance
    i, j = self._select_merge_pair(scores_matrix, rng)
    if i == j:
        return

    # Find common ancestor
    ancestor_idx = self._find_common_ancestor(candidates, i, j)
    if ancestor_idx is None:
        return

    ancestor = candidates[ancestor_idx]

    # Build merged candidate
    merged_overrides = dict(ancestor.overrides)
    for path in self.config.target_section_paths:
        i_changed = candidates[i].overrides.get(path) != ancestor.overrides.get(path)
        j_changed = candidates[j].overrides.get(path) != ancestor.overrides.get(path)

        if i_changed and not j_changed:
            merged_overrides[path] = candidates[i].overrides.get(path, merged_overrides.get(path))
        elif j_changed and not i_changed:
            merged_overrides[path] = candidates[j].overrides.get(path, merged_overrides.get(path))
        elif i_changed and j_changed:
            # Both changed: pick from higher-scoring candidate
            if mean(scores_matrix[i]) >= mean(scores_matrix[j]):
                merged_overrides[path] = candidates[i].overrides.get(path, merged_overrides.get(path))
            else:
                merged_overrides[path] = candidates[j].overrides.get(path, merged_overrides.get(path))

    # Add merged candidate if novel
    if merged_overrides not in [c.overrides for c in candidates]:
        merged = Candidate(overrides=merged_overrides, parent_index=None)
        candidates.append(merged)
        scores_matrix.append(self._evaluate_on_pareto(merged, D_pareto))
```

## Rollout Execution

Each rollout creates an isolated session and evaluates the prompt:

```python
def _run_rollout(
    self,
    candidate: Candidate,
    example: GepaExample,
) -> RolloutRecord:
    # 1. Create isolated session
    rollout_session = Session(tags={"scope": "gepa_rollout"})

    # 2. Clone prompt sections for isolation
    cloned_prompt = self._clone_prompt_for_session(prompt, rollout_session)

    # 3. Build overlay store with candidate overrides
    candidate_store = InMemoryPromptOverridesStore()
    for path, body in candidate.overrides.items():
        candidate_store.set_section_override(cloned_prompt, path=path, body=body)

    overlay_store = OverlayPromptOverridesStore(
        base=self._context.overrides_store,
        overlay=candidate_store,
    )

    # 4. Create prompt with overlay store
    eval_prompt = Prompt(
        cloned_prompt.template,
        overrides_store=overlay_store,
        overrides_tag=self._context.overrides_tag,
    ).bind(*example.params)

    # 5. Evaluate
    try:
        response = self._context.adapter.evaluate(
            eval_prompt,
            session=rollout_session,
            deadline=self._context.deadline,
        )
        feedback = self.config.feedback_fn(example, response, rollout_session)
    except PromptEvaluationError as exc:
        feedback = RolloutFeedback(
            score=0.0,
            feedback=f"Evaluation failed: {exc}",
        )
        response = None

    # 6. Format trace
    formatter = self.config.example_formatter or default_example_formatter
    trace_formatter = self.config.trace_formatter or default_trace_formatter

    inputs_repr, output_repr = formatter(example, response) if response else ("", "")
    tool_trace_repr = trace_formatter(rollout_session)

    return RolloutRecord(
        example_id=example.id,
        inputs_repr=inputs_repr[:self.config.max_example_chars],
        output_repr=output_repr[:self.config.max_example_chars],
        tool_trace_repr=tool_trace_repr[:self.config.max_trace_chars],
        feedback=feedback,
    )
```

## Trace Formatting

### Default Tool Trace Formatter

```python
def default_trace_formatter(session: SessionProtocol) -> str:
    events = session.select_all(ToolInvoked)
    lines = []
    for event in events:
        lines.append(f"[{event.name}]")
        lines.append(f"  params: {_truncate(dump(event.params), 500)}")
        lines.append(f"  result: {_truncate(str(event.result), 500)}")
    return "\n".join(lines)
```

### Default Example Formatter

```python
def default_example_formatter(
    example: GepaExample,
    response: PromptResponse[object] | None,
) -> tuple[str, str]:
    inputs = dump(example.params) if example.params else "{}"
    if response is None:
        return inputs, "<no response>"
    if response.output is not None:
        output = dump(response.output)
    else:
        output = response.text or "<empty>"
    return inputs, output
```

## Robustness Considerations

### Dollar Sign Handling

`MarkdownSection` uses `string.Template` which treats `$` as special. When
`escape_dollar_signs=True` (default), the optimizer escapes `$` as `$$` in
mutated bodies to prevent rendering failures.

```python
def _escape_dollars(self, body: str, path: tuple[str, ...]) -> str:
    # Escape all $ not part of original placeholders
    # For safety, escape all $ as $$ in optimizer-generated content
    return body.replace("$", "$$")
```

### Validation of Mutations

Before accepting a mutation, validate it renders correctly:

```python
def _validate_mutation(self, prompt: Prompt, candidate: Candidate) -> bool:
    try:
        # Attempt dry-run render
        test_prompt = self._build_prompt_with_overrides(prompt, candidate)
        _ = test_prompt.render(session=Session())
        return True
    except Exception:
        return False
```

Mutations that fail validation are rejected immediately.

### Trace Truncation

Large tool results can overwhelm the meta-prompt. Apply truncation:

- Per-tool result: 500-2000 characters
- Total trace: `max_trace_chars` (default 6000)
- Example inputs/outputs: `max_example_chars` (default 6000)

### Determinism

For reproducibility:

- Use a single `random.Random(seed)` instance
- Stable sort by candidate index when comparing floats
- Deterministic iteration order (sorted keys)

## Persistence

When `persist=True` and the context has an `overrides_store`:

```python
def _persist_result(
    self,
    prompt: Prompt,
    best_candidate: Candidate,
) -> None:
    if not self.config.persist:
        return
    if self._context.overrides_store is None:
        return

    for path, body in best_candidate.overrides.items():
        self._context.overrides_store.set_section_override(
            prompt,
            tag=self._context.overrides_tag,
            path=path,
            body=body,
        )
```

## Multiple TaskExamplesSection Patterns

Using multiple `TaskExamplesSection` instances enables **specialist examples**
that GEPA can selectively optimize for different task categories.

### Pattern: Category-Specific Examples

```python
# Define separate example sections for different task categories
security_examples = TaskExamplesSection(
    key="security-examples",
    title="Security Review Examples",
    summary="Examples for security-focused code review tasks.",
    examples=[
        TaskExample(key="sql-injection", objective="Find SQL injection...", ...),
        TaskExample(key="xss-detection", objective="Detect XSS...", ...),
    ],
)

performance_examples = TaskExamplesSection(
    key="performance-examples",
    title="Performance Audit Examples",
    summary="Examples for performance optimization tasks.",
    examples=[
        TaskExample(key="n-plus-one", objective="Find N+1 queries...", ...),
        TaskExample(key="memory-leak", objective="Detect memory leaks...", ...),
    ],
)

template = PromptTemplate(
    ns="agents/reviewer",
    key="code-review",
    sections=[
        instructions_section,
        security_examples,
        performance_examples,
    ],
)
```

### GEPA Optimization of Specialist Examples

Configure GEPA to optimize examples within specific sections:

```python
config = GepaConfig(
    # ...
    target_task_examples=(
        ("security-examples", "sql-injection"),
        ("security-examples", "xss-detection"),
        ("performance-examples", "n-plus-one"),
    ),
    enable_example_generation=True,
    max_task_examples_per_section=3,  # Cap per section
)
```

GEPA may:
- Mutate existing examples to better match training data patterns
- Generate new examples when coverage gaps are detected
- Specialize examples within each section for its category

### Pattern: Progressive Complexity

```python
# Examples ordered by complexity
basic_examples = TaskExamplesSection(
    key="basic-examples",
    title="Basic Workflows",
    summary="Simple single-tool examples for common tasks.",
    examples=[...],  # 1-2 step workflows
)

advanced_examples = TaskExamplesSection(
    key="advanced-examples",
    title="Advanced Workflows",
    summary="Multi-step examples with tool chaining.",
    visibility=SectionVisibility.SUMMARY,  # Collapsed by default
    examples=[...],  # 3+ step workflows
)
```

## Usage Example

### Basic: Instruction Optimization Only

```python
from weakincentives.contrib.optimizers.gepa import (
    GepaConfig,
    GepaExample,
    GepaOptimizer,
    RolloutFeedback,
)
from weakincentives.optimizers import OptimizationContext, PersistenceScope
from weakincentives.prompt.overrides import LocalPromptOverridesStore

# Define training examples
examples = [
    GepaExample(id="1", params=(QueryParams(query="..."),), label=expected_output_1),
    GepaExample(id="2", params=(QueryParams(query="..."),), label=expected_output_2),
    # ... more examples
]

# Define feedback function
def score_response(
    example: GepaExample,
    response: PromptResponse[object],
    session: SessionProtocol,
) -> RolloutFeedback:
    score = compute_similarity(response.output, example.label)
    feedback = generate_feedback(response.output, example.label)
    return RolloutFeedback(score=score, feedback=feedback)

# Configure GEPA for instruction-only optimization
config = GepaConfig(
    rollout_budget=500,
    minibatch_size=4,
    pareto_set_size=10,
    target_section_paths=(
        ("instructions",),
        ("tool-guidance",),
    ),
    training_examples=examples,
    feedback_fn=score_response,
    seed=42,
)

# Run optimization
store = LocalPromptOverridesStore()
context = OptimizationContext(
    adapter=adapter,
    event_bus=bus,
    overrides_store=store,
    overrides_tag="gepa-v1",
)
optimizer = GepaOptimizer(context, config)
result = optimizer.optimize(prompt, session=session)

print(f"Best Pareto mean: {result.best_pareto_mean}")
print(f"Accepted mutations: {result.accepted_mutations}")
```

### Advanced: Combined Instruction and Example Optimization

```python
from weakincentives.contrib.optimizers.gepa import (
    GepaConfig,
    GepaExample,
    GepaOptimizer,
    ToolExampleTarget,
)

# Configure GEPA with example optimization
config = GepaConfig(
    rollout_budget=1000,
    minibatch_size=4,
    pareto_set_size=15,

    # Instruction modules
    target_section_paths=(
        ("instructions",),
        ("tool-guidance",),
    ),

    # Example modules
    target_tool_examples=(
        ToolExampleTarget(tool_name="search_code", example_index=0),
        ToolExampleTarget(tool_name="read_file", example_index=0),
    ),
    target_task_examples=(
        ("task-examples", "security-review"),
        ("task-examples", "refactoring"),
    ),

    # Example optimization settings
    enable_example_generation=True,
    max_task_examples_per_section=5,
    example_mutation_rate=0.3,  # 30% chance to mutate examples

    training_examples=examples,
    feedback_fn=score_response,
    seed=42,
)

optimizer = GepaOptimizer(context, config)
result = optimizer.optimize(prompt, session=session)

# Inspect example optimization results
print(f"Accepted instruction mutations: {result.accepted_mutations}")
print(f"Accepted example mutations: {result.accepted_example_mutations}")
print(f"Generated new examples: {len(result.generated_task_examples)}")

# Review generated examples
for gen_ex in result.generated_task_examples:
    print(f"  - {gen_ex.key}: {gen_ex.objective[:50]}...")
```

## Testing Strategy

### Unit Tests

**Selection algorithm (`test_gepa_selection.py`):**

- Known scores_matrix → expected selection distribution
- Dominance pruning correctness
- Edge cases: single candidate, all identical scores

**Mutation parsing (`test_gepa_prompt_parsing.py`):**

- Extract instruction from various response formats
- Dollar sign escaping
- Malformed response handling

**Example mutation parsing (`test_gepa_example_parsing.py`):**

- Parse ToolExample mutation responses
- Parse TaskExample mutation responses
- Parse new TaskExample generation responses
- Handle malformed JSON in example inputs/outputs

**Merge algorithm (`test_gepa_merge.py`):**

- Ancestry tracking
- Module-level override selection (including examples)
- Novel candidate detection

### Integration Tests

**End-to-end with fake adapter (`test_gepa_optimizer.py`):**

```python
def test_gepa_improves_score():
    # Trivial prompt with one optimizable section
    # Fake adapter: returns output based on instruction keyword
    # Feedback: score 1 if keyword present, 0 otherwise

    # Assert:
    # - At least one mutation accepted
    # - Final override contains target keyword
    # - Pareto mean improved
```

**Example optimization (`test_gepa_example_optimization.py`):**

```python
def test_gepa_mutates_tool_examples():
    # Prompt with tool that has optimizable example
    # Fake adapter: returns better output when example description
    # contains specific keyword

    # Assert:
    # - Example mutation accepted
    # - Final example override contains keyword

def test_gepa_generates_task_examples():
    # Prompt with TaskExamplesSection
    # Feedback indicating coverage gaps
    # enable_example_generation=True

    # Assert:
    # - At least one new TaskExample generated
    # - Generated example passes validation
    # - Generated example persisted to store
```

**Store integration (`test_gepa_stores.py`):**

- InMemoryPromptOverridesStore CRUD operations
- OverlayPromptOverridesStore merging behavior
- Thread safety under parallel access
- Example override storage and retrieval

**Example override integration (`test_example_overrides.py`):**

- ToolExampleOverride applied during tool rendering
- TaskExampleOverride applied during section rendering
- Generated TaskExample appended to container
- Hash validation for stale example overrides

### Robustness Tests

- Mutation validation catches rendering errors
- Example type validation (input/output match tool types)
- Truncation respects character limits
- Seeded RNG produces identical results
- Generated examples respect `max_task_examples_per_section`

## Events

GEPA optimization emits events through the context event bus:

- `GepaOptimizationStarted` - Configuration and prompt descriptor
- `GepaRolloutCompleted` - Individual rollout feedback
- `GepaMutationProposed` - Before/after content (instruction or example)
- `GepaMutationAccepted` / `GepaMutationRejected` - Acceptance decision
- `GepaExampleGenerated` - New TaskExample created (NEW)
- `GepaOptimizationCompleted` - Final result summary

## Limitations

- **Synchronous execution**: Rollouts run sequentially (parallelization deferred)
- **Section-path modules only**: Multi-prompt systems require future extension
- **No auto-retry on provider errors**: Errors convert to zero-score feedback
- **Memory growth**: Large candidate pools consume memory (no eviction)
- **Dollar sign escaping**: May interfere with intentional template variables
- **Example type constraints**: Generated examples must match tool type parameters
- **No example removal optimization**: GEPA can add/mutate but not remove examples
- **JSON serialization required**: Example inputs/outputs must be JSON-serializable
- **Alpha stability**: Interfaces may change without backward compatibility

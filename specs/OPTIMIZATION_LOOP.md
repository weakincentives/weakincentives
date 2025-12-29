# Optimization Loop Specification

## Purpose

Automatic prompt optimization that reduces token usage while preserving
evaluation performance. Changes are applied automatically when they cause no
regression on the evaluation dataset.

## Guiding Principles

- **EvalLoop composition**: OptimizationLoop delegates evaluation to EvalLoop,
  never duplicates scoring logic.
- **Override-native**: Candidates are written to the override store with unique
  tags; no prompt patching in the optimization layer.
- **Regression-gated**: Changes apply automatically only if all passing samples
  continue to pass.
- **Token-focused**: Initial scope is reducing prompt size without quality loss.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          OptimizationLoop                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Prompt + Dataset + Evaluator                                          │
│         │                                                               │
│         ▼                                                               │
│   ┌─────────────┐                                                       │
│   │  Baseline   │  EvalLoop.run(tag="stable")                           │
│   │  Eval       │  → EvalReport (pass_rate, token_count)                │
│   └──────┬──────┘                                                       │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────┐                                                       │
│   │  Compress   │  CompressStrategy.propose()                           │
│   │  Proposal   │  → Write to store with candidate tag                  │
│   └──────┬──────┘                                                       │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────┐                                                       │
│   │  Candidate  │  EvalLoop.run(tag=candidate)                          │
│   │  Eval       │  → EvalReport                                         │
│   └──────┬──────┘                                                       │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────┐     ┌──────────────┐                                  │
│   │ Regression? │─No─▶│ Promote tag  │  copy_tag → "stable"             │
│   │             │     └──────────────┘                                  │
│   └──────┬──────┘                                                       │
│          │Yes                                                           │
│          ▼                                                              │
│   ┌─────────────┐                                                       │
│   │  Discard    │  delete_tag (cleanup)                                 │
│   └─────────────┘                                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Experiment

An Experiment represents an isolated optimization context backed by an overrides
tag.

```python
@dataclass(slots=True, frozen=True)
class Experiment:
    """Isolated optimization context backed by an overrides tag."""

    id: str                          # Unique identifier
    name: str                        # Human-readable name
    overrides_tag: str               # Tag in PromptOverridesStore
    parent_tag: str | None = None    # Tag this derived from
    flags: dict[str, bool] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

Experiment is accessible in tool handlers via ToolContext:

```python
@property
def experiment(self) -> Experiment | None:
    """Current experiment context, if any."""
    return self.resources.get(Experiment)
```

## Compression Strategy

Generates token-reduced prompt variants.

```python
class CompressStrategy(Protocol):
    """Generates compressed prompt variants."""

    def propose(
        self,
        prompt: PromptDescriptor,
        *,
        adapter: ProviderAdapter[object],
        session: Session,
    ) -> OptimizationProposal | None:
        """Generate a compressed variant of the prompt.

        Returns None if no compression is possible.
        """
        ...


@dataclass(slots=True, frozen=True)
class SectionEdit:
    """Proposed edit to a single section."""

    path: tuple[str, ...]
    original_hash: HexDigest
    proposed_body: str
    original_tokens: int
    proposed_tokens: int


@dataclass(slots=True, frozen=True)
class OptimizationProposal:
    """Proposed prompt modification."""

    id: str
    edits: tuple[SectionEdit, ...]
    total_token_reduction: int
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### Built-in: LLMCompressStrategy

Uses an LLM to compress verbose sections while preserving meaning:

```python
class LLMCompressStrategy:
    """LLM-based prompt compression."""

    def __init__(
        self,
        *,
        compression_prompt: Prompt[CompressOutput],
        min_reduction_percent: float = 0.1,  # Skip if < 10% reduction
    ) -> None: ...
```

## Candidate Evaluation

```python
@dataclass(slots=True, frozen=True)
class ExperimentResult:
    """Outcome of evaluating an experiment."""

    experiment: Experiment
    eval_report: EvalReport
    token_count: int                 # Total prompt tokens used
    evaluated_at: datetime


@dataclass(slots=True, frozen=True)
class CandidateComparison:
    """Comparison of candidate against baseline."""

    baseline: ExperimentResult
    candidate: ExperimentResult
    proposal: OptimizationProposal

    @property
    def token_reduction(self) -> int:
        """Tokens saved (positive = smaller)."""
        return self.baseline.token_count - self.candidate.token_count

    @property
    def has_regression(self) -> bool:
        """True if any previously passing sample now fails."""
        baseline_passed = {
            r.sample_id
            for r in self.baseline.eval_report.results
            if r.success and r.score.passed
        }
        for r in self.candidate.eval_report.results:
            if r.sample_id in baseline_passed and not r.score.passed:
                return True
        return False
```

## Override Store Extensions

```python
class PromptOverridesStore(Protocol):
    # Existing methods...

    def copy_tag(
        self,
        *,
        ns: str,
        prompt_key: str,
        from_tag: str,
        to_tag: str,
    ) -> None:
        """Copy all overrides from one tag to another."""
        ...

    def delete_tag(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> None:
        """Delete all overrides for a tag."""
        ...
```

## OptimizationLoop

```python
@dataclass(slots=True, frozen=True)
class OptimizationConfig:
    """Configuration for optimization loop."""

    baseline_tag: str = "stable"
    target_tag: str = "stable"
    experiment_prefix: str = "opt"
    cleanup_on_regression: bool = True


@dataclass(slots=True, frozen=True)
class OptimizationResult:
    """Result of an optimization run."""

    experiment_id: str
    baseline: ExperimentResult
    candidate: ExperimentResult | None      # None if no proposal generated
    applied: bool                           # True if candidate was promoted
    token_reduction: int                    # 0 if not applied


class OptimizationLoop(Generic[InputT, OutputT, ExpectedT]):
    """Automatic prompt optimization with regression gating."""

    def __init__(
        self,
        *,
        eval_loop: EvalLoop[InputT, OutputT, ExpectedT],
        strategy: CompressStrategy,
        overrides_store: PromptOverridesStore,
        config: OptimizationConfig = OptimizationConfig(),
    ) -> None:
        self._eval_loop = eval_loop
        self._strategy = strategy
        self._store = overrides_store
        self._config = config

    def run(
        self,
        prompt: Prompt[InputT],
        dataset: Dataset[InputT, ExpectedT],
        evaluator: Evaluator[OutputT, ExpectedT],
        *,
        flags: dict[str, bool] | None = None,
    ) -> OptimizationResult:
        """Run optimization loop.

        Generates a compressed prompt variant, evaluates it against the full
        dataset, and applies it automatically if there is no regression.
        """
        experiment_id = generate_experiment_id()
        flags = flags or {}
        descriptor = descriptor_for_prompt(prompt)

        # 1. Evaluate baseline
        baseline_exp = Experiment(
            id=f"{experiment_id}-baseline",
            name="baseline",
            overrides_tag=self._config.baseline_tag,
            flags=flags,
        )
        baseline_result = self._evaluate(prompt, dataset, evaluator, baseline_exp)

        # 2. Generate compression proposal
        proposal = self._strategy.propose(
            descriptor,
            adapter=self._eval_loop.adapter,
            session=self._create_session(),
        )

        if proposal is None:
            return OptimizationResult(
                experiment_id=experiment_id,
                baseline=baseline_result,
                candidate=None,
                applied=False,
                token_reduction=0,
            )

        # 3. Write proposal to store and evaluate
        candidate_tag = f"{self._config.experiment_prefix}-{experiment_id}"
        self._write_proposal(prompt, proposal, candidate_tag)

        candidate_exp = Experiment(
            id=f"{experiment_id}-candidate",
            name="candidate",
            overrides_tag=candidate_tag,
            parent_tag=self._config.baseline_tag,
            flags=flags,
        )
        candidate_result = self._evaluate(prompt, dataset, evaluator, candidate_exp)

        # 4. Check for regression
        comparison = CandidateComparison(
            baseline=baseline_result,
            candidate=candidate_result,
            proposal=proposal,
        )

        if comparison.has_regression:
            # Regression detected - discard
            if self._config.cleanup_on_regression:
                self._store.delete_tag(
                    ns=descriptor.ns,
                    prompt_key=descriptor.key,
                    tag=candidate_tag,
                )
            return OptimizationResult(
                experiment_id=experiment_id,
                baseline=baseline_result,
                candidate=candidate_result,
                applied=False,
                token_reduction=0,
            )

        # 5. No regression - promote automatically
        self._store.copy_tag(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            from_tag=candidate_tag,
            to_tag=self._config.target_tag,
        )

        # Cleanup candidate tag after promotion
        self._store.delete_tag(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=candidate_tag,
        )

        return OptimizationResult(
            experiment_id=experiment_id,
            baseline=baseline_result,
            candidate=candidate_result,
            applied=True,
            token_reduction=comparison.token_reduction,
        )

    def _evaluate(
        self,
        prompt: Prompt[InputT],
        dataset: Dataset[InputT, ExpectedT],
        evaluator: Evaluator[OutputT, ExpectedT],
        experiment: Experiment,
    ) -> ExperimentResult:
        """Evaluate with experiment context."""
        report, token_count = self._eval_loop.run(
            prompt,
            dataset,
            evaluator,
            overrides_tag=experiment.overrides_tag,
            experiment=experiment,
        )
        return ExperimentResult(
            experiment=experiment,
            eval_report=report,
            token_count=token_count,
            evaluated_at=datetime.now(UTC),
        )

    def _write_proposal(
        self,
        prompt: Prompt[InputT],
        proposal: OptimizationProposal,
        tag: str,
    ) -> None:
        """Write proposal edits to override store."""
        for edit in proposal.edits:
            self._store.set_section_override(
                prompt,
                tag=tag,
                path=edit.path,
                body=edit.proposed_body,
            )
```

## EvalLoop Integration

EvalLoop must propagate experiment context and return token counts:

```python
class EvalLoop(Generic[InputT, OutputT, ExpectedT]):
    def run(
        self,
        prompt: Prompt[InputT],
        dataset: Dataset[InputT, ExpectedT],
        evaluator: Evaluator[OutputT, ExpectedT],
        *,
        overrides_tag: str = "latest",
        experiment: Experiment | None = None,
    ) -> tuple[EvalReport, int]:
        """Run evaluation, returning report and total token count."""
        ...
```

## Usage Example

```python
from weakincentives.evals import Dataset, EvalLoop, exact_match
from weakincentives.optimizers import OptimizationLoop, LLMCompressStrategy
from weakincentives.prompt.overrides import LocalPromptOverridesStore

# Setup
store = LocalPromptOverridesStore()
eval_loop = EvalLoop(loop=main_loop, evaluator=exact_match, ...)
strategy = LLMCompressStrategy(compression_prompt=compress_prompt)

opt_loop = OptimizationLoop(
    eval_loop=eval_loop,
    strategy=strategy,
    overrides_store=store,
)

# Run - automatically applies if no regression
dataset = Dataset.load(Path("evals/qa.jsonl"), str, str)
result = opt_loop.run(prompt, dataset, exact_match)

if result.applied:
    print(f"Saved {result.token_reduction} tokens with no regression")
else:
    if result.candidate:
        print("Compression caused regression - discarded")
    else:
        print("No compression possible")
```

## Limitations

- **Single iteration**: Runs one compression attempt per call; iterative
  compression requires multiple runs.
- **Binary regression check**: Any single regression blocks; no threshold-based
  acceptance.
- **Token counting**: Relies on EvalLoop to track and return token usage.
- **Alpha stability**: Interfaces may evolve without compatibility shims.

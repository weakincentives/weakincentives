# Optimization Loop Specification

## Purpose

Mailbox-driven prompt optimization that generates validated modifications
reducing token usage without regression. Compatible with LoopGroup for unified
worker deployments alongside MainLoop and EvalLoop.

## Guiding Principles

- **Mailbox-driven**: Receives requests, replies via `Message.reply()`
- **LoopGroup compatible**: Standard `run()`/`shutdown()` interface
- **EvalLoop composition**: Delegates evaluation via EvalLoop mailbox
- **Override-native**: Validates via temporary tags; no prompt patching
- **Shared storage**: Distributed workers share PromptOverridesStore backend

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Worker Process                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   LoopGroup                                                             │
│   ├── MainLoop      ←→ main_requests    (reply via msg.reply())         │
│   ├── EvalLoop      ←→ eval_requests    (reply via msg.reply())         │
│   └── OptimizeLoop  ←→ optimize_requests (reply via msg.reply())        │
│                                                                         │
│   Shared: PromptOverridesStore (Redis/S3/shared filesystem)             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Request and Result Types

```python
@dataclass(slots=True, frozen=True)
class OptimizeRequest(Generic[InputT, ExpectedT]):
    """Request to optimize a prompt."""

    descriptor: PromptDescriptor
    dataset: Dataset[InputT, ExpectedT]
    baseline_tag: str = "stable"
    eval_runs: int = 3  # Number of eval runs per candidate for reliability
    flags: Mapping[str, bool] = field(default_factory=dict)
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True, frozen=True)
class OptimizeResult:
    """Result of an optimization run."""

    request_id: UUID
    report: OptimizationReport
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None
```

## Experiment

An Experiment represents an isolated optimization context backed by an overrides
tag.

```python
@dataclass(slots=True, frozen=True)
class Experiment:
    """Isolated optimization context backed by an overrides tag."""

    id: str
    name: str
    overrides_tag: str
    parent_tag: str | None = None
    flags: Mapping[str, bool] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

Experiment is accessible in tool handlers via ToolContext:

```python
@property
def experiment(self) -> Experiment | None:
    return self.resources.get(Experiment)
```

## Experiment Propagation

For override resolution during prompt rendering, Experiment must flow through
the request chain. **The following changes are required to existing types:**

### MainLoopRequest (requires modification)

Add `experiment` field to `src/weakincentives/runtime/main_loop.py`:

```python
@FrozenDataclass()
class MainLoopRequest[UserRequestT]:
    request: UserRequestT
    budget: Budget | None = None
    deadline: Deadline | None = None
    resources: ResourceRegistry | None = None
    experiment: Experiment | None = None   # ADD THIS FIELD
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### EvalRequest (requires modification)

Add `experiment` field to `src/weakincentives/evals/_types.py`:

```python
@dataclass(slots=True, frozen=True)
class EvalRequest[InputT, ExpectedT]:
    sample: Sample[InputT, ExpectedT]
    experiment: Experiment | None = None   # ADD THIS FIELD
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

### Override Resolution Flow

When `experiment` is present on a request:

1. **MainLoop.prepare()** receives experiment via request
2. **Prompt rendering** uses `experiment.overrides_tag` for store resolution
3. **ToolContext** includes experiment in resources for tool handler access

```python
def prepare(self, request: UserRequestT, experiment: Experiment | None) -> tuple[Prompt[OutputT], Session]:
    prompt = Prompt(
        self._template,
        overrides_tag=experiment.overrides_tag if experiment else "stable",
    ).bind(params)
    # ...
```

## Compression Strategy

```python
class CompressStrategy(Protocol):
    """Generates compressed prompt variants.

    Implementations should prioritize sections by token count (largest first)
    and skip sections below a minimum token threshold where compression
    effort exceeds potential savings.
    """

    min_section_tokens: int  # Sections below this are skipped

    def propose(
        self,
        prompt: PromptDescriptor,
        *,
        adapter: ProviderAdapter[object],
        session: Session,
    ) -> Sequence[SectionEdit]:
        """Generate compressed sections.

        Returns edits ordered by original_tokens descending. Sections with
        fewer than min_section_tokens are excluded.
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
```

## Modification and Report

```python
@dataclass(slots=True, frozen=True)
class Modification:
    """A validated prompt modification ready for application."""

    section_path: tuple[str, ...]
    original_hash: HexDigest
    proposed_body: str
    token_reduction: int
    baseline_pass_rate: float
    candidate_pass_rate: float


@dataclass(slots=True, frozen=True)
class RejectedModification:
    """A modification rejected due to regression."""

    section_path: tuple[str, ...]
    token_reduction: int
    regression_count: int


@dataclass(slots=True, frozen=True)
class OptimizationReport:
    """Result of an optimization run."""

    experiment_id: str
    prompt_ns: str
    prompt_key: str
    baseline_pass_rate: float
    modifications: tuple[Modification, ...]
    rejected: tuple[RejectedModification, ...]

    @property
    def total_token_reduction(self) -> int:
        return sum(m.token_reduction for m in self.modifications)

    @property
    def has_modifications(self) -> bool:
        return len(self.modifications) > 0


@dataclass(slots=True, frozen=True)
class EvalSummary:
    """Aggregated results from multiple evaluation runs."""

    pass_rate: float  # Average pass rate across runs
    consistently_passed: frozenset[str]  # Sample IDs that passed in ALL runs

    @classmethod
    def from_reports(cls, reports: Sequence[EvalReport]) -> EvalSummary:
        """Aggregate multiple eval reports into a summary."""
        if not reports:
            return cls(pass_rate=0.0, consistently_passed=frozenset())

        # Average pass rate
        pass_rate = sum(r.pass_rate for r in reports) / len(reports)

        # Samples that passed in ALL runs
        passed_sets = [
            {r.sample_id for r in report.results if r.success and r.score.passed}
            for report in reports
        ]
        consistently_passed = frozenset.intersection(*passed_sets) if passed_sets else frozenset()

        return cls(
            pass_rate=pass_rate,
            consistently_passed=consistently_passed,
        )
```

## OptimizeLoop

```python
class OptimizeLoop(Generic[InputT, OutputT, ExpectedT]):
    """Mailbox-driven prompt optimization loop.

    Receives OptimizeRequest messages, generates and validates compressed
    prompt variants via EvalLoop, and replies with OptimizeResult.
    """

    def __init__(
        self,
        *,
        strategy: CompressStrategy,
        adapter: ProviderAdapter[object],
        overrides_store: PromptOverridesStore,
        eval_requests: Mailbox[EvalRequest[InputT, ExpectedT], EvalResult],
        requests: Mailbox[OptimizeRequest[InputT, ExpectedT], OptimizeResult],
    ) -> None:
        self._strategy = strategy
        self._adapter = adapter
        self._store = overrides_store
        self._eval_requests = eval_requests
        self._requests = requests
        self._shutdown_event = threading.Event()
        self._running = False
        self._lock = threading.Lock()
        self._current_dataset: Dataset[InputT, ExpectedT] | None = None

    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> None:
        """Process optimization requests from mailbox."""
        with self._lock:
            self._running = True
            self._shutdown_event.clear()

        iterations = 0
        try:
            while max_iterations is None or iterations < max_iterations:
                if self._shutdown_event.is_set():
                    break

                for msg in self._requests.receive(
                    visibility_timeout=visibility_timeout,
                    wait_time_seconds=wait_time_seconds,
                ):
                    if self._shutdown_event.is_set():
                        msg.nack()
                        break

                    try:
                        result = self._process_request(msg.body)
                        msg.reply(result)
                        msg.acknowledge()
                    except Exception as e:
                        self._handle_failure(msg, e)

                iterations += 1
        finally:
            with self._lock:
                self._running = False

    def shutdown(self, *, timeout: float = 30.0) -> bool:
        """Signal shutdown and wait for completion."""
        self._shutdown_event.set()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._running:
                    return True
            time.sleep(0.1)
        return False

    def _process_request(
        self, request: OptimizeRequest[InputT, ExpectedT]
    ) -> OptimizeResult:
        """Process a single optimization request."""
        self._current_dataset = request.dataset  # For combined/greedy validation
        report = self._optimize(
            descriptor=request.descriptor,
            dataset=request.dataset,
            baseline_tag=request.baseline_tag,
            flags=request.flags,
            eval_runs=request.eval_runs,
        )
        return OptimizeResult(request_id=request.request_id, report=report)

    def _optimize(
        self,
        descriptor: PromptDescriptor,
        dataset: Dataset[InputT, ExpectedT],
        baseline_tag: str,
        flags: Mapping[str, bool],
        eval_runs: int,
    ) -> OptimizationReport:
        """Core optimization logic."""
        experiment_id = generate_experiment_id()

        # 1. Evaluate baseline via EvalLoop (multiple runs for reliability)
        baseline_exp = Experiment(
            id=f"{experiment_id}-baseline",
            name="baseline",
            overrides_tag=baseline_tag,
            flags=dict(flags),
        )
        baseline_summary = self._run_eval_multi(dataset, baseline_exp, runs=eval_runs)
        baseline_passed = baseline_summary.consistently_passed

        # 2. Generate compression proposals (ordered by token count descending)
        edits = self._strategy.propose(
            descriptor,
            adapter=self._adapter,
            session=Session(bus=InProcessDispatcher()),
        )

        # 3. Evaluate each edit independently
        modifications: list[Modification] = []
        rejected: list[RejectedModification] = []

        for edit in edits:
            temp_tag = f"opt-{experiment_id}-{_hash_path(edit.path)}"

            try:
                self._store.store(
                    descriptor,
                    SectionOverride(
                        path=edit.path,
                        expected_hash=edit.original_hash,
                        body=edit.proposed_body,
                    ),
                    tag=temp_tag,
                )

                candidate_exp = Experiment(
                    id=f"{experiment_id}-{_hash_path(edit.path)}",
                    name=f"candidate-{edit.path[-1]}",
                    overrides_tag=temp_tag,
                    parent_tag=baseline_tag,
                    flags=dict(flags),
                )
                candidate_summary = self._run_eval_multi(
                    dataset, candidate_exp, runs=eval_runs
                )

                # Regression = passed consistently in baseline but not in candidate
                regression_count = len(
                    baseline_passed - candidate_summary.consistently_passed
                )

                if regression_count == 0:
                    modifications.append(Modification(
                        section_path=edit.path,
                        original_hash=edit.original_hash,
                        proposed_body=edit.proposed_body,
                        token_reduction=edit.original_tokens - edit.proposed_tokens,
                        baseline_pass_rate=baseline_summary.pass_rate,
                        candidate_pass_rate=candidate_summary.pass_rate,
                    ))
                else:
                    rejected.append(RejectedModification(
                        section_path=edit.path,
                        token_reduction=edit.original_tokens - edit.proposed_tokens,
                        regression_count=regression_count,
                    ))
            finally:
                self._store.delete(
                    ns=descriptor.ns, prompt_key=descriptor.key, tag=temp_tag
                )

        # 4. Test combined modifications for interaction effects
        if len(modifications) > 1:
            modifications = self._validate_combined(
                descriptor=descriptor,
                experiment_id=experiment_id,
                baseline_tag=baseline_tag,
                baseline_passed=baseline_passed,
                modifications=modifications,
                rejected=rejected,
                flags=flags,
                eval_runs=eval_runs,
            )

        return OptimizationReport(
            experiment_id=experiment_id,
            prompt_ns=descriptor.ns,
            prompt_key=descriptor.key,
            baseline_pass_rate=baseline_summary.pass_rate,
            modifications=tuple(modifications),
            rejected=tuple(rejected),
        )

    def _run_eval_multi(
        self,
        dataset: Dataset[InputT, ExpectedT],
        experiment: Experiment,
        *,
        runs: int,
    ) -> EvalSummary:
        """Run evaluation multiple times and summarize results.

        Returns summary with pass rates averaged across runs and the set
        of sample IDs that passed consistently in all runs.
        """
        all_results: list[EvalReport] = []
        for i in range(runs):
            run_exp = Experiment(
                id=f"{experiment.id}-run{i}",
                name=experiment.name,
                overrides_tag=experiment.overrides_tag,
                parent_tag=experiment.parent_tag,
                flags=experiment.flags,
            )
            all_results.append(self._run_eval(dataset, run_exp))

        return EvalSummary.from_reports(all_results)

    def _validate_combined(
        self,
        descriptor: PromptDescriptor,
        experiment_id: str,
        baseline_tag: str,
        baseline_passed: frozenset[str],
        modifications: list[Modification],
        rejected: list[RejectedModification],
        flags: Mapping[str, bool],
        eval_runs: int,
    ) -> list[Modification]:
        """Test all modifications applied together for interaction effects.

        If combined application causes regressions, falls back to greedy
        selection: adds modifications one at a time, keeping only those
        that don't regress when combined with previously accepted ones.
        """
        combined_tag = f"opt-{experiment_id}-combined"

        try:
            # Write all modifications to combined tag
            for mod in modifications:
                self._store.store(
                    descriptor,
                    SectionOverride(
                        path=mod.section_path,
                        expected_hash=mod.original_hash,
                        body=mod.proposed_body,
                    ),
                    tag=combined_tag,
                )

            combined_exp = Experiment(
                id=f"{experiment_id}-combined",
                name="combined",
                overrides_tag=combined_tag,
                parent_tag=baseline_tag,
                flags=dict(flags),
            )
            combined_summary = self._run_eval_multi(
                self._current_dataset, combined_exp, runs=eval_runs
            )

            regression_count = len(
                baseline_passed - combined_summary.consistently_passed
            )

            if regression_count == 0:
                # All modifications work together
                return modifications

            # Combined fails - fall back to greedy selection
            return self._greedy_select(
                descriptor=descriptor,
                experiment_id=experiment_id,
                baseline_tag=baseline_tag,
                baseline_passed=baseline_passed,
                candidates=modifications,
                rejected=rejected,
                flags=flags,
                eval_runs=eval_runs,
            )
        finally:
            self._store.delete(
                ns=descriptor.ns, prompt_key=descriptor.key, tag=combined_tag
            )

    def _greedy_select(
        self,
        descriptor: PromptDescriptor,
        experiment_id: str,
        baseline_tag: str,
        baseline_passed: frozenset[str],
        candidates: list[Modification],
        rejected: list[RejectedModification],
        flags: Mapping[str, bool],
        eval_runs: int,
    ) -> list[Modification]:
        """Greedy selection: add modifications one at a time.

        Candidates are processed in order (largest token reduction first).
        Each is added only if it doesn't cause regression when combined
        with already-accepted modifications. If rejected, the tag is rebuilt
        without the rejected modification.
        """
        accepted: list[Modification] = []
        greedy_tag = f"opt-{experiment_id}-greedy"

        try:
            for mod in sorted(candidates, key=lambda m: -m.token_reduction):
                # Add candidate to greedy tag
                self._store.store(
                    descriptor,
                    SectionOverride(
                        path=mod.section_path,
                        expected_hash=mod.original_hash,
                        body=mod.proposed_body,
                    ),
                    tag=greedy_tag,
                )

                greedy_exp = Experiment(
                    id=f"{experiment_id}-greedy-{len(accepted)}",
                    name=f"greedy-{len(accepted)}",
                    overrides_tag=greedy_tag,
                    parent_tag=baseline_tag,
                    flags=dict(flags),
                )
                summary = self._run_eval_multi(
                    self._current_dataset, greedy_exp, runs=eval_runs
                )

                regression_count = len(
                    baseline_passed - summary.consistently_passed
                )

                if regression_count == 0:
                    accepted.append(mod)
                else:
                    # Rejected: rebuild tag with only accepted modifications
                    rejected.append(RejectedModification(
                        section_path=mod.section_path,
                        token_reduction=mod.token_reduction,
                        regression_count=regression_count,
                    ))
                    self._store.delete(
                        ns=descriptor.ns, prompt_key=descriptor.key, tag=greedy_tag
                    )
                    for accepted_mod in accepted:
                        self._store.store(
                            descriptor,
                            SectionOverride(
                                path=accepted_mod.section_path,
                                expected_hash=accepted_mod.original_hash,
                                body=accepted_mod.proposed_body,
                            ),
                            tag=greedy_tag,
                        )

            return accepted
        finally:
            self._store.delete(
                ns=descriptor.ns, prompt_key=descriptor.key, tag=greedy_tag
            )

    def _run_eval(
        self,
        dataset: Dataset[InputT, ExpectedT],
        experiment: Experiment,
        *,
        timeout: float = 600.0,
    ) -> EvalReport:
        """Submit samples to EvalLoop and collect results via reply.

        Args:
            dataset: Samples to evaluate.
            experiment: Experiment context for override resolution.
            timeout: Maximum seconds to wait for all results.

        Raises:
            TimeoutError: If not all results received within timeout.
        """
        # Create reply mailbox for this eval run
        replies: Mailbox[EvalResult, None] = InMemoryMailbox(
            name=f"opt-eval-{experiment.id}"
        )

        # Submit all samples with experiment context, replies go to our mailbox
        for sample in dataset:
            self._eval_requests.send(
                EvalRequest(sample=sample, experiment=experiment),
                reply_to=replies,
            )

        # Collect results with deadline
        results: list[EvalResult] = []
        expected = len(dataset)
        deadline = time.monotonic() + timeout

        while len(results) < expected:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Eval timed out: received {len(results)}/{expected} results"
                )
            for msg in replies.receive(wait_time_seconds=min(30, int(remaining))):
                results.append(msg.body)
                msg.acknowledge()

        return EvalReport(results=tuple(results))

    def _handle_failure(
        self,
        msg: Message[OptimizeRequest[InputT, ExpectedT]],
        error: Exception,
    ) -> None:
        """Handle optimization failure."""
        try:
            msg.reply(OptimizeResult(
                request_id=msg.body.request_id,
                report=OptimizationReport(
                    experiment_id="failed",
                    prompt_ns=msg.body.descriptor.ns,
                    prompt_key=msg.body.descriptor.key,
                    baseline_pass_rate=0.0,
                    modifications=(),
                    rejected=(),
                ),
                error=str(error),
            ))
            msg.acknowledge()
        except Exception:
            msg.nack(visibility_timeout=min(60 * msg.delivery_count, 900))
```

## Applying Modifications

```python
def apply_modifications(
    store: PromptOverridesStore,
    descriptor: PromptDescriptor,
    modifications: Sequence[Modification],
    *,
    tag: str = "stable",
) -> None:
    """Apply validated modifications to the override store."""
    for mod in modifications:
        store.store(
            descriptor,
            SectionOverride(
                path=mod.section_path,
                expected_hash=mod.original_hash,
                body=mod.proposed_body,
            ),
            tag=tag,
        )
```

## Shared Storage Requirement

In distributed deployments, all workers must use a `PromptOverridesStore`
backed by shared storage:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Worker 1   │     │   Worker 2   │     │   Worker 3   │
│ OptimizeLoop │     │   EvalLoop   │     │   MainLoop   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                    ┌───────▼───────┐
                    │ Shared Store  │
                    │ (Redis/S3/NFS)│
                    └───────────────┘
```

**Why shared storage is required:**

1. OptimizeLoop writes temporary override tags during candidate evaluation
2. EvalLoop workers (possibly on different machines) must resolve those tags
3. MainLoop workers render prompts using the same override resolution

**Supported backends:**

- `RedisPromptOverridesStore` - Redis-backed for low-latency distributed access
- `S3PromptOverridesStore` - S3-backed for serverless deployments
- `LocalPromptOverridesStore` - Filesystem-backed (single machine only)

## Usage Example

### Worker with All Three Loops

```python
from weakincentives.runtime import MainLoop, LoopGroup, RedisMailbox
from weakincentives.evals import EvalLoop
from weakincentives.optimizers import OptimizeLoop, LLMCompressStrategy
from weakincentives.prompt.overrides import RedisPromptOverridesStore

# Shared override store (all workers use same Redis)
store = RedisPromptOverridesStore(client=redis)

# Mailboxes
eval_requests = RedisMailbox(name="eval-requests", client=redis)
optimize_requests = RedisMailbox(name="optimize-requests", client=redis)

# Loops
main_loop = MyMainLoop(adapter=adapter, bus=bus, overrides_store=store)
eval_loop = EvalLoop(loop=main_loop, requests=eval_requests)
optimize_loop = OptimizeLoop(
    strategy=LLMCompressStrategy(...),
    adapter=adapter,
    overrides_store=store,
    eval_requests=eval_requests,
    requests=optimize_requests,
)

# Run all loops
group = LoopGroup(loops=[main_loop, eval_loop, optimize_loop])
group.run()
```

### Submitting Optimization Request

```python
from weakincentives.evals import Dataset, Sample
from weakincentives.prompt.overrides import PromptDescriptor

# Get descriptor from prompt
descriptor = PromptDescriptor.from_prompt(prompt)

# Build dataset
dataset = Dataset(samples=tuple(
    Sample(id=str(i), input=inp, expected=exp)
    for i, (inp, exp) in enumerate(test_cases)
))

# Create reply mailbox
my_replies: Mailbox[OptimizeResult, None] = RedisMailbox(
    name=f"opt-replies-{uuid4()}", client=redis
)

# Submit request with reply mailbox
optimize_requests.send(
    OptimizeRequest(
        descriptor=descriptor,
        dataset=dataset,
        baseline_tag="stable",
    ),
    reply_to=my_replies,
)

# Collect result
for msg in my_replies.receive(wait_time_seconds=300):
    result = msg.body
    if result.report.has_modifications:
        apply_modifications(store, descriptor, result.report.modifications)
    msg.acknowledge()
```

## Override Store Integration

Uses the existing `PromptOverridesStore` protocol from `specs/PROMPT_OPTIMIZATION.md`:

- `store(prompt, override, *, tag)` - Write proposed edits via `SectionOverride`
- `delete(*, ns, prompt_key, tag)` - Cleanup temporary experiment tags

## Spec Dependencies

This spec requires updates to:

- **EVALS.md**: Update `EvalRequest` to include `experiment` field
- **MAIN_LOOP.md**: Update `MainLoopRequest` to include `experiment` field

## Limitations

- **Sequential sample submission**: Samples submitted to EvalLoop sequentially
- **Greedy combined selection**: When modifications interact, greedy selection
  may not find the optimal subset (NP-hard in general)
- **Shared storage required**: Distributed deployments need shared override store
- **Evaluation cost**: Multiple runs per candidate (default 3) plus combined
  testing can be expensive for large datasets
- **Alpha stability**: Interfaces may evolve without compatibility shims

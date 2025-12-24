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

    prompt_ns: str
    prompt_key: str
    dataset: Dataset[InputT, ExpectedT]
    baseline_tag: str = "stable"
    flags: dict[str, bool] = field(default_factory=dict)
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
    flags: dict[str, bool] = field(default_factory=dict)
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
    """Generates compressed prompt variants."""

    def propose(
        self,
        prompt: PromptDescriptor,
        *,
        adapter: ProviderAdapter[object],
        session: Session,
    ) -> Sequence[SectionEdit]:
        """Generate compressed sections."""
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
    baseline_token_count: int
    modifications: tuple[Modification, ...]
    rejected: tuple[RejectedModification, ...]

    @property
    def total_token_reduction(self) -> int:
        return sum(m.token_reduction for m in self.modifications)

    @property
    def has_modifications(self) -> bool:
        return len(self.modifications) > 0
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
        overrides_store: PromptOverridesStore,
        prompt_registry: PromptRegistry,
        eval_requests: Mailbox[EvalRequest[InputT, ExpectedT], EvalResult],
        requests: Mailbox[OptimizeRequest[InputT, ExpectedT], OptimizeResult],
    ) -> None:
        self._strategy = strategy
        self._store = overrides_store
        self._prompts = prompt_registry
        self._eval_requests = eval_requests
        self._requests = requests
        self._shutdown_event = threading.Event()
        self._running = False
        self._lock = threading.Lock()

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
        prompt = self._prompts.get(request.prompt_ns, request.prompt_key)
        report = self._optimize(
            prompt=prompt,
            dataset=request.dataset,
            baseline_tag=request.baseline_tag,
            flags=request.flags,
        )
        return OptimizeResult(request_id=request.request_id, report=report)

    def _optimize(
        self,
        prompt: Prompt[InputT],
        dataset: Dataset[InputT, ExpectedT],
        baseline_tag: str,
        flags: dict[str, bool],
    ) -> OptimizationReport:
        """Core optimization logic."""
        experiment_id = generate_experiment_id()
        descriptor = descriptor_for_prompt(prompt)

        # 1. Evaluate baseline via EvalLoop
        baseline_exp = Experiment(
            id=f"{experiment_id}-baseline",
            name="baseline",
            overrides_tag=baseline_tag,
            flags=flags,
        )
        baseline_report = self._run_eval(dataset, baseline_exp)
        baseline_passed = {
            r.sample_id for r in baseline_report.results
            if r.success and r.score.passed
        }

        # 2. Generate compression proposals
        edits = self._strategy.propose(
            descriptor,
            adapter=self._get_adapter(),
            session=Session(bus=InProcessDispatcher()),
        )

        # 3. Evaluate each edit via EvalLoop
        modifications: list[Modification] = []
        rejected: list[RejectedModification] = []

        for edit in edits:
            temp_tag = f"opt-{experiment_id}-{_hash_path(edit.path)}"

            try:
                # Write to shared override store using SectionOverride
                self._store.store(
                    prompt,
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
                    flags=flags,
                )
                candidate_report = self._run_eval(dataset, candidate_exp)

                regression_count = sum(
                    1 for r in candidate_report.results
                    if r.sample_id in baseline_passed and not r.score.passed
                )

                if regression_count == 0:
                    modifications.append(Modification(
                        section_path=edit.path,
                        original_hash=edit.original_hash,
                        proposed_body=edit.proposed_body,
                        token_reduction=edit.original_tokens - edit.proposed_tokens,
                        baseline_pass_rate=baseline_report.pass_rate,
                        candidate_pass_rate=candidate_report.pass_rate,
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

        return OptimizationReport(
            experiment_id=experiment_id,
            prompt_ns=descriptor.ns,
            prompt_key=descriptor.key,
            baseline_pass_rate=baseline_report.pass_rate,
            baseline_token_count=0,  # Aggregated from eval results
            modifications=tuple(modifications),
            rejected=tuple(rejected),
        )

    def _run_eval(
        self,
        dataset: Dataset[InputT, ExpectedT],
        experiment: Experiment,
    ) -> EvalReport:
        """Submit samples to EvalLoop and collect results via reply."""
        # Create reply mailbox for this eval run
        eval_id = f"opt-eval-{experiment.id}"
        replies: Mailbox[EvalResult, None] = InMemoryMailbox(name=eval_id)

        # Submit all samples with experiment context
        for sample in dataset:
            self._eval_requests.send(
                EvalRequest(sample=sample, experiment=experiment),
                reply_to=eval_id,
            )

        # Collect results from replies
        results: list[EvalResult] = []
        expected = len(dataset)
        while len(results) < expected:
            for msg in replies.receive(wait_time_seconds=30):
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
                    prompt_ns=msg.body.prompt_ns,
                    prompt_key=msg.body.prompt_key,
                    baseline_pass_rate=0.0,
                    baseline_token_count=0,
                    modifications=(),
                    rejected=(),
                ),
                error=str(error),
            ))
            msg.acknowledge()
        except Exception:
            msg.nack(visibility_timeout=min(60 * msg.delivery_count, 900))
```

## Prompt Registry

```python
class PromptRegistry(Protocol):
    """Resolves prompt references."""

    def get(self, ns: str, key: str) -> Prompt[Any]: ...
```

## Applying Modifications

```python
def apply_modifications(
    store: PromptOverridesStore,
    prompt: Prompt,
    modifications: Sequence[Modification],
    *,
    tag: str = "stable",
) -> None:
    """Apply validated modifications to the override store."""
    for mod in modifications:
        store.store(
            prompt,
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

# Mailboxes with reply resolver
resolver = RedisMailboxResolver(client=redis)
eval_requests = RedisMailbox(name="eval-requests", client=redis, reply_resolver=resolver)
optimize_requests = RedisMailbox(name="optimize-requests", client=redis, reply_resolver=resolver)

# Loops
main_loop = MyMainLoop(adapter=adapter, bus=bus, overrides_store=store)
eval_loop = EvalLoop(loop=main_loop, requests=eval_requests)
optimize_loop = OptimizeLoop(
    strategy=LLMCompressStrategy(...),
    overrides_store=store,
    prompt_registry=prompts,
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

# Build dataset
dataset = Dataset(samples=tuple(
    Sample(id=str(i), input=inp, expected=exp)
    for i, (inp, exp) in enumerate(test_cases)
))

# Create reply mailbox
my_replies = RedisMailbox(name=f"opt-replies-{uuid4()}", client=redis)
resolver.register(my_replies.name, my_replies)

# Submit request
optimize_requests.send(
    OptimizeRequest(
        prompt_ns="agent",
        prompt_key="code-review",
        dataset=dataset,
        baseline_tag="stable",
    ),
    reply_to=my_replies.name,
)

# Collect result via reply
for msg in my_replies.receive(wait_time_seconds=300):
    result = msg.body
    if result.report.has_modifications:
        apply_modifications(store, prompt, result.report.modifications)
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

Note: The EVALS.md spec text shows the two-mailbox pattern but the code
(`src/weakincentives/evals/_loop.py`) already uses `msg.reply()`. The spec
should be updated to match the implementation.

## Limitations

- **Sequential sample submission**: Samples submitted to EvalLoop sequentially
- **Independent edits**: Each edit evaluated independently
- **Binary regression**: Any single regression rejects the modification
- **Shared storage required**: Distributed deployments need shared override store
- **Alpha stability**: Interfaces may evolve without compatibility shims

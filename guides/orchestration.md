# Orchestration with MainLoop

*Canonical spec: [specs/MAIN_LOOP.md](../specs/MAIN_LOOP.md)*

`MainLoop` exists for one reason:

> Make progressive disclosure and budgets/deadlines easy to handle correctly.

You could write the loop yourself. MainLoop just does it in a tested, consistent
way.

## The Minimal MainLoop

You implement a single method:

- `prepare(request) -> tuple[Prompt[OutputT], Session]`

Then call `loop.execute(request)`.

```python
from weakincentives.runtime import MainLoop, MainLoopConfig, Session
from weakincentives.prompt import Prompt

class MyLoop(MainLoop[RequestType, OutputType]):
    def prepare(self, request: RequestType) -> tuple[Prompt[OutputType], Session]:
        prompt = Prompt(self._template).bind(request)
        session = Session(tags={"loop": "my-loop"})
        return prompt, session
```

`MainLoop` handles `VisibilityExpansionRequired` automatically. When the model
calls `open_sections`, MainLoop applies the visibility overrides and re-evaluates
the prompt. You don't have to handle this yourself.

## Configuring MainLoop with Resources

You can inject custom resources at the loop level via `MainLoopConfig`:

```python
from weakincentives.resources import Binding, ResourceRegistry
from weakincentives.runtime import MainLoopConfig

# Simple case: pre-constructed instances
resources = ResourceRegistry.of(Binding.instance(HTTPClient, http_client))

# Or with lazy construction
resources = ResourceRegistry.of(
    Binding(Config, lambda r: Config()),
    Binding(HTTPClient, lambda r: HTTPClient()),
)

config = MainLoopConfig(resources=resources)
loop = MyLoop(adapter=adapter, dispatcher=dispatcher, config=config)
response, session = loop.execute(request)
```

Resources configured this way are available to all tool handlers during
execution. You can also pass resources directly to
`loop.execute(request, resources=...)` for per-request overrides.

## Deadlines and Budgets

`Deadline` is a wall-clock deadline. `Budget` can include token limits and/or a
deadline. `BudgetTracker` accumulates usage across retries.

**Typical pattern:**

```python
from datetime import timedelta
from weakincentives import Deadline, Budget

deadline = Deadline.from_timeout(timedelta(seconds=30))
budget = Budget(max_total_tokens=20_000)

response, session = loop.execute(request, deadline=deadline, budget=budget)
```

Deadlines prevent runaway agents. Budgets prevent runaway costs. Both are
enforced at the adapter level, so they work consistently across providers.

## What MainLoop Does For You

1. **Prepares the prompt**: Calls your `prepare()` method to get a bound prompt
   and session.

1. **Handles progressive disclosure**: When the model calls `open_sections`,
   MainLoop catches `VisibilityExpansionRequired`, applies visibility overrides
   to the session, and retries the evaluation.

1. **Enforces deadlines**: Passes the deadline to the adapter, which will abort
   if time runs out.

1. **Tracks budgets**: Passes the budget tracker to the adapter to accumulate
   token usage.

1. **Manages resources**: Binds resources to the prompt and handles lifecycle.

1. **Returns results**: Returns the `PromptResponse` and the session for
   inspection.

## When to Use MainLoop

Use `MainLoop` when:

- You're building a request-response agent
- You want progressive disclosure to "just work"
- You need deadline and budget enforcement
- You want consistent resource handling

You might skip `MainLoop` when:

- You're doing simple one-off evaluations during development
- You need custom retry logic that doesn't fit MainLoop's pattern
- You're building something that doesn't fit the request-response model

## Next Steps

- [Evaluation](evaluation.md): Use EvalLoop to test your MainLoop
- [Lifecycle](lifecycle.md): Run multiple loops with LoopGroup
- [Progressive Disclosure](progressive-disclosure.md): Understand visibility

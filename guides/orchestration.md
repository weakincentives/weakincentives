# Orchestration with AgentLoop

*Canonical spec: [specs/AGENT_LOOP.md](../specs/AGENT_LOOP.md)*

`AgentLoop` exists for one reason:

> Make progressive disclosure and budgets/deadlines easy to handle correctly.

You could write the loop yourself. AgentLoop just does it in a tested, consistent
way.

## The Minimal AgentLoop

You implement a single method:

- `prepare(request) -> tuple[Prompt[OutputT], Session]`

Then call `loop.execute(request)`.

```python nocheck
from weakincentives.runtime import AgentLoop, AgentLoopConfig, Session
from weakincentives.prompt import Prompt

class MyLoop(AgentLoop[RequestType, OutputType]):
    def prepare(self, request: RequestType) -> tuple[Prompt[OutputType], Session]:
        prompt = Prompt(self._template).bind(request)
        session = Session(tags={"loop": "my-loop"})
        return prompt, session
```

`AgentLoop` handles `VisibilityExpansionRequired` automatically. When the model
calls `open_sections`, AgentLoop applies the visibility overrides and re-evaluates
the prompt. You don't have to handle this yourself.

## Configuring AgentLoop with Resources

You can inject custom resources at the loop level via `AgentLoopConfig`:

```python nocheck
from weakincentives.resources import Binding, ResourceRegistry
from weakincentives.runtime import AgentLoopConfig

# Simple case: pre-constructed instances
resources = ResourceRegistry.of(Binding.instance(HTTPClient, http_client))

# Or with lazy construction
resources = ResourceRegistry.of(
    Binding(Config, lambda r: Config()),
    Binding(HTTPClient, lambda r: HTTPClient()),
)

config = AgentLoopConfig(resources=resources)
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

```python nocheck
from datetime import timedelta
from weakincentives import Deadline, Budget

deadline = Deadline.from_timeout(timedelta(seconds=30))
budget = Budget(max_total_tokens=20_000)

response, session = loop.execute(request, deadline=deadline, budget=budget)
```

Deadlines prevent runaway agents. Budgets prevent runaway costs. Both are
enforced at the adapter level, so they work consistently across providers.

## What AgentLoop Does For You

1. **Prepares the prompt**: Calls your `prepare()` method to get a bound prompt
   and session.

1. **Handles progressive disclosure**: When the model calls `open_sections`,
   AgentLoop catches `VisibilityExpansionRequired`, applies visibility overrides
   to the session, and retries the evaluation.

1. **Enforces deadlines**: Passes the deadline to the adapter, which will abort
   if time runs out.

1. **Tracks budgets**: Passes the budget tracker to the adapter to accumulate
   token usage.

1. **Manages resources**: Binds resources to the prompt and handles lifecycle.

1. **Returns results**: Returns the `PromptResponse` and the session for
   inspection.

## When to Use AgentLoop

Use `AgentLoop` when:

- You're building a request-response agent
- You want progressive disclosure to "just work"
- You need deadline and budget enforcement
- You want consistent resource handling

You might skip `AgentLoop` when:

- You're doing simple one-off evaluations during development
- You need custom retry logic that doesn't fit AgentLoop's pattern
- You're building something that doesn't fit the request-response model

## Next Steps

- [Evaluation](evaluation.md): Use EvalLoop to test your AgentLoop
- [Lifecycle](lifecycle.md): Run multiple loops with LoopGroup
- [Progressive Disclosure](progressive-disclosure.md): Understand visibility

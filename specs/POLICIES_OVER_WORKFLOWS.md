# Policies Over Workflows

Design philosophy for unattended agents: prefer declarative policies over prescriptive workflows.

## The Problem with Workflows

Workflows encode predetermined sequences:

```
1. Read file → 2. Parse AST → 3. Generate patch → 4. Write file → 5. Run tests
```

When unexpected situations arise (file doesn't exist, AST fails, tests need setup), workflows fail, skip steps, or spawn decision tree branches.

**Fundamental issue**: Workflows encode *how* to accomplish a goal, not *what* the goal requires.

## Policies Preserve Agency

A policy is a declarative constraint:

- "A file must be read before it can be overwritten"
- "Tests must pass before deployment"

Policies describe invariants that must hold. The agent remains free to find any path satisfying the constraints.

| Aspect | Workflow | Policy |
|--------|----------|--------|
| Specifies | Steps to execute | Constraints to satisfy |
| On unexpected input | Fails or branches | Agent reasons about constraints |
| Composability | Sequential coupling | Independent conjunction |
| Agent role | Executor | Reasoner |

## Characteristics of Good Policies

### Declarative, Not Procedural

```python
# Procedural (workflow)
def deploy(): run_tests(); build(); push()

# Declarative (policy)
policy = SequentialDependencyPolicy(
    dependencies={"deploy": frozenset({"test", "build"})}
)
```

### Independently Composable

```python
policies = [ReadBeforeWritePolicy(), RequireReviewPolicy(), BudgetLimitPolicy()]
for policy in policies:
    if not policy.check(tool, params, context=context).allowed:
        return deny()
```

### Fail-Closed by Default

When uncertain, deny. The agent can then reason about why denial occurred.

### Observable and Debuggable

```python
return PolicyDecision.deny(
    f"File '{path}' must be read before overwriting. Read {', '.join(read_paths)} so far."
)
```

## Policies in WINK

- **Tool Policies**: Gate tool invocations based on session state (`TOOL_POLICIES.md`)
- **Feedback Providers**: Soft guidance based on patterns (`FEEDBACK_PROVIDERS.md`)
- **Budget Constraints**: Hard limits on resource consumption (`SESSIONS.md`)
- **Task Completion**: Verify goal achievement without prescribing how (`TASK_COMPLETION.md`)

## Anti-Patterns

- **Workflow in policy clothing**: Sequential dependencies that leave only one valid path
- **Over-constraining**: Policies that eliminate all flexibility
- **State machine policies**: Conflating enforcement with workflow execution

## When Workflows Are Appropriate

- Sequence is truly invariant (protocol handshakes, transaction ordering)
- Failure is preferable to adaptation
- Agent lacks reasoning capability
- Human oversight is continuous

For unattended agents with LLM reasoning, policies are almost always the better default.

## Summary

| Principle | Implication |
|-----------|-------------|
| Preserve agency | Let the agent reason; don't script actions |
| Declare constraints | State invariants, not procedures |
| Compose independently | Policies should be conjunction-friendly |
| Fail closed | Deny when uncertain; let agent adapt |
| Surface reasoning | Explain denials to enable self-correction |

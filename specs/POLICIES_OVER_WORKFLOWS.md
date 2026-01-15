# Policies Over Workflows

## Purpose

Design philosophy for unattended agents: prefer declarative policies over
prescriptive workflows. Policies define constraints that preserve agent
reasoning; workflows encode rigid sequences that fracture on edge cases.

## The Problem with Workflows

A workflow is predetermined steps: read → parse → identify → patch → test.

When encountering unexpected states:
- **Fail** - Abort on error
- **Skip** - Continue with invalid state
- **Branch** - Add conditional logic (spawns decision trees)

**Fundamental issue:** Workflows encode *how* to accomplish goals, not *what*
goals require. When "how" breaks, the agent has no recourse.

## Policies Preserve Agency

Policies describe invariants that must hold, not sequences to execute:

- "A file must be read before it can be overwritten"
- "Tests must pass before deployment"
- "Sensitive operations require confirmation"

The agent remains free to find any path satisfying constraints.

| Aspect | Workflow | Policy |
|--------|----------|--------|
| Specifies | Steps to execute | Constraints to satisfy |
| On unexpected | Fails or branches | Agent reasons |
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

Each policy evaluable in isolation. Compose through conjunction.

### Fail-Closed by Default

When uncertain, deny. Agent reasons about why and adjusts.

### Observable and Debuggable

Expose reasoning. Denial feedback enables self-correction.

## Policies in WINK

| Mechanism | Description | Spec |
|-----------|-------------|------|
| Tool Policies | Gate tool invocations | `TOOL_POLICIES.md` |
| Feedback Providers | Soft guidance over time | `FEEDBACK_PROVIDERS.md` |
| Budget Constraints | Resource limits | `SESSIONS.md` |
| Task Completion | Goal verification | `TASK_COMPLETION.md` |

## Anti-Patterns

- **Workflow in policy clothing**: Sequential chain as dependencies
- **Over-constraining**: Policies that leave only one valid path
- **Stateful policies**: Conflating enforcement with orchestration

## When Workflows Are Appropriate

1. Sequence is truly invariant (protocol handshakes)
2. Failure preferable to adaptation
3. Agent lacks reasoning capability
4. Human oversight is continuous

For unattended agents with LLM reasoning, policies are almost always better.

## Summary

| Principle | Implication |
|-----------|-------------|
| Preserve agency | Let agent reason; don't script |
| Declare constraints | Invariants, not procedures |
| Compose independently | Conjunction-friendly |
| Fail closed | Deny when uncertain |
| Surface reasoning | Explain denials |

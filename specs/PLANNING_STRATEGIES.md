# Planning Strategies

This specification enumerates the supported planning configurations for prompts that expose a dedicated planning section. It focuses on richer orchestration patterns beyond the default ReAct loop and describes how to capture, execute, and inspect structured plans.

## Baseline: ReAct With Plan Integration

The system continues to support a lightweight ReAct-style loop in which the plan serves as a coordination surface. Agents may:

- Draft or update the plan inline before tool calls when additional context is required.
- Reference plan steps during the ReAct cycle to justify tool usage or adjust course.
- Terminate once the plan's exit criteria are satisfied or explicitly note why the plan was abandoned.

The following sections extend this baseline with more formal planning regimes.

## Pattern 1: Plan → Act → Reflect (PAR)

### Overview

The PAR loop enforces a typed planning artifact followed by ordered execution and reflection. It is intended for high-clarity workflows where rollback and auditing matter.

### Required Plan Structure

- **Steps**: Ordered, uniquely identified units of work, each linked to intended tools or reasoning paths.
- **Tools**: Explicit declarations of tool handles (name, version, parameters) expected for each step.
- **Exit Criteria**: Objective conditions that signal completion or trigger reevaluation.

Plans are authored before any tool invocation. Updates to the plan must be versioned (e.g., `plan.v1`, `plan.v2`) to preserve diffs and support rollback.

### Execution Semantics

- **Plan**: The agent emits a `PlanAuthored` event containing the full typed plan (schema described in "Typed event traces").
- **Act**: Each tool call is executed step-by-step. The agent references the corresponding plan step ID and records a `ToolInvoked` event followed by a `ToolReturned` event.
- **Reflect**: After completing a step or upon receiving new evidence, the agent evaluates outcomes against the exit criteria. Reflections are logged as `ReflectionAdded` events, optionally producing a revised plan (`PlanUpdated`).

### Benefits

- Minimizes dead-ends by validating each step against exit criteria.
- Creates clear rollback points through plan versioning.
- Simplifies diffing plan revisions thanks to typed schemas.

## Pattern 2: Multi-stage Policies (DeepAgent-style)

### Overview

Multi-stage policies split the overall reasoning policy into discrete stages, each with its own prompt, schema, and guardrails. This configuration is well-suited for complex tasks that benefit from specialized reasoning modules.

### Stage Configuration

- **Stage Definitions**: Enumerate ordered stages (e.g., `goal_framing`, `decomposition`, `tool_routing`, `synthesis`). Each stage must define:
  - A stage-specific prompt template or instruction block.
  - Input/output schemas describing expected fields and validations.
  - Guardrails (e.g., max tokens, allowed tool namespaces, safety filters).
- **Transition Rules**: Specify how outputs of one stage feed into the next, including failure handling (retry, fallback stage, or termination).
- **Per-stage Tooling**: Stages may have restricted tool access; the planner must note these constraints in the plan artifact.

### Execution Semantics

- The plan describes the stage sequence, required artifacts, and stage-level exit criteria.
- Each stage emits events indicating activation (`StageStarted`), produced outputs, and completion (`StageCompleted`). These can be modeled using the general event schema under the "Typed event traces" pattern.
- If a stage detects irrecoverable issues, it emits a reflection and either rewinds to an earlier stage or terminates with rationale.

### Benefits

- Encourages smaller, specialized prompts that are easier to optimize.
- Enables per-stage evaluation and targeted improvements.
- Supports targeted guardrails, ensuring each stage adheres to its responsibilities.

## Pattern 3: Typed Event Traces (First-class Logs)

### Overview

Typed event traces treat every significant action as an append-only event conforming to a shared schema. This pattern underpins deterministic replay and advanced analytics.

### Event Schema

All events share a core envelope:

```json
{
  "event_id": "uuid",
  "event_type": "PlanAuthored | PlanUpdated | ToolInvoked | ToolReturned | ReflectionAdded | RunTerminated | StageStarted | StageCompleted | ...",
  "timestamp": "ISO-8601",
  "actor": "agent identifier",
  "references": {
    "plan_id": "stable plan/version identifier",
    "step_id": "plan step identifier",
    "prompt_version": "semver or git SHA",
    "tool_version": "version string",
    "section_key": "planning | action | reflection",
    "namespace": "logical grouping"
  },
  "payload": { /* event-specific typed fields */ }
}
```

Recommended payloads include:

- **PlanAuthored / PlanUpdated**: Full plan structure, including steps, tools, dependencies, and exit criteria.
- **ToolInvoked**: Tool name, input parameters, referenced plan step ID, expected outcome signature.
- **ToolReturned**: Success flag, output payload, errors, and latency metrics.
- **ReflectionAdded**: Analysis of observed vs. expected outcomes, proposed adjustments.
- **RunTerminated**: Reason for termination (success, manual stop, guardrail violation).

Events are never mutated; corrections appear as new events referencing prior IDs. Stable references allow downstream systems to reconstruct timelines and perform optimizer-friendly analytics.

### Benefits

- Guarantees deterministic replay by rehydrating the event log.
- Enables time-travel debugging through event inspection at any checkpoint.
- Produces structured data that supports offline optimization and training.

## Configuration Guidance

When enabling planning support in a prompt:

1. **Select a pattern** (or combination) that matches the task's complexity and auditing requirements.
2. **Expose schema hooks** in the prompt so the agent knows how to author plans, invoke tools, and log events.
3. **Surface guardrails** (token budgets, allowed tools, stage constraints) directly in the planning section.
4. **Define termination semantics**—exit criteria for PAR, stage completion rules for multi-stage policies, and run termination events for typed traces.
5. **Instrument telemetry** by wiring event emission into the agent runtime and ensuring IDs remain stable across retries.

Adopting these patterns yields richer coordination, safer tool usage, and higher-quality data for analysis and optimization.

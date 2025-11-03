# Planning Strategies

This specification enumerates the supported planning configurations for prompts that expose a dedicated planning section. It focuses on richer orchestration patterns beyond the default ReAct loop and describes how to capture, execute, and inspect structured plans.

## Baseline: ReAct With Plan Integration

The system continues to support a lightweight ReAct-style loop in which the plan serves as a coordination surface. Agents may:

- Draft or update the plan inline before tool calls when additional context is required.
- Reference plan steps during the ReAct cycle to justify tool usage or adjust course.
- Terminate once the plan's exit criteria are satisfied or explicitly note why the plan was abandoned.

The following sections extend this baseline with more formal planning regimes.

### Example: Basic ReAct Planning Section

```python
from dataclasses import dataclass

from weakincentives.prompt import MarkdownSection, Prompt


@dataclass(slots=True, frozen=True)
class BasicPlanParams:
    summary: str
    next_actions: str
    done_when: str


planning_section = MarkdownSection[BasicPlanParams](
    key="planning",
    title="Plan",
    template="""
    Current summary:
    ${summary}

    Next actions:
    ${next_actions}

    Done when:
    ${done_when}
    """,
)

prompt = Prompt(
    ns="support/triage",
    key="ticket-router",
    sections=(planning_section,),
)

rendered = prompt.render(
    BasicPlanParams(
        summary="Identify the customer's product tier and entitlement.",
        next_actions="1. Inspect CRM notes.\n2. Summarize entitlement findings.",
        done_when="We have a definitive tier plus entitlement summary.",
    )
)
```

The plan section stays lightweight, but it still enforces typed fields and
provides the coordination surface that the ReAct loop references while the
agent reasons.

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

### Example Configuration

```python
from __future__ import annotations

from dataclasses import dataclass

from weakincentives.prompt import MarkdownSection, Prompt


@dataclass(slots=True, frozen=True)
class ParStep:
    identifier: str
    objective: str
    tools: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ParPlan:
    version: str
    steps: tuple[ParStep, ...]
    exit_criteria: str


@dataclass(slots=True, frozen=True)
class ParPlanTemplateParams:
    plan_version: str
    steps_markdown: str
    tools_markdown: str
    exit_criteria: str


def to_template_params(plan: ParPlan) -> ParPlanTemplateParams:
    step_lines = (
        f"- **{step.identifier}**: {step.objective}"
        for step in plan.steps
    )
    tool_lines = (
        f"- {step.identifier}: {', '.join(step.tools)}" for step in plan.steps
    )
    return ParPlanTemplateParams(
        plan_version=plan.version,
        steps_markdown="\n".join(step_lines),
        tools_markdown="\n".join(tool_lines),
        exit_criteria=plan.exit_criteria,
    )


par_planning_section = MarkdownSection[ParPlanTemplateParams](
    key="planning",
    title="Plan → Act → Reflect Plan",
    template="""
    Version: ${plan_version}

    Planned steps:
    ${steps_markdown}

    Tool expectations:
    ${tools_markdown}

    Exit criteria:
    ${exit_criteria}
    """,
)

par_prompt = Prompt(
    ns="research/sweeps",
    key="dataset-curation",
    sections=(par_planning_section,),
)

rendered_par = par_prompt.render(
    to_template_params(
        ParPlan(
            version="v1",
            steps=(
                ParStep(
                    identifier="S1",
                    objective="Fetch baseline metrics from analytics store.",
                    tools=("analytics.query",),
                ),
                ParStep(
                    identifier="S2",
                    objective="Validate anomalies with raw warehouse tables.",
                    tools=("warehouse.sql", "notebook"),
                ),
            ),
            exit_criteria="Every anomaly has either a documented root cause or a Jira ticket.",
        )
    )
)
```

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

### Example Configuration

```python
from __future__ import annotations

from dataclasses import dataclass

from weakincentives.prompt import MarkdownSection, Prompt


@dataclass(slots=True, frozen=True)
class StageDefinition:
    name: str
    guardrails: str
    handoff: str


@dataclass(slots=True, frozen=True)
class MultiStageTemplateParams:
    sequence_overview: str
    stage_table: str
    exit_signal: str


def stage_table(stages: tuple[StageDefinition, ...]) -> str:
    header = "| Stage | Guardrails | Handoff |\n| --- | --- | --- |"
    rows = (
        f"| {stage.name} | {stage.guardrails} | {stage.handoff} |"
        for stage in stages
    )
    return "\n".join((header, *rows))


def to_multi_stage_params(stages: tuple[StageDefinition, ...], exit_signal: str) -> MultiStageTemplateParams:
    return MultiStageTemplateParams(
        sequence_overview=" → ".join(stage.name for stage in stages),
        stage_table=stage_table(stages),
        exit_signal=exit_signal,
    )


multi_stage_section = MarkdownSection[MultiStageTemplateParams](
    key="planning",
    title="Multi-stage Policy",
    template="""
    Stage sequence: ${sequence_overview}

    ${stage_table}

    Terminate when: ${exit_signal}
    """,
)

multi_stage_prompt = Prompt(
    ns="sales/enrichment",
    key="lead-categorizer",
    sections=(multi_stage_section,),
)

rendered_multi_stage = multi_stage_prompt.render(
    to_multi_stage_params(
        stages=(
            StageDefinition(
                name="Goal framing",
                guardrails="Only restate user intent; do not propose tools yet.",
                handoff="Provide a concise mission statement for decomposition.",
            ),
            StageDefinition(
                name="Decomposition",
                guardrails="Produce ≤4 steps with tool hints and dependencies.",
                handoff="Structured checklist for tool routing stage.",
            ),
            StageDefinition(
                name="Tool routing",
                guardrails="Select tools from the allowed namespace; include fallback plans.",
                handoff="Ordered tool invocation plan and guardrail summary.",
            ),
            StageDefinition(
                name="Synthesis",
                guardrails="Do not invoke tools; rely on prior stage outputs only.",
                handoff="Draft final response and note any unresolved items.",
            ),
        ),
        exit_signal="Synthesis confirms deliverables satisfy the mission statement.",
    )
)
```

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

### Example Configuration

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from weakincentives.events import InProcessEventBus
from weakincentives.prompt import MarkdownSection, Prompt


@dataclass(slots=True, frozen=True)
class EventLogTemplateParams:
    plan_identifier: str
    last_event_preview: str
    replay_hint: str


event_log_section = MarkdownSection[EventLogTemplateParams](
    key="planning",
    title="Typed Event Trace",
    template="""
    Plan ID: ${plan_identifier}

    Last event:
    ${last_event_preview}

    Replay guidance:
    ${replay_hint}
    """,
)

eventful_prompt = Prompt(
    ns="ops/runbooks",
    key="cache-mitigation",
    sections=(event_log_section,),
)

event_bus = InProcessEventBus()
plan_id = f"plan::{uuid4()}"

def handle_plan_authored(event: object) -> None:
    print(f"Captured event: {event}")


event_bus.subscribe(dict, handle_plan_authored)  # Example stub subscriber.

rendered_event_log = eventful_prompt.render(
    EventLogTemplateParams(
        plan_identifier=plan_id,
        last_event_preview="PlanAuthored@" + datetime.now().isoformat(timespec="seconds"),
        replay_hint="Replay by hydrating events into the coordinator before resuming tools.",
    )
)

event_bus.publish(
    {
        "event_type": "PlanAuthored",
        "plan_id": plan_id,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
    }
)
```

The configuration ties the planning section to the event stream by surfacing the
active plan identifier and the last observed event, while the bus records every
typed event necessary to replay the run.

## Configuration Guidance

When enabling planning support in a prompt:

1. **Select a pattern** (or combination) that matches the task's complexity and auditing requirements.
1. **Expose schema hooks** in the prompt so the agent knows how to author plans, invoke tools, and log events.
1. **Surface guardrails** (token budgets, allowed tools, stage constraints) directly in the planning section.
1. **Define termination semantics**—exit criteria for PAR, stage completion rules for multi-stage policies, and run termination events for typed traces.
1. **Instrument telemetry** by wiring event emission into the agent runtime and ensuring IDs remain stable across retries.

Adopting these patterns yields richer coordination, safer tool usage, and higher-quality data for analysis and optimization.

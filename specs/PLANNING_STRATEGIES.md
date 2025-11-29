# Planning Strategy Templates Specification

## Overview

This document extends the built-in planning prompt section so orchestrators can tailor
its instructions to different reasoning styles. The current `PlanningToolsSection`
always emits the same guidance block; this spec introduces a strategy enum that
swaps the descriptive copy while keeping the tool surface identical. Treat this
document as both the design rationale (why strategies exist and how they should
feel) and a reference guide (where to hook strategies into prompts, configuration,
and testing).

## Guiding Principles

- **Prompt house style is non-negotiable**: Strategy-specific copy must respect the
  rules in `specs/PROMPTS.md` about ASCII-only text, concise headings, imperative
  phrasing, and quiet tone. Do not reformat list structures or heading counts when
  swapping strategies.
- **Planning remains tool-centric**: The mindset text can change, but every
  strategy must still foreground the planning tool invocation flow described in
  `specs/PLANNING_TOOL.md` and `specs/PROMPTS_COMPOSITION.md`. Planning is a tool
  engagement, not a free-form essay.
- **State transitions stay observable**: Each strategy should reinforce status and
  note updates on the plan steps so downstream agents can read what happened.
  Avoid copy that encourages inline stream-of-consciousness outside the plan
  object.
- **Default path must be backwards compatible**: Selecting no strategy should emit
  the existing guidance block to protect all legacy prompt templates and tests.

## Scope and Use Cases

Planning strategies are only responsible for tuning the instructional copy inside
the planning tools section of a prompt. They do **not**:

- Alter the structure or schema of plan objects.
- Change tool schemas or the routing logic for planning vs. direct tool calls.
- Replace prompt sections outside the planning block (e.g., safety sections,
  workspace overviews, or structured output constraints).

Use a strategy when you need to steer the model toward a specific workflow without
rewriting prompt templates. Pair strategy selection with existing prompt
composition APIs described in `specs/PROMPTS_COMPOSITION.md` so the same prompt
layout can support multiple reasoning flavours.

## Goals

- Allow callers to select from a small set of planning mindsets without changing
  tool definitions or session wiring.
- Encode each mindset as a reusable template so the prompt can swap copy while
  preserving the existing markdown structure and house style.
- Keep the default behaviour identical to the current output so no change is
  observed unless a strategy is explicitly chosen.

## Configuration and Integration Points

- **Section entrypoint**: Strategies are selected via the `strategy` argument on
  `PlanningToolsSection` in `weakincentives.tools.planning.section`. The section
  wiring and prompt assembly patterns live in `specs/PROMPTS_COMPOSITION.md`—reuse
  those patterns rather than inventing bespoke templates.
- **Prompt templates**: The base planning copy currently mirrors the planning
  templates documented in `specs/PLANNING_TOOL.md` and the broader prompt
  conventions in `specs/PROMPTS.md`. When editing strategy text, cross-check the
  rendered output against those templates to avoid drift.
- **Runtime configuration**: Strategy selection should flow from the session or
  prompt configuration surface (e.g., prompt constructor options). Avoid hard
  coding strategy choices inside tools or adapters so orchestration layers can
  override defaults.
- **Testing**: Snapshot tests in the prompt module should assert the rendered
  markdown for each strategy. Add or update fixtures when the copy changes to keep
  coverage meaningful.

## API Additions

- Define a `PlanningStrategy` `Enum` in `weakincentives.tools.planning.section`
  with the following members:
  - `REACT`
  - `PLAN_ACT_REFLECT`
  - `GOAL_DECOMPOSE_ROUTE_SYNTHESISE`
- Extend `PlanningToolsSection` to accept an optional `strategy` argument
  defaulting to `PlanningStrategy.REACT` to preserve legacy behaviour.
- Store the selected strategy on the section instance so `render()` can select
  the correct template at generation time.

## Template Behaviour

Each strategy represents a mindset we want to nudge the LLM toward. The emitted
markdown must continue to observe the rules from `PROMPTS.md` (ASCII, short
intro, imperative voice, quiet tone). Within those constraints, layer in the
following guidance:

### ReAct (`reason -> act -> observe -> repeat`)

- Mirrors the existing copy: encourage the agent to alternate between short
  reasoning bursts, tool calls, and observation updates.
- Remind the agent to capture observations as plan step notes when relevant so
  the session history reflects what happened.

### Plan -> Act -> Reflect (PAR)

- Instruct the agent to outline the entire plan first, execute steps, then add a
  short reflection after each tool call or completed step.
- Highlight that reflections should be appended as plan notes or brief status
  updates so the plan shows what was learned.

### Goal framing -> Decomposition -> Tool routing -> Synthesis

- Coach the agent to start by restating the goal in their own words.
- Break the goal into concrete sub-problems before assigning tools to each one;
  make explicit that tool routing should be documented in the plan steps.
- Close with guidance to synthesise the results back into a cohesive answer once
  all tools have run, updating the plan status as part of the synthesis step.

## Rendering Rules

- The section continues to emit a single markdown heading and ordered lists for
  the workflow instructions. Swap only the body text that describes the mindset;
  do not alter the tool usage references or ordering of the tool bullet list.
- Keep the call-to-action about when to engage planning, how to seed a plan, and
  how to manage step status identical across strategies; only the mindset
  paragraph(s) should vary.
- Update regression tests (or add new ones) to snapshot the rendered markdown for
  each strategy so future edits remain intentional.

## Known Caveats

- **Template drift**: Because strategy copy is free text, it can silently diverge
  from the canonical planning tool description. Periodically compare the rendered
  markdown with the examples in `specs/PLANNING_TOOL.md` to ensure the plan
  lifecycle remains aligned.
- **Section ordering**: The planning section may appear alongside safety or
  workspace context sections. Strategies must not assume they are the first or
  only instructions the model sees; avoid references like "above" or "previous"
  that depend on placement.
- **Session constraints**: Some orchestration flows may disable planning tools in
  favour of direct tool calls. Strategy selection should no-op when planning is
  unavailable rather than producing contradictory guidance.

## Backwards Compatibility

- The default constructor path (`PlanningToolsSection(session=Session(...))`)
  must render exactly the existing copy so no external prompts change without
  opting into a strategy.
- Strategy selection is optional; omitting it maintains current behaviour.

## Open Questions

- Should we allow user-defined templates in addition to the enum? Out of scope
  for this iteration but consider the extension points while implementing the
  enum to avoid locking ourselves out of future custom strategies.

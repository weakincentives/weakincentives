# Planning Strategy Templates Specification

## Overview

This document extends the built-in planning prompt section so orchestrators can tailor
its instructions to different reasoning styles. The current `PlanningToolsSection`
always emits the same guidance block; this spec introduces a strategy enum that
swaps the descriptive copy while keeping the tool surface identical.

## Goals

- Allow callers to select from a small set of planning mindsets without changing
  tool definitions or session wiring.
- Encode each mindset as a reusable template so the prompt can swap copy while
  preserving the existing markdown structure and house style.
- Keep the default behaviour identical to the current output so no change is
  observed unless a strategy is explicitly chosen.

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

## Backwards Compatibility

- The default constructor path (`PlanningToolsSection()`) must render exactly the
  existing copy so no external prompts change without opting into a strategy.
- Strategy selection is optional; omitting it maintains current behaviour.

## Open Questions

- Should we allow user-defined templates in addition to the enum? Out of scope
  for this iteration but consider the extension points while implementing the
  enum to avoid locking ourselves out of future custom strategies.

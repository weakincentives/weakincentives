# Roadmap

## Near-Term Initiatives

### Structured Output Prompts

- Define prompt scaffolding that yields machine-parseable responses without manual post-processing.
- Align prompt templates with `specs/PROMPTS.md` requirements and extend tests covering JSON and key-value formats.
- Document fallback behavior when models cannot guarantee structure.

### Notes System Retrospectives

- Establish a notes pattern that captures retrospectives for individual prompt invocations and entire sessions.
- Model notes as entities that can attach to `Section` objects and `Tool` objects to preserve context.
- Outline lifecycle and storage expectations so notes integrate cleanly with existing session state abstractions.

### Single Turn Prompt Optimizations

- Profile current single-turn flows to surface latency and token usage hot spots.
- Experiment with prompt compression techniques and instruction restructuring to maintain quality while reducing cost.
- Add benchmarks or harness scripts that assert improvements before regression.

### Named Entities Handling (Input and Output)

- Introduce utilities for detecting and tagging named entities across inputs.
- Preserve, normalize, or obfuscate entities in outputs according to privacy and compliance guidelines.
- Validate the pipeline with targeted tests that cover multilingual and domain-specific vocabularies.

### Session State Container

- Formalize a `Session` abstraction that captures conversation state, tool outputs, and transient metadata.
- Define serialization hooks so sessions can persist across process restarts without leaking sensitive data.
- Thread the session object through existing prompt and tool layers, backed by integration tests that assert idempotent replay.

### Built-In Planning & Virtual Filesystem Tools

- Provide first-class tool definitions for planning/todo workflows and virtual filesystem operations for agents.
- Establish section templates that ensure tools render consistently in prompts and downstream telemetry.
- Ship representative examples and regression tests demonstrating safe defaults and extensibility points.

### Agentic Reasoning Loop

- Design an orchestrator that coordinates system prompts, user turns, tool routing, and session state updates.
- Integrate entity resolvers and named-entity policies to normalize inputs before tool calls and responses.
- Document the execution phases (think, act, observe) with diagrams and tests that enforce correct transitions.

### Tracing & Observability

- Capture structured trace data from agent runs, including tool calls and message content classification.
- Export telemetry to persistent storage with configurable redaction for sensitive fields and PII.
- Provide replay and visualization utilities that allow developers to inspect state transitions and timing.

### Subagents & Parallel Execution

- Enable the primary agent to spawn scoped subagents with dedicated sessions for independent objectives.
- Coordinate concurrent execution, result aggregation, and conflict resolution when subagents touch shared resources.
- Provide lifecycle hooks so subagents inherit policies, tools, and logging while remaining cancellable.

## Out of Scope (For Now)

- Graph-based agent composers—current focus is flexible orchestration over rigid node/edge pipelines.
- Retrieval and memory connectors—the library assumes ambient context rather than external knowledge stores.
- Human-in-the-loop gating—agents should operate autonomously once launched inside controlled environments.
- Evaluation frameworks beyond prompt optimization—the only supported eval loops will target prompt tuning scenarios.

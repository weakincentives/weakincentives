# Roadmap

## Near-Term Initiatives

### Tracing & Observability

- Capture structured trace data from agent runs, including tool calls and message content classification.
- Export telemetry to persistent storage with configurable redaction for sensitive fields and PII.
- Provide replay and visualization utilities that allow developers to inspect state transitions and timing.

### GEPA Prompt Overrides

- Integrate GEPA as a prompt optimization layer that emits override payloads for existing single-turn prompts.
- Apply GEPA overrides within the current prompt execution flow without reworking the underlying runtime.
- Benchmark GEPA-driven overrides to confirm quality and cost improvements while guarding against regressions.

### Session Durability & Prompt Resumption

- Persist agent session state, including prompt stacks, tool artifacts, and checkpoint metadata, so long-running evaluations survive restarts.
- Expose APIs and CLI flows to resume suspended prompt evaluations safely and idempotently after interruptions.
- Cover resumable workflows with targeted integration tests that assert state continuity and guard against duplicate side effects.

### Named Entities Handling (Input and Output)

- Introduce utilities for detecting and tagging named entities across inputs.
- Preserve, normalize, or obfuscate entities in outputs according to privacy and compliance guidelines.
- Validate the pipeline with targeted tests that cover multilingual and domain-specific vocabularies.

## Out of Scope (For Now)

- Graph-based agent composers—current focus is flexible orchestration over rigid node/edge pipelines.
- Retrieval and memory connectors—the library assumes ambient context rather than external knowledge stores.
- Human-in-the-loop gating—agents should operate autonomously once launched inside controlled environments.
- Evaluation frameworks beyond prompt optimization—the only supported eval loops will target prompt tuning scenarios.

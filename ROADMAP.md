# Roadmap

## Near-Term Initiatives

### ACP Integration

- Integrate with OpenCode via the Agent Communication Protocol (ACP) to enable WINK agents within the OpenCode harness.
- Build generic ACP support that allows WINK agents to run inside other execution harnesses adopting the protocol.
- Define clear boundaries between agent definition (WINK-owned) and harness concerns (ACP-mediated).

### Codex App Server Integration

- Integrate with [Codex App Server](https://developers.openai.com/codex/app-server) as an execution environment for WINK agents.
- Adapt the runtime to operate within Codex's sandboxed execution model.
- Enable WINK agent definitions to be deployed and executed through the Codex infrastructure.

### Checkpointer Mechanism

- Implement time-based checkpointing that captures full session snapshots every minute during agent execution.
- Persist checkpoint data including prompt state, tool artifacts, and workspace contents.
- Support resumption by restoring the latest checkpoint and prompting the agent to continue with:
  - Details of the original prompt and goal
  - Guidance that execution was interrupted
  - Warning that workspace may contain partial state from incomplete operations
- Cover checkpoint/restore workflows with integration tests that assert state continuity and handle partial artifacts gracefully.

### GEPA Prompt Overrides

- Integrate GEPA as a prompt optimization layer that emits override payloads for existing single-turn prompts.
- Apply GEPA overrides within the current prompt execution flow without reworking the underlying runtime.
- Benchmark GEPA-driven overrides to confirm quality and cost improvements while guarding against regressions.

### Basic Metrics

- Capture lightweight metrics inspired by Google SRE golden signals (latency, traffic, errors, saturation) scoped to tool calls and prompt evaluations.
- Expose aggregate statistics without deep instrumentation overhead.
- Rely on debug bundles as the primary instrument for in-depth analysis and troubleshooting rather than extensive runtime telemetry.

## Out of Scope (For Now)

- Native SDK integrations (Claude Agent SDK, OpenAI Agents SDK, etc.)—these are not proper execution harnesses for unattended agents and lack the planning loop, sandboxing, and recovery mechanisms that production agent deployments require.
- Graph-based agent composers—current focus is flexible orchestration over rigid node/edge pipelines.
- Retrieval and memory connectors—the library assumes ambient context rather than external knowledge stores.
- Human-in-the-loop gating—agents should operate autonomously once launched inside controlled environments.
- Evaluation frameworks beyond prompt optimization—the only supported eval loops will target prompt tuning scenarios.

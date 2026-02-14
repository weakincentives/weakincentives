# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Document descriptions for wink docs list command.

Descriptions are sourced from CLAUDE.md spec table and guides/README.md tables.
"""

from __future__ import annotations

SPEC_DESCRIPTIONS: dict[str, str] = {
    "ADAPTERS": "Provider integrations, structured output, throttling",
    "CLAUDE_AGENT_SDK": "Claude Agent SDK adapter, MCP tool bridging, skill mounting",
    "CODEX_APP_SERVER": "Codex App Server adapter, stdio JSON-RPC, thread/turn lifecycle",
    "CLOCK": "Controllable time abstractions, clock injection, testing patterns",
    "DATACLASSES": "Serde utilities, frozen dataclass patterns",
    "DBC": "DbC decorators, exhaustiveness checking, assert_never patterns",
    "DEBUG_BUNDLE": "Debug bundles, log capture, session snapshots, debug web UI",
    "DLQ": "Dead letter queues, poison message handling, AgentLoop/EvalLoop DLQ config",
    "EVALS": "Evaluation framework, datasets, evaluators, session evaluators",
    "EXAMPLES": "Code review agent reference implementation",
    "EXPERIMENTS": "Experiment configuration, A/B testing, prompt overrides, feature flags",
    "FILESYSTEM": "Filesystem protocol, backend implementations, ToolContext integration",
    "GUARDRAILS": "Tool policies, feedback providers, task completion checking",
    "FORMAL_VERIFICATION": "Embedding TLA+ in Python, @formal_spec decorator, TLC verification",
    "HEALTH": "Health endpoints, watchdog, stuck worker detection, process termination",
    "LEASE_EXTENDER": "Automatic message visibility extension during processing",
    "LIFECYCLE": "LoopGroup, ShutdownCoordinator, graceful shutdown patterns",
    "LOGGING": "Logging surfaces",
    "MAILBOX": "Message queue abstraction, SQS/Redis semantics, reply patterns, resolver",
    "AGENT_LOOP": "Agent loop orchestration, visibility handling, event-driven execution",
    "MODULE_BOUNDARIES": "Module organization and import boundaries",
    "ACP_ADAPTER": "Generic ACP adapter, protocol flow, MCP HTTP bridging",
    "OPENCODE_ACP_ADAPTER": "Superseded — see ACP_ADAPTER and OPENCODE_ADAPTER",
    "OPENCODE_ADAPTER": "OpenCode ACP adapter, model validation, quirk handling",
    "POLICIES_OVER_WORKFLOWS": "Philosophy of declarative policies vs rigid workflows",
    "PROMPTS": "Prompt system, composition, overrides, structured output, resources",
    "RESOURCE_REGISTRY": "Dependency injection, resource scopes, transactional snapshots",
    "RUN_CONTEXT": "Execution metadata, request correlation, distributed tracing",
    "SESSIONS": "Session lifecycle, events, deadlines, budgets",
    "SKILLS": "Agent Skills specification and WINK skill mounting",
    "SLICES": "Slice storage backends, factory configuration, JSONL persistence",
    "TESTING": "Test harnesses, fault injection, fuzzing, coverage standards",
    "THREAD_SAFETY": "Concurrency and shared state",
    "TOOLS": "Tool runtime, failure semantics, transactional rollback",
    "TRANSCRIPT": "Unified transcript format, entry schema, adapter mapping, debug bundle integration",
    "VERIFICATION": "Redis mailbox detailed specification, invariants, property tests",
    "VERIFICATION_TOOLBOX": "Verification toolchain (check.py), checker protocol, failure reporting",
    "WINK_DEBUG": "Debug bundle viewer UI, proposed enhancements, timeline visualization",
    "WINK_DOCS": "CLI docs command, bundled documentation access",
    "WINK_QUERY": "SQL CLI for debug bundle exploration, dynamic schemas, agent diagnostics",
    "WORKSPACE": "Claude Agent SDK workspace, host mounts, workspace digest",
}

GUIDE_DESCRIPTIONS: dict[str, str] = {
    "README": "Table of contents and overview",
    "philosophy": "The 'weak incentives' approach and why WINK exists",
    "quickstart": "Get a working agent running quickly",
    "prompts": "Build typed, testable prompts",
    "tools": "Define tool contracts and handlers",
    "sessions": "Manage state with reducers",
    "serialization": "Parse and serialize dataclasses without Pydantic",
    "adapters": "Connect to agentic harnesses like Claude Agent SDK",
    "claude-agent-sdk": "Production integration with Claude Code",
    "skills-authoring": "Create and mount custom skills",
    "orchestration": "Use AgentLoop for request handling",
    "evaluation": "Test agents with datasets and evaluators",
    "lifecycle": "Manage shutdown, health checks, and watchdogs",
    "resources": "Dependency injection, scopes, and lifecycle management",
    "progressive-disclosure": "Control context size with summaries",
    "prompt-overrides": "Iterate on prompts without code changes",
    "debugging": "Inspect sessions and use the debug UI",
    "query": "SQL-based exploration of debug bundles",
    "testing": "Test prompts, tools, and reducers",
    "code-quality": "Types, contracts, coverage, security",
    "recipes": "Common patterns for agents",
    "troubleshooting": "Debug common errors",
    "api-reference": "Quick lookup for key types",
    "migration-from-langgraph": "Coming from LangGraph/LangChain",
    "migration-from-dspy": "Coming from DSPy",
    "formal-verification": "TLA+ specs for critical code",
    "code-review-agent": "End-to-end walkthrough of the code reviewer",
    "concurrency": "Thread safety guarantees and multi-threaded agent patterns",
    "experiments": "A/B test agent variants with baselines and treatments",
    "guardrails-and-feedback": "Tool policies, feedback providers, and task completion",
    "mailbox-and-dlq": "Message queuing, dead letter queues, and lease extension",
    "module-boundaries": "Layer architecture, dependency rules, and where new code goes",
    "observability": "Run context, tracing, logging, and transcript collection",
    "workspaces-and-filesystem": "Sandboxed file access, host mounts, and workspace digests",
}

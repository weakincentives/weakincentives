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
    "CLOCK": "Controllable time abstractions, clock injection, testing patterns",
    "DATACLASSES": "Serde utilities, frozen dataclass patterns",
    "DBC": "DbC decorators, exhaustiveness checking, assert_never patterns",
    "DEBUG_BUNDLE": "Debug bundles, log capture, session snapshots, debug web UI",
    "DLQ": "Dead letter queues, poison message handling, MainLoop/EvalLoop DLQ config",
    "EVALS": "Evaluation framework, datasets, evaluators, session evaluators",
    "EXAMPLES": "Code review agent reference implementation",
    "EXPERIMENTS": "Experiment configuration, A/B testing, prompt overrides, feature flags",
    "FEEDBACK_PROVIDERS": "Ongoing progress feedback, stall/drift detection, context injection",
    "FILESYSTEM": "Filesystem protocol, backend implementations, ToolContext integration",
    "FORMAL_VERIFICATION": "Embedding TLA+ in Python, @formal_spec decorator, TLC verification",
    "HEALTH": "Health endpoints, watchdog, stuck worker detection, process termination",
    "LEASE_EXTENDER": "Automatic message visibility extension during processing",
    "LIFECYCLE": "LoopGroup, ShutdownCoordinator, graceful shutdown patterns",
    "LOGGING": "Logging surfaces",
    "MAILBOX": "Message queue abstraction, SQS/Redis semantics, reply patterns, resolver",
    "MAIN_LOOP": "Main loop orchestration, visibility handling, event-driven execution",
    "MODULE_BOUNDARIES": "Module organization and import boundaries",
    "POLICIES_OVER_WORKFLOWS": "Philosophy of declarative policies vs rigid workflows",
    "PROMPTS": "Prompt system, composition, overrides, structured output, resources",
    "RESOURCE_REGISTRY": "Dependency injection, resource scopes, transactional snapshots",
    "RUN_CONTEXT": "Execution metadata, request correlation, distributed tracing",
    "SESSIONS": "Session lifecycle, events, deadlines, budgets",
    "SKILLS": "Agent Skills specification and WINK skill mounting",
    "SLICES": "Slice storage backends, factory configuration, JSONL persistence",
    "TASK_COMPLETION": "Task completion checking, PlanBasedChecker, composite verification",
    "TESTING": "Test harnesses, fault injection, fuzzing, coverage standards",
    "THREAD_SAFETY": "Concurrency and shared state",
    "TOOLS": "Tool runtime, policies, sequential dependencies, planning tools",
    "VERIFICATION": "Redis mailbox detailed specification, invariants, property tests",
    "VERIFICATION_TOOLBOX": "Verification toolchain (check.py), checker protocol, failure reporting",
    "WINK_DEBUG": "Debug bundle viewer UI, proposed enhancements, timeline visualization",
    "WINK_DOCS": "CLI docs command, bundled documentation access",
    "WINK_QUERY": "SQL CLI for debug bundle exploration, dynamic schemas, agent diagnostics",
    "WORKSPACE": "VFS, Podman, asteval, workspace digest",
}

GUIDE_DESCRIPTIONS: dict[str, str] = {
    "README": "Table of contents and overview",
    "philosophy": "The 'weak incentives' approach and why WINK exists",
    "quickstart": "Get a working agent running quickly",
    "prompts": "Build typed, testable prompts",
    "tools": "Define tool contracts and handlers",
    "sessions": "Manage state with reducers",
    "serialization": "Parse and serialize dataclasses without Pydantic",
    "adapters": "Connect to OpenAI, LiteLLM, Claude Agent SDK",
    "claude-agent-sdk": "Production integration with Claude Code",
    "skills-authoring": "Create and mount custom skills",
    "orchestration": "Use MainLoop for request handling",
    "evaluation": "Test agents with datasets and evaluators",
    "lifecycle": "Manage shutdown, health checks, and watchdogs",
    "guardrails": "Feedback providers, task completion, and lease extension",
    "resources": "Dependency injection, scopes, and lifecycle management",
    "progressive-disclosure": "Control context size with summaries",
    "prompt-overrides": "Iterate on prompts without code changes",
    "workspace-tools": "Use VFS, Podman, planning, workspace digests",
    "debugging": "Inspect sessions and use the debug UI",
    "testing": "Test prompts, tools, and reducers",
    "code-quality": "Types, contracts, coverage, security",
    "recipes": "Common patterns for agents",
    "troubleshooting": "Debug common errors",
    "api-reference": "Quick lookup for key types",
    "migration-from-langgraph": "Coming from LangGraph/LangChain",
    "migration-from-dspy": "Coming from DSPy",
    "formal-verification": "TLA+ specs for critical code",
    "code-review-agent": "End-to-end walkthrough of the code reviewer",
}

# WINK Guides

Welcome to the Weak Incentives (WINK) guides. These documents are designed to
help you understand not just *how* to use the library, but *why* it works the
way it does. The goal is to build a correct mental model for building agents
with WINK.

## How to Read These Guides

If you're new to WINK, start with the **Quickstart** to get something running,
then read **Philosophy** to understand the design principles. From there, follow
the guides in order or jump to specific topics as needed.

The guides assume you'll be delegating much of the coding to AI assistants. They
focus on building understanding so you can effectively direct that work and
debug issues when they arise.

## Core Concepts

| Guide | What You'll Learn |
| --- | --- |
| [Philosophy](philosophy.md) | The "weak incentives" approach and why WINK exists |
| [Quickstart](quickstart.md) | Get a working agent running in minutes |
| [Prompts](prompts.md) | Build typed, testable prompts as first-class objects |
| [Tools](tools.md) | Define tool contracts and handlers |
| [Sessions](sessions.md) | Manage state with reducers and event-driven updates |

## Provider Integration

| Guide | What You'll Learn |
| --- | --- |
| [Adapters](adapters.md) | Connect to OpenAI, LiteLLM, and Claude Agent SDK |

## Production Patterns

| Guide | What You'll Learn |
| --- | --- |
| [Orchestration](orchestration.md) | Use MainLoop for request handling |
| [Evaluation](evaluation.md) | Test agents with datasets and evaluators |
| [Lifecycle](lifecycle.md) | Manage shutdown, health checks, and watchdogs |

## Advanced Topics

| Guide | What You'll Learn |
| --- | --- |
| [Progressive Disclosure](progressive-disclosure.md) | Control context size with summaries and expansion |
| [Prompt Overrides](prompt-overrides.md) | Iterate on prompts without code changes |
| [Workspace Tools](workspace-tools.md) | Use VFS, Podman, planning, and workspace digests |
| [Debugging](debugging.md) | Inspect sessions, dump snapshots, use the debug UI |

## Code Quality

| Guide | What You'll Learn |
| --- | --- |
| [Testing](testing.md) | Test prompts, tools, and reducers without a model |
| [Code Quality](code-quality.md) | Types, contracts, coverage, and security scanning |
| [Formal Verification](formal-verification.md) | Embed TLA+ specifications for critical code |

## Practical Reference

| Guide | What You'll Learn |
| --- | --- |
| [Recipes](recipes.md) | Common patterns for code review, Q&A, and patching agents |
| [Troubleshooting](troubleshooting.md) | Debug common errors and issues |
| [API Reference](api-reference.md) | Quick lookup for key types and functions |

## Migration Guides

| Guide | What You'll Learn |
| --- | --- |
| [Coming from LangGraph](migration-from-langgraph.md) | Concept mapping and migration patterns |
| [Coming from DSPy](migration-from-dspy.md) | Concept mapping and migration patterns |

## Examples

| Guide | What You'll Learn |
| --- | --- |
| [Code Review Agent](code-review-agent.md) | End-to-end walkthrough of a complete agent |

## Where to Go Next

- **Specs**: For precise guarantees and implementation details, see the `specs/`
  directory. Guides tell you *how to think about* WINK; specs tell you *exactly
  what it does*.
- **AGENTS.md**: For contributors, the canonical handbook for working in this
  codebase.
- **llms.md**: The full public API reference, suitable for LLM consumption.

## Status

WINK is **alpha software**. APIs may change without backward compatibility. The
guides will evolve alongside the library.

# WINK Guides

Welcome to the Weak Incentives (WINK) guides. These documents are designed to
help you understand not just *how* to use the library, but *why* it works the
way it does. The goal is to build a correct mental model for building agents
with WINK.

## How to Read These Guides

The guides assume you'll be delegating much of the coding to AI assistants. They
focus on building understanding so you can effectively direct that work and
debug issues when they arise.

### For Beginners

1. [Quickstart](quickstart.md) - Get something running first
1. [Philosophy](philosophy.md) - Understand *why* WINK works this way
1. [Prompts](prompts.md) - Learn the core abstraction
1. [Tools](tools.md) - Add capabilities to your agent
1. [Sessions](sessions.md) - Manage state correctly
1. [Serialization](serialization.md) - Parse and dump dataclasses
1. [Adapters](adapters.md) - Connect to your preferred provider

### For Production Deployments

1. [Claude Agent SDK](claude-agent-sdk.md) - Recommended adapter for production
1. [Skills Authoring](skills-authoring.md) - Create and mount custom skills
1. [Orchestration](orchestration.md) - Handle requests at scale
1. [Lifecycle](lifecycle.md) - Health checks, shutdown, watchdogs
1. [Debugging](debugging.md) - Inspect and troubleshoot sessions

### For Advanced Users

1. [Resources](resources.md) - Dependency injection and lifecycle management
1. [Progressive Disclosure](progressive-disclosure.md) - Manage context size
1. [Workspace Tools](workspace-tools.md) - VFS, Podman, sandboxed execution
1. [Prompt Overrides](prompt-overrides.md) - A/B test prompts without deploys
1. [Formal Verification](formal-verification.md) - TLA+ for critical code

### For Contributors

1. [Code Quality](code-quality.md) - Standards and tooling
1. [Testing](testing.md) - How to test prompts, tools, reducers
1. See also: `AGENTS.md` for the canonical contributor handbook

## Core Concepts

| Guide | What You'll Learn |
| --- | --- |
| [Philosophy](philosophy.md) | The "weak incentives" approach and why WINK exists |
| [Quickstart](quickstart.md) | Get a working agent running in minutes |
| [Prompts](prompts.md) | Build typed, testable prompts as first-class objects |
| [Tools](tools.md) | Define tool contracts and handlers |
| [Sessions](sessions.md) | Manage state with reducers and event-driven updates |
| [Serialization](serialization.md) | Parse and serialize dataclasses without Pydantic |

## Provider Integration

| Guide | What You'll Learn |
| --- | --- |
| [Adapters](adapters.md) | Connect to OpenAI, LiteLLM, and Claude Agent SDK |
| [Claude Agent SDK](claude-agent-sdk.md) | Production integration with Claude Code's native tooling |
| [Skills Authoring](skills-authoring.md) | Create and mount custom skills for Claude Code |

## Production Patterns

| Guide | What You'll Learn |
| --- | --- |
| [Orchestration](orchestration.md) | Use MainLoop for request handling |
| [Evaluation](evaluation.md) | Test agents with datasets and evaluators |
| [Lifecycle](lifecycle.md) | Manage shutdown, health checks, and watchdogs |

## Advanced Topics

| Guide | What You'll Learn |
| --- | --- |
| [Resources](resources.md) | Dependency injection, scopes, and lifecycle management |
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

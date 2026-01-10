# The WINK Book

**A comprehensive guide to building deterministic, side-effect-free background agents**

---

## Status

This library is in **alpha**. APIs may change without notice.

---

## About This Book

This book is your comprehensive guide to WINK (Weak Incentives), a Python library for building deterministic, side-effect-free background agents. You'll learn the philosophy behind weak incentives, master the core abstractions, and gain practical experience building production-ready agent systems.

### What You'll Learn

- **Philosophy**: Why "weak incentives" lead to more reliable agents than strict orchestration
- **Core Abstractions**: Prompts as first-class programs, session-driven state, and type-safe tooling
- **Production Patterns**: Evaluation loops, lifecycle management, and prompt optimization
- **Advanced Features**: Progressive disclosure, workspace sandboxing, and formal verification
- **Practical Skills**: End-to-end recipes for code review, Q&A, and research agents

### Who This Book Is For

- Developers building autonomous agents that must run unattended
- Teams that need deterministic, auditable agent behavior
- Anyone frustrated with the unpredictability of traditional LLM orchestration frameworks

---

## Table of Contents

### Part I: Foundations

1. [**Philosophy**](01-philosophy.md) - The weak incentives paradigm
2. [**Quickstart**](02-quickstart.md) - Your first agent in 10 minutes

### Part II: Core Abstractions

3. [**Prompts**](03-prompts.md) - Type-safe prompt composition
4. [**Tools**](04-tools.md) - Sandboxed, deterministic tool execution
   - 4.5 [**Tool Policies**](04.5-tool-policies.md) - Declarative constraints and custom policy development
   - 4.6 [**Task Monitoring**](04.6-task-monitoring.md) - Completion verification and progress tracking
5. [**Sessions**](05-sessions.md) - Event-driven state management

### Part III: Integration & Orchestration

6. [**Adapters**](06-adapters.md) - Provider integrations (OpenAI, LiteLLM, Claude Agent SDK)
7. [**Main Loop**](07-main-loop.md) - Event loop orchestration
   - 7.5 [**Distributed Orchestration**](07.5-distributed-orchestration.md) - Message queues and distributed agents
8. [**Evaluation**](08-evaluation.md) - Testing and quality assurance

### Part IV: Advanced Features

9. [**Lifecycle Management**](09-lifecycle.md) - Production deployment patterns
10. [**Progressive Disclosure**](10-progressive-disclosure.md) - Cost optimization through selective context
11. [**Prompt Optimization**](11-prompt-optimization.md) - Version management and A/B testing
12. [**Workspace Tools**](12-workspace-tools.md) - Planning, VFS, sandboxing, and more

### Part V: Operations & Quality

13. [**Debugging & Observability**](13-debugging.md) - Instrumentation and troubleshooting
14. [**Testing & Reliability**](14-testing.md) - Test strategies and harnesses
15. [**Code Quality**](15-code-quality.md) - Type safety, DbC, and coverage requirements

### Part VI: Practical Applications

16. [**Recipes**](16-recipes.md) - Complete agent patterns
17. [**Troubleshooting**](17-troubleshooting.md) - Common errors and solutions

### Appendices

- [**Appendix A**: Coming from LangGraph/LangChain](appendix-a-from-langgraph.md)
- [**Appendix B**: Coming from DSPy](appendix-b-from-dspy.md)
- [**Appendix C**: Formal Verification with TLA+](appendix-c-formal-verification.md)

### Reference

- [**API Reference**](18-api-reference.md) - Complete API documentation

---

## How to Read This Book

### If you're new to WINK

Start with **Part I** to understand the philosophy, then work through **Part II** to master the core abstractions. Skip to **Chapter 16** for complete examples when you're ready to build.

### If you're migrating from another framework

Read **Chapter 1** (Philosophy) to understand WINK's design principles, then jump to the relevant appendix (A for LangGraph/LangChain, B for DSPy).

### If you're looking for specific features

Use the table of contents to navigate directly to the relevant chapter. Each chapter is self-contained with cross-references to related topics.

---

## Getting Help

- **Source code**: [github.com/weakincentives/weakincentives](https://github.com/weakincentives/weakincentives)
- **Issues**: [github.com/weakincentives/weakincentives/issues](https://github.com/weakincentives/weakincentives/issues)
- **Specs**: See the `/specs` directory for detailed design documents

---

## Technical Strategy

WINK makes five key bets about agent systems:

1. **Determinism over flexibility**: Reproducible event logs beat ad-hoc state management
2. **Prompts as programs**: Versioned, type-safe prompt templates with structured output
3. **Weak incentives over orchestration**: Guide with suggestions, enforce with constraints
4. **Session-driven evaluation**: Test against real execution traces, not synthetic datasets
5. **Sandboxed tools**: Copy-on-write filesystems and containerized execution prevent side effects

If these resonate with you, WINK might be the right fit.

---

**Let's build reliable agents together.**

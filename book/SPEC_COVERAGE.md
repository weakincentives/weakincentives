# Spec Coverage Analysis

**Last Updated**: 2026-01-09

This document tracks which specs in `/specs/` are covered by book chapters in `/book/`.

---

## Coverage Summary

- **Total specs**: 32
- **Fully covered**: 14 specs (44%)
- **Partially covered**: 8 specs (25%)
- **Not covered**: 10 specs (31%)

---

## Fully Covered Specs ✅

These specs have comprehensive book chapter coverage:

| Spec | Book Chapter(s) | Notes |
|------|----------------|-------|
| ADAPTERS.md | Chapter 6: Adapters | Complete coverage of provider integrations |
| DBC.md | Chapter 15: Code Quality | Design-by-contract philosophy and usage |
| EVALS.md | Chapter 8: Evaluation | Comprehensive evaluation framework coverage |
| EXAMPLES.md | Chapter 16: Recipes | Code review agent fully documented |
| FORMAL_VERIFICATION.md | Appendix C | Complete TLA+ integration guide |
| LIFECYCLE.md | Chapter 9: Lifecycle Management | LoopGroup and shutdown coordination |
| MAIN_LOOP.md | Chapter 7: Main Loop | Event loop orchestration well covered |
| PROMPTS.md | Chapter 3: Prompts | Excellent comprehensive coverage |
| PROMPT_OPTIMIZATION.md | Chapter 11: Prompt Optimization | A/B testing and overrides |
| SESSIONS.md | Chapter 5: Sessions | Complete session lifecycle coverage |
| TESTING.md | Chapter 14: Testing & Reliability | Test harnesses and strategies |
| TOOLS.md | Chapter 4: Tools | Comprehensive tool system coverage |
| TOOL_POLICIES.md | Chapter 4.5: Tool Policies | Dedicated chapter with custom policy guide |
| WORKSPACE.md | Chapter 12: Workspace Tools | VFS, Podman, planning tools |

---

## Partially Covered Specs ⚠️

These specs are mentioned but need expansion:

| Spec | Current Coverage | Gap | Recommendation |
|------|-----------------|-----|----------------|
| CLAUDE_AGENT_SDK.md | Ch. 6 (brief) | Missing isolation details, MCP bridging examples | Expand Ch. 6 with dedicated Claude SDK section |
| DATACLASSES.md | Ch. 15 (mention) | Missing serde patterns, FrozenDataclass guide | Add section to Ch. 18 (API Reference) |
| EXHAUSTIVENESS.md | Ch. 15 (brief) | Missing assert_never examples, match coverage | Expand Ch. 15 with pattern examples |
| FILESYSTEM.md | Ch. 12 (VFS only) | Missing protocol abstraction, custom backends | Add filesystem protocol section to Ch. 4 or 12 |
| HEALTH.md | Ch. 9 (brief) | Missing watchdog details, health endpoint patterns | Expand health monitoring in Ch. 9 or 13 |
| LOGGING.md | Ch. 13 (brief) | Missing structured logging guide, best practices | Expand logging section in Ch. 13 |
| POLICIES_OVER_WORKFLOWS.md | Ch. 1 (implied) | Philosophy not explicitly stated | Add dedicated section to Ch. 1 |
| RESOURCE_REGISTRY.md | Ch. 3 (section) | Missing DI patterns, scope lifecycle details | Expand resource section in Ch. 3 |

---

## Not Covered Specs ❌

**High Priority** - User-facing features critical for production:

| Spec | Description | Impact | Recommendation |
|------|-------------|--------|----------------|
| **MAILBOX.md** | Message queue abstraction, distributed orchestration | HIGH - Distributed deployments | **NEW CHAPTER** (7.5 or Part III) |
| **MAILBOX_RESOLVER.md** | Mailbox routing, reply-to patterns | HIGH - Service discovery | Include in mailbox chapter |
| **SKILLS.md** | Agent Skills spec, progressive disclosure | HIGH - Claude Code integration | **NEW SECTION** in Ch. 6 or Appendix |
| **TASK_COMPLETION.md** | Completion verification, PlanBasedChecker | HIGH - Runtime verification | Add to Ch. 4 or Ch. 12 |
| **TRAJECTORY_OBSERVERS.md** | Progress monitoring, stall detection | MEDIUM - Production observability | Add to Ch. 4 or Ch. 13 |
| **WINK_DEBUG.md** | Debug UI, snapshot explorer | MEDIUM - Debugging workflows | Expand Ch. 13 |

**Lower Priority** - Internal/advanced features:

| Spec | Description | Recommendation |
|------|-------------|----------------|
| SLICES.md | Storage backends, JSONL persistence | Add to Ch. 5 as advanced topic |
| THREAD_SAFETY.md | Concurrency patterns | Add to Ch. 9 or Ch. 15 |
| VERIFICATION.md | Redis mailbox verification | Include in mailbox chapter |
| WINK_DOCS.md | CLI docs command | Add to Ch. 2 or Ch. 17 |

---

## Recommended Actions

### Immediate (High Impact)

1. **Create Chapter on Distributed Orchestration**
   - Cover: MAILBOX.md, MAILBOX_RESOLVER.md, VERIFICATION.md
   - Position: After Ch. 7 (Main Loop) or Ch. 8 (Evaluation)
   - Audience: Users deploying distributed agent systems

2. **Add Skills Coverage**
   - Cover: SKILLS.md
   - Position: Expand Ch. 6 (Adapters) or new appendix
   - Audience: Claude Code integration users

3. **Expand Chapter 13 (Debugging)**
   - Add: WINK_DEBUG.md UI walkthrough
   - Add: LOGGING.md structured logging guide
   - Add: Debug workflow recipes

4. **Add Task Monitoring Section**
   - Cover: TASK_COMPLETION.md, TRAJECTORY_OBSERVERS.md
   - Position: Ch. 4 (Tools) or Ch. 13 (Debugging)
   - Audience: Production monitoring needs

### Secondary (Medium Impact)

5. **Expand Chapter 6 (Adapters)**
   - Add dedicated Claude Agent SDK section with:
     - Isolation configuration deep dive
     - MCP tool bridging examples
     - Workspace security patterns

6. **Expand Chapter 5 (Sessions)**
   - Add SLICES.md storage backend configuration
   - Add persistence patterns and serialization

7. **Expand Chapter 9 (Lifecycle)**
   - Add THREAD_SAFETY.md concurrency patterns
   - Add HEALTH.md health endpoint details

8. **Make Philosophy Explicit (Chapter 1)**
   - Add POLICIES_OVER_WORKFLOWS.md as dedicated section
   - Contrast with traditional orchestration frameworks

### Low Priority (Nice-to-Have)

9. **Expand Chapter 18 (API Reference)**
   - Add DATACLASSES.md serde utilities
   - Add EXHAUSTIVENESS.md type safety patterns

10. **Add CLI Discovery (Chapter 2 or 17)**
    - Cover WINK_DOCS.md
    - Help users discover built-in documentation

---

## Coverage Gaps by Book Part

### Part I: Foundations
- Missing: POLICIES_OVER_WORKFLOWS.md (should be in Ch. 1)

### Part II: Core Abstractions
- Missing: TASK_COMPLETION.md, TRAJECTORY_OBSERVERS.md (could fit in Ch. 4)
- Partial: RESOURCE_REGISTRY.md, FILESYSTEM.md

### Part III: Integration & Orchestration
- **Missing: MAILBOX.md, MAILBOX_RESOLVER.md** (major gap - no distributed orchestration coverage)
- Partial: CLAUDE_AGENT_SDK.md, SKILLS.md

### Part IV: Advanced Features
- Missing: SLICES.md (belongs in sessions/storage)

### Part V: Operations & Quality
- Missing: WINK_DEBUG.md details, THREAD_SAFETY.md
- Partial: HEALTH.md, LOGGING.md

---

## Metrics

### Coverage by Category

**Core Abstractions**: 85% coverage
- Prompts: ✅ Full
- Tools: ✅ Full
- Sessions: ⚠️ Partial (missing SLICES.md)
- Policies: ✅ Full

**Integration**: 60% coverage
- Adapters: ⚠️ Partial (Claude SDK needs expansion)
- Main Loop: ✅ Full
- Evaluation: ✅ Full
- Mailbox: ❌ Not covered
- Skills: ❌ Not covered

**Production Features**: 50% coverage
- Lifecycle: ✅ Full
- Testing: ✅ Full
- Debugging: ⚠️ Partial (WINK_DEBUG, LOGGING need expansion)
- Monitoring: ❌ Not covered (TRAJECTORY_OBSERVERS, TASK_COMPLETION)
- Concurrency: ❌ Not covered (THREAD_SAFETY)

**Advanced**: 75% coverage
- Optimization: ✅ Full
- Workspace: ✅ Full
- Formal Verification: ✅ Full
- Progressive Disclosure: ✅ Full

---

## Next Steps

1. **Review this analysis** with maintainers
2. **Prioritize** which specs need book coverage most urgently
3. **Create issues** for each missing/partial spec
4. **Plan chapters** for high-priority gaps (Mailbox, Skills, Monitoring)
5. **Expand** existing chapters for partial coverage
6. **Update this document** as coverage improves

---

## Notes

- Some specs (like VERIFICATION.md) are internal implementation details and may not need user-facing book coverage
- Others (like MAILBOX.md, SKILLS.md) are critical user-facing features that absolutely need comprehensive coverage
- The 31% "not covered" figure is concerning but manageable with focused effort on high-priority specs
- Consider whether some specs should be combined into single chapters (e.g., MAILBOX + MAILBOX_RESOLVER + VERIFICATION)

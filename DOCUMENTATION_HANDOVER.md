# Documentation Handover: Review and Recommendations

This document provides a comprehensive analysis of all markdown documentation in the
`weakincentives` repository, identifying updates needed, new documentation to create,
and consolidation opportunities.

---

## Executive Summary

The documentation is generally well-structured with clear separation between:
- **Root-level files**: Project overview, contributor guides, AI assistant instructions
- **guides/**: User-facing how-to documentation (25 files)
- **specs/**: Technical specifications (38 files)
- **Demo-skills**: Example skill definitions (3 skills)

**Key Findings:**
1. Several missing guides that would improve onboarding
2. Some content duplication between files that could be consolidated
3. A few outdated references and inconsistencies
4. Gaps in coverage for certain features

---

## Section 1: Updates to Existing Files

### 1.1 Root-Level Files

#### README.md
- **Location**: `/README.md`
- **Issues**:
  - References `gpt-5.2` model (line ~284) - should update to a current model name
  - The Debug UI image reference (`debug_ui.png`) should be verified to exist
  - Consider adding a "Quick Links" section at the top for navigation
- **Action**: Update model reference, verify image exists

#### AGENTS.md
- **Location**: `/AGENTS.md`
- **Issues**:
  - Currently only contains "See CLAUDE.md" - too sparse
  - Should expand to be a proper contributor guide for AI agents OR consolidate into CLAUDE.md
- **Action**: Either expand with agent-specific workflows or merge into CLAUDE.md

#### GLOSSARY.md
- **Location**: `/GLOSSARY.md`
- **Issues**: File appears to exist but wasn't fully reviewed
- **Action**: Verify completeness and ensure all key terms are defined

#### ROADMAP.md
- **Location**: `/ROADMAP.md`
- **Issues**: Review for stale items that may have been completed
- **Action**: Update to reflect current priorities

#### llms.md
- **Location**: `/llms.md`
- **Issues**:
  - Very comprehensive (~1150 lines) - serves as both API reference AND guide
  - References `gpt-4o` and `claude-3-sonnet-20240229` which may need updating
  - Some code blocks marked `nocheck` could benefit from being runnable
- **Action**: Review model references, consider if some content should move to dedicated guides

#### GEMINI.md and WARP.md
- **Location**: Root level
- **Issues**: Purpose unclear - appear to be AI-specific instruction files
- **Action**: Consider consolidating AI assistant instructions into fewer files

### 1.2 Guides

#### guides/README.md
- **Issues**:
  - Good structure but could link to specs more explicitly
  - Missing "Resources" and "Serialization" guides mentioned in some cross-references
- **Action**: Verify all cross-referenced guides exist

#### guides/adapters.md
- **Issues**:
  - References `gpt-4.1-mini` (line 47) - unusual model name, verify
  - Good content but could use more error handling examples
- **Action**: Verify model names, add error handling section

#### guides/quickstart.md
- **Issues**: Generally good, ensure code examples are tested
- **Action**: Verify all code examples run

#### guides/code-review-agent.md
- **Issues**:
  - References `gpt-5.2` (line 284) - future model reference
  - Very detailed but could benefit from a "Common Issues" section
- **Action**: Update model reference

#### guides/claude-agent-sdk.md
- **Issues**:
  - References `claude-sonnet-4-5-20250929` - future date model
  - Very comprehensive (~600 lines), well-structured
- **Action**: Review model name for accuracy

#### guides/debugging.md
- **Issues**:
  - Good coverage but `wink debug` command examples could be expanded
  - Missing coverage of common debugging scenarios
- **Action**: Add more debugging scenario examples

#### guides/evaluation.md
- **Issues**:
  - Code blocks using `text` language instead of `python nocheck` for some examples
  - Could benefit from more real-world evaluation examples
- **Action**: Standardize code block language tags

#### guides/migration-from-langgraph.md and migration-from-dspy.md
- **Issues**: Good conceptual content, but code examples could be more complete
- **Action**: Expand migration code examples

#### guides/formal-verification.md
- **Issues**: Good content, but TLA+ installation instructions could be more detailed
- **Action**: Expand installation section for different platforms

### 1.3 Specs

#### specs/PROMPTS.md
- **Issues**: Well-structured, comprehensive
- **Action**: No immediate updates needed

#### specs/TOOLS.md
- **Issues**: Very detailed (~350 lines), well-structured
- **Action**: No immediate updates needed

#### specs/SESSIONS.md
- **Issues**: Good coverage
- **Action**: No immediate updates needed

#### specs/CLAUDE_AGENT_SDK.md
- **Issues**: Model references may need updating
- **Action**: Review model name references

#### specs/DEBUG_BUNDLE.md
- **Issues**: Recently updated, appears current
- **Action**: No immediate updates needed

---

## Section 2: New Documentation to Create

### 2.1 Missing Guides (High Priority)

#### guides/resources.md (NEW)
- **Purpose**: Document the dependency injection system
- **Content**:
  - `ResourceRegistry` and `Binding` concepts
  - Scope types (SINGLETON, TOOL_CALL, PROTOTYPE)
  - Lifecycle protocols (Closeable, PostConstruct, Snapshotable)
  - Integration with prompts and tools
  - Common patterns and anti-patterns
- **Why**: Referenced in api-reference.md and llms.md but no dedicated guide exists

#### guides/serialization.md (NEW)
- **Purpose**: Document the serde module
- **Content**:
  - `dump()`, `parse()`, `schema()`, `clone()` functions
  - Constraint annotations (`Annotated[type, {"ge": 0}]`)
  - `__type__` field for polymorphic unions
  - Session snapshot serialization
  - Common pitfalls (e.g., mutable defaults)
- **Why**: Serialization is critical for session state but lacks a guide

#### guides/time-and-clock.md (NEW)
- **Purpose**: Document time handling patterns
- **Content**:
  - `Deadline` and `Budget` concepts
  - `WallClock`, `MonotonicClock`, `Sleeper` protocols
  - `FakeClock` for testing
  - Timezone best practices (always UTC)
  - Integration with adapters and tools
- **Why**: Time handling is mentioned in CLAUDE.md but not documented

#### guides/mailbox.md (NEW)
- **Purpose**: Document the message queue abstraction
- **Content**:
  - `Mailbox` protocol
  - `InMemoryMailbox` for testing
  - `RedisMailbox` for production
  - Reply-to patterns
  - Dead letter queue handling
- **Why**: Referenced in specs but no user guide exists

#### guides/errors-and-exceptions.md (NEW)
- **Purpose**: Comprehensive error handling guide
- **Content**:
  - Error hierarchy (from llms.md)
  - When each error type is raised
  - Error handling patterns for tools
  - Recovery strategies
  - Debugging error traces
- **Why**: Error handling is scattered across multiple docs

### 2.2 Missing Guides (Medium Priority)

#### guides/security.md (NEW)
- **Purpose**: Security best practices for agent development
- **Content**:
  - Input validation patterns
  - Sandbox configuration (VFS, Podman, Claude Agent SDK)
  - Network isolation
  - Credential handling
  - OWASP considerations for agents
- **Why**: Security is critical but spread across multiple docs

#### guides/multi-model.md (NEW)
- **Purpose**: Working with multiple models/providers
- **Content**:
  - Provider comparison
  - Model selection strategies
  - Fallback patterns
  - Cost optimization
  - LiteLLM configuration
- **Why**: Multi-model support exists but isn't well-documented

#### guides/deployment.md (NEW)
- **Purpose**: Production deployment guide
- **Content**:
  - Container configuration
  - Kubernetes deployment (with LoopGroup)
  - Health endpoints configuration
  - Watchdog tuning
  - Logging in production
  - Scaling considerations
- **Why**: Lifecycle guide covers some, but deployment needs dedicated coverage

#### guides/custom-sections.md (NEW)
- **Purpose**: Creating custom section types
- **Content**:
  - Section protocol
  - WorkspaceSection pattern
  - Resource contribution
  - Tool registration
  - Visibility and progressive disclosure
- **Why**: Extending the section system is advanced but not documented

### 2.3 Example/Tutorial Additions

#### guides/examples/README.md (NEW)
- **Purpose**: Index of runnable examples
- **Content**:
  - Link to code_reviewer_example.py with explanation
  - Additional example scenarios:
    - Simple Q&A agent
    - Document analysis agent
    - Multi-step workflow agent
- **Why**: One complex example exists, but simpler examples would help onboarding

---

## Section 3: Consolidation Opportunities

### 3.1 AI Assistant Instructions

**Current State**: Multiple files for different AI assistants
- CLAUDE.md (comprehensive)
- AGENTS.md (minimal, just redirects)
- GEMINI.md (Gemini-specific)
- WARP.md (Warp-specific)
- IDENTITY.md (unclear purpose)

**Recommendation**:
- Keep CLAUDE.md as the canonical AI assistant guide
- Merge AGENTS.md content into CLAUDE.md or expand it properly
- Consider consolidating GEMINI.md and WARP.md into CLAUDE.md with sections for model-specific notes
- Clarify IDENTITY.md purpose or remove

### 3.2 API Reference Overlap

**Current State**:
- `llms.md` - Very comprehensive API reference (~1150 lines)
- `guides/api-reference.md` - Curated API reference (~267 lines)

**Recommendation**:
- Keep both but clarify their purposes:
  - `llms.md` → "Dense reference for AI agents" (PyPI README)
  - `guides/api-reference.md` → "Quick reference for humans"
- Add cross-references between them
- Consider extracting common patterns to avoid duplication

### 3.3 Progressive Disclosure Documentation

**Current State**:
- `guides/progressive-disclosure.md` - Concept guide
- `specs/PROMPTS.md` - Contains progressive disclosure section
- Various tool guides mention it

**Recommendation**:
- Keep separate but add more cross-references
- Consider a "see also" section in each

### 3.4 Tool Documentation

**Current State**:
- `guides/tools.md` - General tool guide
- `guides/workspace-tools.md` - VFS/Podman/Planning tools
- `specs/TOOLS.md` - Technical specification
- `specs/WORKSPACE.md` - Workspace specification

**Recommendation**:
- Current structure is good, just ensure cross-references are complete
- Consider adding a "Tool Catalog" page listing all built-in tools

---

## Section 4: Structural Improvements

### 4.1 Navigation Improvements

**Current Issue**: Finding information requires knowing where to look

**Recommendations**:
1. Add a `docs/` or documentation index page with:
   - Quick navigation by topic
   - "I want to..." task-based index
   - Search guidance

2. Add "Related Documentation" sections to each guide

3. Consider a visual documentation map

### 4.2 Versioning

**Current Issue**: No versioning information in documentation

**Recommendations**:
1. Add version badges to README
2. Note API stability in guides
3. Reference CHANGELOG more prominently

### 4.3 Code Examples

**Current Issue**: Some code examples use inconsistent markers

**Recommendations**:
1. Standardize: Use `python` for runnable code, `python nocheck` for pseudo-code
2. Add `# noqa` or similar markers consistently
3. Consider automated example testing

---

## Section 5: Content Gaps

### 5.1 Missing Topics

1. **Async/Streaming**: Currently "no async yet" - when added, needs documentation
2. **Multi-agent patterns**: Mentioned briefly but not covered
3. **Custom adapter creation**: No guide for implementing new adapters
4. **Optimizer framework**: `WorkspaceDigestOptimizer` exists but optimizer creation isn't documented
5. **Skill authoring**: Demo skills exist but no comprehensive authoring guide

### 5.2 Under-documented Features

1. **Experiments** (`specs/EXPERIMENTS.md`): No corresponding guide
2. **Feedback Providers** (`specs/FEEDBACK_PROVIDERS.md`): Limited guide coverage
3. **Lease Extender** (`specs/LEASE_EXTENDER.md`): No guide coverage
4. **Run Context** (`specs/RUN_CONTEXT.md`): No guide coverage
5. **Thread Safety** (`specs/THREAD_SAFETY.md`): Should have guide coverage

---

## Section 6: Demo Skills Review

### Current Skills

1. **code-review/** - Good example of structured skill
2. **ascii-art/** - Simple example
3. **python-style/** - Python-specific example

### Recommendations

1. Add a **guides/skills-authoring.md** covering:
   - SKILL.md format
   - YAML frontmatter options
   - Directory structure
   - Validation requirements
   - Mounting and configuration

2. Consider adding more example skills:
   - Documentation skill
   - Testing skill
   - Refactoring skill

---

## Section 7: Priority Actions

### Immediate (P0)
1. Update outdated model references (gpt-5.2, future-dated Claude models)
2. Verify all cross-referenced files exist
3. Create `guides/resources.md`
4. Create `guides/serialization.md`

### Short-term (P1)
1. Create `guides/errors-and-exceptions.md`
2. Create `guides/time-and-clock.md`
3. Create `guides/mailbox.md`
4. Consolidate or clarify AI assistant instruction files
5. Add "Related Documentation" sections to all guides

### Medium-term (P2)
1. Create `guides/security.md`
2. Create `guides/deployment.md`
3. Create `guides/custom-sections.md`
4. Add more example agents
5. Create skills authoring guide

### Long-term (P3)
1. Create documentation index/navigation page
2. Add automated example testing
3. Create video tutorials or interactive examples
4. Add versioning information

---

## Appendix A: Complete File Inventory

### Root-Level Markdown Files (10)
| File | Lines | Purpose |
|------|-------|---------|
| README.md | ~360 | Project overview |
| CLAUDE.md | ~100 | AI assistant instructions |
| AGENTS.md | ~2 | Redirect to CLAUDE.md |
| GLOSSARY.md | ? | Term definitions |
| IDENTITY.md | ? | Identity information |
| ROADMAP.md | ? | Future plans |
| llms.md | ~1150 | Dense API reference |
| CHANGELOG.md | ? | Version history |
| GEMINI.md | ? | Gemini-specific |
| WARP.md | ? | Warp-specific |

### Guides (25 files)
| File | Lines | Topic |
|------|-------|-------|
| README.md | ~140 | Guide index |
| adapters.md | ~160 | Provider adapters |
| api-reference.md | ~267 | Curated API reference |
| claude-agent-sdk.md | ~600 | Claude SDK integration |
| code-quality.md | ~205 | Quality practices |
| code-review-agent.md | ~330 | Example walkthrough |
| debugging.md | ~225 | Debug and observability |
| evaluation.md | ~250 | Eval framework |
| formal-verification.md | ~185 | TLA+ integration |
| lifecycle.md | ~215 | LoopGroup and health |
| migration-from-dspy.md | ~160 | DSPy migration |
| migration-from-langgraph.md | ~130 | LangGraph migration |
| orchestration.md | ~120 | MainLoop |
| philosophy.md | ~260 | Core concepts |
| progressive-disclosure.md | ~160 | Visibility system |
| prompt-overrides.md | ~145 | Override system |
| prompts.md | ~280 | Prompt authoring |
| quickstart.md | ~270 | Getting started |
| recipes.md | ~200 | Common patterns |
| sessions.md | ~315 | Session management |
| testing.md | ~195 | Testing patterns |
| tools.md | ~380 | Tool authoring |
| troubleshooting.md | ~190 | Common issues |
| workspace-tools.md | ~295 | VFS/Podman/Planning |

### Specs (38 files)
| File | Topic |
|------|-------|
| ADAPTERS.md | Provider adapters |
| CLAUDE_AGENT_SDK.md | Claude SDK spec |
| CLOCK.md | Time handling |
| DATACLASSES.md | Dataclass patterns |
| DBC.md | Design-by-contract |
| DEBUG_BUNDLE.md | Debug bundles |
| DLQ.md | Dead letter queue |
| EVALS.md | Evaluation framework |
| EXAMPLES.md | Example patterns |
| EXPERIMENTS.md | Experiment tracking |
| FEEDBACK_PROVIDERS.md | Feedback system |
| FILESYSTEM.md | Filesystem protocol |
| FORMAL_VERIFICATION.md | TLA+ spec |
| HEALTH.md | Health endpoints |
| LEASE_EXTENDER.md | Lease extension |
| LIFECYCLE.md | Lifecycle management |
| LOGGING.md | Logging spec |
| MAILBOX.md | Message queue |
| MAIN_LOOP.md | MainLoop spec |
| MODULE_BOUNDARIES.md | Module organization |
| POLICIES_OVER_WORKFLOWS.md | Design philosophy |
| PROMPTS.md | Prompt system |
| RESOURCE_REGISTRY.md | DI system |
| RUN_CONTEXT.md | Run context |
| SESSIONS.md | Session spec |
| SKILLS.md | Skills spec |
| SLICES.md | Slice storage |
| TASK_COMPLETION.md | Completion checking |
| TESTING.md | Testing spec |
| THREAD_SAFETY.md | Thread safety |
| TOOLS.md | Tool spec |
| VERIFICATION.md | Verification |
| VERIFICATION_TOOLBOX.md | Verification tools |
| WINK_DEBUG.md | Debug CLI |
| WINK_DOCS.md | Docs CLI |
| WINK_QUERY.md | Query spec |
| WORKSPACE.md | Workspace spec |

### Other Markdown Files
| File | Purpose |
|------|---------|
| demo-skills/code-review/SKILL.md | Example skill |
| demo-skills/ascii-art/SKILL.md | Example skill |
| demo-skills/python-style/SKILL.md | Example skill |
| debug_bundles/README.md | Debug output info |
| tests/plugins/README.md | Test plugin info |
| test-repositories/sunfish/README.md | Test fixture |
| test-repositories/sunfish/LICENSE.md | License |

---

## Appendix B: Cross-Reference Matrix

Key cross-references that should exist:

| From | Should Link To |
|------|---------------|
| guides/tools.md | specs/TOOLS.md, guides/workspace-tools.md |
| guides/sessions.md | specs/SESSIONS.md, specs/SLICES.md |
| guides/adapters.md | specs/ADAPTERS.md, guides/claude-agent-sdk.md |
| guides/evaluation.md | specs/EVALS.md |
| guides/lifecycle.md | specs/HEALTH.md, specs/LIFECYCLE.md |
| guides/debugging.md | specs/DEBUG_BUNDLE.md, specs/LOGGING.md |
| README.md | guides/README.md, llms.md |

---

*Document generated: 2026-01-21*
*Review scope: All .md files in weakincentives repository*

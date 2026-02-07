# SOLID Principles Review

**Date:** 2026-02-07
**Scope:** All major subsystems under `src/weakincentives/`
**Methodology:** File-by-file analysis of every subsystem against each SOLID principle

---

## Executive Summary

The weakincentives codebase demonstrates **strong SOLID adherence at architectural
boundaries** — protocols, layering, and composition are used correctly throughout.
Foundation-layer subsystems (resources, serde, dbc) are nearly exemplary.
Core-layer subsystems (runtime, prompt) show excellent boundary design but
accumulate too many responsibilities inside key classes. Adapter-layer code
suffers most from god-object anti-patterns and hard-coded dependencies.

### Scorecard

| Subsystem | SRP | OCP | LSP | ISP | DIP | Overall |
|-----------|-----|-----|-----|-----|-----|---------|
| resources | A+  | A-  | A+  | A+  | A+  | **A**   |
| serde     | A-  | A+  | A+  | A+  | A+  | **A**   |
| dbc       | A+  | A-  | A+  | A+  | A   | **A-**  |
| filesystem| B+  | B   | A   | A   | B+  | **B+**  |
| skills    | A-  | B+  | A   | A   | B+  | **A-**  |
| evals     | B+  | B+  | A   | A   | A-  | **B+**  |
| runtime   | B+  | B   | B+  | A-  | B   | **B+**  |
| prompt    | C+  | B-  | B   | C+  | B-  | **C+**  |
| adapters  | C   | C+  | B+  | C+  | C   | **C+**  |
| contrib   | B-  | B-  | A   | B+  | B   | **B**   |

### Violation Count by Severity

| Severity | Count | Distribution |
|----------|-------|--------------|
| Critical | 3     | prompt (2), adapters (1) |
| High     | 12    | prompt (5), adapters (3), runtime (3), contrib (1) |
| Medium   | 22    | Spread across all subsystems |
| Low      | 18    | Minor design improvements |

---

## Principle-by-Principle Analysis

### 1. Single Responsibility Principle (SRP)

**Overall assessment: Mixed.** Foundation layers are clean; core and adapter
layers have god objects.

#### Exemplary Components

| Component | Location | Why It Works |
|-----------|----------|--------------|
| `Binding[T]` | `resources/binding.py:27-117` | One concept: resource binding configuration |
| `Score` | `evals/_types.py:53-70` | Pure data: represents evaluation outcome |
| `SliceStore` | `runtime/session/slice_store.py:35-210` | Manages slice lifecycle with factory pattern |
| `ToolResult` rendering | `prompt/tool_result.py:71-164` | Converts tool payloads to text, nothing else |
| `ThrottlePolicy` | `adapters/throttle.py:34-81` | Rate limiting concern only |
| `ReducerRegistry` | `runtime/session/reducer_registry.py:47-189` | Event-to-reducer routing only |
| Each DBC decorator | `dbc/__init__.py` | `@require`, `@ensure`, `@invariant`, `@pure` each own one contract type |

#### Critical Violations

**1. `Section` — 8 concerns in one abstract base class**
`prompt/section.py:49-374` (374 lines, 18+ public methods)

Responsibilities entangled in a single class:
- Rendering pipeline (`render`, `render_body`, `render_override`, `format_heading`)
- Visibility control (`effective_visibility`, `is_enabled`)
- Tool/policy management (`tools`, `policies`, `render_tool_examples`)
- Metadata extraction (`placeholder_names`, `original_body_template`)
- Resource provisioning (`resources`)
- Lifecycle cleanup (`cleanup`)
- Configuration (`accepts_overrides`, `summary`, `visibility`)
- Child section management (`children`)

This is the most impactful SRP violation because Section is the fundamental
building block — every extension and every test must contend with this surface
area.

**Recommended decomposition:**
```
Section (core identity + rendering contract only)
├── SectionRenderer (render methods)
├── SectionVisibilityProvider (visibility logic)
├── SectionToolRegistry (tools/policies access)
├── SectionMetadataProvider (placeholder extraction)
└── SectionResourceProvider (resources)
```

**2. `ClaudeAgentSDKAdapter` — god object at 1773 lines**
`adapters/claude_agent_sdk/adapter.py:317-1773`

10+ distinct responsibilities in one class:
- Constructor/configuration
- Prompt evaluation orchestration
- Async execution management
- SDK options building and schema normalization
- Hook configuration (6+ hooks)
- Token statistics tracking
- Task completion verification
- Message collection/streaming
- Output parsing and structured output validation

**Recommended decomposition:**
```
ClaudeAgentSDKAdapter (thin orchestrator)
├── SchemaValidator (normalize output schemas)
├── HookManager (configure and manage SDK hooks)
├── OutputParser (parse structured output)
└── TokenTracker (extract usage metrics)
```

**3. `Prompt` — 6 tangled responsibilities**
`prompt/prompt.py:280-519`

Mixes parameter binding, resource lifecycle management, rendering
coordination, section/tool inspection, cleanup lifecycle, and metadata
access. Each subgroup could be a separate collaborator.

#### Other Notable Violations

| Component | Location | Concerns Mixed | Severity |
|-----------|----------|----------------|----------|
| `PromptRegistry` | `prompt/registry.py:337-582` | Registration + validation + index building | High |
| `PromptRenderer` | `prompt/rendering.py:122-437` | 6 concerns in 147-line render() method | High |
| `EphemeralHome` | `adapters/claude_agent_sdk/isolation.py:743-1240` | Settings gen + AWS config + env filtering + skill mounting | High |
| `RedisMailbox` | `contrib/mailbox/_redis.py:632-1180` | Lua scripts + serialization + reaper + queue ops | Medium |
| `HostFilesystem` | `filesystem/_host.py:75-785` | Path resolution + git subprocess + I/O + snapshots | Medium |
| `InMemoryFilesystem` | `contrib/tools/filesystem_memory.py:110-682` | File I/O + directory ops + snapshot management | Medium |
| `EvalLoop` | `evals/_loop.py:76-498` | Message processing + evaluation + scoring + DLQ | Medium |
| `FeedbackContext` | `prompt/feedback.py:156-301` | Session state queries + tool queries + computed metrics | Medium |

---

### 2. Open/Closed Principle (OCP)

**Overall assessment: Strong at boundaries, weak in business logic.** Protocol
design enables extension; internal decision points use hardcoded dispatch.

#### Exemplary Components

| Component | Location | Extension Mechanism |
|-----------|----------|---------------------|
| `ProviderAdapter` ABC | `adapters/core.py:41-82` | New adapters extend without modification |
| `FeedbackProvider` protocol | `prompt/feedback.py:308-365` | `name`, `should_run`, `provide` contract |
| `ToolPolicy` protocol | `prompt/policy.py:62-102` | `check` and `on_result` contract |
| Serde coercion chain | `serde/parse.py:758-778` | Coercers return result or sentinel; new types append |
| Schema builder chain | `serde/schema.py:98-114` | Builders return schema or None; new types append |
| Slice backends | `runtime/session/slices/_protocols.py` | Protocol-based; MemorySlice and JsonlSlice extend cleanly |
| `Filesystem` protocol | `filesystem/_protocol.py:49-281` | New backends without modifying protocol |
| `PromptOverridesStore` | `prompt/_overrides_protocols.py` | Multiple store implementations possible |
| Evaluator combinators | `evals/_evaluators.py:196-281` | `all_of`, `any_of` compose without modification |

#### Key Violations

**Hardcoded dispatch chains (require modification to extend):**

| Location | Pattern | Impact |
|----------|---------|--------|
| `adapters/claude_agent_sdk/_errors.py:361-377` | `isinstance` chain for error handlers | New SDK errors require code change |
| `prompt/rendering.py:212-245` | `SectionVisibility.SUMMARY` checks at 3 points | New visibility types require renderer modification |
| `prompt/feedback.py:456-505` | Hardcoded trigger checks (`every_n_calls`, `every_n_seconds`, `on_file_created`) | New trigger types require modification |
| `adapters/claude_agent_sdk/adapter.py:879-939` | Hardcoded unsupported options set | SDK changes require code update |
| `adapters/claude_agent_sdk/adapter.py:188-266` | Hardcoded schema combinator names | New JSON schema features require modification |
| `runtime/session.py:671-678` | Hardcoded default reducer (`append_all`) | Cannot provide alternative defaults |
| `runtime/session.py:745-754` | Fixed telemetry subscriptions (4 event types) | New events require modifying Session |
| `contrib/mailbox/_redis.py:91-238` | 7 hardcoded Lua script strings | New operations require class modification |
| `skills/_validation.py:204-354` | Hardcoded frontmatter field validators | New SKILL.md fields require modification |

**Recommended pattern for dispatch chains:**
```python
# Instead of:
if isinstance(error, CLINotFoundError):
    return _handle_cli_not_found
if isinstance(error, ProcessError):
    return _handle_process_error

# Use registry:
_ERROR_HANDLERS: dict[type, Callable] = {
    CLINotFoundError: _handle_cli_not_found,
    ProcessError: _handle_process_error,
}
```

---

### 3. Liskov Substitution Principle (LSP)

**Overall assessment: Generally strong.** Protocol implementations are
faithful. A few base-class contract issues exist.

#### Exemplary Components

| Component | Evidence |
|-----------|----------|
| `MarkdownSection` | Correctly implements full Section contract; `clone()` returns Self |
| Policy implementations | Both `SequentialDependencyPolicy` and `ReadBeforeWritePolicy` fully honor `ToolPolicy` |
| Slice backends | `MemorySlice` and `JsonlSlice` substitute cleanly; callers never check concrete type |
| `InMemoryFilesystem` | Properly implements `Filesystem` protocol with correct signatures |
| Error hierarchies | All error subtypes (skills, resources, adapters) properly substitute for base |
| Evaluators | All return `Score` per contract; combinators preserve semantics |

#### Violations

| # | Location | Issue | Severity |
|---|----------|-------|----------|
| 1 | `prompt/section.py:232-265` | `Section.render_override()` raises `PromptRenderError` by default — base instances cannot substitute for `MarkdownSection` in override contexts | High |
| 2 | `runtime/session/session_view.py:73-75` | `SessionView` accesses private `_session._select_all()` — breaks encapsulation; method should be in `SessionProtocol` | High |
| 3 | `adapters/throttle.py:84-117` | `ThrottleError` requires `details` parameter that parent `PromptEvaluationError` doesn't — breaks factory substitutability | Medium |
| 4 | `prompt/rendering.py:200-204` | Implicit `accepts_overrides` attribute — uses `getattr` with default True; sections without attribute silently get wrong behavior | Medium |
| 5 | `runtime/mailbox/_types.py:182-204` | `Message.reply()` hardcodes finalization — subtypes cannot override finalization semantics | Medium |

---

### 4. Interface Segregation Principle (ISP)

**Overall assessment: Foundation layers exemplary; core classes expose too
much.** Protocols are lean; implementation classes are fat.

#### Exemplary Interfaces

| Interface | Methods | Assessment |
|-----------|---------|------------|
| `ToolPolicy` | 1 property + 2 methods | Minimal and focused |
| `FeedbackProvider` | 1 property + 2 methods | Clients use exactly what they need |
| `Closeable` | 1 method (`close`) | Single purpose |
| `PostConstruct` | 1 method (`post_construct`) | Single hook |
| `Snapshotable[T]` | 2 methods (`snapshot`, `restore`) | Cohesive pair |
| `SliceView` | 6 methods | Read-only query operations only |
| `SliceFactory` | 1 method (`create`) | Minimal |
| `Evaluator` type | 2 parameters | Clean separation from `SessionEvaluator` (3 params) |
| `Skill` | 3 fields | Minimal data structure |

#### Violations

| # | Location | Issue | Severity |
|---|----------|-------|----------|
| 1 | `prompt/section.py:49-374` | 18+ public methods — rendering clients don't need `resources()`, `cleanup()`, `tools()`; metadata clients don't need `render()` | Critical |
| 2 | `prompt/prompt.py:280-330` | 15+ public members — rendering clients don't need `bind()`, `cleanup()`; cleanup doesn't need `render()` | High |
| 3 | `prompt/rendering.py:163-171` | `render()` method requires 5 complex parameters; high coupling to internal structure | High |
| 4 | `adapters/claude_agent_sdk/adapter.py:317-325` | Constructor takes 5 params covering different concerns (model, client config, model config, tool lists) | Medium |
| 5 | `adapters/claude_agent_sdk/config.py:54-104` | `ClaudeAgentSDKClientConfig` — 10+ fields mixing permission, budget, isolation, transcript concerns | Medium |
| 6 | `adapters/_shared/_bridge.py:179-215` | `BridgedTool` constructor takes 14 parameters | Medium |
| 7 | `contrib/mailbox/_redis.py:699-737` | `RedisMailbox.__init__()` takes 8 parameters | Medium |
| 8 | `runtime/events/types.py:30-46` | `Dispatcher` conflates subscription management with event delivery | Low |
| 9 | `prompt/registry.py:44-133` | `RegistrySnapshot` exposes internal optimization indices | Low |

**Recommended pattern for Section:**
```python
# Segregated protocols for different clients:
RenderableSection  = {render, render_body, render_override, format_heading}
ToolBearingSection = {tools, policies, render_tool_examples}
VisibleSection     = {is_enabled, effective_visibility}
MetadataSection    = {placeholder_names, original_body_template}
CloneableSection   = {clone}
# Section combines all via protocol composition
```

---

### 5. Dependency Inversion Principle (DIP)

**Overall assessment: Abstractions at boundaries are correct; internal wiring
uses direct instantiation.** Foundation layers are protocol-driven; higher
layers hard-code concrete implementations.

#### Exemplary Usage

| Component | Abstraction Used |
|-----------|-----------------|
| `Session` composition | Delegates to `SliceStore`, `ReducerRegistry`, `SessionSnapshotter` |
| `SliceStore` | Uses `SliceFactoryConfig` abstraction, not concrete factories |
| `InMemoryMailbox` | Accepts `MonotonicClock` protocol; enables `FakeClock` injection |
| `MailboxWorker` | Depends on abstract `Mailbox` protocol |
| `ProviderAdapter.evaluate()` | Accepts `Prompt`, `SessionProtocol`, `Deadline`, `Budget` — all abstractions |
| `create_bridged_tools()` | All parameters are protocols |
| `llm_judge()` | Depends on `ProviderAdapter[JudgeOutput]`, not concrete adapter |
| Serde modules | `dump.py` and `parse.py` are independent; `clone()` bridges via public APIs |
| Resources framework | Context depends on `ResourceResolver` protocol, not concrete |

#### Key Violations

**Hard-coded concrete instantiation (should be injected):**

| Location | Dependency | Impact |
|----------|-----------|--------|
| `adapters/claude_agent_sdk/adapter.py:80-95` | Direct `import claude_agent_sdk` at module level | Cannot test with mock SDK |
| `adapters/claude_agent_sdk/adapter.py:44,445` | Direct import and use of `run_async` | Cannot inject alternative async runners |
| `adapters/codex_app_server/adapter.py:50` | Direct import of `CodexAppServerClient` | Cannot swap for protocol-based client |
| `runtime/session.py:740-743` | Creates `InProcessDispatcher()` directly | Session cannot use alternative dispatchers |
| `runtime/session.py:747-754` | Hard-codes visibility reducer registration | Visibility logic baked into Session core |
| `prompt/prompt.py:361-370` | Directly instantiates `PromptRenderer` | Cannot swap renderer implementation |
| `prompt/rendering.py:260-285` | Hard-codes synthetic tool creation functions | Alternative tool injection requires modification |
| `prompt/feedback.py:193-197` | Imports concrete `Filesystem` class in `FeedbackContext` | Should use Filesystem protocol |
| `adapters/claude_agent_sdk/isolation.py:783` | Direct `tempfile.mkdtemp()` | Cannot use in-memory FS for testing |
| `adapters/claude_agent_sdk/workspace.py:377-388` | Always creates `HostFilesystem` | Cannot inject test filesystems |
| `filesystem/_host.py:138-171` | Direct `subprocess.run()` for git | No git backend abstraction |
| `contrib/mailbox/_redis.py:754-760` | Creates default reply resolver internally | Should require injection |

---

## Subsystem-Specific Findings

### Resources (Grade: A)

Near-exemplary SOLID adherence. Minimal, focused protocols (`Closeable`,
`PostConstruct`, `Snapshotable` — each 1-2 methods). Clean layering from
`Binding` to `ResourceRegistry` to `ScopedResourceContext`. Only minor OCP
concern: scope-type dispatch in `context.py:176-182` uses if/elif on `Scope`
enum values, but the enum is intentionally fixed at 3 values.

### Serde (Grade: A)

Chain-of-responsibility patterns in both `parse.py` and `schema.py` are
textbook OCP. Each coercer/builder returns a result or sentinel, allowing new
type handlers to be appended. One SRP concern: `_utils.py` (359 lines) mixes
constraint validation, field manipulation, type identification, and string
normalization — could split into submodules if it grows further.

### DBC (Grade: A-)

Each decorator (`@require`, `@ensure`, `@invariant`, `@pure`) has a single,
focused contract type. The `@pure` implementation necessarily patches global
builtins — a justified DIP violation with well-managed reference-counted
activation. Minor OCP concern: `_normalize_contract_result` uses isinstance
dispatch on result types, but the set of result formats is stable.

### Filesystem (Grade: B+)

Protocol design is clean (`Filesystem`, `SnapshotableFilesystem` extend
naturally). `HostFilesystem` at 710 lines accumulates too many concerns —
particularly git subprocess management for snapshots, which could be extracted
into a `GitSnapshotBackend`. Direct `subprocess.run()` and `os.environ` access
violate DIP.

### Skills (Grade: A-)

Clean dataclasses (`Skill`, `SkillMount`, `SkillConfig` — 3 fields each).
Good use of `_YAMLModule` protocol for YAML abstraction. Hardcoded frontmatter
field validators limit OCP — adding new SKILL.md fields requires modification
of validation functions.

### Evals (Grade: B+)

Evaluator type aliases and combinators (`all_of`, `any_of`, `adapt`) are
textbook OCP. `EvalLoop` mixes 5+ responsibilities (message processing,
evaluation, scoring, failure handling, DLQ). Runtime type dispatch on evaluator
kind (`is_session_aware` check) should use polymorphism.

### Runtime/Session (Grade: B+)

Session properly delegates to `SliceStore`, `ReducerRegistry`, and
`SessionSnapshotter` — preventing a god object. Excellent protocol design for
slice backends and mailbox abstraction. Key issues: hard-coded
`InProcessDispatcher`, fixed telemetry subscriptions (4 event types baked into
Session init), and `SessionView` accessing private `_select_all` method.

### Prompt (Grade: C+)

Strong protocol design at subsystem boundaries (`FeedbackProvider`,
`ToolPolicy`, `PromptOverridesStore`). However, the three core classes
(`Section`, `Prompt`, `PromptRenderer`) each consolidate far too many
responsibilities. Section alone has 18+ public methods serving 8 distinct
concerns. This is the highest-impact area for refactoring — Section is the
fundamental building block and its bloated interface propagates complexity to
every dependent.

### Adapters (Grade: C+)

`ProviderAdapter` ABC and shared bridging layer are well-designed.
`ClaudeAgentSDKAdapter` at 1773 lines is the largest SRP violation in the
codebase with 10+ responsibilities. Error dispatch uses non-extensible
isinstance chains. Multiple hard-coded dependencies (`claude_agent_sdk` import,
`run_async`, `tempfile.mkdtemp`, `HostFilesystem`). Configuration objects are
oversized with 8-14 parameters.

### Contrib (Grade: B)

`RedisMailbox` accumulates 7+ concerns (Lua scripts, serialization, reaper,
queue operations). LSP is clean — all protocol implementations properly
substitute. DIP issues with internally-created reply resolvers and direct OS
dependencies.

---

## Recommended Refactoring Priorities

### Phase 1 — Critical (Architectural Impact)

1. **Decompose `Section`** (`prompt/section.py`)
   - Extract rendering, visibility, tool management, metadata, and resource
     concerns into focused collaborators
   - Define segregated protocols for different client needs
   - Impact: Unblocks cleaner extension and testing of the entire prompt system

2. **Decompose `ClaudeAgentSDKAdapter`** (`adapters/claude_agent_sdk/adapter.py`)
   - Extract `SchemaValidator`, `HookManager`, `OutputParser`, `TokenTracker`
   - Reduce the adapter to a thin orchestrator
   - Impact: Makes the most complex adapter testable and maintainable

3. **Fix `SessionView` encapsulation** (`runtime/session/session_view.py:73-75`)
   - Move `_select_all()` to `SessionProtocol`
   - Impact: Fixes LSP violation at the core session boundary

### Phase 2 — High (Extensibility)

4. **Inject `Dispatcher` into `Session`** (`runtime/session.py:740-743`)
   - Accept dispatcher as required parameter or factory
   - Impact: Enables alternative dispatch strategies

5. **Split `Prompt` responsibilities** (`prompt/prompt.py:280-519`)
   - Extract `PromptResourceManager`, `PromptInspector`
   - Accept `RendererFactory` instead of directly instantiating `PromptRenderer`
   - Impact: Makes prompt lifecycle independently testable

6. **Extract `PromptRegistry` validation** (`prompt/registry.py:337-582`)
   - Separate `PromptValidator` class with per-concern validation methods
   - Impact: Validation independently extensible and testable

7. **Replace error dispatch chains with registries**
   - `adapters/claude_agent_sdk/_errors.py:361-377` → handler registry
   - `prompt/feedback.py:456-505` → trigger evaluator strategy
   - Impact: New error/trigger types don't require framework modification

### Phase 3 — Medium (Code Quality)

8. **Extract `EphemeralHome` concerns** (`adapters/claude_agent_sdk/isolation.py`)
   - `SettingsGenerator`, `AwsConfigManager`, `EnvironmentSanitizer`, `SkillMounter`

9. **Extract `RedisMailbox` helpers** (`contrib/mailbox/_redis.py`)
   - `RedisScriptManager`, `RedisReaper`, `RedisSerializer`

10. **Extract `HostFilesystem` git backend** (`filesystem/_host.py`)
    - `GitSnapshotBackend` protocol and implementation
    - Inject `subprocess` runner

11. **Make telemetry subscriptions pluggable** (`runtime/session.py:745-754`)
    - Registry-based subscription instead of hard-coded event types

12. **Reduce parameter counts** via config grouping
    - `BridgedTool` (14 params) → `BridgedToolConfig` dataclass
    - `ClaudeAgentSDKClientConfig` (10+ fields) → segregated sub-configs

### Phase 4 — Low Priority (Polish)

13. Split `serde/_utils.py` into constraint, field, and type submodules
14. Abstract `SectionVisibility` dispatch in renderer via visitor pattern
15. Segregate `Dispatcher` protocol into `SubscriptionManager` + `EventDelivery`
16. Split `Mailbox` protocol into producer/consumer/admin interfaces
17. Inject YAML module in skills validation

---

## Cross-Cutting Patterns

### What Works Well

1. **Protocol-first design** — Subsystem boundaries are clean abstractions
   that enable substitution and testing
2. **Frozen dataclasses** — Immutability prevents accidental state corruption
   and simplifies reasoning
3. **Chain-of-responsibility** — Serde coercion/schema chains are textbook OCP
4. **Composition over inheritance** — Session delegates to specialized
   components rather than accumulating behavior
5. **Layered architecture** — Foundation → Core → Adapters → High-level
   dependency direction is consistently maintained
6. **Design-by-contract** — Runtime contract enforcement catches violations
   at execution boundaries

### What Needs Improvement

1. **God objects in core and adapter layers** — Section, Prompt, PromptRenderer,
   ClaudeAgentSDKAdapter each consolidate too many responsibilities
2. **Hardcoded dispatch** — isinstance chains and if/elif on types appear in
   error handling, visibility, triggers, and options filtering
3. **Direct instantiation** — Concrete classes created inside constructors
   instead of injected via protocols/factories
4. **Oversized parameter lists** — Configuration objects and constructors with
   8-14 parameters violate ISP and hinder testability
5. **Implicit contracts** — Some base class methods raise by default (Section.render_override)
   or use `getattr` fallbacks, creating fragile substitution boundaries

---

## Conclusion

The codebase has **correct architectural instincts** — protocols at boundaries,
immutable data, layered dependencies, and composition patterns are all present
and well-applied. The primary area for improvement is **internal decomposition**:
the code that sits behind the well-designed boundaries has accumulated too many
responsibilities in key classes. Addressing the Phase 1 items (Section, adapter,
and SessionView) would yield the highest return on investment, as these are the
most-touched abstractions in the system.

# SOLID Refactoring Projects

Project definitions for refactoring work identified by the SOLID principles
review (`reviews/SOLID_PRINCIPLES_REVIEW.md`). Each project is self-contained,
specifies exact files, success criteria, and risk.

---

## Project 1: Decompose Section ABC

**Priority:** P0
**Principles:** SRP, ISP
**Estimated scope:** ~400 lines changed across ~8 files

### Problem

`Section` (`prompt/section.py:49-407`) is a 374-line abstract base class with
18+ public methods serving 8 distinct concerns: rendering, visibility control,
tool/policy management, metadata extraction, resource provisioning, lifecycle
cleanup, configuration, and child section management. Every client of Section
must depend on all 18 methods regardless of which concern it needs.

### Goal

Split Section into a core identity class and focused collaborators so that:
- Rendering clients depend only on rendering methods
- Tool aggregation code depends only on tool-bearing methods
- Resource collectors depend only on resource methods

### Files

| File | Role |
|------|------|
| `prompt/section.py` | **Primary target** — decompose into focused interfaces |
| `prompt/markdown.py` | Sole concrete subclass — must stay substitutable |
| `prompt/rendering.py` | Consumer — uses render, visibility, tools, metadata |
| `prompt/registry.py` | Consumer — uses tools, policies, placeholder_names |
| `prompt/prompt.py` | Consumer — uses resources, cleanup, tools, policies |
| `prompt/feedback.py` | Consumer — uses tools (indirectly) |

### Approach

1. **Define segregated protocols** in a new `prompt/_section_protocols.py`:
   - `RenderableSection`: `render()`, `render_body()`, `render_override()`,
     `format_heading()`, `render_tool_examples()`
   - `ToolBearingSection`: `tools()`, `policies()`
   - `VisibleSection`: `is_enabled()`, `effective_visibility()`
   - `MetadataSection`: `placeholder_names()`, `original_body_template()`,
     `original_summary_template()`
   - `ResourceSection`: `resources()`
   - `CleanableSection`: `cleanup()`

2. **Have `Section` implement all protocols** (preserves backward compat).

3. **Update consumers** to depend on the narrowest protocol:
   - `PromptRenderer` accepts `RenderableSection & VisibleSection & ToolBearingSection`
   - `PromptRegistry` accepts `ToolBearingSection & MetadataSection`
   - `Prompt.cleanup()` accepts `CleanableSection`
   - `Prompt._collected_resources()` accepts `ResourceSection`

4. **Fix the LSP violation** in `render_override()` (line 232-265): change the
   base class default from raising `PromptRenderError` to returning the
   override body unchanged, or make the method abstract and require all
   subclasses to implement it. This fixes the substitutability gap between
   Section and MarkdownSection.

### Success Criteria

- `make check` passes
- No consumer imports all 18 methods when it uses fewer than 5
- `render_override()` no longer raises by default
- `MarkdownSection` passes all existing tests unchanged

### Risks

- `Section` is the most-extended type in the framework; protocol introduction
  must be strictly additive
- `GenericParamsSpecializer` base class complicates the type hierarchy

---

## Project 2: Decompose ClaudeAgentSDKAdapter

**Priority:** P0
**Principles:** SRP, ISP, DIP
**Estimated scope:** ~800 lines extracted across ~5 new files

### Problem

`ClaudeAgentSDKAdapter` (`adapters/claude_agent_sdk/adapter.py:317-1773`) is
1773 lines with 10+ distinct responsibilities: prompt evaluation orchestration,
async execution management, SDK options building, JSON Schema normalization,
hook configuration (6+ hooks), token statistics tracking, task completion
verification, message collection/streaming, output parsing, and structured
output validation.

### Goal

Reduce the adapter to a thin orchestrator (~300 lines) that delegates to
focused collaborators, each independently testable.

### Files

| File | Role |
|------|------|
| `adapters/claude_agent_sdk/adapter.py` | **Primary target** — extract collaborators |
| `adapters/claude_agent_sdk/_errors.py` | Related — error dispatch (Project 6) |
| `adapters/claude_agent_sdk/_hooks.py` | Already extracted — good example |
| `adapters/claude_agent_sdk/config.py` | Config — may need sub-config grouping |
| New: `adapters/claude_agent_sdk/_schema.py` | JSON Schema normalization |
| New: `adapters/claude_agent_sdk/_output_parser.py` | Structured output parsing |
| New: `adapters/claude_agent_sdk/_token_stats.py` | Token/cost extraction |
| New: `adapters/claude_agent_sdk/_options_builder.py` | SDK options construction |

### Approach

1. **Extract `_schema.py`**: Move all JSON Schema combinator normalization
   (the `_normalize_schema`, `_remove_unsupported_combinators`, and related
   helpers around lines 188-266) into a `SchemaTransformer` class.

2. **Extract `_output_parser.py`**: Move structured output parsing, validation,
   and the `_extract_content_block` / `_normalize_response` logic into an
   `OutputParser` class.

3. **Extract `_token_stats.py`**: Move `_extract_usage_from_result` and token
   counting logic into a `TokenStatsExtractor`.

4. **Extract `_options_builder.py`**: Move `_build_sdk_options` and the
   unsupported-options filtering (lines 879-939) into an `SDKOptionsBuilder`.

5. **Inject collaborators** into adapter constructor. Each collaborator should
   be defined as a protocol so tests can mock them independently.

6. **Split `ClaudeAgentSDKClientConfig`** (config.py:54-104, 10+ fields) into
   sub-configs:
   - `PermissionConfig` (allowed_tools, disallowed_tools, permission_mode)
   - `BudgetConfig` (max_turns)
   - `IsolationConfig` (already exists — reference it)
   - `TranscriptConfig` (transcript_output_dir)

### Success Criteria

- `make check` passes
- `ClaudeAgentSDKAdapter.__init__` takes ≤6 parameters
- Adapter class body ≤400 lines
- Each extracted class has independent unit tests

### Risks

- The SDK imports (`claude_agent_sdk`) are only available when the extra is
  installed; extracted modules must use the same lazy-import pattern
- Hook functions reference adapter state — may need adapter-state protocol

---

## Project 3: Fix SessionView Encapsulation

**Priority:** P0
**Principles:** LSP, DIP
**Estimated scope:** ~20 lines changed across 3 files

### Problem

`SessionView._select_all()` (`runtime/session/session_view.py:73-75`) accesses
the private `_session._select_all()` method, breaking encapsulation. The
`pyright: reportPrivateUsage=false` pragma at the top of the file confirms this
is a known workaround. `SessionProtocol` (`runtime/session/protocols.py:33-104`)
does not expose `_select_all`, so `SessionView` can't be backed by any
conforming implementation — only by the concrete `Session` class.

### Goal

Make `_select_all` part of the protocol contract so `SessionView` works with
any `SessionProtocol` implementation.

### Files

| File | Role |
|------|------|
| `runtime/session/protocols.py:33-104` | Add method to `SessionProtocol` |
| `runtime/session/session_view.py:73-75` | Remove private access |
| `runtime/session/session.py` | Already implements it — verify signature |

### Approach

**Option A (preferred):** Add a public `select_all` method to
`SessionProtocol` and `SessionViewProtocol`:

```python
class SessionProtocol(Protocol):
    def select_all[S: SupportsDataclass](self, slice_type: type[S]) -> tuple[S, ...]: ...
```

Then update `SessionView._select_all` to call `self._session.select_all()` and
remove the pyright pragma.

**Option B:** If exposing `select_all` is too broad, add a narrower internal
protocol `_SliceQueryable` that both `Session` and `SessionView` implement, and
have `SessionView.__init__` accept that protocol instead of `SessionProtocol`.

### Success Criteria

- `make check` passes
- `runtime/session/session_view.py` has no `reportPrivateUsage=false` pragma
- `SessionView` can be backed by any `SessionProtocol` conformer

### Risks

- `select_all` becomes part of the public API, which may be too broad
- Other code may also access `_select_all` privately — search first

---

## Project 4: Inject Dispatcher into Session

**Priority:** P1
**Principles:** DIP, OCP
**Estimated scope:** ~40 lines changed across 4 files

### Problem

`Session.__init__` (`runtime/session/session.py`) creates an
`InProcessDispatcher` directly and hardcodes 4 telemetry subscriptions
(lines 745-754: `ToolInvoked`, `PromptExecuted`, `PromptRendered`,
`RenderedTools`). This means:
- Session cannot use an alternative dispatcher (e.g., async, distributed)
- Adding new telemetry event types requires modifying Session internals

### Goal

Accept a dispatcher as a parameter and make telemetry subscriptions
configurable.

### Files

| File | Role |
|------|------|
| `runtime/session/session.py` | Accept dispatcher parameter |
| `runtime/session/session_telemetry.py` | Already extracted — good boundary |
| `runtime/events/types.py` | Dispatcher protocol definition |
| Tests touching Session construction | Update to pass dispatcher |

### Approach

1. Add `dispatcher: TelemetryDispatcher | None = None` parameter to
   `Session.__init__`. When `None`, create `InProcessDispatcher()` as default.

2. Extract `_attach_to_dispatcher` subscriptions into a
   `default_telemetry_subscriptions()` function in `session_telemetry.py` that
   returns a list of `(event_type, handler)` tuples.

3. Add optional `subscriptions` parameter to `Session.__init__` that defaults
   to `default_telemetry_subscriptions(self)`.

### Success Criteria

- `make check` passes
- `Session` can be instantiated with a custom dispatcher
- New telemetry events can be subscribed without modifying `Session`

### Risks

- Session cloning must propagate the dispatcher — verify `clone()` path

---

## Project 5: Split Prompt Responsibilities

**Priority:** P1
**Principles:** SRP, ISP, DIP
**Estimated scope:** ~200 lines refactored across ~4 files

### Problem

`Prompt` (`prompt/prompt.py:280-571`) mixes 6 concerns: parameter binding,
resource lifecycle management, rendering coordination, section/tool inspection,
cleanup lifecycle, and metadata access. It also directly instantiates
`PromptRenderer` (line 367) instead of accepting a factory.

### Goal

Reduce `Prompt` to parameter binding + rendering delegation. Move resource
lifecycle and section inspection into collaborators.

### Files

| File | Role |
|------|------|
| `prompt/prompt.py` | **Primary target** |
| `prompt/_prompt_resources.py` | Already exists — `PromptResources` |
| `prompt/rendering.py` | Renderer — should be injected |
| `prompt/protocols.py` | May need `RendererFactory` protocol |

### Approach

1. **Inject renderer factory:** Define a `RendererFactory` protocol in
   `prompt/protocols.py`. `Prompt.__init__` accepts an optional factory;
   defaults to `DefaultRendererFactory` that wraps the current `PromptRenderer`
   constructor.

2. **Extract `Prompt.policies_for_tool`** (lines 340-359) and
   `Prompt.find_section` (lines 477-482) into a `PromptInspector` helper
   class that accepts a `RegistrySnapshot`. `Prompt` delegates to it.

3. **Extract `Prompt.filesystem()`** (lines 484-502): This is a convenience
   method that does `isinstance(section, WorkspaceSection)` — move it to
   `PromptInspector` and have it depend on the `WorkspaceSection` protocol.

4. **Simplify `Prompt.cleanup()`** (lines 504-518) to iterate and call
   `cleanup()` on `CleanableSection` (from Project 1).

### Success Criteria

- `make check` passes
- `Prompt` body ≤250 lines
- `PromptRenderer` instantiation goes through factory
- `Prompt.filesystem()` no longer does `isinstance` checks

### Risks

- `PromptResources` already wraps `Prompt` — adding another collaborator
  may create circular references. Use lazy initialization.

---

## Project 6: Extract PromptRegistry Validation

**Priority:** P1
**Principles:** SRP, OCP
**Estimated scope:** ~300 lines moved to new file

### Problem

`PromptRegistry` (`prompt/registry.py:337-582`) tangles three concerns:
section registration and tree building, validation across multiple aspects
(enabled predicates, section params, tools/policies, task examples), and
index construction for optimization. The validation logic alone spans
~250 lines and cannot be extended without modifying the registry.

### Goal

Extract validation into a separate `PromptValidator` class that can be
extended (new validation rules added) without modifying the registry.

### Files

| File | Role |
|------|------|
| `prompt/registry.py` | **Primary target** — extract validation |
| New: `prompt/_validation.py` | Extracted validation rules |

### Approach

1. Create `prompt/_validation.py` with a `PromptValidator` class containing:
   - `validate_enabled_predicates(section, params_type)`
   - `validate_section_params(section, params_type)`
   - `validate_tools_and_policies(section, tool_names)`
   - `validate_task_examples(section, structured_output_type)`

2. Each method raises `PromptValidationError` on failure, same as today.

3. `PromptRegistry.register_sections()` calls
   `validator.validate(section, ...)` at each step instead of inline checks.

4. Validator can be subclassed or composed to add custom rules.

### Success Criteria

- `make check` passes
- `PromptRegistry.register_sections()` contains no validation logic
- Each validation rule has independent unit tests

### Risks

- Low — this is a pure extraction with no behavioral change

---

## Project 7: Replace Error and Trigger Dispatch Chains

**Priority:** P1
**Principles:** OCP
**Estimated scope:** ~60 lines changed across 2 files

### Problem

Two dispatch chains use if/isinstance patterns that require modification to
extend:

1. **Error dispatch** (`adapters/claude_agent_sdk/_errors.py:361-377`):
   `_get_error_handler()` uses a chain of `isinstance` checks against 4 SDK
   exception types plus a string check for `MaxTurnsExceededError`.

2. **Feedback triggers** (`prompt/feedback.py:456-505`): `_should_trigger()`
   hardcodes checks for `every_n_calls`, `every_n_seconds`, and
   `on_file_created`. Adding a new trigger type requires modifying this
   function.

### Goal

Replace both with registry-based dispatch so new types can be added without
modifying existing code.

### Files

| File | Role |
|------|------|
| `adapters/claude_agent_sdk/_errors.py` | Error handler registry |
| `prompt/feedback.py` | Trigger evaluator strategy |

### Approach

**Error dispatch:**

Replace `_get_error_handler` with a registry:

```python
_ERROR_HANDLERS: dict[type, _ErrorHandler] = {
    CLINotFoundError: _handle_cli_not_found,
    CLIConnectionError: _handle_cli_connection_error,
    ProcessError: _handle_process_error,
    CLIJSONDecodeError: _handle_json_decode_error,
}

def _get_error_handler(error, error_type):
    for exc_type, handler in _ERROR_HANDLERS.items():
        if isinstance(error, exc_type):
            return handler
    if error_type == "MaxTurnsExceededError":
        return _handle_max_turns_exceeded
    return _handle_unknown_error
```

**Feedback triggers:**

Replace the monolithic `_should_trigger` with a per-field evaluator:

```python
_TRIGGER_EVALUATORS: list[Callable[[FeedbackTrigger, FeedbackContext, str], bool]] = [
    _check_call_count_trigger,
    _check_time_trigger,
    _check_file_created_trigger,
]

def _should_trigger(trigger, context, provider_name):
    return any(evaluate(trigger, context, provider_name)
               for evaluate in _TRIGGER_EVALUATORS)
```

### Success Criteria

- `make check` passes
- New error types or trigger types can be added by appending to a collection
- Existing behavior unchanged

### Risks

- Error handler ordering may matter for exception hierarchy (e.g.,
  `CLIConnectionError` before `ProcessError` if one inherits from the other)
  — preserve ordering in the registry

---

## Project 8: Decompose EphemeralHome

**Priority:** P2
**Principles:** SRP
**Estimated scope:** ~400 lines refactored across ~4 new files

### Problem

`EphemeralHome` (`adapters/claude_agent_sdk/isolation.py:743-1240`) mixes
settings generation, AWS config management, environment variable filtering,
and skill mounting. Direct `tempfile.mkdtemp()` and `os.environ` access make
testing difficult.

### Goal

Extract each concern into a focused class with injected dependencies.

### Files

| File | Role |
|------|------|
| `adapters/claude_agent_sdk/isolation.py` | **Primary target** |
| New: `adapters/claude_agent_sdk/_settings_gen.py` | Settings file generation |
| New: `adapters/claude_agent_sdk/_env_filter.py` | Environment sanitization |
| New: `adapters/claude_agent_sdk/_skill_mounter.py` | Skill mounting |

### Approach

1. Extract `_write_settings()` and related methods → `SettingsGenerator`
2. Extract `_write_aws_config()` → `AwsConfigGenerator`
3. Extract environment variable filtering → `EnvironmentFilter`
4. Extract skill YAML writing → `SkillMounter`
5. `EphemeralHome` becomes an orchestrator that composes these collaborators
6. Accept `tmp_dir_factory: Callable[[], str]` instead of calling
   `tempfile.mkdtemp()` directly

### Success Criteria

- `make check` passes
- `EphemeralHome` body ≤150 lines
- Each collaborator has independent unit tests
- No direct `tempfile.mkdtemp()` calls

### Risks

- Order-of-operations between settings, env, and skills may be fragile

---

## Project 9: Extract RedisMailbox Helpers

**Priority:** P2
**Principles:** SRP, OCP
**Estimated scope:** ~300 lines refactored across ~3 new files

### Problem

`RedisMailbox` (`contrib/mailbox/_redis.py:632-1180`) mixes Lua script
management (7 hardcoded scripts at lines 91-238), serialization/deserialization,
visibility reaper logic, and queue operations.

### Goal

Extract Lua scripts, serialization, and reaper into focused classes.

### Files

| File | Role |
|------|------|
| `contrib/mailbox/_redis.py` | **Primary target** |
| New: `contrib/mailbox/_redis_scripts.py` | Lua script registry |
| New: `contrib/mailbox/_redis_reaper.py` | Visibility reaper |
| New: `contrib/mailbox/_redis_codec.py` | Message serialization |

### Approach

1. Extract Lua scripts into a `RedisScriptRegistry` class with named script
   accessors and lazy loading of SHA digests.
2. Extract `_reap_expired_messages()` and visibility timeout logic into
   `VisibilityReaper`.
3. Extract `_serialize_message()` / `_deserialize_message()` into
   `RedisMessageCodec` with a protocol interface.
4. `RedisMailbox` composes these three collaborators.

### Success Criteria

- `make check` passes
- `RedisMailbox` body ≤300 lines
- Lua scripts are maintained in one place
- Codec is independently testable

### Risks

- Lua scripts reference Redis keys that are constructed by the mailbox
  — need shared key-naming convention

---

## Project 10: Extract HostFilesystem Git Backend

**Priority:** P2
**Principles:** SRP, DIP
**Estimated scope:** ~200 lines extracted to new file

### Problem

`HostFilesystem` (`filesystem/_host.py:75-785`) directly calls
`subprocess.run()` for git operations during snapshot/restore. This mixes
I/O operations with version control and makes testing without a real git
repo impossible.

### Goal

Extract git subprocess calls into a `GitBackend` protocol with a concrete
`SubprocessGitBackend` and a `FakeGitBackend` for tests.

### Files

| File | Role |
|------|------|
| `filesystem/_host.py` | **Primary target** |
| New: `filesystem/_git.py` | `GitBackend` protocol + subprocess impl |

### Approach

1. Define `GitBackend` protocol: `init()`, `add()`, `commit()`, `stash()`,
   `stash_pop()`, `diff()`, `restore()`.
2. Extract all `subprocess.run(["git", ...])` calls into
   `SubprocessGitBackend`.
3. `HostFilesystem.__init__` accepts optional `git_backend` parameter;
   defaults to `SubprocessGitBackend`.
4. Create `FakeGitBackend` for test use.

### Success Criteria

- `make check` passes
- No `subprocess.run` calls in `_host.py`
- Tests that previously needed git can use `FakeGitBackend`

### Risks

- Git operations have side effects on the working directory — backend must
  correctly manage the working directory context

---

## Project 11: Make Telemetry Subscriptions Pluggable

**Priority:** P2
**Principles:** OCP
**Estimated scope:** ~30 lines changed

### Problem

Session hardcodes 4 telemetry subscriptions in `_attach_to_dispatcher`
(lines 745-754 of `runtime/session/session.py`). Adding a new telemetry
event type requires modifying Session.

### Goal

Subscriptions are registered via a list that can be extended without
modifying Session.

### Files

| File | Role |
|------|------|
| `runtime/session/session.py:745-754` | Subscription registration |
| `runtime/session/session_telemetry.py` | Telemetry handlers |

### Approach

1. Define `TelemetrySubscription = tuple[type, Callable[[object], None]]`
2. Add `default_telemetry_subscriptions(session: Session)` function in
   `session_telemetry.py` returning `list[TelemetrySubscription]`
3. `Session.__init__` accepts optional `subscriptions` parameter
4. `_attach_to_dispatcher` iterates the subscriptions list

### Success Criteria

- `make check` passes
- New event types can be subscribed without touching Session

### Risks

- Very low — pure refactoring

---

## Project 12: Reduce Parameter Counts via Config Grouping

**Priority:** P2
**Principles:** ISP
**Estimated scope:** ~100 lines changed across ~4 files

### Problem

Several constructors/functions take too many parameters:
- `BridgedTool` constructor: 14 parameters
  (`adapters/_shared/_bridge.py:179-215`)
- `ClaudeAgentSDKClientConfig`: 10+ fields mixing permission, budget,
  isolation, transcript concerns (`adapters/claude_agent_sdk/config.py:54-104`)
- `RedisMailbox.__init__()`: 8 parameters (`contrib/mailbox/_redis.py:699-737`)

### Goal

Group related parameters into frozen dataclass sub-configs.

### Files

| File | Role |
|------|------|
| `adapters/_shared/_bridge.py` | `BridgedToolConfig` extraction |
| `adapters/claude_agent_sdk/config.py` | Sub-config grouping |
| `contrib/mailbox/_redis.py` | `RedisMailboxConfig` extraction |

### Approach

For each target, create a frozen dataclass grouping cohesive parameters:

- `BridgedToolConfig`: group `filesystem`, `session`, `run_context`,
  `heartbeat`, `deadline`, `budget_tracker` → `ToolExecutionContext`
- `ClaudeAgentSDKClientConfig`: split into `PermissionConfig`,
  `TranscriptConfig`, reuse existing `IsolationConfig`
- `RedisMailboxConfig`: group `queue_name`, `visibility_timeout`,
  `max_delivery_count`, `reap_interval` → single config object

### Success Criteria

- `make check` passes
- No constructor takes >6 positional parameters
- Sub-configs are reusable across call sites

### Risks

- Changing constructor signatures is a breaking change — must update all
  call sites including tests

---

## Dependency Graph

Projects can be worked in parallel within each tier. Cross-tier dependencies
are noted.

```
Tier 0 (independent, start immediately):
  [3] SessionView encapsulation
  [6] Extract PromptRegistry validation
  [7] Replace dispatch chains

Tier 1 (independent of Tier 0):
  [1] Decompose Section  (unblocks Project 5)
  [2] Decompose ClaudeAgentSDKAdapter
  [4] Inject Dispatcher

Tier 2 (depends on Tier 1):
  [5] Split Prompt responsibilities  (depends on: Project 1)
  [8] Decompose EphemeralHome
  [9] Extract RedisMailbox helpers
  [10] Extract HostFilesystem git backend
  [11] Make telemetry pluggable  (depends on: Project 4)
  [12] Reduce parameter counts
```

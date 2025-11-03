# Weak Incentives

**Lean, typed building blocks for side-effect-free background agents.**
Compose deterministic prompts, run typed tools, and parse strict JSON replies without
heavy dependencies. Optional adapters snap in when you need a model provider.

## Why now?

This library was built out of frustration with LangGraph and DSPy to explore
better ways to do state and context management when building apps with LLMs
while allowing the prompts to be automatically optimized.

## Highlights

- Namespaced prompt trees with deterministic Markdown renders, placeholder
  verification, and tool-aware versioning metadata.
- Stdlib-only dataclass serde (`parse`, `dump`, `clone`, `schema`) keeps request and
  response types honest end-to-end.
- Session state container and event bus collect prompt and tool telemetry for
  downstream automation.
- Built-in planning and virtual filesystem tool suites give agents durable plans and
  sandboxed edits backed by reducers and selectors.
- Optional OpenAI and LiteLLM adapters integrate structured output parsing, tool
  orchestration, and telemetry hooks.

## Requirements

- Python 3.12+ (the repository pins 3.14 in `.python-version` for development)
- [`uv`](https://github.com/astral-sh/uv) CLI

## Install

```bash
uv add weakincentives
# optional provider adapters
uv add "weakincentives[openai]"
uv add "weakincentives[litellm]"
# cloning the repo? use: uv sync --extra openai --extra litellm
```

## Tutorial: Explore the Code Review Example

If you are new to agents or to this library, start with
[`code_reviewer_example.py`](code_reviewer_example.py). The file implements a
single-player "pair review" session: you paste a code snippet, the agent drafts
feedback, and you can inspect every prompt, plan, and tool invocation it uses
along the way. Treat the script as an annotated tour of the core abstractions:

- **Session orchestration** keeps track of user turns, plan state, and tool
  outputs.
- **Prompt composition** defines what the model sees—including instructions,
  planning helpers, and virtual filesystem mounts.
- **Model adapters** translate between provider SDKs and the strongly typed
  prompts/results used by Weak Incentives.

The snippets below highlight the main building blocks so you can follow along
directly in the source before running it yourself.

### 1. Start the interactive session

The example introduces a `SunfishReviewSession` helper that wires the entire
runtime together. Even if you have never built an agent loop before, you can
read the constructor to see which pieces matter: an adapter ("how do we talk to
the model?"), an override tag ("which prompt edits should be applied?"), and the
session state that records every interaction (see
[Session State](specs/SESSIONS.md) and [Prompt Event Emission](specs/EVENTS.md)).

```python
session = SunfishReviewSession(
    build_adapter(),
    override_tag=_resolve_override_tag(),
)
```

Inside the class, `_PromptWithOverrides` wraps the base prompt in a
`LocalPromptOverridesStore`. That store watches the filesystem for Markdown
edits, so you can tweak instructions without editing Python code (outlined in
[Prompt Versioning & Persistence](specs/PROMPTS_VERSIONING.md)).

### 2. Compose prompts and tooling

Next, the `build_sunfish_prompt` function assembles the actual conversation
guide. Think of a prompt here as a namespaced tree of Markdown sections. Each
section can inject guidance or register tools; the library renders them into a
deterministic string before every model call. Even if you have never used
planning or VFS tools, skimming the factory shows how they are wired in (see the
specs for the [Prompt Class](specs/PROMPTS.md),
[Planning Tools](specs/PLANNING_TOOL.md), and
[Virtual Filesystem Tools](specs/VFS_TOOLS.md)).

```python
vfs_section = VfsToolsSection(
    session=session,
    mounts=(
        HostMount(
            host_path='sunfish',
            mount_path=VfsPath(('sunfish',)),
            include_glob=SUNFISH_MOUNT_INCLUDE_GLOBS,
            exclude_glob=SUNFISH_MOUNT_EXCLUDE_GLOBS,
            max_bytes=SUNFISH_MOUNT_MAX_BYTES,
        ),
    ),
    allowed_host_roots=(TEST_REPOSITORIES_ROOT,),
)
...
return Prompt[ReviewResponse](
    ns='examples/code-review',
    key='code-review-session',
    name='sunfish_code_review_agent',
    sections=(
        guidance_section,
        planning_section,
        vfs_section,
        asteval_section,
        user_turn_section,
    ),
)
```

### 3. Inspect telemetry and plans

Agent tooling becomes much easier to debug once you can see what happened. The
session captures every tool invocation, stores the latest plan, and exposes
helper methods like `render_tool_history` and `render_plan_snapshot` that turn
the raw state into readable summaries (described in
[Session Snapshots](specs/SESSION_SNAPSHOTS.md)). When you call `history` inside
the running script, it executes code like this:

```python
for index, record in enumerate(self._history, start=1):
    lines.append(
        f"{index}. {record.name} ({record.prompt_name}) → {record.message}"
    )
...
plan = select_latest(self._session, Plan)
if plan is None:
    return 'No active plan.'
```

### 4. Choose a provider and run it

Finally, `build_adapter` selects which model API to call. The logic is ordinary
Python conditionals, so you can follow it even if you have never touched an SDK
before: check an environment variable, ensure the right API key is present, and
instantiate an adapter. The example defaults to OpenAI, but LiteLLM is available
too (see [Adapter Evaluation](specs/ADAPTERS.md) and
[Native OpenAI Structured Outputs](specs/NATIVE_OPENAI_STRUCTURED_OUTPUTS.md)).

```python
if provider == 'openai':
    if 'OPENAI_API_KEY' not in os.environ:
        raise SystemExit('Set OPENAI_API_KEY before running this example.')
    model = os.getenv('OPENAI_MODEL', 'gpt-5')
    return OpenAIAdapter(model=model)
```

### Try it locally

With the structure in mind, invite yourself to poke around: sync dependencies
(`uv sync --extra openai`), ensure the `test-repositories/sunfish` fixture is in
place, and launch the script with your OpenAI API key:

```bash
OPENAI_API_KEY=sk-... uv run python code_reviewer_example.py
```

Inside the REPL, type your own code review prompt, inspect the evolving plan
with `plan`, or replay tool usage with `history`. When something looks worth
customizing, edit `code_reviewer_example.py` directly—the file is compact on
purpose so you can copy patterns into your own project.

## Development Setup

1. Install Python 3.14 (for example with `pyenv install 3.14.0`).

1. Install `uv`, then bootstrap the environment and hooks:

   ```bash
   uv sync
   ./install-hooks.sh
   ```

1. Run checks with `uv run` so everything shares the managed virtualenv:

   - `make format` / `make format-check`
   - `make lint` / `make lint-fix`
   - `make typecheck` (Ty + Pyright, warnings fail the build)
   - `make test` (pytest via `build/run_pytest.py`, 100% coverage enforced)
   - `make check` (aggregates the quiet checks above plus Bandit, Deptry, pip-audit,
     and markdown linting)

## Documentation

- `AGENTS.md` — operational handbook and contributor workflow.
- `specs/` — design docs for prompts, planning tools, and adapters.
- `ROADMAP.md` — upcoming feature sketches.
- `docs/api/` — API reference material.

## License

Apache 2.0 • Status: Alpha (APIs may change between releases)

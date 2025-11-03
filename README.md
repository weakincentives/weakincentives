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

Instead of a separate walkthrough, the tutorial now orients you around
[`code_reviewer_example.py`](code_reviewer_example.py). That script wires the
session, prompt, and adapters together so you can inspect every moving part in
one place. Drop into the file, skim the highlighted snippets below, and then run
it locally—an OpenAI API key is all you need to talk to the default provider.

### 1. Start the interactive session

`SunfishReviewSession` glues together the adapter, session state, overrides
store, and tool logging. The constructor seeds prompt overrides and subscribes
to tool events so you can replay activity during a run (see
[Session State](specs/SESSIONS.md) and [Prompt Event Emission](specs/EVENTS.md)).

```python
session = SunfishReviewSession(
    build_adapter(),
    override_tag=_resolve_override_tag(),
)
```

Inside the class, `_PromptWithOverrides` wraps the base prompt to render through
`LocalPromptOverridesStore`, giving you hot-swappable sections without touching
code ([Prompt Versioning & Persistence](specs/PROMPTS_VERSIONING.md)).

### 2. Compose prompts and tooling

`build_sunfish_prompt` shows how Markdown sections and built-in tool suites are
assembled into one deterministic prompt tree. Planning, VFS, and Python
evaluation tools are registered on the session so subsequent renders automatically
expose them to the model (review the specs for
[Prompt Class](specs/PROMPTS.md),
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

The session captures every tool invocation and the latest plan snapshot. The
helper methods `render_tool_history` and `render_plan_snapshot` turn that state
into readable summaries, making it easy to debug an interaction without digging
into logs ([Session Snapshots](specs/SESSION_SNAPSHOTS.md)).

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

`build_adapter` picks between the OpenAI and LiteLLM adapters based on
environment variables (see [Adapter Evaluation](specs/ADAPTERS.md) and
[Native OpenAI Structured Outputs](specs/NATIVE_OPENAI_STRUCTURED_OUTPUTS.md)).
With no extra configuration, providing `OPENAI_API_KEY` is enough to evaluate the
prompt end to end.

```python
if provider == 'openai':
    if 'OPENAI_API_KEY' not in os.environ:
        raise SystemExit('Set OPENAI_API_KEY before running this example.')
    model = os.getenv('OPENAI_MODEL', 'gpt-5')
    return OpenAIAdapter(model=model)
```

### Try it locally

Once your environment is synced (`uv sync --extra openai`) and the
`test-repositories/sunfish` fixture is present, launch the example:

```bash
OPENAI_API_KEY=sk-... uv run python code_reviewer_example.py
```

Type review prompts, inspect the plan with `plan`, or replay tool usage with
`history`. When you are ready to customize behavior, edit
`code_reviewer_example.py` directly—the script is intentionally compact so you
can lift pieces into your own agent workflow.

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

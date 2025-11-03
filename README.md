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
along the way.

To keep things approachable, the script is now heavily commented with numbered
section headers. Skim them in order—each block mirrors a topic in this tutorial
and links to the supporting specs so you know where to dig deeper next:

| Step | What it covers | Why it matters |
| --- | --- | --- |
| 0 | Project paths and prompt override tagging | Mounts the `sunfish` repo and explains how Markdown overrides load from disk (see [Prompt Versioning & Persistence](specs/PROMPTS_VERSIONING.md)). |
| 1 | `_PromptWithOverrides` proxy | Shows how prompt renders pick up local Markdown edits without patching Python code. |
| 2 | `build_sunfish_prompt` factory | Walks through the prompt tree, planning tools, VFS mounts, and Python evaluation helpers (see [Prompt Class](specs/PROMPTS.md), [Planning Tools](specs/PLANNING_TOOL.md), and [Virtual Filesystem Tools](specs/VFS_TOOLS.md)). |
| 3 | `SunfishReviewSession` orchestration | Demonstrates session state, telemetry collection, and helper renderers (see [Session State](specs/SESSIONS.md) and [Session Snapshots](specs/SESSION_SNAPSHOTS.md)). |
| 4 | `build_adapter` provider selection | Explains how to switch between OpenAI and LiteLLM adapters, validate API keys, and pass through model names (see [Adapter Evaluation](specs/ADAPTERS.md) and [Native OpenAI Structured Outputs](specs/NATIVE_OPENAI_STRUCTURED_OUTPUTS.md)). |
| 5 | `main` REPL loop | Ties everything together in a friendly command-line interface. |

The snippets below correspond to those sections so you can orient yourself in
the source while reading.

> **Heads-up:** Step 0 sits at the top of the file and does not have its own
> snippet below. It simply defines mount paths, glob filters, and the
> `CODE_REVIEW_PROMPT_TAG` default so the later steps stay focused on logic.

### Step 1: Manage prompt overrides

Before any model call happens, the script wraps the base prompt with a
`LocalPromptOverridesStore`. That proxy ensures prompt renders pick up Markdown
edits from `.weakincentives/prompts/overrides/<tag>/...`—handy when you want to
tweak instructions without redeploying code.

```python
self._prompt = cast(
    Prompt[ReviewResponse],
    _PromptWithOverrides(
        base_prompt,
        overrides_store=self._overrides_store,
        tag=self._override_tag,
    ),
)
```

Call `_resolve_override_tag()` to swap override tags at runtime. The helper
normalizes values read from the `CODE_REVIEW_PROMPT_TAG` environment variable so
typos fall back to the `latest` tag gracefully.

### Step 2: Compose prompts and tooling

Next, `build_sunfish_prompt` assembles the conversation guide. Think of a prompt
as a namespaced tree of Markdown sections. Each section injects guidance or
registers tools; the library renders them into a deterministic string before
every model call.

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

### Step 3: Inspect telemetry and plans

`SunfishReviewSession` wires the adapter, session state, and event bus together.
As tool invocations stream in, the event handler records each call so you can
replay it with `history`. Plans get similar treatment—the helper uses
`select_latest` to snapshot the most recent `Plan` dataclass.

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

### Step 4: Choose a provider and run it

`build_adapter` selects which model API to call. The logic is ordinary Python
conditionals: check the `CODE_REVIEW_EXAMPLE_PROVIDER` environment variable,
ensure the right API key is present, and instantiate the adapter.

```python
if provider == 'openai':
    if 'OPENAI_API_KEY' not in os.environ:
        raise SystemExit('Set OPENAI_API_KEY before running this example.')
    model = os.getenv('OPENAI_MODEL', 'gpt-5')
    return OpenAIAdapter(model=model)
```

### Step 5: Try it locally

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

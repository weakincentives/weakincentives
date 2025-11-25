# Weak Incentives (Is All You Need)

Weak Incentives (WINK) is a Python library for building "background agents" (automated
AI systems). It provides lean, typed, and composable building blocks that keep
determinism, testability, and safe execution front and center without relying on
heavy dependencies or hosted services.

The core philosophy treats agent development as a structured engineering
discipline rather than ad-hoc scripting. WINK favors typed prompts,
dataclass-backed outputs, observable sessions, sandboxed tools, provider-agnostic
adapters, and configurable overrides so deterministic, testable behavior comes
first.

## What's novel?

While other agent frameworks provide a toolbox of loose components, WINK
offers an opinionated chassis that emphasizes determinism, type
contracts, and observable workflows:

- **Redux-like state management with reducers.** Every state change is a
  traceable consequence of a published event processed by a pure reducer,
  delivering replayability and visibility far beyond free-form dictionaries or
  mutable object properties. A Redux-like session ledger and in-process event
  bus keep every tool call and prompt render replayable. Built-in planning,
  virtual filesystem, and Python-evaluation sections ship with reducers that
  enforce domain rules while emitting structured telemetry. See [Session
  State](https://github.com/weakincentives/weakincentives/blob/main/specs/SESSIONS.md),
  [Prompt Event Emission](https://github.com/weakincentives/weakincentives/blob/main/specs/EVENTS.md),
  [Planning Tools](https://github.com/weakincentives/weakincentives/blob/main/specs/PLANNING_TOOL.md),
  [Virtual Filesystem Tools](https://github.com/weakincentives/weakincentives/blob/main/specs/VFS_TOOLS.md),
  and [Asteval Integration](https://github.com/weakincentives/weakincentives/blob/main/specs/ASTEVAL.md).
- **Composable prompt blueprints with typed contracts.** Prompts are built from
  reusable sections and chapters backed by dataclasses, so composition and
  parameter binding feel like standard software engineering instead of string
  concatenation. Dataclass-backed sections compose into reusable blueprints that
  render validated Markdown and expose tool contracts automatically. Specs:
  [Prompt Overview](https://github.com/weakincentives/weakincentives/blob/main/specs/PROMPTS.md),
  [Prompt Composition](https://github.com/weakincentives/weakincentives/blob/main/specs/PROMPTS_COMPOSITION.md),
  and [Structured Output](https://github.com/weakincentives/weakincentives/blob/main/specs/STRUCTURED_OUTPUT.md).
- **Integrated, hash-based prompt overrides.** `PromptDescriptor` content
  hashes, tool contracts, and chapter descriptors ensure overrides only apply to
  the intended section version while describing the declared chapter layout.
  `LocalPromptOverridesStore` keeps the JSON artifacts in version control so
  teams can collaborate without risking stale edits. Prompt definitions ship
  with hash-based descriptors and on-disk overrides that stay in sync through
  schema validation and Git-root discovery, laying the groundwork for iterative
  optimization. Review
  [Prompt Overrides](https://github.com/weakincentives/weakincentives/blob/main/specs/PROMPT_OVERRIDES.md)
  for the full contract.
- **First-class in-memory virtual filesystem.** The sandboxed VFS ships as a
  core tool, giving agents a secure workspace whose state is tracked like any
  other session slice and avoiding accidental host access. Everything runs
  locally without hosted dependencies, and prompt renders stay diff-friendly so
  version control captures intent instead of churn. The code-review example
  ties it together with override-aware prompts, session telemetry, and
  replayable tooling.
- **Provider-agnostic adapters.** Adapters connect the framework to providers
  like OpenAI or LiteLLM by handling API calls, tool negotiation, and response
  parsing while keeping the agent logic model-agnostic. Shared conversation
  loops negotiate tool calls, apply JSON-schema response formats, and normalize
  structured payloads so the runtime stays model-agnostic. See
  [Adapter Specification](https://github.com/weakincentives/weakincentives/blob/main/specs/ADAPTERS.md)
  and provider-specific docs such as the
  [LiteLLM Adapter](https://github.com/weakincentives/weakincentives/blob/main/specs/LITE_LLM_ADAPTER.md).
- **Lean dependency surface.** Avoiding heavyweight stacks such as Pydantic
  keeps the core lightweight. Custom serde modules provide the needed
  functionality without saddling users with sprawling dependency trees.

In short, WINK favors software-engineering discipline—determinism,
type safety, testability, and clear state management—over maximizing the number
of exposed knobs.

## Requirements

- Python 3.12+ (the repository pins 3.12 in `.python-version` for development)
- [`uv`](https://github.com/astral-sh/uv) CLI

## Install

```bash
uv add weakincentives
# optional tool extras
uv add "weakincentives[asteval]"
# optional provider adapters
uv add "weakincentives[openai]"
uv add "weakincentives[litellm]"
# optional CLI extras (FastAPI debug UI)
uv add "weakincentives[wink]"
# cloning the repo? use: uv sync --extra asteval --extra openai --extra litellm --extra wink
```

### Debugging snapshots with `wink`

The `wink` CLI ships a debug subcommand that serves a FastAPI-based UI for
exploring session snapshot JSON files. Install the extra and start the server:

```bash
uv run --extra wink wink debug snapshots/5de6bba7-d699-4229-9747-d68664d8f91e.json \
  --host 127.0.0.1 --port 8000
```

The UI is tuned for quick inspection of captured runs:

- Snapshot metadata (path, created timestamp, schema version) is pinned to the
  header so you always know what file is being viewed.
- A slice sidebar lists every slice type with item counts; selecting one streams
  the items into a JSON viewer with copy-to-clipboard and collapse controls.
- A reload action re-reads the snapshot from disk so you can iterate on
  reproducible runs without restarting the server, and a raw download button
  fetches the full JSON for archival or diffing.

![Snapshot Explorer UI (1902x1572)](snapshot_explorer.png)

## Tutorial: An Interactive Code Review Assistant

Let's build a simple, interactive code review assistant. This agent will be able to browse a codebase, answer questions about it, and create plans for more complex reviews. We'll see how WINK helps build this in a structured, observable, and safe way.

The full source for this example is in
[`code_reviewer_example.py`](https://github.com/weakincentives/weakincentives/blob/main/code_reviewer_example.py),
and a high-level walkthrough of the architecture lives in
[`specs/code_reviewer_example.md`](https://github.com/weakincentives/weakincentives/blob/main/specs/code_reviewer_example.md).

### 1. Define the Agent's Task with Typed Dataclasses

Instead of dealing with messy string outputs from the LLM, we'll define our expected output using a Python `dataclass`. The library ensures the model's response is parsed into this structure.

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class ReviewResponse:
    """The structured response we expect from our agent."""
    summary: str
    issues: list[str]
    next_steps: list[str]
```

This `ReviewResponse` class is our contract with the agent. We're telling it exactly what we need: a summary, a list of issues, and next steps.

### 2. Compose a "Blueprint" for the Agent's Brain

In WINK, prompts are not just f-strings; they are composable, versioned objects. We build a `Prompt` from `Section`s, which are like building blocks for the agent's reasoning process.

Here, we create a main prompt that includes:

- `guidance_section`: General instructions for the agent.
- `planning_section`: Gives the agent the ability to create and manage plans.
- `workspace_section`: Provides tools for interacting with a virtual filesystem.
- `user_turn_section`: A placeholder for the user's interactive request.

```python
from weakincentives import MarkdownSection, Prompt
from weakincentives.tools.planning import PlanningToolsSection
from weakincentives.tools.vfs import VfsToolsSection

# Sections are reusable components for building prompts.
guidance_section = MarkdownSection(...)
planning_section = PlanningToolsSection(session=session)
workspace_section = VfsToolsSection(session=session, mounts=...)
user_turn_section = MarkdownSection[ReviewTurnParams](...) # Takes user input

# The Prompt object is the blueprint for the agent.
review_prompt = Prompt[ReviewResponse](
    ns="examples/code-review",
    key="code-review-session",
    name="sunfish_code_review_agent",
    sections=(
        guidance_section,
        planning_section,
        workspace_section,
        user_turn_section,
    ),
)
```

This "prompt as code" approach makes our agent's logic modular, reusable, and easier to test.

### 3. Provide a Safe Workspace with a Virtual Filesystem

To let the agent review code, we need to give it access to the files. But we don't want it to have unrestricted access to the host machine. The `VfsToolsSection` provides a sandboxed in-memory filesystem. We can `mount` a real directory into this virtual workspace.

```python
from weakincentives.tools.vfs import HostMount, VfsPath, VfsToolsSection

# Mount the 'sunfish' test repository into the agent's virtual workspace.
# The agent will see it at the path 'sunfish/'.
mounts = (
    HostMount(
        host_path="sunfish",
        mount_path=VfsPath(("sunfish",)),
    ),
)

vfs_section = VfsToolsSection(
    session=session,
    mounts=mounts,
    allowed_host_roots=(TEST_REPOSITORIES_ROOT,), # Limit host access
)
```

Now the agent can use tools like `vfs_list_files` and `vfs_read_file` to explore the code inside its sandbox, without any risk to the host system.

### 4. Run the Agent and Get a Structured Result

With the prompt defined, we need a `Session` to track state and an `Adapter` to communicate with an LLM provider (like OpenAI).

The `Session` is a central state store. All events, like tool calls and prompt evaluations, are recorded in the session. This makes the agent's execution fully observable and replayable.

```python
from weakincentives.runtime.session import Session
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.adapters.openai import OpenAIAdapter

# The event bus allows us to listen to events from the session.
bus = InProcessEventBus()
# The session tracks all state changes.
session = Session(bus=bus)

# The adapter connects to the LLM provider.
adapter = OpenAIAdapter(model="gpt-5.1")

# This is the main evaluation loop.
response = adapter.evaluate(
    review_prompt,
    ReviewTurnParams(request="Are there any obvious bugs in sunfish.py?"),
    bus=bus,
    session=session,
)

# The output is a typed dataclass object, not a raw string.
review: ReviewResponse = response.output
print(review.summary)
```

If the model's output doesn't match our `ReviewResponse` dataclass, the adapter will raise an error, preventing corrupted data from flowing through the system.

### 5. Observe the Agent's Thought Process

Because every action is tracked in the `Session`, we can inspect the agent's state at any time. For example, we can retrieve the final plan the agent came up with.

```python
from weakincentives.runtime.session import select_latest
from weakincentives.tools.planning import Plan

# Select the latest plan from the session state.
latest_plan = select_latest(session, Plan)

if latest_plan:
    print(f"Plan objective: {latest_plan.objective}")
    for step in latest_plan.steps:
        print(f"- [{step.status}] {step.title}")
```

This observability is crucial for debugging and understanding the agent's behavior. You can see exactly what tools it ran, what files it read, and what conclusions it drew at each step.

### 6. Evolve Prompts without Changing Code

What if you want to tweak the agent's instructions? Instead of editing the Python code, you can use **Prompt Overrides**. WINK can load modified prompt sections from external JSON files.

This allows you to iterate on prompts, A/B test different instructions, and tune the agent's behavior without redeploying your application.

```python
# When rendering, specify a tag to look for overrides.
rendered = review_prompt.render(
    ...,
    overrides_store=LocalPromptOverridesStore(),
    tag="assertive-feedback",
)
```

The `LocalPromptOverridesStore` will look for a JSON file in `.weakincentives/prompts/overrides/` that matches the prompt's namespace, key, and the "assertive-feedback" tag. This makes prompt engineering a data-driven process, separate from application logic.

### You've built a reviewer!

That's it. You now have a deterministic, observable, and safe code review assistant that:

1. Returns structured, typed data.
1. Interacts with files in a sandboxed environment.
1. Creates and follows plans to solve complex tasks.
1. Whose every action is recorded and can be inspected.
1. Can be easily tweaked and improved via external configuration.

This approach turns agent development from a scripting exercise into a structured engineering discipline.

## Logging

WINK ships a structured logging adapter so hosts can add contextual
metadata to every record without manual dictionary plumbing. Call
`configure_logging()` during startup to install the default handler and then
bind logger instances wherever you need telemetry:

```python
from weakincentives.runtime.logging import configure_logging, get_logger

configure_logging(json_mode=True)
logger = get_logger("demo").bind(component="cli")
logger.info("boot", event="demo.start", context={"attempt": 1})
```

The helper respects any existing root handlers—omit `force=True` if your
application already configures logging and you only want WINK to
honor the selected level. When you do want to take over the pipeline, call
`configure_logging(..., force=True)` and then customize the root handler list
with additional sinks (for example, forwarding records to Cloud Logging or a
structured log shipper). Each emitted record contains an `event` field plus a
`context` mapping, so downstream processors can make routing decisions without
parsing raw message strings.

## Development Setup

1. Install Python 3.12 (for example with `pyenv install 3.12.0`).

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

### Integration tests

Provider integrations require live credentials, so the suite stays opt-in. Export the
necessary OpenAI configuration and then run the dedicated `make` target, which disables
coverage enforcement automatically:

```bash
export OPENAI_API_KEY="sk-your-key"
# Optionally override the default model (`gpt-5.1`).
export OPENAI_TEST_MODEL="gpt-5.1"

make integration-tests
```

`make integration-tests` forwards `--no-cov` to pytest so you can exercise the adapter
scenarios without tripping the 100% coverage gate configured for the unit test suite. The
tests remain skipped when `OPENAI_API_KEY` is not present.

## Documentation

- `AGENTS.md` — operational handbook and contributor workflow.
- `specs/` — design docs for prompts, planning tools, and adapters.
- `ROADMAP.md` — upcoming feature sketches.
- `docs/api/` — API reference material.

## License

Apache 2.0 • Status: Alpha (APIs may change between releases)

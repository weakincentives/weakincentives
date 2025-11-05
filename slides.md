---
marp: true
paginate: true
class: lead
---

# Weak Incentives

## Typed building blocks for side-effect-free background agents

---

## Quickstart

1. **Install** the package and optional extras with [`uv`](https://github.com/astral-sh/uv):
   ```bash
   uv add weakincentives
   uv add "weakincentives[asteval]"
   uv add "weakincentives[openai]"
   uv add "weakincentives[litellm]"
   ```
2. **Sync the repo** (if developing locally):
   ```bash
   uv sync --extra asteval --extra openai --extra litellm
   ```
3. **Explore the examples** in `code_reviewer_example.py` and the prompts under `specs/` to understand the runtime patterns.

---

## Observable session state

- Redux-inspired session ledger captures every prompt and tool interaction.
- Reducers attach domain-specific validation while keeping runs deterministic.
- In-process event bus emits `ToolInvoked` and `PromptExecuted` events for telemetry.
- Built-in planning, virtual filesystem, and Python evaluation sections register reducers automatically.

---

## Composable prompt blueprints

- Prompt sections are typed dataclasses with validated placeholders.
- Blueprints assemble reusable trees of sections while enforcing strict contracts.
- Markdown renders stay predictable and version-control-friendly.
- Tool contracts surface alongside prompts to keep structured replies consistent.

---

## Override-friendly workflows

- Prompt overrides enable experimentation without changing source-controlled defaults.
- Hash-based descriptors keep overrides aligned with prompt schema changes.
- On-disk overrides are validated and resolved relative to the Git root.
- Optimization loops plug into the same override surface as manual tweaks.

---

## Provider adapters

- Conversation loop negotiates tool calls across model providers.
- JSON Schema-enforced response formats normalize structured payloads.
- Runtime stays model-agnostic while adapters share the same negotiation contract.

---

## Local-first, deterministic execution

- No mandatory hosted services—everything runs locally by default.
- Reproducible renders keep diffs meaningful and easy to review.
- Code review example combines overrides, session telemetry, and replayable tooling.

---

## Next steps

- Read the specs in `specs/` for deep dives into sessions, prompts, tooling, and overrides.
- Extend the library with typed tools and prompts tailored to your workflow.
- Wire the Marp workflow to publish these slides via GitHub Pages.

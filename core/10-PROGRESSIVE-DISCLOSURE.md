# Progressive Disclosure

**Progressive disclosure** is the convention that sections of a prompt
should default to terse summaries and expand on demand. The agent decides
what depth it needs; the framework re-renders the prompt with the
expanded content; tokens stay focused on what matters now.

This is how WINK keeps prompts compact without amputating capability.

______________________________________________________________________

## The problem it solves

A prompt that includes every tool, every reference document, and every
edge-case instruction in full text on every turn is wasteful in three
ways:

- **Cost.** Every token is paid for on every turn.
- **Attention.** The model's reasoning quality degrades as the prompt
  expands; relevant detail is buried in irrelevant detail.
- **Coupling.** Stuffing low-frequency information into the main prompt
  ties context size to feature count.

Progressive disclosure flips the default. Most sections render as one-line
summaries; the agent expands the ones it needs; the rest stay folded.

______________________________________________________________________

## The mechanism

A section declares its visibility — `FULL` or `SUMMARY`. In summary
mode, only the abridged form renders, and the section's tools are
*withheld* from the rendered prompt.

When the agent decides it needs the full content, it calls a built-in
expansion tool that names the section path. The framework catches this as
a typed signal — not a normal tool result — applies a visibility
override to the session, and re-renders the prompt for the next turn.

After expansion, the section renders fully and its tools become available
in the same turn. The agent does not have to ask twice.

There are two flavors of the expansion tool, depending on what the
section is for:

- For sections that *contain tools*, expansion is permanent: the tools
  must remain accessible for the agent to call them, so the framework
  raises a re-render signal that persists the override.
- For sections that are pure reference material with no tools, expansion
  can be transient: a "read this section" tool returns the content
  without state change.

______________________________________________________________________

## Why withhold tools, not just text

A section's tools are part of its capability. If a tool's documentation is
hidden in summary mode, the agent has no information about how to use the
tool — calling it correctly becomes guesswork. Worse, if the tool is
visible without its documentation, the agent may misuse it.

So summary mode hides the tool itself, not just the prose. The capability
is *latent* until the section is expanded. This keeps the invariant from
[Prompt-is-the-Agent](02-PROMPT-IS-THE-AGENT.md): what the agent can do
is exactly what the prompt currently expresses.

______________________________________________________________________

## Re-render as a typed signal

When expansion happens, the framework does not silently mutate the prompt
mid-turn. It raises a typed exception that the adapter catches, applies
the override to the session's visibility map, and retries the evaluation
from the start of the turn with the new prompt rendered.

This has two important properties:

- **Determinism.** The new render is computed from the section tree and
  the visibility overrides, exactly like any other render. There is no
  "patched mid-stream" state.
- **Boundedness.** The retry loop has a fixed cap. A pathological prompt
  that requests expansion every turn cannot run forever.

______________________________________________________________________

## How summaries should be written

A summary is a *pointer*, not a teaser. It tells the agent what is in the
expanded section and roughly when to ask for it. Good summaries answer:

- What does this section cover?
- When would the agent want the full content?
- What capability does expansion grant?

A bad summary tries to convey the section's information in compressed
form. That defeats the purpose: the agent now has incomplete information
in the summary *and* will not expand because it thinks it has enough.

______________________________________________________________________

## Visibility and tools, together

The visibility model only works because tools live on sections. If tools
existed in a separate registry, hiding their description in the prompt
would not affect their availability — the agent could still call them,
just without instructions.

Because tools are a property of the section, hiding the section hides the
tool. Expanding the section reveals the tool. The two states are coherent
because they have one source of truth.

______________________________________________________________________

## What progressive disclosure is not

- **Not lazy loading of data.** Section content is rendered text, not a
  remote fetch. Heavy data should still be paginated or summarized.
- **Not a cache.** Expansions are session-scoped. They persist for the
  rest of the session unless explicitly cleared, but they are not stored
  beyond the session.
- **Not a workflow.** Expansion does not predetermine an order. The
  agent expands what it needs, in whatever order the task takes.
- **Not for static sections.** A section that is always relevant should
  not be summarized. Reserve `SUMMARY` for sections that *might* matter
  but often don't.

______________________________________________________________________

## Anti-patterns

- **Summaries that lie.** A summary that does not faithfully describe the
  expanded content makes the agent's expand-or-not decision worse, not
  better.
- **Summaries that include the answer.** If the summary is enough, the
  section should not be summarized — render it in full.
- **Summarizing the prompt's main instructions.** The agent needs those
  every turn. Summary is for tangential or specialized content.
- **Many small summarized sections.** A flurry of one-line summaries with
  unclear distinctions makes the agent's choice harder. Group related
  content into one section.

______________________________________________________________________

## Pointers

- [SECTIONS](03-SECTIONS.md) — visibility is a section property.
- [TOOLS](04-TOOLS.md) — why tools must be withheld with their section.
- [PROMPT-IS-THE-AGENT](02-PROMPT-IS-THE-AGENT.md) — the invariant
  progressive disclosure preserves.
- [AGENT-LOOP](15-AGENT-LOOP.md) — where the visibility-expansion
  retry lives.

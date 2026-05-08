# Sections

A **section** is the unit a prompt is built from. Every section bundles
instructions, tools, skills, child sections, and the conditions under which
all of those become visible. Sections are why "the prompt is the agent" is
a structural fact rather than a slogan.

______________________________________________________________________

## What a section bundles

A section carries, in one place:

- A title and a stable key.
- Instruction text — typically markdown with typed placeholders.
- Tools attached to it (this section's contributions to the capability
  surface).
- Skills attached to it (capability bundles mounted into the runtime).
- Child sections it composes with.
- An `enabled` predicate that decides whether the section is active for
  this render.
- A visibility — full text or summary.
- Resources the section requires (e.g., a workspace, a filesystem).
- A flag controlling whether the section's body can be overridden.

Everything that depends on a section being present — its instructions, its
tools, its skills, its children, its resources — is reachable through the
section object. There is no shadow registry.

______________________________________________________________________

## Composition

Sections compose into trees. A prompt template owns a top-level ordered
tuple of sections; each section can have children; each child can have
grandchildren. Rendering walks depth-first and produces markdown with
deterministic numbered headings: `## 1. Title`, `### 1.1. Subtitle`, and so
on.

Composition is structural, not syntactic. Two prompts can share section
*types* (instances are typically clones, not aliases) so that a `Workspace`
section behaves the same way wherever it appears. When you reuse a section,
you reuse its capability — text, tools, and lifecycle.

______________________________________________________________________

## Co-location

This is the central design move. The section that explains a capability is
the same section that *provides* the capability. The text describing how
to read files lives next to the read-file tool definition. They are the
same object.

The consequence: the documentation cannot drift from the implementation
because there are not two artifacts to drift between. Updating the tool
description updates the section. Removing the tool removes the explanation
along with it. New contributors do not have to discover that a tool exists;
the section that motivates it is the same section that declares it.

______________________________________________________________________

## Conditional presence

Two mechanisms control whether a section is in the rendered prompt at all.

**Enabled predicates.** Every section carries a predicate function. If
the predicate returns false, the entire section — its text, its tools,
its skills, its children — is omitted. This is the cleanest way to model
a capability that should only appear under specific session conditions:
scope it inside a section and gate the section.

**Visibility.** A section can render in `FULL` or `SUMMARY` form. In summary
mode, only the abridged text is shown and the section's tools are withheld.
The agent can request expansion through a dedicated tool; the framework
catches the request as a typed exception, applies a visibility override,
and re-renders. After expansion the full section — including its tools —
becomes available.

These mechanisms preserve the invariant that *what the prompt expresses is
what the agent can do*. There is no capability hiding behind a flag the
agent cannot see.

______________________________________________________________________

## Visibility and tokens

Progressive disclosure exists because tokens are the working memory of the
agent. Stuffing every reference document and every advanced tool into every
turn diffuses the model's attention and balloons cost. Sections marked as
summaries default to a short pointer; the agent expands them only when
relevant. After a turn finishes, those expansions can persist for the rest
of the session or revert, depending on policy. (See
[Progressive Disclosure](10-PROGRESSIVE-DISCLOSURE.md).)

______________________________________________________________________

## Resources

Sections may declare resources they need (e.g., a filesystem when the
section provides file tools). These resources merge into the prompt's
resource registry alongside template-level and bind-time bindings.
Lifetimes are tied to the prompt's resource context: when the prompt's
context exits, the section's resources are cleaned up. This avoids the
common pattern of leaking temp directories or open clients across runs.
(See [Resources](09-RESOURCES.md).)

______________________________________________________________________

## Cloning

Sections are immutable but cloneable. Cloning produces a fully decoupled
copy with independent children. This is how you reuse a section across
prompts (or across multiple instances within the same prompt) without
sharing mutable state. Cloning is also how a section participates in
overrides: the override applies to a specific cloned instance identified by
its path within the tree.

______________________________________________________________________

## Why sections are the right unit

- **Cohesion.** A capability is a tight bundle of text, tools, skills, and
  lifecycle. Sections capture that bundle directly.
- **Locality.** Changing a capability changes one node in the tree. There
  is no other place it leaks into.
- **Reachability.** Every active capability is reachable from the prompt
  root. Static analysis on the section tree tells you exactly what the
  agent can do.
- **Override granularity.** Overrides target a section by stable path. A
  small change to instruction text does not require rewriting the whole
  prompt.
- **Visibility coherence.** Visibility is a section property, so collapsing
  or expanding context is a one-line change with predictable downstream
  effects.

______________________________________________________________________

## Anti-patterns

- **Tools without a section.** A tool that exists outside the prompt tree
  has no documented motivation and no visibility control. The framework
  rejects this shape.
- **Sections that share mutable state.** Sections are clonable specifically
  to avoid this. If you want shared state, model it as a session slice or a
  shared resource, not as section-level state.
- **Workflow logic in `enabled` predicates.** The predicate is a clean
  yes/no on whether the capability is currently in scope. Encoding
  multi-step decisions in it pushes orchestration into the wrong layer.

______________________________________________________________________

## Pointers

- [PROMPT-IS-THE-AGENT](02-PROMPT-IS-THE-AGENT.md) — why sections exist
  in the first place.
- [TOOLS](04-TOOLS.md) — what a section contributes when it has
  invokable capability.
- [PROGRESSIVE-DISCLOSURE](10-PROGRESSIVE-DISCLOSURE.md) — visibility,
  expansion, and how it interacts with tools.
- [RESOURCES](09-RESOURCES.md) — what a section can require from the
  resource registry.

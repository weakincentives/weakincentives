# The Prompt Is the Agent

Most frameworks treat a prompt as a string template glued to a separately
registered list of tools. WINK inverts this: a prompt is a typed,
hierarchical, immutable document that *fully determines* what the agent can
think and what it can do. There is no second source of truth.

______________________________________________________________________

## The inversion

Consider what a typical agent framework keeps in separate places:

- The prompt text (a string or markdown file).
- The tool catalog (a registry of name → handler bindings).
- The schema for structured output (a separate schema declaration).
- The conditions under which sections should appear (config flags).
- The instructions for using each tool (more text, often in another file).

Each of these is independently maintained. They drift. When something
breaks, you hunt across files to figure out what was actually sent to the
model.

WINK collapses these into a single artifact. The prompt is a tree. Each node
in the tree — a *section* — carries:

- The instructional text it renders.
- The tools it contributes (with their typed parameters and results).
- The skills it mounts.
- Its child sections.
- A predicate for whether it should be enabled.
- Its visibility (full text vs. summary).
- The resources it requires.

A section that documents how to use a tool *is* the section that provides
that tool. A section that disables itself takes its tools with it. A
section that renders as a summary withholds its tools until the agent
expands it.

______________________________________________________________________

## Templates and prompts

Two related ideas form the spine of the system.

A **prompt template** is the immutable description of an agent: namespace,
key, ordered section tree, attached policies, attached feedback providers,
attached completion checker, attached resources, optional structured output
type. Templates are shareable and composable.

A **prompt** is a template plus runtime bindings — the record parameter
values that fill section placeholders, the resource overrides supplied at
bind time, and the lifecycle context that owns the resources. Prompts are
the things you actually evaluate.

The split lets you describe an agent once and instantiate it many times
under different inputs without copying anything.

______________________________________________________________________

## Single source of truth

Because the prompt *is* the agent:

- There is no tool registry to keep in sync.
- There is no router to maintain.
- There is no documentation file describing tool semantics — the section
  above the tool *is* that documentation.
- Capabilities cannot exist outside the prompt; tools without a section are
  tools nothing can call.
- Disabling part of the prompt removes capability. There is no other place
  that capability could live.

This is what allows the definition layer to be portable. A prompt can be
serialized, rendered, hashed, overridden, snapshotted, and replayed because
it is a closed object graph. Nothing relevant lives outside it.

______________________________________________________________________

## Hierarchy and rendering

Sections are nested. Rendering walks the tree depth-first, producing
markdown with deterministic numbered headings. Two prompts with the same
section tree and same parameters render to byte-identical text. This
determinism is what makes prompt overrides safe (see
[Prompt Overrides](17-PROMPT-OVERRIDES.md)) and snapshots meaningful (see
[Observability](14-OBSERVABILITY.md)).

The hierarchy is also semantic. Higher-level sections set context;
lower-level sections refine. Sections at the same depth are siblings and
render in declared order. Children render under their parent. Parents and
children compose meaning, not just text.

______________________________________________________________________

## Typed all the way down

The prompt is a typed object graph. Section parameters are typed
records. Tool parameters and results are typed records. Structured
output — when the prompt declares one — is a typed record type.
Validation happens at construction, not at the boundary with the model.
By the time the model sees a prompt, every contract has already been
checked.

______________________________________________________________________

## Dynamic without losing rigor

Two mechanisms let prompts adapt to runtime state without abandoning the
"single source of truth" property:

- **Enabled predicates.** Each section carries a predicate function
  that returns whether the section is currently active. The whole
  subtree disappears if it returns false — including its tools.
- **Visibility.** Sections may declare themselves as `SUMMARY`, in which
  case only an abridged form renders until the agent explicitly expands
  them. Tools attached to a summarized section are withheld until the
  expansion happens.

In both cases, the *capability the agent has* is exactly what the *prompt
currently expresses*. There is no hidden capability waiting in a registry.

______________________________________________________________________

## Why it matters

- **Determinism.** Same inputs, same prompt text, same tool set. You can
  write tests that assert exact rendered text.
- **Safe iteration.** Prompt overrides are validated against content
  hashes, so they apply only to the version they were authored against.
- **Inspectability.** Every render is observable. There is nothing
  "happening in the framework" that is not visible in the prompt object.
- **Portability.** Because the prompt is a self-contained graph, any
  adapter can render and execute it without runtime-specific glue.

______________________________________________________________________

## Pointers

- [SECTIONS](03-SECTIONS.md) — the building block of every prompt.
- [TOOLS](04-TOOLS.md) — what a section contributes when it has work for
  the agent to do.
- [PROGRESSIVE-DISCLOSURE](10-PROGRESSIVE-DISCLOSURE.md) — how visibility
  expansion works.
- [TYPED-CONTRACTS](12-TYPED-CONTRACTS.md) — why everything is a
  typed record.
- [PROMPT-OVERRIDES](17-PROMPT-OVERRIDES.md) — how prompts iterate
  safely without source changes.

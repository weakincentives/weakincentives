# Prompt Overrides

A **prompt override** is a hash-validated text replacement applied to
a section, tool description, or example *without changing source*. It
is how prompts iterate safely: optimizers can author overrides
programmatically, humans can tune via JSON, and the framework
guarantees overrides apply only to the version they were authored
against.

This is the mechanism that turns prompt engineering from "edit and
hope" into a versioned, validated workflow.

______________________________________________________________________

## The two problems overrides solve

**Iteration without code changes.** Every prompt edit going through
source means a pull request, a code review, a deploy. For tuning —
trying a slightly different phrasing, adjusting a tool description,
adding an example — the round-trip is too heavy. Overrides decouple
prompt iteration from source changes.

**Drift without surprises.** If you store an override as plain text
and apply it later, you have a problem: was the override authored
against the *current* version of the prompt, or some older version
where the section read differently? Plain-text overrides drift
silently. WINK overrides do not.

______________________________________________________________________

## Hash-validated overrides

Every overridable element of a prompt has a content hash:

- A section's body produces a hash over its text and structure.
- A tool's description produces a hash over its declaration —
  including parameters, results, and other contract surface.
- The whole prompt produces an aggregate descriptor combining all of
  the above.

Every override carries the hash it was authored against. When the
framework loads overrides, it filters out any whose hash no longer
matches the current prompt. The override is not silently applied
against a changed target; it is silently *skipped*.

The author then sees that the override is no longer applying and
either updates it (against the new content) or accepts the current
version. There is no third option where stale overrides corrupt the
prompt.

______________________________________________________________________

## What can be overridden

Five kinds of element accept overrides:

- **Section body.** The instructional text rendered for a section.
- **Tool description.** The short summary the model sees.
- **Tool parameter description.** Each parameter's description text.
- **Tool example.** A specific example by index.
- **Task example.** A specific in-section example by path and index.

The override targets each by stable identifier — the section's path,
the tool's name, the parameter's name, the example's index.

______________________________________________________________________

## What cannot be overridden

Structural shape is *not* overridable:

- Tool names — they are identifiers the model uses to invoke.
- Parameter and result types — they are typed contracts.
- Section structure — children, nesting, attachment order.
- Tool availability — whether a tool exists at all.

The reasoning: structural changes affect what the agent can *do*, not
how the agent is instructed. They belong in source. Text-level
changes affect how the agent is instructed without changing what is
possible. They belong in overrides.

______________________________________________________________________

## Storage and tags

Overrides live in a JSON store, addressed by namespace, key, and
**tag**. A tag groups a related set of overrides — typically an
experiment name, a tuning iteration, or a deployment target. Multiple
tags can exist for the same prompt; one is selected at bind time.

The default location is a directory under the project root, but the
store is a protocol. A custom store can read from a database, an
object store, or any other backing service.

______________________________________________________________________

## Tags and experiments

Tags are how A/B testing without code changes works. Define two tags:

- `baseline` — empty, or the canonical prompt.
- `treatment-assertive-feedback` — overrides that change one
  section's text.

Run an evaluation with both tags as separate experiments. Compare the
pass rates. Promote the winner to baseline by adjusting the source,
or keep iterating with new tags.

(See [Evaluation](16-EVAL-LOOP.md) for how experiments use tags.)

______________________________________________________________________

## Authoring vs applying

Two operations matter.

**Authoring.** A `seed` operation captures the current prompt's
descriptor and writes it to the store as the starting point for a
tag. From there, an author edits the JSON to change specific fields.
The hashes match because the seed was taken from the current prompt;
edits applied to the text fields keep the hashes valid until the
prompt source itself changes.

**Applying.** At bind time, the override store is queried for the
selected tag. Returned overrides are filtered against current hashes;
matches apply, mismatches drop. The applied prompt has the override
text wherever a match was found; original text wherever no override
existed or the override was filtered.

______________________________________________________________________

## Why hashes, not version numbers

A version number is a human convention. It is too easy for two edits
to share a version, or for a prompt to evolve faster than humans
remember to bump version. Hashes are derived from the content; they
change automatically when the content does. There is no manual step
to forget.

This makes overrides robust to refactoring. If a section is moved
within the tree, its hash is preserved (the hash is over content, not
position). If its text changes, the hash changes, and overrides
authored against the old text become inactive — exactly the behavior
you want.

______________________________________________________________________

## Programmatic authoring

Because overrides are JSON, they can be generated by code. An
optimizer that searches over phrasings can author overrides under a
generated tag, run a dataset, and pick the winner. A workflow can
take a complaint, propose a refinement, and store it as an override
for review. Anything that produces text against a known descriptor
can produce overrides.

The framework does not prescribe the optimizer; it just makes the
override format portable.

______________________________________________________________________

## What overrides are not

- **Not a templating engine.** Overrides replace text wholesale; they
  do not interpolate variables. Variables are the prompt's parameter
  bindings.
- **Not a feature-flag system.** They modify prompt content; they do
  not turn capabilities on or off. Capability gating belongs in
  section `enabled` predicates or experiment flags.
- **Not a versioning system for prompts.** They are a layer on top of
  the source prompt. The prompt itself is versioned in code.
- **Not retroactive.** An override authored today affects evaluations
  from today forward. It does not change historical bundle outputs.

______________________________________________________________________

## Anti-patterns

- **Editing the JSON without re-checking the hash.** If you change
  the source prompt and copy old overrides forward, the hash
  validation will silently drop them. Re-seed the tag and re-apply
  edits.
- **Using overrides for structural changes.** "I want a new tool" or
  "I want a new section" is a code change, not an override.
- **Hard-coding tags in source.** A tag selection is a deployment
  concern. Pass it in at bind time, or read it from configuration.
- **Treating overrides as a config bypass.** Overrides are for prompt
  iteration. Operational config (model name, budgets, deadlines)
  belongs in adapter config, not in overrides.
- **Letting tags accumulate.** Old tags become noise. Prune them as
  experiments resolve.

______________________________________________________________________

## Pointers

- [PROMPT-IS-THE-AGENT](02-PROMPT-IS-THE-AGENT.md) — what gets
  overridden.
- [SECTIONS](03-SECTIONS.md) — section-level overrides.
- [TOOLS](04-TOOLS.md) — tool-level overrides.
- [TYPED-CONTRACTS](12-TYPED-CONTRACTS.md) — why hashes are derived
  from typed structure.
- [EVAL-LOOP](16-EVAL-LOOP.md) — how Experiments use override tags.

# Typed Contracts

WINK is a typed system end to end. Section parameters, tool inputs and
outputs, structured outputs, events, slice payloads, adapter
configurations — all are **typed records** with strict types. Validation
happens at construction, not at the model boundary. By the time anything
reaches the LLM, every contract has already been checked.

This is what lets the rest of the system stay small. Without typed
contracts, every layer would need its own runtime validation; with them,
each layer can trust its inputs.

A *typed record* here is whatever the host language calls a named
fixed-shape value with declared fields and per-field types: a
`dataclass` in Python, a `struct` in Rust or Go, a `record` in Java
or C# or TypeScript, a `case class` in Scala, and so on. The vocabulary
in this doc is deliberately neutral — the discipline is the same in
every language WINK could be ported to.

______________________________________________________________________

## The discipline

A typed contract has three properties:

- **The shape is a named record type.** Not a free-form map, not an
  unvalidated structurally-typed map, not raw text. A type with declared
  fields, declared field types, and an explicit name.
- **Construction validates.** Required fields must be present; types are
  honored; semantic invariants (positive integers, well-formed paths,
  matching identifiers) are checked at the moment of creation.
- **A strict static type checker enforces it.** The project runs the
  language's strict type checker in CI. Type mismatches surface before
  the code ever runs.

This is not a stylistic preference. It is what allows WINK to skip
runtime defensive coding.

______________________________________________________________________

## Why named records, not other shapes

Several alternatives exist; each has a structural problem.

- **Free-form maps.** Allow typos, missing keys, and silent type drift.
  The type checker cannot help.
- **Structurally-typed maps** (e.g., shape-only types over key-value
  containers). Typed at definition but bypass runtime validation — a
  wrong shape can still arrive at runtime through any deserialization
  path.
- **Third-party schema libraries.** Powerful, but introduce a parallel
  type surface that competes with the language's own type system. WINK
  prefers to lean on the language's standard typed-record facility plus
  a small internal serialization layer.
- **Plain unstructured types** (e.g., generic objects without declared
  fields). Lose introspection: no automatic equality, no useful
  printable form, no easy serialization.

Named record types give the right tradeoff: language-native, type-checked
at edit time, validated at construction, serializable through
reflection.

______________________________________________________________________

## Two tiers of immutability

Most records in WINK fall into one of two tiers.

- **Tier 1 — Simple value objects.** Fields stored exactly as
  declared, no validation beyond type hints. Direct construction works
  normally. The plain immutable-record convention suffices.
- **Tier 2 — Validated / normalized records.** Direct construction is
  blocked at runtime; a `create(...)` factory runs the validation and
  then produces the instance. This is for types where invariants must
  hold — deadlines that must be in the future, identifiers that must
  match a pattern, parameters that must agree with each other.

The split keeps the simple case simple — most types just need the basic
immutable-record convention — and forces validation on the types that
actually need it.

The mechanism for "block direct construction" varies by language: a
guard in the constructor, a private constructor with a public factory,
a sealed type with smart constructors. The discipline is uniform; the
implementation is per-language.

______________________________________________________________________

## Immutable by default

WINK records are immutable unless there is an explicit reason
otherwise. Mutability is a special case, not the default. The reasoning
chain:

- Reducers must produce new state, not mutate it. Immutable state makes
  that structurally enforced.
- Snapshots assume their captured values won't change underneath them.
  Immutable instances guarantee this without copying.
- Equality and hashing become reliable — immutable records can be used
  as map keys, set members, and cache keys without surprises.
- Concurrent code can share immutable values without locks.

Together, immutability is what makes the event-driven architecture work.
Mutable state at the boundaries would defeat reducers, snapshots, and
transactions.

______________________________________________________________________

## Validation at the boundary, not in handlers

Tool handlers do not validate their parameters. The framework parses the
incoming wire format (typically JSON, but the contract is the type, not
the encoding) into the parameter record with strict mode (no extra keys
allowed) and rejects anything that doesn't fit. By the time the handler
runs, the parameters are already correct shapes with correct types.

The handler's job is the *operation*. Validation is the framework's job.
This is a deliberate division: handlers stay small and focused, while
validation lives in one place that every adapter shares.

The same applies in reverse: a tool result is constructed as a typed
record, and the framework serializes it for transmission to the model.
The handler does not format the wire payload.

______________________________________________________________________

## Structured output

When a prompt declares a structured output type, the framework attaches
a schema derived from the record type to the request (in the formats
that support it) and parses the model's response back into the record.
Failed parses surface as a typed error with the raw response attached
for debugging.

This is not a separate validation layer; it is the same record
discipline applied at one more boundary. The same rules — required
fields, type checking, invariants — apply to structured output that
apply to tool parameters.

A prompt may declare a *list* of a record type as well, in which case
the framework expects an array. Validation, parsing, and error reporting
are uniform.

______________________________________________________________________

## Polymorphism and serialization

Some events form union types — different variants of "what happened"
that share a slot in a slice. The serialization layer uses a
discriminator field on serialized payloads to route deserialization to
the right record type. This is what allows event histories to round-trip
through the wire format without losing variant information.

This is the only piece of "magic" in the type story, and it lives at the
serialization boundary, not in the type system itself. Sum types,
tagged unions, sealed hierarchies, and discriminated unions all
implement the same idea — the host language picks the spelling.

______________________________________________________________________

## Design by contract for invariants types can't express

Some constraints are stronger than types alone can capture: "this method
preserves a balance ≥ 0", "this function's result is no shorter than its
input." For these, WINK uses a small in-house design-by-contract layer:
preconditions, postconditions, and invariants attached at the
declaration site of functions, methods, and types. They run by default;
they can be temporarily suspended in performance-critical paths; they
fail loud with diagnostic detail.

This is a *complement* to types, not a substitute. Types check shape;
contracts check semantics. Languages with native contract support
(Eiffel, Ada/SPARK, dependent types) get this for free; languages
without it use lightweight runtime checks.

______________________________________________________________________

## Anti-patterns

- **Reaching for free-form maps when a record would do.** Once a
  payload is a key-value map, every consumer needs its own validation.
  The language can't help.
- **Using "any"-typed fields.** This is the type-system equivalent of
  reaching for a free-form map. If the type is genuinely
  unconstrained, document why.
- **Validating in handlers.** The framework parses the parameters; the
  handler trusts them. Adding a manual null/shape check is a sign the
  type is wrong.
- **Mutable record payloads.** Reducers must produce new values; mutable
  payloads break determinism and snapshot semantics.
- **Accepting "extra" keys silently.** Extra keys in serialized payloads
  are usually typos or version drift. WINK rejects them. Deserialize
  with strict mode.

______________________________________________________________________

## Pointers

- [TOOLS](04-TOOLS.md) — typed parameters and results.
- [STATE](05-STATE.md) — typed events and slice payloads.
- [PROMPT-IS-THE-AGENT](02-PROMPT-IS-THE-AGENT.md) — typed structured
  output.
- [PRINCIPLES](PRINCIPLES.md) §6 — typed contracts everywhere.

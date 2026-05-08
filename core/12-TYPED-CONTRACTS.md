# Typed Contracts

WINK is a typed system end to end. Section parameters, tool inputs and
outputs, structured outputs, events, slice payloads, adapter
configurations — all are dataclasses with strict types. Validation
happens at construction, not at the model boundary. By the time anything
reaches the LLM, every contract has already been checked.

This is what lets the rest of the system stay small. Without typed
contracts, every layer would need its own runtime validation; with them,
each layer can trust its inputs.

______________________________________________________________________

## The discipline

A typed contract has three properties:

- **The shape is a dataclass.** Not a free-form dictionary, not a
  TypedDict, not a string of JSON. A real Python class with declared
  fields.
- **Construction validates.** Required fields must be present; type
  hints are honored; semantic invariants (positive integers, well-formed
  paths) are checked at the moment of creation.
- **The strict type checker enforces it.** Pyright runs in strict mode in
  CI. Type mismatches surface before the code ever runs.

This is not a stylistic preference. It is what allows WINK to skip
runtime defensive coding.

______________________________________________________________________

## Why dataclasses, not other shapes

Several alternatives exist; each has a structural problem.

- **Free-form dicts** allow typos, missing keys, and silent type drift.
  They cannot be validated by a type checker.
- **TypedDicts** are typed at definition but bypass runtime validation —
  a wrong shape can still arrive at runtime through any deserialization
  path.
- **Pydantic / external schema libraries** are powerful but introduce a
  third-party dependency surface that competes with the language's own
  type system. WINK chooses to lean on stdlib dataclasses plus a small
  internal serde layer.
- **Plain classes** lose introspection: no automatic equality, no
  reasonable `__repr__`, no easy serialization.

Dataclasses give the right tradeoff: language-native, type-checked at
edit time, validated at construction, serializable through reflection.

______________________________________________________________________

## Two tiers of immutability

Most dataclasses in WINK fall into one of two tiers.

- **Tier 1 — Simple value objects.** Fields stored exactly as
  declared, no validation beyond type hints. Direct construction works
  normally. Frozen and slotted by convention to make immutability
  structural.
- **Tier 2 — Validated / normalized classes.** Inherit from a
  construction-controlled base. Direct construction is blocked at
  runtime; a `create(...)` classmethod runs the validation and then
  produces the instance. This is for types where invariants must hold —
  deadlines that must be in the future, identifiers that must match a
  pattern, parameters that must agree with each other.

The split keeps the simple case simple — most types just need
`@FrozenDataclass` and they are done — and forces validation on the
types that actually need it.

______________________________________________________________________

## Frozen by default

WINK dataclasses are frozen unless there is an explicit reason otherwise.
Mutability is a special case, not the default. The reasoning chain:

- Reducers must produce new state, not mutate it. Frozen state makes that
  structurally enforced.
- Snapshots assume their captured values won't change underneath them.
  Frozen instances guarantee this without copying.
- Equality and hashing become reliable — frozen dataclasses can be used
  as dict keys, set members, and cache keys without surprises.
- Concurrent code can share immutable values without locks.

Together, immutability is what makes the event-driven architecture work.
Mutable state at the boundaries would defeat reducers, snapshots, and
transactions.

______________________________________________________________________

## Validation at the boundary, not in handlers

Tool handlers do not validate their parameters. The framework parses the
incoming JSON-or-equivalent into the parameter dataclass with strict
mode (no extra keys allowed) and rejects anything that doesn't fit. By
the time the handler runs, the parameters are already correct shapes
with correct types.

The handler's job is the *operation*. Validation is the framework's
job. This is a deliberate division: handlers stay small and focused,
while validation lives in one place that every adapter shares.

The same applies in reverse: a tool result is constructed as a
dataclass, and the framework serializes it for transmission to the
model. The handler does not format JSON.

______________________________________________________________________

## Structured output

When a prompt declares a structured output type, the framework attaches
the dataclass's JSON Schema to the request (in the formats that support
it) and parses the model's response back into the dataclass. Failed
parses surface as a typed error with the raw response attached for
debugging.

This is not a separate validation layer; it is the same dataclass
discipline applied at one more boundary. The same rules — required
fields, type checking, invariants — apply to structured output that
apply to tool parameters.

A prompt may declare a *list* of a dataclass type as well, in which case
the framework expects an array. Validation, parsing, and error reporting
are uniform.

______________________________________________________________________

## Polymorphism and serde

Some events form union types — different variants of "what happened"
that share a slot in a slice. The serde layer uses a `__type__` field
on serialized payloads to route deserialization to the right dataclass.
This is what allows event histories to round-trip through JSON without
losing variant information.

This is the only piece of "magic" in the type story, and it lives at the
serialization boundary, not in the type system itself.

______________________________________________________________________

## Design by contract for invariants types can't express

Some constraints are stronger than types alone can capture: "this method
preserves a balance ≥ 0", "this function's result is no shorter than its
input." For these, WINK uses a small in-house design-by-contract layer:
preconditions, postconditions, and class invariants attached as
decorators. They run by default; they can be temporarily suspended in
performance-critical paths; they fail loud with diagnostic detail.

This is a *complement* to types, not a substitute. Types check shape;
contracts check semantics.

______________________________________________________________________

## Anti-patterns

- **Reaching for dictionaries when a dataclass would do.** Once a
  payload is a dict, every consumer needs its own validation. The
  language can't help.
- **Using `Any` as a type hint.** This is the type-system equivalent of
  reaching for a dictionary. If the type is genuinely unconstrained,
  document why.
- **Validating in handlers.** The framework parses the parameters; the
  handler trusts them. Adding `if params.x is None: raise` is a sign
  the type is wrong.
- **Mutable slice payloads.** Reducers must produce new values; mutable
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

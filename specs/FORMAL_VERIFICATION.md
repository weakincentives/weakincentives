# Formal Verification with Embedded TLA+

## Purpose

Enable embedding TLA+ formal specifications directly in Python code. Specs live
next to implementation, verified via TLC model checker.

**Implementation:** `src/weakincentives/formal/`

## Components

1. **`@formal_spec` decorator** - Embed TLA+ metadata in Python classes
2. **Test utilities** - Extract specs and run TLC
3. **CI integration** - `make verify-formal`

## @formal_spec Decorator

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `str` | TLA+ module name (required) |
| `state_vars` | `Sequence[StateVar]` | State variables |
| `actions` | `Sequence[Action]` | Actions/transitions |
| `invariants` | `Sequence[Invariant]` | Safety properties |
| `constants` | `dict[str, int \| str]` | Model constants |
| `constraint` | `str \| None` | State constraint |
| `extends` | `Sequence[str]` | TLA+ modules to extend |

### StateVar

| Field | Description |
|-------|-------------|
| `name` | Variable name |
| `type` | TLA+ type (Nat, Seq(...), Set(...)) |
| `description` | Human-readable description |
| `initial_value` | Override type-based default |

### Action

| Field | Description |
|-------|-------------|
| `name` | Action name |
| `parameters` | Bounded parameters |
| `preconditions` | Enabling conditions |
| `updates` | State variable updates |

### Invariant

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (INV-1) |
| `name` | TLA+ name |
| `predicate` | Boolean expression |

## Testing Utilities

| Function | Description |
|----------|-------------|
| `extract_spec(cls)` | Extract spec from decorated class |
| `write_spec(spec, path)` | Write .tla and .cfg files |
| `model_check(spec)` | Run TLC model checker |
| `extract_and_verify(cls, ...)` | All-in-one test entry point |

### Model Check Result

| Field | Description |
|-------|-------------|
| `passed` | All invariants held |
| `states_generated` | States checked |
| `stdout` | TLC stdout output |
| `stderr` | TLC stderr output |
| `returncode` | Process exit code |

## State Space Optimization

- **Small constants**: `MaxMessages: 2` not `100`
- **State constraints**: `constraint="depth <= 5"`
- **Narrow domains**: `"0..2"` not `"0..10"`

## Verification

```bash
make verify-formal
pytest formal-tests/ -v
```

## Best Practices

1. Start small - model 2-3 core actions first
2. Use smallest constants that test interesting behavior
3. Each invariant tests one specific property
4. Verify incrementally - add one action at a time

## Related Specifications

- `specs/VERIFICATION.md` - RedisMailbox verification

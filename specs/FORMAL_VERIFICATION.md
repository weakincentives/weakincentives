# Formal Verification with Embedded TLA+

## Purpose

Enable embedding TLA+ formal specifications directly in Python code. Specs live
next to implementation, verified via TLC model checker.

**Implementation:** `src/weakincentives/formal/`

## Components

1. **`@formal_spec` decorator** - Embed TLA+ metadata in Python classes
1. **`FormalSpec` dataclass** - Complete specification with TLA+ generation
1. **Test utilities** - Extract specs and run TLC
1. **CI integration** - `make verify-formal`

## @formal_spec Decorator

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `str` | TLA+ module name (required) |
| `state_vars` | `list[StateVar]` | State variables |
| `actions` | `list[Action]` | Actions/transitions |
| `invariants` | `list[Invariant]` | Safety properties |
| `constants` | `dict[str, Any]` | Model constants |
| `constraint` | `str \| None` | State constraint |
| `extends` | `tuple[str, ...]` | TLA+ modules to extend |
| `helpers` | `dict[str, str]` | Helper operator definitions (raw TLA+) |

### StateVar

| Field | Description |
|-------|-------------|
| `name` | Variable name |
| `type` | TLA+ type (Nat, Seq(...), Set(...)) |
| `description` | Human-readable description (optional) |
| `initial_value` | Override type-based default (optional) |

### ActionParameter

| Field | Description |
|-------|-------------|
| `name` | Parameter name (e.g., "consumer") |
| `domain` | TLA+ domain expression (e.g., "1..NumConsumers") |

### Action

| Field | Description |
|-------|-------------|
| `name` | Action name |
| `parameters` | Tuple of `ActionParameter` with domains |
| `preconditions` | Enabling conditions (tuple of strings) |
| `updates` | State variable updates (dict) |
| `description` | Human-readable description (optional) |

### Invariant

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (INV-1) |
| `name` | TLA+ name |
| `predicate` | Boolean expression |
| `description` | Human-readable description (optional) |

## FormalSpec Class

The `FormalSpec` dataclass holds all specification metadata and provides TLA+
generation methods:

| Method | Description |
|--------|-------------|
| `to_tla()` | Generate complete TLA+ module as string |
| `to_tla_config(...)` | Generate TLC configuration file content |

The `to_tla_config()` method accepts optional parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `init` | `str \| None` | Initial state formula (for simulation) |
| `next` | `str \| None` | Next-state formula (for simulation) |
| `check_deadlock` | `bool` | Whether to check for deadlocks (default: False) |
| `state_constraint` | `str \| None` | Override state constraint expression |

## Testing Utilities

Located in `weakincentives.formal.testing`:

| Function | Description |
|----------|-------------|
| `extract_spec(cls)` | Extract spec from decorated class |
| `write_spec(spec, path)` | Write .tla and .cfg files |
| `model_check(spec, *, tlc_config)` | Run TLC model checker (3-minute timeout) |
| `extract_and_verify(cls, ...)` | All-in-one test entry point |

### ModelCheckResult

| Field | Description |
|-------|-------------|
| `passed` | All invariants held |
| `states_generated` | States checked |
| `stdout` | TLC stdout output |
| `stderr` | TLC stderr output |
| `returncode` | Process exit code |

### ModelCheckError

Exception raised when TLC is not found or model checking fails.

## State Space Optimization

- **Small constants**: `MaxMessages: 2` not `100`
- **State constraints**: `constraint="depth <= 5"`
- **Narrow domains**: `"0..2"` not `"0..10"`

## Verification

```bash
make verify-formal         # Full TLC model checking (~30s)
make verify-formal-fast    # Fast extraction only, no model check (~1s)
make verify-formal-persist # Full check + persist specs to specs/tla/extracted/
make verify-all            # Full verification + property-based tests
```

Direct pytest invocation:

```bash
uv run pytest formal-tests/ -v --no-cov
uv run pytest formal-tests/ --skip-model-check  # Extraction only
uv run pytest formal-tests/ --persist-specs     # Save extracted specs
```

## Best Practices

1. Start small - model 2-3 core actions first
1. Use smallest constants that test interesting behavior
1. Each invariant tests one specific property
1. Verify incrementally - add one action at a time

## Related Specifications

- `specs/VERIFICATION.md` - RedisMailbox verification

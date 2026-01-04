# Makefile Formal Verification Targets

## Overview

This document describes the Makefile targets for embedded TLA+ specification
verification. These targets are already implemented in the repository root
Makefile.

## Implemented Targets

The following targets support embedded TLA+ specifications:

```makefile
# =============================================================================
# Embedded TLA+ Specifications
# =============================================================================

# Full formal verification with TLC model checking (~30s)
# Uses temp directory (no filesystem pollution)
verify-formal:
	@uv run --all-extras pytest formal-tests/ --no-cov -q --timeout=240

# Fast extraction only (development, ~1s)
# Skips model checking - use only for rapid iteration
verify-formal-fast:
	@uv run --all-extras pytest formal-tests/ --no-cov -q --skip-model-check

# Full verification + persist specs to specs/tla/extracted/
verify-formal-persist:
	@uv run --all-extras pytest formal-tests/ --no-cov -v --persist-specs

# Run Hypothesis property-based tests
property-tests:
	@uv run --all-extras pytest tests/contrib/mailbox/test_redis_mailbox_properties.py \
		tests/contrib/mailbox/test_redis_mailbox_invariants.py \
		--no-cov -v --hypothesis-show-statistics

# Run concurrent stress tests
stress-tests:
	@uv run --all-extras pytest tests/contrib/mailbox/test_redis_mailbox_stress.py \
		--no-cov -v -m slow --timeout=120

# Run all mailbox verification
verify-mailbox: verify-formal property-tests
	@echo "All mailbox verification checks passed"

# Run all formal verification (embedded specs + property tests)
verify-all: verify-formal property-tests
	@echo "✓ All formal verification passed"

# Remove extracted TLA+ specs
clean-extracted:
	@echo "Cleaning extracted TLA+ specs..."
	@rm -rf specs/tla/extracted/
	@echo "✓ Cleaned specs/tla/extracted/"
```

## Usage Examples

### Development Workflow

```bash
# During development - fast extraction without model checking
make verify-formal-fast

# Before commit - full verification with TLC model checker
make verify-formal

# Persist specs to specs/tla/extracted/ for inspection
make verify-formal-persist

# Run all verification (embedded specs + property tests)
make verify-all

# Clean up extracted files
make clean-extracted
```

### CI Integration

For CI, run the full verification suite:

```yaml
# .github/workflows/verify.yml

- name: Run formal verification
  run: make verify-all
```

This ensures:

1. Embedded `@formal_spec` decorators are extracted and model-checked
1. Property-based tests validate implementation
1. Liveness properties are verified via Hypothesis

## Target Descriptions

| Target | Description | Speed | When to Use |
| ----------------------- | ------------------------------ | ------------ | -------------------------- |
| `verify-formal` | Extract and model check | ~30s | Before commit, CI |
| `verify-formal-fast` | Extract only (no TLC) | ~1s | Development, syntax check |
| `verify-formal-persist` | Extract + save to disk | ~30s | Spec inspection, debugging |
| `property-tests` | Hypothesis stateful tests | ~10s | Implementation validation |
| `stress-tests` | Concurrent stress tests | ~120s | Pre-release |
| `verify-mailbox` | verify-formal + property-tests | ~40s | Mailbox changes |
| `verify-all` | Complete verification suite | ~40s | CI, pre-release |
| `clean-extracted` | Remove extracted specs | Instant | Cleanup |

## Troubleshooting

### "TLC not found"

**Problem:** TLA+ tools not installed.

**Solution:**

```bash
make setup-tlaplus
```

Or manually:

```bash
brew install tlaplus  # macOS
```

### "No @formal_spec decorators found"

**Problem:** No embedded specs in codebase.

**Solution:**

- Add `@formal_spec` decorator to your class
- See `specs/FORMAL_VERIFICATION.md` for decorator usage

### "Model checking timeout"

**Problem:** State space too large.

**Solution:**

- Reduce constants in `@formal_spec` (e.g., `MaxMessages: 3` → `MaxMessages: 2`)
- Add state constraints via `constraint=` parameter
- Use `make verify-formal-fast` for syntax-only validation

## Integration with Existing Targets

The verification targets integrate with the existing workflow:

```
make verify-formal       → Extract + TLC model check (~30s)
make verify-formal-fast  → Extract only, no TLC (~1s)
make property-tests      → Hypothesis stateful tests
make verify-all          → Complete verification suite
```

## See Also

- `specs/FORMAL_VERIFICATION.md` - `@formal_spec` decorator documentation
- `specs/VERIFICATION.md` - RedisMailbox detailed verification spec
- `formal-tests/` - Test examples

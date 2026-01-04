# Makefile Updates for TLA+ Embedding

## Overview

This document shows the complete Makefile updates needed to support embedded
TLA+ specifications alongside the existing verification targets.

## New Targets

Add these targets to the Makefile after the existing `verify-mailbox` target:

```makefile
# =============================================================================
# Embedded TLA+ Specifications
# =============================================================================

.PHONY: extract-tla
extract-tla: ## Extract TLA+ specs from @formal_spec decorators
	@echo "Extracting embedded TLA+ specifications..."
	@uv run pytest --extract-tla -v
	@echo "✓ Specs extracted to specs/tla/extracted/"

.PHONY: check-tla
check-tla: ## Extract and model check embedded TLA+ specs
	@echo "Extracting and validating embedded TLA+ specifications..."
	@uv run pytest --check-tla -v
	@echo "✓ All embedded specs passed model checking"

.PHONY: check-tla-fast
check-tla-fast: ## Extract embedded specs without model checking
	@echo "Extracting embedded TLA+ specifications (no model checking)..."
	@uv run pytest --extract-tla -q
	@echo "✓ Specs extracted (skipped model checking)"

.PHONY: verify-embedded
verify-embedded: check-tla ## Alias for check-tla
	@echo "✓ Embedded formal verification complete"

.PHONY: verify-all
verify-all: tlaplus-check check-tla property-tests ## Run all formal verification
	@echo "✓ All formal verification passed:"
	@echo "  - Legacy TLA+ specs (specs/tla/*.tla)"
	@echo "  - Embedded TLA+ specs (@formal_spec decorators)"
	@echo "  - Property-based tests (Hypothesis)"

# =============================================================================
# Verification Helpers
# =============================================================================

.PHONY: compare-specs
compare-specs: ## Compare legacy and embedded TLA+ specs
	@echo "Comparing legacy and embedded specifications..."
	@pytest --extract-tla -q
	@echo ""
	@echo "Legacy spec vs Embedded spec:"
	@if [ -f specs/tla/RedisMailbox.tla ] && [ -f specs/tla/extracted/RedisMailbox.tla ]; then \
		diff -u specs/tla/RedisMailbox.tla specs/tla/extracted/RedisMailbox.tla || true; \
	else \
		echo "One or both spec files not found"; \
	fi

.PHONY: clean-extracted
clean-extracted: ## Remove extracted TLA+ specs
	@echo "Cleaning extracted TLA+ specs..."
	@rm -rf specs/tla/extracted/
	@echo "✓ Cleaned specs/tla/extracted/"
```

## Updated `.PHONY` Declaration

Update the first line of the Makefile to include the new targets:

```makefile
.PHONY: format check test lint ty pyright typecheck type-coverage bandit vulture deptry pip-audit markdown-check integration-tests redis-tests redis-standalone-tests redis-cluster-tests validate-integration-tests mutation-test mutation-check tlaplus-check tlaplus-check-exhaustive property-tests stress-tests verify-mailbox extract-tla check-tla check-tla-fast verify-embedded verify-all compare-specs clean-extracted setup setup-tlaplus setup-redis demo demo-podman demo-claude-agent sync-docs all clean
```

## Updated `check` Target

Modify the `check` target to optionally include embedded spec extraction:

```makefile
# Run all checks (the kitchen sink)
check: format-check lint typecheck type-coverage bandit deptry pip-audit test markdown-check
	@echo "All checks passed!"

# Optional: Run all checks including formal verification
check-all: check extract-tla
	@echo "All checks including embedded TLA+ extraction passed!"
```

## Updated `clean` Target

Add extraction cleanup to the `clean` target:

```makefile
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf .coverage htmlcov .pytest_cache .mypy_cache .ruff_cache .mutmut-cache
	@rm -rf build dist *.egg-info
	@rm -rf specs/tla/extracted/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleaned"
```

## Complete Makefile Diff

Here's the complete diff showing all changes:

```diff
--- a/Makefile
+++ b/Makefile
@@ -1,4 +1,4 @@
-.PHONY: format check test lint ty pyright typecheck type-coverage bandit vulture deptry pip-audit markdown-check integration-tests redis-tests redis-standalone-tests redis-cluster-tests validate-integration-tests mutation-test mutation-check tlaplus-check tlaplus-check-exhaustive property-tests stress-tests verify-mailbox setup setup-tlaplus setup-redis demo demo-podman demo-claude-agent sync-docs all clean
+.PHONY: format check test lint ty pyright typecheck type-coverage bandit vulture deptry pip-audit markdown-check integration-tests redis-tests redis-standalone-tests redis-cluster-tests validate-integration-tests mutation-test mutation-check tlaplus-check tlaplus-check-exhaustive property-tests stress-tests verify-mailbox extract-tla check-tla check-tla-fast verify-embedded verify-all compare-specs clean-extracted setup setup-tlaplus setup-redis demo demo-podman demo-claude-agent sync-docs all clean

 # Format code with ruff
 format:
@@ -150,6 +150,42 @@ verify-mailbox: tlaplus-check property-tests
 	@echo "All mailbox verification checks passed"

 # =============================================================================
+# Embedded TLA+ Specifications
+# =============================================================================
+
+extract-tla: ## Extract TLA+ specs from @formal_spec decorators
+	@echo "Extracting embedded TLA+ specifications..."
+	@uv run pytest --extract-tla -v
+	@echo "✓ Specs extracted to specs/tla/extracted/"
+
+check-tla: ## Extract and model check embedded TLA+ specs
+	@echo "Extracting and validating embedded TLA+ specifications..."
+	@uv run pytest --check-tla -v
+	@echo "✓ All embedded specs passed model checking"
+
+check-tla-fast: ## Extract embedded specs without model checking
+	@echo "Extracting embedded TLA+ specifications (no model checking)..."
+	@uv run pytest --extract-tla -q
+	@echo "✓ Specs extracted (skipped model checking)"
+
+verify-embedded: check-tla ## Alias for check-tla
+	@echo "✓ Embedded formal verification complete"
+
+verify-all: tlaplus-check check-tla property-tests ## Run all formal verification
+	@echo "✓ All formal verification passed:"
+	@echo "  - Legacy TLA+ specs (specs/tla/*.tla)"
+	@echo "  - Embedded TLA+ specs (@formal_spec decorators)"
+	@echo "  - Property-based tests (Hypothesis)"
+
+compare-specs: ## Compare legacy and embedded TLA+ specs
+	@echo "Comparing legacy and embedded specifications..."
+	@pytest --extract-tla -q
+	@if [ -f specs/tla/RedisMailbox.tla ] && [ -f specs/tla/extracted/RedisMailbox.tla ]; then \
+		diff -u specs/tla/RedisMailbox.tla specs/tla/extracted/RedisMailbox.tla || true; \
+	fi
+
+# =============================================================================
 # Demos
 # =============================================================================

@@ -180,6 +216,7 @@ clean:
 	@echo "Cleaning build artifacts..."
 	@rm -rf .coverage htmlcov .pytest_cache .mypy_cache .ruff_cache .mutmut-cache
 	@rm -rf build dist *.egg-info
+	@rm -rf specs/tla/extracted/
 	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
 	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
 	@echo "✓ Cleaned"
```

## Usage Examples

### Development Workflow

```bash
# During development - extract specs to check syntax
make extract-tla

# Before commit - validate specs with model checker
make check-tla

# Quick extraction without waiting for TLC
make check-tla-fast

# Compare legacy spec with embedded version
make compare-specs

# Run all verification (legacy + embedded + property tests)
make verify-all

# Clean up extracted files
make clean-extracted
```

### CI Integration

For CI, you'll want to run both legacy and embedded verification:

```yaml
# .github/workflows/verify.yml

- name: Run formal verification
  run: |
    make verify-all
```

This ensures:

1. Legacy TLA+ specs in `specs/tla/*.tla` are checked
1. Embedded `@formal_spec` decorators are extracted and checked
1. Property-based tests validate implementation

### Migration Workflow

During migration from legacy to embedded specs:

```bash
# 1. Add @formal_spec decorator to Python class
vim src/weakincentives/contrib/mailbox/_redis.py

# 2. Extract the embedded spec
make extract-tla

# 3. Compare with legacy spec
make compare-specs

# 4. Iterate until they match semantically
# (edit decorator, extract, compare, repeat)

# 5. Validate with model checker
make check-tla

# 6. Once passing, archive legacy spec
mkdir -p specs/tla/archive
mv specs/tla/RedisMailbox.tla specs/tla/archive/
```

## Target Descriptions

| Target | Description | Speed | When to Use |
| ----------------- | ------------------------------ | ---------------- | ------------------------- |
| `extract-tla` | Extract specs without checking | Fast (1-2s) | Development, syntax check |
| `check-tla` | Extract and model check | Slow (5-60s) | Before commit, CI |
| `check-tla-fast` | Extract only (alias) | Fast (1-2s) | Quick validation |
| `verify-embedded` | Alias for check-tla | Slow (5-60s) | Semantic clarity |
| `verify-all` | All verification methods | Very slow (1-5m) | CI, pre-release |
| `compare-specs` | Diff legacy vs embedded | Fast (1-2s) | Migration validation |
| `clean-extracted` | Remove extracted files | Instant | Cleanup |

## Incremental Adoption

You can adopt embedded specs incrementally:

### Phase 1: Add extraction only

```makefile
# Just extract, don't check
check: format-check lint typecheck test extract-tla
```

### Phase 2: Add optional checking

```makefile
# Check on demand
check: format-check lint typecheck test
check-strict: check check-tla
```

### Phase 3: Make checking mandatory

```makefile
# Always check (once migration complete)
check: format-check lint typecheck test check-tla
```

## Troubleshooting

### "pytest: command not found"

**Problem:** Pytest not installed.

**Solution:**

```bash
uv sync --all-extras
```

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

**Problem:** No embedded specs in codebase yet.

**Solution:**

- This is expected during migration
- Add `@formal_spec` decorator following the migration guide
- Or skip with: `make check || true` (non-fatal)

### "Model checking timeout"

**Problem:** State space too large.

**Solution:**

- Reduce constants in `@formal_spec` (e.g., `MaxMessages: 3` → `MaxMessages: 2`)
- Add state constraints
- Use `make extract-tla` for syntax-only validation

## Integration with Existing Targets

The new targets integrate cleanly with existing verification:

```
Existing:
  make tlaplus-check       → Check specs/tla/*.tla
  make property-tests      → Run Hypothesis tests
  make verify-mailbox      → Both of the above

New:
  make extract-tla         → Extract @formal_spec
  make check-tla           → Extract + TLC check
  make verify-all          → All three methods

Migration:
  make compare-specs       → Diff legacy vs embedded
```

## Performance Optimization

For large codebases, extraction can be slow. Optimize with:

```makefile
# Only extract from specific paths
extract-tla-mailbox:
	@uv run pytest --extract-tla src/weakincentives/contrib/mailbox/ -v

# Parallel extraction (if multiple modules)
extract-tla-parallel:
	@uv run pytest --extract-tla -n auto -v
```

## Summary

After applying these changes:

1. ✅ `make extract-tla` - Extract specs during development
1. ✅ `make check-tla` - Validate specs before commit
1. ✅ `make verify-all` - Complete verification suite
1. ✅ `make compare-specs` - Aid migration from legacy
1. ✅ Incremental adoption path (extract → check → require)
1. ✅ CI integration with `make verify-all`

See `specs/FORMAL_VERIFICATION.md` for step-by-step migration instructions.

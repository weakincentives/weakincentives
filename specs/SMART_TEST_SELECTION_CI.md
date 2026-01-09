# Smart Test Selection - CI Integration Guide

## Quick Reference

### Workflow Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Branch Push                          │
│                                                              │
│  1. Run full test suite with coverage context               │
│  2. Build .coverage-cache/ database                          │
│  3. Upload to GitHub Actions cache                           │
│     - Key: coverage-cache-{sha}                             │
│     - Key: coverage-cache-main-latest                       │
└──────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Pull Request                              │
│                                                              │
│  1. Restore coverage cache from main                         │
│  2. If cache available:                                      │
│     → Run smart test selection                               │
│     → Only test files affected by PR changes                 │
│  3. If cache unavailable or smart tests fail:                │
│     → Fall back to targeted tests (by module)                │
│     → Or full test suite (for core changes)                  │
└──────────────────────────────────────────────────────────────┘
```

## Decision Tree

```
PR opened/updated
    │
    ├─→ Core/CI files changed?
    │   └─→ YES → Run full test suite ✓
    │
    ├─→ Coverage cache available?
    │   ├─→ YES → Try smart test selection
    │   │         ├─→ Success → Done ✓
    │   │         └─→ Failed → Run full test suite ✓
    │   │
    │   └─→ NO → Run targeted tests by module ✓
    │
    └─→ All paths run core tests first (smoke tests)
```

## Cache Management

### Cache Keys

**Primary key**: `coverage-cache-main-latest`
- Updated on every main branch push
- Used by all PRs for smart test selection
- Always contains the most recent coverage data

**Commit-specific key**: `coverage-cache-{sha}`
- Allows restoring exact historical coverage
- Useful for debugging or bisecting
- Not typically used in normal workflow

### Cache Lifecycle

1. **Creation**: Main branch push → `build-coverage-cache` job
2. **Storage**: GitHub Actions cache (7-day retention, 10GB limit)
3. **Restoration**: PR → `restore-cache` step
4. **Invalidation**: Automatic after 7 days or manual cache deletion

### Cache Size

- Typical size: 5-20 MB (compressed)
- Contains: SQLite database with test-to-code mappings
- Minimal impact on CI/CD bandwidth

## Performance Impact

### Main Branch Pushes

**Before smart test selection:**
- Run full test suite: ~60s

**After smart test selection:**
- Run full test suite: ~60s (same)
- Build coverage cache: +10-15s (context tracking overhead)
- **Total**: ~75s (+25% on main branch)

### Pull Requests

**Before smart test selection:**
- Run targeted tests: ~30-60s (depends on change scope)

**After smart test selection:**
- Restore cache: ~5s
- Smart test selection: ~5-20s (5-50 tests)
- **Total**: ~10-25s (50-80% faster for typical PRs)

**Net benefit**: Saves 30-40s per PR, worth the 15s overhead on main.

## Monitoring and Metrics

### Key Metrics to Track

1. **Cache hit rate**: % of PRs that successfully use smart selection
2. **Test reduction**: Average # of tests run (smart vs full)
3. **Time savings**: PR test time before/after
4. **False negative rate**: PRs that should have run more tests

### GitHub Actions Insights

View cache performance in Actions:
- Cache hit/miss rates
- Cache download times
- Test execution times

## Troubleshooting

### Cache Not Found

**Symptoms:**
```
Restore coverage cache: Cache not found for key: coverage-cache-main-latest
```

**Solutions:**
1. Wait for next main branch push to build cache
2. Manually trigger cache build workflow
3. Fall back to targeted tests (automatic)

### Smart Test Selection Failed

**Symptoms:**
```
Error querying coverage database: ...
Running full test suite (fallback)
```

**Solutions:**
1. Check coverage.py version compatibility
2. Verify git history is available (fetch-depth: 0)
3. Review error logs for specific issues
4. Fallback to full tests happens automatically

### Cache Size Limit Exceeded

**Symptoms:**
```
Cache size exceeded: 10GB limit
```

**Solutions:**
1. Review cache retention policy
2. Delete old caches manually
3. Consider cache size optimization

## Migration Guide

### Enabling Smart Test Selection

Already enabled in `.github/workflows/ci.yml`! No action needed.

### Disabling Smart Test Selection

To temporarily disable:

1. **Option 1**: Comment out `build-coverage-cache` job
2. **Option 2**: Add condition to skip smart tests:
   ```yaml
   - name: Run smart test selection
     if: false  # Disable smart tests
   ```

### Customization

Adjust thresholds in `build/select_tests.py`:

```python
# Example: Only use smart selection for small changes
if len(changed_files) > 20:
    # Too many changes, run all tests
    return [], 2
```

## Best Practices

### Do's

✅ Let cache build automatically on main pushes
✅ Use smart selection for isolated feature PRs
✅ Monitor cache hit rates and test times
✅ Trust the fallback mechanisms

### Don'ts

❌ Don't manually build cache locally (use `make build-coverage-cache` for testing only)
❌ Don't skip core tests (always run as smoke tests)
❌ Don't disable fallbacks (safety is critical)
❌ Don't cache from non-main branches

## Advanced Usage

### Local Testing

Simulate CI behavior locally:

```bash
# Build cache as if on main branch
git checkout main
make build-coverage-cache

# Test as if on PR
git checkout feature/my-pr
BASE=main make test-smart
```

### Cache Debugging

Inspect cache contents:

```bash
# View cache metadata
cat .coverage-cache/metadata.json

# Query coverage database
sqlite3 .coverage-cache/.coverage "SELECT COUNT(*) FROM context"
```

### Performance Tuning

Optimize for your codebase:

```python
# build/select_tests.py

# Adjust path filtering
python_files = [
    f for f in changed_files
    if f.startswith("src/") and f.endswith(".py")
    # Add more filters as needed
]
```

## FAQ

**Q: How often is the cache rebuilt?**
A: Every main branch push automatically rebuilds and updates the cache.

**Q: What if the cache is stale?**
A: Smart selection falls back to full tests when uncertain. Cache is usually fresh (<1 day old).

**Q: Does this work with matrix builds?**
A: Yes, each matrix job can restore the same cache independently.

**Q: Can we use this for integration tests?**
A: Not recommended. Integration tests often have external dependencies that coverage tracking can't capture.

**Q: What's the cache retention period?**
A: GitHub Actions caches are retained for 7 days or until 10GB limit is reached.

## See Also

- [SMART_TEST_SELECTION.md](./SMART_TEST_SELECTION.md) - Full technical specification
- [TESTING.md](./TESTING.md) - Testing standards and best practices
- [CLAUDE.md](../CLAUDE.md) - Development workflow guide

## Summary

Smart test selection is **automatically enabled** in the CI workflow and provides:

- ⚡ **50-80% faster PR tests** for isolated changes
- 🛡️ **Safe fallbacks** when cache unavailable or uncertain
- 🔄 **Zero maintenance** - cache updates automatically
- 📊 **Transparent operation** - clear logs show what's running

No configuration needed - it just works! 🎉

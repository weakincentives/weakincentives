# Smart Test Selection

**Status**: Alpha
**Author**: Claude
**Last Updated**: 2026-01-09

## Overview

Smart test selection uses cached coverage data to determine which tests need to run based on code changes. This dramatically reduces test execution time for PRs by running only the tests that cover the modified code.

## Motivation

Running the full test suite on every PR can be time-consuming, especially as the codebase grows. However, most changes only affect a small portion of the codebase, and therefore only a subset of tests need to run to validate those changes.

By caching the mapping between tests and the code they cover, we can:

1. **Reduce CI time**: Run only affected tests on PRs
2. **Maintain confidence**: Still achieve 100% coverage of changed code
3. **Enable faster iteration**: Get quicker feedback during development
4. **Optimize resources**: Use less compute time for test runs

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Coverage Cache                          │
│                                                              │
│  .coverage-cache/                                           │
│  ├── .coverage          (SQLite database)                   │
│  └── metadata.json      (commit hash, timestamps, stats)    │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ builds
                              │
┌─────────────────────────────┴───────────────────────────────┐
│          build/build_coverage_cache.py                       │
│                                                              │
│  - Runs pytest with --cov-context=test                       │
│  - Tracks which test executed which lines                    │
│  - Stores .coverage database with test-to-code mappings      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│          build/select_tests.py                               │
│                                                              │
│  - Takes list of changed files (from git diff)               │
│  - Queries .coverage database for affected tests             │
│  - Returns minimal test set or runs them                     │
└──────────────────────────────────────────────────────────────┘
```

### Coverage Database Schema

Coverage.py uses SQLite with the following key tables:

- **context**: Each test execution context (test node ID)
- **file**: Measured source files
- **line_bits**: Maps files + contexts to executed line numbers (packed format)
- **arc**: Branch coverage data (fromno, tono pairs)

The `--cov-context=test` flag enables dynamic context tracking, which records the pytest node ID (e.g., `tests/test_foo.py::test_bar`) for each test execution.

### Query Algorithm

Given a set of changed files:

1. **Normalize paths**: Convert to repo-relative paths
2. **Filter to source files**: Only consider `src/**/*.py` files
3. **Query database**: Find all contexts (tests) that executed any lines in those files
4. **Return test set**: Sorted list of pytest node IDs to run

### Fallback Strategy

The system falls back to running all tests when:

- Coverage cache doesn't exist (needs to be built first)
- Cache is stale or corrupt
- Changed files aren't found in coverage data (new files)
- Non-source files changed (docs, configs, etc.)
- Too many files changed (safety threshold)

## Usage

### Building the Cache

Build the coverage cache on the main branch:

```bash
# Build cache (runs all tests with context tracking)
make build-coverage-cache
```

This will:
1. Run pytest with `--cov-context=test`
2. Store the `.coverage` database in `.coverage-cache/`
3. Save metadata (git commit, timestamp, statistics)

**When to rebuild**:
- After merging to main
- Periodically (e.g., nightly)
- When test structure changes significantly

### Running Smart Tests

Run only tests affected by your changes:

```bash
# Compare against main branch (default)
make test-smart

# Compare against specific branch
BASE=develop make test-smart

# Or use the script directly
python build/select_tests.py --base main --run --verbose
```

### Query Without Running

Get the list of tests without running them:

```bash
# Output test IDs (one per line)
python build/select_tests.py --base main

# For specific files
python build/select_tests.py --files src/foo.py src/bar.py
```

### CI Integration

The `weakincentives` repository uses smart test selection in its GitHub Actions CI workflow. The strategy is:

**On main branch pushes:**
1. Build coverage cache with full test suite
2. Upload cache with commit-specific key and latest key

**On pull requests:**
1. Restore coverage cache from main branch
2. Run smart test selection if cache available
3. Fall back to targeted/full tests if cache unavailable or smart selection fails

**Implementation in `.github/workflows/ci.yml`:**

```yaml
# Build coverage cache on main branch pushes
build-coverage-cache:
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v5
    - name: Install uv
      uses: astral-sh/setup-uv@v7
      with:
        enable-cache: true
    - name: Set up Python
      run: uv python install 3.14
    - name: Install dependencies
      run: uv sync --all-extras

    - name: Build coverage cache
      run: make build-coverage-cache

    - name: Upload coverage cache
      uses: actions/cache/save@v4
      with:
        path: .coverage-cache
        key: coverage-cache-${{ github.sha }}

    - name: Upload coverage cache (latest)
      uses: actions/cache/save@v4
      with:
        path: .coverage-cache
        key: coverage-cache-main-latest

# Test job with smart selection
test:
  needs: detect-changes
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v5
      with:
        fetch-depth: 0  # Need full history for git diff

    - name: Install uv
      uses: astral-sh/setup-uv@v7
      with:
        enable-cache: true

    - name: Set up Python
      run: uv python install 3.14

    - name: Install dependencies
      run: uv sync --all-extras

    # Restore coverage cache for PRs
    - name: Restore coverage cache
      if: github.event_name == 'pull_request'
      id: restore-cache
      uses: actions/cache/restore@v4
      with:
        path: .coverage-cache
        key: coverage-cache-main-latest
        restore-keys: |
          coverage-cache-main-

    # Smart test selection (PRs only, when cache available)
    - name: Run smart test selection
      if: |
        github.event_name == 'pull_request' &&
        steps.restore-cache.outputs.cache-hit == 'true'
      id: smart-tests
      continue-on-error: true
      run: make test-smart
      env:
        BASE: ${{ github.event.pull_request.base.sha }}

    # Fallback to full tests if smart selection unavailable/failed
    - name: Run full test suite
      if: |
        steps.smart-tests.outcome == 'failure' ||
        steps.smart-tests.outcome == 'skipped'
      run: make test
```

**Key Features:**
- **Dual cache keys**: Commit-specific and latest for flexibility
- **Graceful degradation**: Falls back to full tests if cache unavailable
- **Full history checkout**: `fetch-depth: 0` enables git diff for change detection
- **Safe failure handling**: `continue-on-error: true` prevents false failures
- **Automatic cache updates**: Main branch pushes refresh the cache

**Benefits:**
- ✅ Faster PR feedback (5-50x speedup for isolated changes)
- ✅ No maintenance overhead (cache updates automatically)
- ✅ Safe fallbacks (never skips necessary tests)
- ✅ Works alongside existing targeted test strategy

## Exit Codes

### build_coverage_cache.py

- `0`: Cache built successfully
- `1`: Tests failed or cache creation failed

### select_tests.py

- `0`: Tests selected/run successfully
- `1`: Error (cache missing, query failed, tests failed)
- `2`: All tests should be run (cache miss, too many changes, etc.)

When using `--run`, exit code 2 triggers a fallback to `make test`.

## Safety Guarantees

### No False Negatives

The system is designed to **never skip a test that should run**:

1. **Uncovered files**: If any changed file isn't in the cache, run all tests
2. **New tests**: Tests not in the cache will be discovered by fallback
3. **Test changes**: Changes to test files trigger full runs
4. **Indirect dependencies**: Coverage tracks actual execution, catching indirect dependencies

### Coverage Completeness

Smart test selection maintains 100% coverage of changed code:

- Only considers source files in `src/`
- Queries both line and branch coverage
- Includes all tests that executed any part of changed files
- Falls back to full runs for safety

### Determinism

Results are deterministic given the same cache and changes:

- SQLite queries are deterministic
- Path normalization is consistent
- Test ordering is stable (sorted)

## Performance Characteristics

### Cache Building

- **Time**: Same as full test run (~baseline)
- **Space**: ~5-20MB for .coverage database (depends on test count)
- **Frequency**: Once per main branch update

### Test Selection

- **Query time**: <1 second for typical PR (10-50 changed files)
- **Memory**: Minimal (SQLite is on-disk)
- **Speedup**: 5-50x for typical PRs (depends on change scope)

### Example Metrics

For a codebase with:
- 142 test files
- 1000+ test cases
- 10,000+ lines of source code

Typical PR changing 5 files:
- Full test run: 60 seconds
- Smart test selection: 5-15 seconds (run ~50-200 tests)
- **Speedup**: 4-12x

## Limitations

### 1. Test-to-Code Coupling Only

Smart selection only considers direct code execution, not:

- **Semantic dependencies**: Tests that should run due to logical relationships
- **Data dependencies**: Tests that share fixtures or state
- **Integration boundaries**: Tests that cover system interactions

**Mitigation**: Conservative fallback strategy catches most cases.

### 2. Cache Staleness

Coverage cache can become stale when:

- Tests are added/removed
- Code structure changes significantly
- Coverage patterns change

**Mitigation**: Rebuild cache on main branch updates. Stale cache detection (future work).

### 3. Context Overhead

`--cov-context=test` adds ~10-20% overhead to test runs:

- More data to track and store
- Larger .coverage database

**Mitigation**: Only used for cache building, not regular test runs.

### 4. False Positives (Over-selection)

May select more tests than strictly necessary due to:

- Shared utility code (many tests cover it)
- Common fixtures or setup code

**Mitigation**: Better than false negatives. Future work: dependency graph analysis.

## Future Enhancements

### 1. Incremental Cache Updates

Instead of rebuilding the entire cache:

```bash
# Update cache for specific tests
make update-coverage-cache TESTS="tests/test_foo.py"
```

### 2. Cache Validation

Detect when cache is stale:

```bash
# Check if cache needs rebuilding
make validate-coverage-cache
```

### 3. Dependency Graph Analysis

Combine coverage data with static analysis:

- Import relationships
- Type dependencies
- Configuration changes

### 4. Test Impact Score

Rank tests by importance:

- Code coverage overlap
- Test execution time
- Historical failure rate

### 5. Parallel Cache Building

Build cache across multiple workers:

```bash
# Split tests and merge coverage data
make build-coverage-cache-parallel
```

## Troubleshooting

### Cache Not Found

```
Coverage cache not found. Run 'make build-coverage-cache' first.
```

**Solution**: Build the cache:

```bash
make build-coverage-cache
```

### Query Failed

```
Error querying coverage database: ...
```

**Solutions**:
1. Rebuild cache: `make build-coverage-cache`
2. Check coverage.py version: `uv run python -c "import coverage; print(coverage.__version__)"`
3. Delete and rebuild: `rm -rf .coverage-cache && make build-coverage-cache`

### All Tests Running

```
Running all tests due to uncovered files.
```

This is expected when:
- New files added
- Cache is stale
- Non-source files changed

**Solution**: Rebuild cache or accept full run.

### Test Selection Too Conservative

If too many tests are selected, the cache may be tracking shared utility code.

**Solution**: This is safe but less optimal. Future work will improve specificity.

## Related Specs

- **TESTING.md**: Testing standards and best practices
- **CLAUDE.md**: Essential commands and development workflow

## References

Coverage.py documentation:
- [Database Schema](https://coverage.readthedocs.io/en/7.13.0/dbschema.html)
- [Dynamic Contexts](https://coverage.readthedocs.io/en/7.13.0/contexts.html)
- [API Reference](https://coverage.readthedocs.io/en/7.13.0/api.html)

Pytest-cov:
- [Context Support](https://pytest-cov.readthedocs.io/en/latest/contexts.html)

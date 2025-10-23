# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview
**weakincentives** is a Python library for developing and optimizing side effect free background agents. This is an early-stage project with minimal code currently implemented.

## Development Environment

### Prerequisites
- **Python 3.14** (specified in `.python-version`)
- **uv** package manager (used for all commands)

### Setup
Install dependencies:
```bash
uv sync
```

## Common Commands

### Testing
```bash
# Run all tests
make test
# or directly:
uv run pytest

# Run a single test file
uv run pytest tests/test_example.py

# Run a specific test function
uv run pytest tests/test_example.py::test_example
```

### Linting and Formatting
```bash
# Check code formatting (no changes)
make format-check

# Auto-format code with ruff
make format

# Run linter
make lint

# Run linter with auto-fixes
make lint-fix
```

### Type Checking
```bash
# Run type checker (ty)
make typecheck
```

### Running All Checks
```bash
# Run format check, lint, typecheck, and tests
make check

# Run format, lint-fix, typecheck, and tests
make all
```

### Cleaning
```bash
# Remove cache files
make clean
```

## Code Structure

```
src/weakincentives/    # Main package source code
  __init__.py          # Package entry point
  py.typed             # PEP 561 type marker file

tests/                 # Test suite
  test_example.py      # Example/placeholder test
```

## Technical Details

### Build System
- **Build backend**: hatchling
- **Package manager**: uv (not pip or poetry)

### Code Quality Tools
- **Linter**: ruff (target: Python 3.14, line length: 88)
- **Type checker**: ty (v0.0.1a23)
- **Test framework**: pytest (v8.4.2+)

### Type Checking
This package includes a `py.typed` marker file, indicating it supports PEP 561 type hints for external type checkers.

## Development Workflow
1. Make code changes in `src/weakincentives/`
2. Write tests in `tests/`
3. Run `make check` before committing to ensure all checks pass
4. Use `make format` and `make lint-fix` to auto-fix issues when possible

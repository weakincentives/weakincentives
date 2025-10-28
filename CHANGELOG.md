# Changelog

All notable changes to this project will be documented in this file.

## v0.2.0 - Unreleased

### Features

- Added a stdlib-only `weakincentives.serde` module with `parse`, `dump`, `clone`, and `schema` helpers for dataclass validation and JSON serialization.
- Introduced a complete prompt composition framework with a `Section` base class, `TextSection` rendering, dedicated
  exceptions, placeholder validation, conditional nesting, improved render ergonomics, and 100% coverage backed by new
  tests.
- Added prompt tool orchestration primitives: a spec-aligned `Tool` dataclass with shared dataclass support and prompt-
  level tool aggregation with duplicate detection.
- Shipped an optional OpenAI adapter behind the `openai` extra, including adapter tests and usage documentation.

### Tooling

- Hardened the automation pipeline by tightening `make check`, enforcing full coverage, quieting wrapper scripts, and
  expanding Ruff lint rule coverage.
- Added security and dependency auditing via Bandit, pip-audit, and deptry wrappers, wiring the new runners into the
  Makefile targets.

### Packaging

- Switched releases to Hatch VCS-derived versioning, removed the manual bump script, bumped the package version to
  0.2.0, and enriched `pyproject.toml` metadata and extras.

### Documentation

- Replaced `WARP.md` with a comprehensive `AGENTS.md` handbook, adding guiding principles, TDD workflow guidance, and
  prompt integration expectations.
- Captured prompt and tools contracts under `specs/`, and refreshed the README to emphasize agent-focused positioning
  and prompt ergonomics.

### Changed

- Simplified tool registration by removing `ToolsSection`; every `Section` now accepts a `tools` sequence.
- Allow prompts to reuse the same params dataclass across multiple sections while honoring per-section defaults.

### Removed

- Deleted the deprecated `plans/` directory now that prompt and tools specifications live under `specs/`.

## v0.1.0 - 2025-10-22

### Added

- Bootstrapped the `weakincentives` package with initial module scaffold, typing marker, and example pytest to keep
  packaging sane.
- Introduced Python 3.14 toolchain configuration, `uv` lockfile, and Makefile tasks covering formatting, linting,
  testing, and release workflows.
- Added git hook installer and guidance so `make check` runs before commits land.
- Established GitHub Actions CI pipeline with coverage enforced at 80% via `pytest-cov`.
- Provisioned automated PyPI release workflow, including dedicated environment configuration and metadata updates.

### Documentation

- Added project license, `.gitignore`, and README placeholder.
- Authored the initial agent operations guide (`WARP.md`) to capture contributor workflows.

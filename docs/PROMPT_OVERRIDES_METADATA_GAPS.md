# Prompt Overrides Store: Metadata Gap Evaluation

## Executive Summary

This document evaluates the metadata currently collected and stored in the
`LocalPromptOverridesStore` and identifies gaps that may limit observability,
auditability, and operational capabilities.

## Current Metadata Inventory

### Override File Level (persisted to JSON)

| Field | Type | Description |
|-------|------|-------------|
| `version` | `int` | Format version for migrations (currently `1`) |
| `ns` | `str` | Namespace grouping related prompts |
| `prompt_key` | `str` | Machine identifier for the prompt |
| `tag` | `str` | Override variant label (e.g., `latest`, `stable`) |
| `sections` | `dict` | Section path → SectionOverride mapping |
| `tools` | `dict` | Tool name → ToolOverride mapping |

### Per Section Override

| Field | Type | Description |
|-------|------|-------------|
| `expected_hash` | `HexDigest` | SHA-256 of original body template |
| `body` | `str` | Replacement template content |

### Per Tool Override

| Field | Type | Description |
|-------|------|-------------|
| `expected_contract_hash` | `HexDigest` | SHA-256 of description + schemas |
| `description` | `str \| None` | Optional replacement description |
| `param_descriptions` | `dict[str, str]` | Per-parameter description overrides |

### Inspection Metadata (computed at read time, not persisted)

| Field | Type | Description |
|-------|------|-------------|
| `path` | `Path` | Absolute path to override file |
| `relative_segments` | `tuple[str, ...]` | Path components relative to root |
| `modified_time` | `float` | File modification timestamp |
| `content_hash` | `HexDigest` | SHA-256 of entire file contents |
| `section_count` | `int` | Number of section overrides |
| `tool_count` | `int` | Number of tool overrides |

---

## Identified Metadata Gaps

### 1. Temporal Metadata (High Priority)

**Gap**: No creation or modification timestamps are persisted in the override
JSON file.

| Missing Field | Description | Use Case |
|---------------|-------------|----------|
| `created_at` | ISO 8601 timestamp when override was first created | Audit trail, lifecycle management |
| `updated_at` | ISO 8601 timestamp of last modification | Change tracking, staleness detection |

**Current Workaround**: `modified_time` is computed from filesystem stat during
inspection but relies on filesystem preservation and is not part of the stored
contract.

**Impact**: Cannot reliably track override age without filesystem metadata;
git history becomes the only source of truth.

---

### 2. Authorship & Provenance (High Priority)

**Gap**: No information about who or what created/modified an override.

| Missing Field | Description | Use Case |
|---------------|-------------|----------|
| `created_by` | Identifier of creator (user, service, CLI) | Attribution, audit |
| `updated_by` | Identifier of last modifier | Change tracking |
| `source` | Origin type: `manual`, `optimization-service`, `cli-seed` | Provenance classification |
| `source_prompt_commit` | Git commit SHA of prompt when override was derived | Reproducibility |

**Impact**: Cannot distinguish manually-authored overrides from
machine-generated ones; no way to trace optimization experiments back to their
origin.

---

### 3. Change History & Rationale (Medium Priority)

**Gap**: No changelog or reason for override creation/modification.

| Missing Field | Description | Use Case |
|---------------|-------------|----------|
| `reason` | Free-form explanation for the override | Documentation, review |
| `changelog` | Array of `{timestamp, author, description}` entries | Audit trail |
| `parent_tag` | Tag this override was derived from | Lineage tracking |

**Impact**: Reviewers cannot understand why an override exists without
consulting external documentation or git history.

---

### 4. Validation Status (Medium Priority)

**Gap**: Override staleness is computed at resolve time but not cached.

| Missing Field | Description | Use Case |
|---------------|-------------|----------|
| `last_validated_at` | Timestamp of last successful validation | Health monitoring |
| `validation_status` | `valid`, `stale`, `partial` | Quick status checks |
| `stale_sections` | List of section paths with hash mismatches | Targeted remediation |
| `stale_tools` | List of tool names with contract mismatches | Targeted remediation |

**Impact**: CLI tooling must re-derive descriptors and re-validate to determine
override health; no cached indicator of staleness.

---

### 5. Original Content Reference (Medium Priority)

**Gap**: Override files do not store the original content being replaced.

| Missing Field | Description | Use Case |
|---------------|-------------|----------|
| `original_body` (per section) | Original template before override | Diffing, rollback |
| `original_description` (per tool) | Original tool description | Comparison |

**Impact**: To compare current override with original, consumers must load the
live prompt and render it; no self-contained diff capability.

---

### 6. Experimentation Support (Lower Priority)

**Gap**: No structured fields for A/B testing or experiment tracking.

| Missing Field | Description | Use Case |
|---------------|-------------|----------|
| `experiment_id` | Identifier linking to an experiment | Experiment tracking |
| `variant` | Variant label (A, B, control) | Traffic splitting |
| `enabled` | Whether override is active | Feature flags |
| `effective_from` | Start of validity period | Scheduled rollouts |
| `effective_until` | End of validity period | Scheduled sunsetting |

**Impact**: Experiment metadata must be managed externally; no native support
for time-based or percentage-based rollouts.

---

### 7. Section-Level Granular Metadata (Lower Priority)

**Gap**: Section overrides lack individual metadata.

| Missing Field | Description | Use Case |
|---------------|-------------|----------|
| `section_updated_at` | When this specific section was last modified | Granular tracking |
| `section_author` | Who authored this section override | Attribution |
| `section_notes` | Notes about this section override | Documentation |

**Impact**: All sections share file-level metadata; cannot track individual
section changes without git diff analysis.

---

### 8. Usage Metrics (Lower Priority)

**Gap**: No runtime usage tracking.

| Missing Field | Description | Use Case |
|---------------|-------------|----------|
| `resolve_count` | Number of times override was resolved | Usage analytics |
| `last_resolved_at` | Last resolution timestamp | Staleness detection |

**Impact**: Cannot identify unused or rarely-used overrides for cleanup.

---

## Recommendations

### Immediate (Low Effort, High Value)

1. **Add `created_at` and `updated_at`** to the persisted JSON format. Update
   `FORMAT_VERSION` to `2` and implement a migration path.

2. **Add `source` field** with enum values like `manual`, `cli-seed`,
   `external-service` to distinguish origin.

### Short-Term (Medium Effort)

3. **Add `reason` field** for human-readable explanation of the override.

4. **Persist `stale_sections` and `stale_tools`** during resolve to cache
   validation results.

### Longer-Term (Higher Effort)

5. **Store original content** alongside overrides to enable self-contained
   diffing.

6. **Add experiment metadata** if A/B testing becomes a requirement.

7. **Implement changelog array** for full audit trail within the file.

---

## Migration Considerations

Any additions to the persisted format should:

1. Increment `FORMAT_VERSION` appropriately.
2. Treat new fields as optional with sensible defaults for backward
   compatibility.
3. Provide a migration utility to backfill metadata for existing files.
4. Update the spec document (`specs/PROMPT_OVERRIDES.md`) with the new schema.

---

## Conclusion

The current metadata model is minimal and functional for basic override
storage but lacks temporal, authorship, and validation caching metadata that
would improve operational visibility. The highest-value additions are
`created_at`, `updated_at`, and `source`, which require modest schema changes
and provide immediate benefits for audit and lifecycle management.

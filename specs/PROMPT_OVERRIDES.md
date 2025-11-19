# Prompt Overrides Specification

## Purpose

External optimization services need a stable contract for identifying prompts and
overriding their content without touching source files. This document consolidates
the descriptor schema, tool-aware hashing rules, override structures, and the
filesystem-backed store required to persist those overrides in local workflows.

## Scope

- Applies to every prompt that participates in the override ecosystem.
- Covers descriptor generation, override authoring, storage, and mutation
  behaviors.
- Omits remote synchronization, conflict resolution between machines, and
  templating semantics.

## Terminology

- **Namespace (`ns`)** – Required string that groups related prompts (for example
  `webapp/agents`). Namespaces partition override storage and lookup.
- **Prompt key** – Required machine identifier for every `Prompt` inside a
  namespace. Distinct from the optional human-readable `name`.
- **Section key** – Required machine identifier for every `Section`. Keys compose
  into ordered tuples (`SectionPath`) that uniquely identify a location inside a
  prompt.
- **Hash-aware section** – A section type (currently `MarkdownSection` and
  subclasses such as `ResponseFormatSection`) whose original body template
  participates in content hashing.
- **Content hash** – Deterministic SHA-256 digest of a hash-aware section’s
  original body template as committed in source control. Runtime overrides never
  change this value.
- **Lookup key** – The quadruple `(ns, prompt_key, section_path, expected_hash)`
  used to decide whether an override still applies.
- **Tag** – Free-form label (for example `latest`, `stable`, `experiment-a`) that
  callers use to request a specific family of overrides.

## Descriptor Contract

Descriptor generation walks the prompt tree and produces deterministic metadata
for both sections and tools.

```python
from dataclasses import dataclass, field

@dataclass(slots=True)
class SectionDescriptor:
    path: tuple[str, ...]
    content_hash: str

@dataclass(slots=True)
class ToolDescriptor:
    path: tuple[str, ...]
    name: str
    contract_hash: str

@dataclass(slots=True)
class ChapterDescriptor:
    key: str
    title: str
    description: str | None
    parent_path: tuple[str, ...]

@dataclass(slots=True)
class PromptDescriptor:
    ns: str
    key: str
    sections: list[SectionDescriptor]  # ordered depth-first
    tools: list[ToolDescriptor]
    chapters: list[ChapterDescriptor]
```

Rules:

1. Every `Prompt` constructor must accept `key: str` and `ns: str` (non-empty).
1. Every `Section` constructor must accept `key: str`. Keys remain stable even
   for sections excluded from hashing.
1. `SectionPath` is the tuple of section keys from root to target.
1. Only hash-aware sections appear in `sections`.
1. Compute `content_hash` solely from the section’s original body template text.
   Ignore defaults, enable predicates, tools, children, and runtime params.
1. The descriptor omits non-hash-aware sections but other APIs may expose them.
1. Hash-aware sections must expose their original body template string so the
   hashing utility can operate without rendering parameters.
1. Collect tool descriptors during the same depth-first traversal used for
   sections.
1. Compute a tool’s contract hash with SHA-256 helpers:
   - `description_hash = hash_text(tool.description)`
   - `params_schema_hash = hash_json(schema(tool.params_type, extra="forbid"))`
   - `result_schema_hash = hash_json(schema(tool.result_type, extra="ignore"))`
   - `contract_hash = hash_text("::".join((description_hash, params_schema_hash, result_schema_hash)))`
1. Record the originating section path for every tool.
1. Descriptor construction ignores runtime enablement predicates.
1. Chapters appear in descriptor order so adapters can audit which visibility
   boundaries were declared.
1. `PromptDescriptor.from_prompt(prompt: Prompt) -> PromptDescriptor` gathers
   both section and tool descriptors while computing the hashes.

## Override Model

Overrides define the replacement content for sections and optional edits for
prompt-exposed tools.

```python
@dataclass(slots=True)
class SectionOverride:
    expected_hash: str
    body: str

@dataclass(slots=True)
class ToolOverride:
    name: str
    expected_contract_hash: str
    description: str | None = None
    param_descriptions: dict[str, str] = field(default_factory=dict)

@dataclass(slots=True)
class PromptOverride:
    ns: str
    prompt_key: str
    tag: str
    sections: dict[tuple[str, ...], SectionOverride] = field(default_factory=dict)
    tool_overrides: dict[str, ToolOverride] = field(default_factory=dict)
```

Rules:

1. Each dict entry in `sections` represents the new body template to use when
   the lookup key matches. Section overrides always store the descriptor hash in
   `expected_hash` alongside the replacement body text.
1. Replacement templates may hash to any value after application; descriptors
   continue to publish the in-code hash.
1. Tool overrides are keyed by tool name. They apply only when the stored
   contract hash matches the descriptor value and allow callers to replace the
   model-facing description and optionally patch per-field parameter
   descriptions.
1. Overrides are valid only when `expected_hash` (sections) or
   `expected_contract_hash` (tools) matches the descriptor entry.
1. Callers render prompts with overrides via `Prompt.render`, which should:
   - Build the prompt descriptor and resolve overrides.
   - Substitute section bodies whose hashes match the descriptor.
   - Copy tools with replacement descriptions when provided, capture
     `param_descriptions` for adapters, and preserve original ordering.
   - Fall back to default bodies when overrides are missing or mismatched.
1. Override consumers return the same `RenderedPrompt` structure used by
   `Prompt.render`, optionally surfacing a map of field description patches keyed
   by tool name.
1. Tool names remain the identity; overrides cannot rename tools. Hash matching
   ensures overrides stop applying after code changes to descriptions or schemas.

## Storage Layout

`LocalPromptOverridesStore` provides a filesystem-backed implementation of
`PromptOverridesStore` so prompts can be customized without editing source
files. The store targets single-repository development workflows where overrides
live alongside the codebase and participate in version control.

### Project Root Detection

1. The store locates the project root automatically.
1. Root discovery precedence:
   1. If the caller provides `root_path`, resolve it to an absolute path and use
      it, skipping auto-detection.
   1. Otherwise, walk upward from the current working directory until a Git
      repository boundary is found.
      - Prefer `git rev-parse --show-toplevel` when Git is available.
      - Fall back to traversing parents until a `.git` directory or file is
        encountered. Treat the containing directory as the root.
   1. If traversal reaches the filesystem root without finding Git, raise
      `PromptOverridesError` with guidance to pass `root_path` explicitly.
1. The default overrides directory resolves to
   `<project-root>/.weakincentives/prompts/overrides` and is created lazily on
   first write.

### On-Disk Layout

```
.weakincentives/
  prompts/
    overrides/
      {ns_segments...}/
        {prompt_key}/
          {tag}.json
```

- Namespaces may include forward slashes to express hierarchy (for example
  `webapp/agents`). Split the namespace on `/` and create one directory per
  segment. Each segment must match the section key regex
  `^[a-z0-9][a-z0-9._-]{0,63}$`.
- `prompt_key` and `tag` are sanitized with the same regex. Invalid identifiers
  raise `PromptOverridesError` before filesystem access.
- Section overrides inside a file are keyed by the serialized section path
  joined with `/`, such as `system/intro` for `("system", "intro")`.
- Tool overrides are stored under a `tools` object keyed by tool name.

#### File Format

Each `{tag}.json` file stores a single `PromptOverride` payload:

```json
{
  "version": 1,
  "ns": "demo",
  "prompt_key": "welcome_prompt",
  "tag": "stable",
  "sections": {
    "system": {
      "expected_hash": "…",
      "body": "You are an enthusiastic assistant."
    }
  },
  "tools": {
    "search": {
      "expected_contract_hash": "…",
      "description": "Use the vector index.",
      "param_descriptions": {
        "query": "User provided keywords."
      }
    }
  }
}
```

- The `version` field enables future format migrations.
- `expected_hash` stores the section `content_hash`; tool entries capture the
  descriptor’s `contract_hash`.
- Section bodies are written verbatim; callers are responsible for formatting.

### Read Behavior

1. `resolve` loads the `{tag}.json` file for the descriptor; missing files return
   `None`.
1. Validate payloads before constructing `PromptOverride`:
   - `ns`, `prompt_key`, and `tag` must match the descriptor inputs.
   - Section overrides must include `expected_hash` matching the descriptor.
     Stale overrides are ignored with a log message while processing continues.
   - Tool overrides follow the same hash validation logic.
1. Return `None` when no valid sections or tools remain after filtering.

### Write Behavior

1. `upsert` receives a `PromptOverride` from callers and must:
   - Validate that `override.ns`, `override.prompt_key`, and `override.tag`
     match the descriptor.
   - Confirm each section path exists in `descriptor.sections` and its stored
     `expected_hash` matches the descriptor’s current hash.
   - Confirm each tool override references a known tool descriptor and that
     `expected_contract_hash` matches the descriptor.
1. Writes are atomic: persist to a temporary file within the target directory and
   replace the final `{tag}.json` via `os.replace`.
1. Directory creation is idempotent; intermediate folders are created with
   `exist_ok=True` semantics.
1. `seed_if_necessary` wraps `upsert` to bootstrap overrides:
   - Derive the descriptor from the provided `Prompt` to ensure hashes correspond
     to the concrete prompt body supplying the data.
   - Build a `PromptOverride` enumerating every section path with the actual
     section body text read from the `Prompt` instance.
   - Include every tool descriptor with the tool descriptions and parameter
     descriptions sourced from the `Prompt`, paired with the descriptor’s
     contract hash.
   - If an override already exists for `(ns, prompt_key, tag)`, return the
     persisted override unchanged without touching the filesystem. Future
     enhancements may add an `overwrite=True` flag if clobbering becomes
     desirable.

## Mutation APIs

`PromptOverridesStore` mediates both lookup and persistence.

```python
class PromptOverridesStore(Protocol):
    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None: ...

    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride: ...

    def delete(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> None: ...

    def seed_if_necessary(
        self,
        prompt: Prompt,
        *,
        tag: str = "latest",
    ) -> PromptOverride: ...
```

Responsibilities:

1. `resolve` matches overrides by `(ns, prompt_key, section_path, expected_hash)`
   and `(ns, prompt_key, tool_name)`, dropping entries whose hashes do not match.
   Returning `None` signals that no applicable overrides exist.
1. `upsert` replaces any existing override file for the `(ns, prompt_key, tag)`
   tuple and returns the validated payload written to disk.
1. `delete` removes the override file for the tag. Missing files should not raise
   errors.
1. Both methods raise `PromptOverridesError` when validation fails (hash
   mismatches, invalid identifiers, serialization issues).
1. `seed_if_necessary` materializes a pristine override snapshot so CLI tooling
   can surface it immediately.

## Operational Flow

1. **Bootstrap** – Enumerate descriptors for all prompts and publish them to the
   optimization service. The service stores overrides keyed by
   `(ns, prompt_key, section_path, expected_hash)` and tagged as desired.
1. **Runtime** – Call `prompt.render(..., overrides_store=store, tag=...)`. Overrides whose lookup key matches replace the in-code bodies; the
   rest are ignored. Tool overrides apply when their contract hashes match.
1. **Author workflow** – Developers edit prompt bodies in source control. Hash
   changes automatically invalidate stale overrides because the lookup keys no
   longer match.

## Error Handling and Logging

- Introduce `PromptOverridesError` (if not already present) for recoverable
  filesystem or validation issues.
- When encountering malformed JSON, raise `PromptOverridesError` with the
  original exception chain preserved.
- Log debug-level messages on reads (cache hits, filtered sections) to aid
  troubleshooting without polluting normal output.

## Testing Recommendations

- Unit tests covering:
  - Root discovery via explicit path, git command, and manual traversal.
  - Reading overrides with matching and mismatched hashes.
  - `upsert` validation failures (invalid keys, unknown sections, hash drift).
  - Atomic write behavior using temporary directories.
  - `delete` idempotence.
  - `seed_if_necessary` generating a complete override snapshot and refusing to
    clobber existing files, including a case where the prompt body differs from
    cached descriptor state to prove the method reads from the `Prompt`
    instance.
- Integration tests wiring the store into `Prompt.render` to
  ensure mutations appear during subsequent renders.

## Non-Goals

- Introducing new templating features or changing section rendering semantics.
- Persisting analytics or override history; external systems control storage and
  retention.
- Allowing tool renaming, handler swapping, schema/type changes, or runtime
  reordering.

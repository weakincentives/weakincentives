# Local Prompt Overrides Store Specification

## Purpose

`LocalPromptOverridesStore` provides a filesystem-backed implementation of the
`PromptOverridesStore` protocol so prompts can be customized without editing the
source prompt definitions. It targets single-repository development workflows
where overrides live alongside the codebase and participate in version control.

## Scope

- Define how the local store discovers its backing directory.
- Specify the on-disk layout and serialization format for overrides.
- Extend the `PromptOverridesStore` contract with mutation APIs required by
  optimizer tooling and seeding helpers for first-time overrides.
- Describe validation and error handling rules for both reads and writes.

Out of scope: remote synchronization, conflict resolution across machines, and
binary diff tooling for override files.

## Project Root Detection

1. The store must locate the project root automatically.
2. Root discovery should obey the following precedence order:
   1. If the caller provides an explicit `root_path` argument, use it (after
      resolving to an absolute path) and skip auto-detection.
   2. Otherwise, walk upward from the current working directory until a Git
      repository boundary is found. The detection MUST succeed when invoked from
      any path inside the repository.
      - Prefer `git rev-parse --show-toplevel` for speed when Git is available.
      - Fall back to manually traversing parents until a `.git` directory or
        file is encountered. Treat the directory containing `.git` as the root.
   3. If the traversal reaches the filesystem root without finding a Git
      repository, raise `PromptOverridesError` with guidance to pass `root_path`
      explicitly.
3. The default overrides directory resolves to
   `<project-root>/.weakincentives/prompts/overrides`.
4. The store must create the directory tree lazily on first write.

## On-Disk Layout

-```
.weakincentives/
  prompts/
    overrides/
      {ns_segments...}/
        {prompt_key}/
          {tag}.json
```

- `ns` MAY include forward slashes to express hierarchy (e.g. `webapp/agents`).
  Split the namespace on `/` and create one directory per segment. Each segment
  **must** match the section key regex (`^[a-z0-9][a-z0-9._-]{0,63}$`).
- `prompt_key` and `tag` are individually sanitized with the same regex. Invalid
  identifiers raise `PromptOverridesError` before any filesystem access.
- Section overrides within a file are keyed by the serialized section path joined
  with `/`, e.g. `system/intro` for `("system", "intro")`.
- Tool overrides are stored under a `tools` object keyed by tool name.

### File Format

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
- `expected_hash` captures either the section `content_hash` or the tool contract
  hash from the descriptor. Mismatches trigger validation failures during both
  reads and writes.
- `body` stores the override markdown for the section. Section bodies are written
  verbatim; callers are responsible for formatting.

## Protocol Extensions

Update `PromptOverridesStore` to support mutations. `Prompt` refers to the
runtime prompt object defined in `weakincentives.prompt.prompt`:

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

- `upsert` replaces any existing override for the `(ns, prompt_key, tag)` tuple.
  The returned value contains the data written to disk after validation.
- `delete` removes the override file for the tag. Missing files should not raise
  errors.
- Both methods raise `PromptOverridesError` when validation fails (e.g. hash
  mismatches, invalid identifiers, or serialization issues).
- `seed_if_necessary` materializes a pristine override by inspecting the concrete `Prompt`
  instance so the rendered section bodies and tool descriptions are captured,
  even though the descriptor only tracks hashes. The method returns the
  persisted override so CLI tooling can surface it to the user immediately.

Future optimizers will call `upsert` after computing new section bodies.

## Read Path Behavior

1. When `resolve` is invoked, load the `{tag}.json` file for the descriptor.
2. If the file is missing, return `None`.
3. Validate the payload before constructing `PromptOverride`:
   - `ns`, `prompt_key`, and `tag` must match the descriptor inputs.
   - Every section override must include `expected_hash` matching the current
     descriptor section hash. Stale overrides are ignored with a log message, and
     the store should continue processing other sections.
   - Tool overrides follow the same hash validation logic.
4. Return `None` if no valid sections or tools remain after filtering.

## Write Path Behavior

1. `upsert` receives a `PromptOverride` produced by callers. The store MUST:
   - Validate that `override.ns`, `override.prompt_key`, and `override.tag`
     match the descriptor.
   - Confirm that every section path exists in `descriptor.sections` and the
     stored `expected_hash` values match the descriptor's current hashes.
   - Confirm that each tool override references a known tool descriptor and that
     `expected_contract_hash` aligns with the descriptor's hash.
2. Writes are atomic: persist to a temporary file within the target directory and
   `os.replace` the final `{tag}.json`.
3. Directory creation should be idempotent; intermediate folders are created with
   `exist_ok=True` semantics.
4. `seed_if_necessary` wraps `upsert` to bootstrap overrides:
   - Derive the `PromptDescriptor` from the provided `Prompt` (e.g. via
     `PromptDescriptor.from_prompt(prompt)`), ensuring the hashes correspond to
     the concrete prompt body that supplied the data.
   - Build a `PromptOverride` that enumerates every section path exposed by the
     descriptor, copying the actual section body text read from the `Prompt`
     instance alongside the descriptor's hash into each entry.
   - Include every tool descriptor with the tool descriptions and parameter
     descriptions sourced from the `Prompt`, paired with the descriptor's
     contract hash.
   - If an override already exists for the `(ns, prompt_key, tag)` tuple,
     return the persisted override unchanged without touching the filesystem.
     This makes the "seed" semantics explicit—no data is overwritten unless the
     caller performs a subsequent `upsert`. Future enhancements may add an
     `overwrite=True` flag if clobbering becomes desirable.

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
  - `seed_if_necessary` generating a complete override snapshot and refusing to clobber
    existing files, including a case where the prompt body differs from cached
    descriptor state to prove the method reads from the `Prompt` instance.
- Integration test wiring the store into `Prompt.render_with_overrides` to ensure
  mutations appear during subsequent renders.

## Migration Notes

- Update existing in-memory stores or test doubles to implement the new protocol
  methods. Provide default no-op implementations where necessary to keep tests
  simple.
- Document the default path and usage in developer-facing docs once the store is
  implemented.

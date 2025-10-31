# Tool-Aware Prompt Versioning

## Purpose

Extend prompt versioning so tools become describable, hashable artifacts. Optimizers can safely override the textual descriptions that shape a tool's presentation without touching code. Parameter schemas remain immutable but can receive optional description patches for provider adapters.

## Scope

- Applies to prompts that expose tools through sections.
- Builds on the existing prompt descriptor and override store.
- Does not allow tool renaming, handler swapping, schema/type changes, or runtime reordering.

## Descriptor Extensions

Augment the prompt descriptor produced today:

```python
@dataclass(slots=True)
class ToolDescriptor:
    path: tuple[str, ...]
    name: str
    contract_hash: str

@dataclass(slots=True)
class PromptDescriptor:
    key: str
    sections: list[SectionDescriptor]
    tools: list[ToolDescriptor]
```

Descriptor rules:

1. Collect tools during the same depth-first traversal used for sections.
1. Compute the tool's contract hash with SHA-256 helpers:
   - `description_hash = hash_text(tool.description)`
   - `params_schema_hash = hash_json(schema(tool.params_type, extra="forbid"))`
   - `result_schema_hash = hash_json(schema(tool.result_type, extra="ignore"))`
   - `contract_hash = hash_text("::".join((description_hash, params_schema_hash, result_schema_hash)))`
1. Record the originating section path.
1. Descriptor construction ignores runtime enablement predicates.

## Override Model

Extend overrides with tool entries:

```python
@dataclass(slots=True)
class ToolOverride:
    name: str
    expected_contract_hash: str
    description: str | None = None
    param_descriptions: dict[str, str] = field(default_factory=dict)

@dataclass(slots=True)
class PromptOverride:
    prompt_key: str
    tag: str
    overrides: dict[tuple[str, ...], str]
    tool_overrides: dict[str, ToolOverride] = field(default_factory=dict)
```

Override rules:

1. Overrides are keyed by tool name inside `tool_overrides`.
1. Apply an override only when `expected_contract_hash` matches the descriptor entry.
1. `description` replaces the model-facing description without mutating the original tool object.
1. `param_descriptions` map schema property names to replacement descriptions; adapters may opt in later.

## Store Resolution

`PromptVersionStore.resolve` already receives a descriptor and tag. Update implementations to:

1. Match tool overrides by `(prompt_key, tool_name)`.
1. Drop overrides whose contract hash does not match.
1. Return `None` when no section or tool overrides survive filtering.

Stores that ignore `tool_overrides` remain valid.

## Rendering Behavior

`Prompt.render_with_overrides` gains tool handling:

1. Build the prompt descriptor (now with tools) and resolve overrides.
1. For each collected tool:
   - Copy it with a replacement description when provided.
   - Capture `param_descriptions` for adapters (optional surface).
1. Return a `RenderedPrompt` whose `.tools` reflect the original order while carrying any re-described tools, plus an optional map of field description patches keyed by tool name.

Adapters that already respect tool order and description require no changes for v1. Adapter support for field description patches is optional and may ship later.

## Safety Guarantees

- Tool names remain the identity; overrides cannot rename tools.
- Hash matching ensures overrides stop applying after code changes to descriptions or schemas.
- Tool handlers and type definitions remain code-driven and immutable at runtime.
- Param/result schemas continue to use canonical JSON for hashing to maintain determinism across environments.

## Migration Notes

1. Implement the new dataclasses and hashing helpers alongside the existing descriptor code.
1. Extend descriptor construction to populate `tools`.
1. Update the in-repo version store to persist `tool_overrides`.
1. Update prompt rendering to apply tool overrides as described above.
1. (Optional) Surface `_tool_param_descriptions` on `RenderedPrompt` and teach adapters to merge patches into provider schemas.

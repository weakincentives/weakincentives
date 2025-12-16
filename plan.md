# Progressive Disclosure: Markdown Context Files

## Overview

Expand progressive disclosure so that when `open_sections` is called for sections
that **don't expose additional tools** (content-only sections), instead of raising
`VisibilityExpansionRequired`, the handler writes the section content to a
`context/` folder as markdown files that the agent can read if needed.

## Current Behavior

```
Agent calls open_sections(["reference-docs"], reason="...")
    │
    ▼
Handler validates section keys
    │
    ▼
_raise_visibility_expansion() raises VisibilityExpansionRequired
    │
    ▼
Caller catches exception, broadcasts SetVisibilityOverride events
    │
    ▼
Prompt re-renders with expanded sections (including any new tools)
```

## Proposed Behavior

```
Agent calls open_sections(["reference-docs"], reason="...")
    │
    ▼
Handler validates section keys
    │
    ▼
For each section, check if it has tools (section.tools() or children with tools)
    │
    ├── Has tools → Collect for exception path
    │
    └── Content-only → Write to context/<section-key>.md via filesystem
    │
    ▼
If any sections have tools:
    → Raise VisibilityExpansionRequired (existing path)

If all sections are content-only:
    → Return ToolResult with file paths, success=True
```

## Key Components

### 1. Section Tool Detection

Add helper to determine if a section (including children) exposes tools:

```python
# progressive_disclosure.py

def _section_has_tools(
    section_path: SectionPath,
    registry: RegistrySnapshot,
) -> bool:
    """Check if a section or any of its children expose tools.

    Args:
        section_path: Path tuple identifying the section.
        registry: Registry snapshot to query.

    Returns:
        True if the section or any descendant has tools.
    """
    for node in registry.sections:
        # Check if this node is the target or a descendant
        if node.path == section_path or _is_descendant(node.path, section_path):
            if node.section.tools():
                return True
    return False


def _is_descendant(candidate: SectionPath, parent: SectionPath) -> bool:
    """Check if candidate path is a descendant of parent path."""
    return (
        len(candidate) > len(parent)
        and candidate[:len(parent)] == parent
    )
```

### 2. Context File Writer

Add helper to render and write section content:

```python
# progressive_disclosure.py

from weakincentives.contrib.tools.filesystem import Filesystem

CONTEXT_FOLDER = "context"


def _write_section_context(
    section_path: SectionPath,
    registry: RegistrySnapshot,
    param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass],
    filesystem: Filesystem,
) -> str:
    """Render section content and write to context folder.

    Args:
        section_path: Path tuple identifying the section.
        registry: Registry snapshot with section tree.
        param_lookup: Parameters for rendering.
        filesystem: Filesystem to write to.

    Returns:
        The file path where content was written.
    """
    # Find section node
    node = _find_section_node(section_path, registry)
    if node is None:
        raise PromptValidationError(
            f"Section '{'.'.join(section_path)}' not found."
        )

    # Resolve params for this section
    section_params = registry.resolve_section_params(node, dict(param_lookup))

    # Render section content with FULL visibility
    content = node.section.render(
        section_params,
        depth=0,  # Top-level heading for standalone file
        number="",  # No numbering for standalone file
        path=node.path,
        visibility=SectionVisibility.FULL,
    )

    # Build file path: context/<dot-notation-key>.md
    file_key = ".".join(section_path)
    file_path = f"{CONTEXT_FOLDER}/{file_key}.md"

    # Ensure context directory exists and write
    filesystem.mkdir(CONTEXT_FOLDER, parents=True, exist_ok=True)
    filesystem.write(file_path, content, mode="overwrite")

    return file_path
```

### 3. Result Dataclass for Content-Only Sections

```python
# progressive_disclosure.py

@dataclass(slots=True, frozen=True)
class OpenSectionsResult:
    """Result from opening content-only sections."""

    written_files: tuple[str, ...] = field(
        metadata={
            "description": (
                "Paths to context files written. Read these files to view "
                "the expanded section content."
            ),
        },
    )
```

### 4. Modified Handler Logic

Update `create_open_sections_handler` to branch based on tool presence:

```python
# progressive_disclosure.py

def create_open_sections_handler(
    *,
    registry: RegistrySnapshot,
    current_visibility: Mapping[SectionPath, SectionVisibility],
    param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass] | None = None,
) -> Tool[OpenSectionsParams, OpenSectionsResult | None]:
    """Create an open_sections tool bound to the current prompt state.

    Args:
        registry: The prompt's section registry snapshot.
        current_visibility: Current visibility state for all sections.
        param_lookup: Parameters for rendering sections to files.

    Returns:
        A Tool instance configured for progressive disclosure.
    """
    # Freeze param_lookup for closure
    frozen_params = dict(param_lookup or {})

    def handler(
        params: OpenSectionsParams, *, context: ToolContext
    ) -> ToolResult[OpenSectionsResult | None]:
        """Handle open_sections requests."""
        # Validate all section keys first
        requested_overrides = _validate_section_keys(
            params.section_keys,
            registry=registry,
            current_visibility=current_visibility,
        )

        # Partition sections: those with tools vs content-only
        sections_with_tools: dict[SectionPath, SectionVisibility] = {}
        content_only_sections: list[SectionPath] = []

        for path, visibility in requested_overrides.items():
            if _section_has_tools(path, registry):
                sections_with_tools[path] = visibility
            else:
                content_only_sections.append(path)

        # If any section has tools, we must raise for re-render
        # Include all sections in the exception to maintain consistency
        if sections_with_tools:
            _raise_visibility_expansion(
                params.section_keys,
                requested_overrides,  # All sections, not just those with tools
                params.reason,
            )

        # All sections are content-only: write to context files
        if context.filesystem is None:
            return ToolResult(
                message="Cannot write context files: no filesystem available.",
                value=None,
                success=False,
            )

        written_files: list[str] = []
        for path in content_only_sections:
            try:
                file_path = _write_section_context(
                    path,
                    registry,
                    frozen_params,
                    context.filesystem,
                )
                written_files.append(file_path)
            except Exception as e:
                return ToolResult(
                    message=f"Failed to write context for '{'.'.join(path)}': {e}",
                    value=None,
                    success=False,
                )

        # Return success with file paths
        files_list = ", ".join(written_files)
        return ToolResult(
            message=f"Section content written to: {files_list}. "
                    "Read these files to view the expanded content.",
            value=OpenSectionsResult(written_files=tuple(written_files)),
            success=True,
        )

    return Tool[OpenSectionsParams, OpenSectionsResult | None](
        name="open_sections",
        description="Expand summarized sections to view their full content.",
        handler=handler,
        accepts_overrides=False,
    )
```

### 5. Update Rendering to Pass Parameters

Update `PromptRenderer.render()` to pass `param_lookup` to the handler factory:

```python
# rendering.py (in the render method, where open_sections is created)

if has_summarized:
    current_visibility = compute_current_visibility(
        self._registry,
        param_lookup,
        session=session,
    )
    open_sections_tool = create_open_sections_handler(
        registry=self._registry,
        current_visibility=current_visibility,
        param_lookup=param_lookup,  # NEW: pass for rendering to files
    )
    collected_tools.append(...)
```

### 6. Update Summary Suffix Message

Modify `build_summary_suffix` to indicate the file-based option:

```python
# progressive_disclosure.py

def build_summary_suffix(
    section_key: str,
    child_keys: tuple[str, ...],
    has_tools: bool = False,
) -> str:
    """Build the instruction suffix appended to summarized sections.

    Args:
        section_key: The dot-notation key for the summarized section.
        child_keys: Keys of child sections revealed on expansion.
        has_tools: Whether this section or children have tools.

    Returns:
        Formatted instruction text for the model.
    """
    if has_tools:
        # Existing message for tool-bearing sections
        base_instruction = (
            f"[This section is summarized. To view full content and access "
            f'additional tools, call `open_sections` with key "{section_key}".]'
        )
    else:
        # New message for content-only sections
        base_instruction = (
            f"[This section is summarized. To view full content, call "
            f'`open_sections` with key "{section_key}". The content will be '
            f'written to context/{section_key}.md for you to read.]'
        )

    if child_keys:
        children_str = ", ".join(child_keys)
        if has_tools:
            base_instruction = (
                f"[This section is summarized. Call `open_sections` with key "
                f'"{section_key}" to view full content including subsections: '
                f'{children_str}. Additional tools may become available.]'
            )
        else:
            base_instruction = (
                f"[This section is summarized. Call `open_sections` with key "
                f'"{section_key}" to write content (including subsections: '
                f'{children_str}) to context/{section_key}.md.]'
            )

    return f"\n\n---\n{base_instruction}"
```

## File Changes Summary

| File | Changes |
|------|---------|
| `src/weakincentives/prompt/progressive_disclosure.py` | Add `OpenSectionsResult`, `_section_has_tools`, `_write_section_context`, update `create_open_sections_handler`, update `build_summary_suffix` |
| `src/weakincentives/prompt/rendering.py` | Pass `param_lookup` to `create_open_sections_handler` |
| `tests/prompts/test_progressive_disclosure.py` | Add tests for content-only sections, file writing |

## Implementation Order

1. **Add `_section_has_tools` helper** - Pure function, easy to test in isolation
2. **Add `OpenSectionsResult` dataclass** - Data structure for new return path
3. **Add `_write_section_context` helper** - Depends on registry, filesystem
4. **Update `create_open_sections_handler`** - Core logic change
5. **Update `build_summary_suffix`** - Message updates for better UX
6. **Update `rendering.py`** - Pass param_lookup through
7. **Update tests** - Cover all new paths

## Edge Cases

1. **Mixed request (some with tools, some without)**
   - If ANY section has tools, raise exception for ALL sections
   - This ensures consistent behavior and avoids partial state

2. **Filesystem unavailable**
   - Return `ToolResult(success=False)` with clear message
   - This can happen if no workspace section is configured

3. **Section rendering fails**
   - Return `ToolResult(success=False)` with error details
   - Don't write partial files

4. **Nested sections with varying tool status**
   - `_section_has_tools` checks entire subtree
   - Parent with no tools but child with tools → has_tools=True

5. **Re-opening already-written context files**
   - Use `mode="overwrite"` to replace existing files
   - Idempotent behavior

## Testing Strategy

### Unit Tests

```python
# test_progressive_disclosure.py

def test_section_has_tools_returns_true_for_section_with_tools():
    """Section with tools is detected."""
    ...

def test_section_has_tools_returns_true_for_nested_tools():
    """Section with child that has tools is detected."""
    ...

def test_section_has_tools_returns_false_for_content_only():
    """Content-only section is correctly identified."""
    ...

def test_write_section_context_creates_file():
    """Content is written to context folder."""
    ...

def test_handler_writes_files_for_content_only_sections():
    """Handler writes files and returns paths."""
    ...

def test_handler_raises_when_section_has_tools():
    """Handler raises exception when tools present."""
    ...

def test_handler_raises_for_mixed_request():
    """Handler raises for all sections if any has tools."""
    ...

def test_handler_fails_gracefully_without_filesystem():
    """Handler returns failure when filesystem unavailable."""
    ...
```

### Integration Tests

```python
# test_progressive_disclosure_integration.py

def test_content_only_section_writes_context_file():
    """Full flow: model opens content section, file is written."""
    # Setup prompt with content-only summarized section
    # Evaluate and verify context file exists
    ...

def test_tool_section_still_raises_exception():
    """Full flow: tool-bearing section triggers re-render."""
    # Setup prompt with tool-bearing summarized section
    # Verify VisibilityExpansionRequired is raised
    ...
```

## Compatibility Notes

- **Backward compatible**: Tool-bearing sections continue to use exception path
- **Filesystem dependency**: Content-only path requires `context.filesystem`
- **No session state changes**: Content-only path doesn't modify visibility overrides
- **Idempotent**: Re-opening writes same content (overwrite mode)

## Documentation Updates

1. Update `specs/PROMPTS.md` - Document new behavior
2. Update `build_summary_suffix` docstring
3. Add example in `code_reviewer_example.py` showing content-only sections

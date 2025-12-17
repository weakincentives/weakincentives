# VFS Context Refactor Specification

## Overview

Refactor VFS tool handlers to access the `Filesystem` instance via
`context.filesystem` instead of capturing it in a `FilesystemToolHandlers`
closure. This aligns with the pattern documented in `specs/FILESYSTEM.md` and
eliminates stateful handler classes.

## Current State

### Closure Pattern (`vfs.py:953-1105`)

```python
def _build_tools(*, filesystem: Filesystem, ...) -> tuple[Tool[...], ...]:
    handlers = FilesystemToolHandlers(filesystem=filesystem)
    return (
        Tool[...](handler=handlers.list_directory, ...),
        Tool[...](handler=handlers.read_file, ...),
        # ... 7 handlers total
    )

class FilesystemToolHandlers:
    def __init__(self, *, filesystem: Filesystem) -> None:
        self._fs = filesystem

    def list_directory(self, params: ListDirectoryParams, *, context: ToolContext) -> ...:
        # Uses self._fs, ignores context.filesystem
        ...
```

**Problems**:

1. Handler state captured via closure, not discoverable from signature
2. `context.filesystem` field unused despite being available
3. Testing requires instantiating `FilesystemToolHandlers` class
4. Inconsistent with `specs/FILESYSTEM.md` handler examples

## Target State

### Context-Based Handlers

```python
def _build_tools(*, accepts_overrides: bool) -> tuple[Tool[...], ...]:
    return (
        Tool[...](handler=list_directory_handler, ...),
        Tool[...](handler=read_file_handler, ...),
        # ... 7 handlers as module-level functions
    )

def list_directory_handler(
    params: ListDirectoryParams,
    *,
    context: ToolContext,
) -> ToolResult[tuple[FileInfo, ...]]:
    fs = _require_filesystem(context)
    # Use fs directly
    ...

def _require_filesystem(context: ToolContext) -> Filesystem:
    """Extract filesystem from context or raise validation error."""
    if context.filesystem is None:
        raise ToolValidationError("No filesystem available in this context.")
    return context.filesystem
```

## Scope

### Files to Modify

| File | Changes |
|------|---------|
| `src/weakincentives/contrib/tools/vfs.py` | Remove `FilesystemToolHandlers` class, convert 7 methods to module functions |
| `src/weakincentives/contrib/tools/vfs.py` | Update `_build_tools` to not instantiate handlers class |
| `src/weakincentives/contrib/tools/vfs.py` | Update `VfsSection` to pass filesystem via context, not closure |
| `tests/tools/test_vfs.py` | Update tests to provide `context.filesystem` |

### Handlers to Refactor

1. `list_directory` → `list_directory_handler`
2. `read_file` → `read_file_handler`
3. `write_file` → `write_file_handler`
4. `edit_file` → `edit_file_handler`
5. `glob` → `glob_handler`
6. `grep` → `grep_handler`
7. `remove` → `remove_handler`

## Implementation Details

### 1. Add Filesystem Extraction Helper

```python
def _require_filesystem(context: ToolContext) -> Filesystem:
    """Extract filesystem from context, raising if unavailable."""
    if context.filesystem is None:
        raise ToolValidationError(
            "Filesystem tools require a filesystem in the tool context."
        )
    return context.filesystem
```

### 2. Convert Handler Methods to Functions

Before:
```python
class FilesystemToolHandlers:
    def list_directory(
        self, params: ListDirectoryParams, *, context: ToolContext
    ) -> ToolResult[tuple[FileInfo, ...]]:
        del context
        path = _normalize_optional_path(params.path)
        # ... uses self._fs
```

After:
```python
def list_directory_handler(
    params: ListDirectoryParams,
    *,
    context: ToolContext,
) -> ToolResult[tuple[FileInfo, ...]]:
    fs = _require_filesystem(context)
    path = _normalize_optional_path(params.path)
    # ... uses fs
```

### 3. Update `_build_tools` Factory

Before:
```python
def _build_tools(
    *,
    filesystem: Filesystem,
    accepts_overrides: bool,
) -> tuple[Tool[...], ...]:
    handlers = FilesystemToolHandlers(filesystem=filesystem)
    return (Tool[...](handler=handlers.list_directory, ...), ...)
```

After:
```python
def _build_tools(
    *,
    accepts_overrides: bool,
) -> tuple[Tool[...], ...]:
    return (Tool[...](handler=list_directory_handler, ...), ...)
```

### 4. Update Section to Provide Filesystem via Context

The adapter must populate `context.filesystem` when invoking tools from a
`VfsSection`. This requires verifying the adapter's `_build_tool_context`
method includes the filesystem.

Check `src/weakincentives/adapters/shared.py` for context construction:

```python
def _build_tool_context(..., filesystem: Filesystem | None = None) -> ToolContext:
    return ToolContext(
        prompt=prompt,
        session=session,
        filesystem=filesystem,  # Must be passed
        ...
    )
```

### 5. Section Filesystem Exposure

`VfsSection` must expose its filesystem so adapters can include it in context:

```python
class VfsSection(MarkdownSection[VfsSectionParams]):
    @property
    def filesystem(self) -> Filesystem:
        return self._filesystem
```

Adapters then pass `section.filesystem` when building context for tools owned
by that section.

## Test Updates

### Before (closure-based)

```python
def test_list_directory() -> None:
    fs = InMemoryFilesystem()
    handlers = FilesystemToolHandlers(filesystem=fs)
    context = make_stub_context()
    result = handlers.list_directory(ListDirectoryParams(path="."), context=context)
    ...
```

### After (context-based)

```python
def test_list_directory() -> None:
    fs = InMemoryFilesystem()
    context = make_stub_context(filesystem=fs)
    result = list_directory_handler(ListDirectoryParams(path="."), context=context)
    ...
```

## Migration Checklist

- [ ] Add `_require_filesystem` helper function
- [ ] Convert `FilesystemToolHandlers.list_directory` to `list_directory_handler`
- [ ] Convert `FilesystemToolHandlers.read_file` to `read_file_handler`
- [ ] Convert `FilesystemToolHandlers.write_file` to `write_file_handler`
- [ ] Convert `FilesystemToolHandlers.edit_file` to `edit_file_handler`
- [ ] Convert `FilesystemToolHandlers.glob` to `glob_handler`
- [ ] Convert `FilesystemToolHandlers.grep` to `grep_handler`
- [ ] Convert `FilesystemToolHandlers.remove` to `remove_handler`
- [ ] Remove `FilesystemToolHandlers` class
- [ ] Update `_build_tools` to use module-level handlers
- [ ] Verify `VfsSection.filesystem` property exists
- [ ] Verify adapter passes filesystem to context
- [ ] Update all VFS tool tests
- [ ] Run `make check`

## Out of Scope

- **Planning tools**: Separate refactor (use `context.session`)
- **Podman tools**: Larger refactor, depends on VFS pattern
- **Asteval sandbox closures**: Different pattern (interpreter injection)
- **Claude SDK bridge**: Adapter pattern, keep as-is

## Risks

1. **Adapter integration**: Adapters must correctly populate `context.filesystem`
   for tools from workspace sections. Verify `_execute_tool` in adapters.

2. **Section discovery**: Adapter needs to know which section owns a tool to
   extract its filesystem. Current dispatch may not track this.

3. **Backward compatibility**: External code instantiating
   `FilesystemToolHandlers` directly will break. This is internal API, so
   acceptable.

## Success Criteria

1. All VFS handlers are module-level functions
2. `FilesystemToolHandlers` class is deleted
3. Handlers access filesystem via `context.filesystem`
4. `make check` passes
5. No behavior changes in tool execution

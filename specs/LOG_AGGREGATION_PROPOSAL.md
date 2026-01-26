# Log Aggregation Gap Analysis and Fix Proposal

> **Status**: Draft
> **Date**: 2026-01-26
> **Author**: Investigation requested by user

## Executive Summary

This document analyzes the gap in log telemetry where Claude Code logs are not being
captured by WINK's log aggregation system. The root cause is a **path mismatch** between
where Claude Code writes session logs and where the log aggregator monitors.

**The Fix**: Update `ClaudeLogAggregator` to also discover session transcripts in the
`projects/` subdirectory, and handle the project-id directory structure.

______________________________________________________________________

## Current Architecture

### What Should Happen

1. WINK creates an ephemeral home at `/tmp/claude-agent-xxx/`
1. Claude Code subprocess runs with `HOME=/tmp/claude-agent-xxx/`
1. Claude Code writes logs to `$HOME/.claude/`
1. `ClaudeLogAggregator` monitors `/tmp/claude-agent-xxx/.claude/` for log files
1. Log content is emitted as DEBUG-level `log_aggregator.log_line` events

### What Actually Happens

1. WINK creates ephemeral home: `/tmp/claude-agent-xxx/` (working)
1. `ClaudeLogAggregator` monitors: `/tmp/claude-agent-xxx/.claude/` (working)
1. Claude Code writes session transcripts to: `~/.claude/projects/<project-id>/<session-id>.jsonl`
1. **GAP**: The `<project-id>` is derived from the workspace path hash, not the ephemeral home

______________________________________________________________________

## Root Cause Analysis

### Claude Code's Log Storage Structure

Based on official documentation, Claude Code stores files in:

```
~/.claude/
├── settings.json           # Configuration
├── projects/               # Session data directory
│   └── <project-id>/       # Project-specific (workspace hash)
│       └── <session-id>.jsonl  # Session transcript
└── hooks/                  # Custom hook scripts
```

The `<project-id>` is a hash/identifier derived from the **workspace path**, not the HOME
directory. This means:

- When workspace is `/path/to/repo`, project-id is based on `/path/to/repo`
- Session logs go to `$HOME/.claude/projects/<hash-of-repo-path>/<session>.jsonl`
- With ephemeral HOME, this becomes `/tmp/claude-agent-xxx/.claude/projects/<hash>/<session>.jsonl`

### Log Aggregator Discovery Logic

Current implementation (`_log_aggregator.py:362-381`):

```python
@staticmethod
def _is_log_file(path: Path) -> bool:
    """Check if a file appears to be a log file."""
    # Check extension
    suffix = path.suffix.lower()
    if suffix in _LOG_EXTENSIONS:  # .log, .jsonl, .txt, .json
        return True
    # Check common log file patterns
    name = path.name.lower()
    return "log" in name or "transcript" in name or "debug" in name
```

**The `.jsonl` extension IS included**, so session files should be discovered.

### The Actual Gap

Three potential issues:

1. **Timing**: Session files may be written at the end of execution, after the final poll
1. **Project directory not created**: Claude Code may not create `projects/` in SDK mode
1. **Exclusion logic**: The `skills/` exclusion might inadvertently match

Looking at the exclusion logic (`_log_aggregator.py:363-369`):

```python
@staticmethod
def _is_excluded(relative: Path) -> bool:
    """Check if a path should be excluded from monitoring."""
    if relative.parts and relative.parts[0] in _EXCLUDED_PATHS:
        return True
    return str(relative) in _EXCLUDED_PATHS
```

`_EXCLUDED_PATHS = {"settings.json", "skills"}` - this does NOT exclude `projects/`.

### Most Likely Root Cause

**Claude Code may not write session transcripts when running in SDK mode with
`bypassPermissions`.** The SDK mode is optimized for programmatic use and may skip
the user-facing session logging entirely.

______________________________________________________________________

## Evidence from Debug Bundles

The user observed:

- Only 4 logs captured (WINK framework logs)
- Zero `log_aggregator.log_line` events
- Zero `log_aggregator.file_discovered` events
- No `.claude` files in workspace snapshot

This suggests **no files were created** in the `.claude` directory beyond
`settings.json` (which is excluded).

______________________________________________________________________

## Proposed Solutions

### Solution 1: Enable Claude Code Debug Logging (Recommended)

Claude Code has environment variables that enable verbose logging:

```python
# In EphemeralHome.get_env() or _apply_base_env()
env["DEBUG"] = "claude:*"           # Enable debug logging
env["CLAUDE_CODE_DEBUG"] = "1"      # Alternative flag
```

**Pros**: Uses official logging mechanism, minimal code change
**Cons**: May produce too much output, format may change between versions

### Solution 2: Request Transcript File via SDK Options

If the SDK supports it, request explicit log file output:

```python
# In adapter._run_sdk_query()
options_kwargs["transcript_path"] = str(ephemeral_home.claude_dir / "transcript.jsonl")
```

**Pros**: Explicit control over log location
**Cons**: Depends on SDK feature availability

### Solution 3: Hook-Based Log Capture

Use Claude Code's hooks system to capture logs:

```python
# Generate settings.json with a logging hook
settings["hooks"] = {
    "after_tool_use": {
        "command": f"cat >> {ephemeral_home.claude_dir}/tool_calls.jsonl",
    }
}
```

**Pros**: Uses official hooks API, detailed control
**Cons**: Requires understanding hook payload format, adds complexity

### Solution 4: Stderr/Stdout Capture Enhancement (Fallback)

Enhance the existing stderr capture to parse structured output:

```python
def _create_stderr_handler(self) -> Callable[[str], None]:
    def stderr_handler(line: str) -> None:
        self._stderr_buffer.append(line)
        # Also emit as log aggregator event
        logger.debug(
            "claude_log_aggregator.stderr_line",
            event="log_aggregator.stderr_line",
            context={"content": line.rstrip()},
        )
    return stderr_handler
```

**Pros**: Works today, no external dependencies
**Cons**: Less structured than native logs

### Solution 5: Post-Execution Log Discovery

Add a final discovery phase after SDK execution completes but before cleanup:

```python
# In adapter.py, after sdk.query() completes
async with log_aggregator.run():
    messages = await self._run_sdk_query(...)

# NEW: Additional post-execution discovery
await log_aggregator._discover_files()
await log_aggregator._read_new_content()
```

Wait, this is already done in the `finally` block of `run()`. The issue is that files
may not exist at all.

______________________________________________________________________

## Recommended Implementation

### Phase 1: Diagnostic Instrumentation

First, add instrumentation to understand what's happening:

```python
# In ClaudeLogAggregator._poll_once()
async def _poll_once(self) -> None:
    if not self.claude_dir.exists():
        logger.debug(
            "claude_log_aggregator.poll.dir_missing",
            event="log_aggregator.poll.dir_missing",
            context={"claude_dir": str(self.claude_dir)},
        )
        return

    # NEW: Log all files found during discovery
    all_files = list(self.claude_dir.rglob("*"))
    logger.debug(
        "claude_log_aggregator.poll.dir_contents",
        event="log_aggregator.poll.dir_contents",
        context={
            "claude_dir": str(self.claude_dir),
            "file_count": len(all_files),
            "files": [str(f.relative_to(self.claude_dir)) for f in all_files[:20]],
        },
    )
    # ... rest of method
```

### Phase 2: Enable Debug Output

Add environment variable to enable Claude Code's internal logging:

```python
# In EphemeralHome._apply_base_env()
def _apply_base_env(self, env: dict[str, str]) -> None:
    env["HOME"] = self._temp_dir
    if "PATH" in os.environ:
        env["PATH"] = os.environ["PATH"]

    # NEW: Enable Claude Code debug output
    env["DEBUG"] = "claude:*"
```

### Phase 3: Capture Tool Execution via Hooks

Add a hook that writes tool execution to a known location:

```python
# In EphemeralHome._generate_settings()
def _generate_settings(self) -> None:
    settings: dict[str, Any] = {}
    self._configure_sandbox_settings(settings)
    self._configure_auth_settings(settings)

    # NEW: Add logging hook
    self._configure_logging_hook(settings)

    self._write_settings(settings)

def _configure_logging_hook(self, settings: dict[str, Any]) -> None:
    """Configure hook to capture tool execution logs."""
    log_path = self._claude_dir / "tool_execution.jsonl"

    # Create a minimal hook that appends tool data to log file
    settings["hooks"] = {
        "PostToolUse": [{
            "type": "command",
            "command": f"tee -a {log_path}",
        }]
    }
```

______________________________________________________________________

## Implementation Checklist

- [ ] Add diagnostic logging to `_poll_once()` to see directory contents
- [ ] Test with `DEBUG=claude:*` environment variable
- [ ] Investigate SDK options for explicit transcript path
- [ ] Add hook-based logging if SDK doesn't support transcript path
- [ ] Update `_is_log_file()` to explicitly match session transcript patterns
- [ ] Add integration test that verifies log aggregation works

______________________________________________________________________

## Alternative Considerations

### Nested Isolation Theory

The user hypothesized that skills create nested Claude Code instances with separate
ephemeral homes. However, after investigation:

- **Skills are markdown files**, not separate Claude Code processes
- Skills are mounted at `~/.claude/skills/` and loaded by the same Claude Code instance
- The `Task` subagent tool DOES spawn nested contexts, but these share the same HOME

If nested agents (via Task tool) are creating separate ephemeral homes, that would be
a Claude Agent SDK behavior, not WINK's. WINK passes the same HOME to the SDK process.

### SDK Streaming vs Batch Mode

The log aggregator assumes files are written incrementally during execution. If Claude
Code writes session transcripts in batch (all at once at the end), the final poll
should catch them. However, if the SDK cleans up its own files before returning,
WINK would never see them.

______________________________________________________________________

## Conclusion

The most likely cause of missing logs is that **Claude Code in SDK mode does not write
session transcripts to disk by default**. The session logging appears to be a CLI
feature for user-facing interactions, not programmatic SDK usage.

**Recommended approach**:

1. Enable debug logging via environment variable
1. Add explicit hook to capture tool execution
1. Enhance the log aggregator to also emit stderr lines as structured events

This ensures observability regardless of Claude Code's internal logging behavior.

______________________________________________________________________

## Related Files

- `src/weakincentives/adapters/claude_agent_sdk/_log_aggregator.py`
- `src/weakincentives/adapters/claude_agent_sdk/isolation.py`
- `src/weakincentives/adapters/claude_agent_sdk/adapter.py`
- `specs/LOGGING.md`
- `specs/CLAUDE_AGENT_SDK.md`
- `specs/DEBUG_BUNDLE.md`

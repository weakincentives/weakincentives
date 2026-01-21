# WINK Agent Documentation Tools

Instructions for AI agents helping users develop and optimize WINK agents.

______________________________________________________________________

## wink docs

Access WINK documentation from the command line. Use this to find implementation
patterns, API details, and design guidance.

### Commands

```bash
# List available documents
wink docs list              # All documents
wink docs list specs        # Design specifications only
wink docs list guides       # Usage guides only

# Search documentation
wink docs search "pattern"           # Search all docs
wink docs search "pattern" --specs   # Search specs only
wink docs search "pattern" --guides  # Search guides only
wink docs search "pattern" --regex   # Regex search
wink docs search "pattern" --context 5  # More context lines

# Show table of contents
wink docs toc spec SESSIONS    # TOC for a spec
wink docs toc guide quickstart # TOC for a guide

# Read full documents
wink docs read reference       # API reference (this conceptual doc)
wink docs read changelog       # Release history
wink docs read example         # Complete code review agent example
wink docs read spec SESSIONS   # Read a specific spec
wink docs read guide quickstart # Read a specific guide
```

### When to Use

**Use `wink docs search`** when:

- User asks "how do I..." questions about WINK
- Looking for specific API patterns or examples
- Finding which spec covers a topic

**Use `wink docs read spec <NAME>`** when:

- Implementing a feature covered by that spec
- Understanding design constraints before modifying code
- Reviewing expected behavior during debugging

**Use `wink docs read guide <NAME>`** when:

- Walking user through a workflow
- Setting up a new agent from scratch
- Learning recommended patterns

### Key Specs

| Spec | When to Read |
|------|--------------|
| PROMPTS | Building or modifying prompts, sections, tools |
| SESSIONS | Working with state, events, reducers |
| TOOLS | Implementing custom tools, policies |
| ADAPTERS | Configuring providers (OpenAI, LiteLLM, Claude) |
| CLAUDE_AGENT_SDK | Using Claude Code native capabilities |
| WORKSPACE | File operations, sandboxing |
| FEEDBACK_PROVIDERS | Progress feedback for long tasks |
| TASK_COMPLETION | Verifying agents complete all work |

______________________________________________________________________

## wink query

Query debug bundles via SQL. Use this to investigate agent failures and
analyze execution traces.

### Commands

```bash
# Always start with schema to discover tables
wink query ./bundle.zip --schema

# Query with JSON output (default)
wink query ./bundle.zip "SELECT * FROM errors"

# Query with ASCII table output
wink query ./bundle.zip "SELECT * FROM errors" --table
```

### Available Tables

| Table | Contents |
|-------|----------|
| `manifest` | Bundle metadata (bundle_id, status, created_at) |
| `errors` | Aggregated errors from all sources |
| `logs` | Log entries from execution |
| `tool_calls` | Tool invocations with timing and results |
| `session_slices` | Session state items |
| `files` | Workspace files captured in bundle |
| `config` | Flattened configuration |
| `metrics` | Token usage and timing |

### Essential Queries

```sql
-- What went wrong?
SELECT error_type, message FROM errors

-- Which tools failed?
SELECT tool_name, error_code, params FROM tool_calls WHERE success = 0

-- Tool performance
SELECT tool_name, COUNT(*) as calls, AVG(duration_ms) as avg_ms
FROM tool_calls GROUP BY tool_name

-- Error logs
SELECT timestamp, message FROM logs WHERE level = 'ERROR'

-- Token usage
SELECT input_tokens, output_tokens, total_ms FROM metrics

-- Session state summary
SELECT slice_type, COUNT(*) FROM session_slices GROUP BY slice_type

-- Find content in files
SELECT path FROM files WHERE content LIKE '%pattern%'
```

### Debugging Workflow

1. **Start with schema**: `wink query bundle.zip --schema`
1. **Check for errors**: `SELECT * FROM errors`
1. **Review failed tools**: `SELECT * FROM tool_calls WHERE success = 0`
1. **Examine logs around failure**: `SELECT * FROM logs WHERE level IN ('ERROR', 'WARNING')`
1. **Inspect session state**: Query `session_slices` or typed `slice_*` tables

______________________________________________________________________

## Agent Development Workflow

When helping a user build or debug a WINK agent:

### 1. Understanding Requirements

```bash
# Find relevant specs
wink docs search "topic user mentioned"

# Read the appropriate spec
wink docs read spec RELEVANT_SPEC
```

### 2. Writing Code

- Follow patterns from `wink docs read example`
- Check spec constraints before implementing
- Use typed dataclasses with `slots=True, frozen=True`

### 3. Debugging Failures

```bash
# Load the debug bundle
wink query ./bundle.zip --schema

# Investigate errors
wink query ./bundle.zip "SELECT * FROM errors" --table
wink query ./bundle.zip "SELECT * FROM tool_calls WHERE success = 0" --table
```

### 4. Optimizing Performance

```bash
# Analyze tool usage
wink query ./bundle.zip "SELECT tool_name, COUNT(*), AVG(duration_ms) FROM tool_calls GROUP BY tool_name" --table

# Check token consumption
wink query ./bundle.zip "SELECT * FROM metrics" --table
```

______________________________________________________________________

## Quick Reference

```bash
# Documentation
wink docs list                    # See all available docs
wink docs search "keyword"        # Find relevant sections
wink docs read spec SPEC_NAME     # Read full spec
wink docs read guide GUIDE_NAME   # Read full guide

# Debugging
wink query bundle.zip --schema    # Discover tables
wink query bundle.zip "SQL"       # JSON output
wink query bundle.zip "SQL" --table  # Table output
```

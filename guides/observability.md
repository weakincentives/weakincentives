# Observability

This guide explains the mental model behind WINK's observability
story: how run context, tracing, transcripts, and logging fit together
to give you visibility into agent execution.

For concrete debugging tools and commands, see [Debugging](debugging.md).
This guide focuses on the *why* and the conceptual framework.

## Why Observability Is Hard for Agents

Traditional services handle discrete requests with predictable
lifetimes. An agent execution is different:

- A single user request may trigger dozens of model calls and tool
  invocations
- Tool calls have side effects that change the world the agent
  operates in
- Retries and dead-letter handling mean the same logical request can
  execute multiple times
- Sub-agents spawn their own conversation threads
- The interesting failures are semantic, not just crashes

Standard request tracing tells you *that* something happened.
Agent observability needs to tell you *why* the agent behaved the
way it did. This requires correlating across multiple layers:
structured logs, execution context, conversation transcripts, and
captured state.

## The Two IDs: run_id vs request_id

Every execution carries two identifiers that serve different
purposes:

| ID | Lifecycle | Question it answers |
|----|-----------|---------------------|
| `run_id` | Fresh per execution attempt | "Which specific execution am I looking at?" |
| `request_id` | Stable across retries | "Which logical request does this belong to?" |

When things work on the first try, both IDs point to the same
execution and the distinction does not matter. The distinction
becomes critical when retries enter the picture.

Consider a request that fails and gets retried twice:

```
request_id: abc-123
  run_id: run-001 (attempt 1, failed)
  run_id: run-002 (attempt 2, failed)
  run_id: run-003 (attempt 3, succeeded)
```

Filtering logs by `request_id` shows you the full history of a
logical request across all attempts. Filtering by `run_id` isolates
a single attempt. Use `request_id` when diagnosing "why did this
request take so long?" and `run_id` when diagnosing "what went
wrong in this specific attempt?"

The `attempt` field tells you which delivery this is (1 = first),
so you can distinguish retries from first attempts without
comparing timestamps.

## RunContext: The Correlation Backbone

`RunContext` is a frozen dataclass that flows from the AgentLoop
through adapters, into tool handlers, and onto every telemetry
event. It carries:

- `run_id` and `request_id` (as above)
- `session_id` for session correlation
- `worker_id` identifying which worker processed the request
- `trace_id` and `span_id` for distributed tracing integration

The key property is that RunContext is **immutable and ubiquitous**.
Once created at the start of an execution, the same context
propagates everywhere. This means every log line, every tool
invocation event, and every telemetry record can be correlated back
to the same execution without guesswork.

### How it flows

```
AgentLoop receives request
  -> builds RunContext (fresh run_id, preserves request_id)
  -> binds RunContext to logger (all logs get correlation IDs)
  -> passes RunContext to adapter
     -> adapter passes to InnerLoop
        -> passes to ToolExecutor
           -> available in tool handlers via ToolContext
     -> attached to PromptRendered, ToolInvoked, PromptExecuted events
  -> attached to AgentLoopResult
```

You do not need to manually thread RunContext through your code.
The harness does it. If you are writing a tool handler, access it
via `context.run_context`. If you are reading logs, filter by
`run_id` or `request_id`.

## Structured Logging

WINK uses structured logging with stable event names and typed
context payloads. Every log record carries:

- A `timestamp` in ISO-8601 UTC
- A severity `level` (DEBUG through CRITICAL)
- An `event` key following `module.action` naming
- A `context` dict with typed fields relevant to the event

When RunContext is bound to the logger (which happens automatically
during execution), every log record also carries `run_id`,
`request_id`, `attempt`, and `worker_id`. This means you can
correlate any log line back to its execution without parsing
timestamps or filenames.

The event taxonomy provides a stable vocabulary for filtering:

| Pattern | What it covers |
|---------|----------------|
| `tool.execution.*` | Tool invocation lifecycle |
| `session.*` | State mutations |
| `adapter.*`, `evaluate.*` | Provider communication |
| `transcript.collector.*` | Transcript discovery and entries |

See [Debugging](debugging.md) for how to configure log levels
and enable JSON output.

## Transcript Collection

When using the Claude Agent SDK adapter, WINK collects the SDK's
conversation transcripts in real time. This gives you the canonical
record of what the agent said, what tools it called natively, and
what it was thinking.

### The discovery model

Transcript collection is **hook-driven, not polling-based**. The
Claude Agent SDK fires hooks at key lifecycle points (prompt
submission, tool use, stop). The transcript collector registers
callbacks on these hooks and extracts the `transcript_path` from
the hook input data. Once it knows where the transcript file lives,
it starts tailing it.

This matters because it means:

1. Discovery is immediate -- no scanning directories hoping to find
   the right file
1. The collector knows exactly which files are transcripts vs other
   log output
1. Sub-agent transcripts are discovered by scanning a known
   directory derived from the main transcript path

### Background tailing

Once a transcript path is discovered, the collector tails it in a
background task. File I/O runs in a thread executor so it never
blocks the event loop. The tailer handles rotation (inode changes)
and truncation (compaction) gracefully by resetting its read
position.

Each JSONL line is parsed and emitted as a DEBUG-level structured
log event with context fields identifying the source (`main` or
`subagent:{id}`), entry type, and sequence number. Because these
are regular log events, they flow into debug bundles automatically
and are queryable via `wink query`.

### What transcripts capture

| Entry type | Content |
|------------|---------|
| `user` | User messages |
| `assistant` | Assistant responses, tool use |
| `tool_result` | Tool execution results |
| `thinking` | Extended thinking blocks |
| `summary` | Compaction summaries |
| `system` | System events |

Transcript collection is enabled by default in the Claude Agent
SDK adapter. Set `transcript_collection=None` in the client config
to disable it.

## How the Pieces Fit Together

The observability stack has four layers, each serving a different
need:

```
+------------------+--------------------------------------+
| Layer            | What it provides                     |
+------------------+--------------------------------------+
| RunContext       | Correlation IDs across all layers    |
| Structured Logs  | Event stream with typed context      |
| Transcripts      | Canonical conversation record        |
| Debug Bundles    | Self-contained execution snapshot    |
+------------------+--------------------------------------+
```

**RunContext** is the thread that ties everything else together.
Without consistent correlation IDs, logs from concurrent executions
would be impossible to separate.

**Structured logs** are the real-time event stream. They tell you
what happened, when, and in what context. Filtering by event name
and run_id gives you a focused view of any execution.

**Transcripts** are the conversation record. They tell you what
the model saw and said, including tool calls made through the SDK's
native runtime (which do not go through WINK's tool executor).
Transcripts are emitted as log events, so they are captured in
bundles and queryable via SQL.

**Debug bundles** are the post-mortem artifact. They package
everything -- logs, transcripts, session state, configuration,
filesystem snapshots -- into a single zip file. When you need to
understand what happened after the fact, the bundle is your
starting point.

### The typical investigation flow

1. Start with the debug bundle (`wink debug <bundle>`)
1. Check the error panel or error summary
1. Look at the tool timeline to see what the agent did
1. Check the transcript for the conversation flow
1. Query specific log events with `wink query` for details
1. Use `run_context.json` to find the run_id and request_id
   for cross-referencing with external systems

## OpenTelemetry Integration

RunContext includes `trace_id` and `span_id` fields for
integration with distributed tracing systems. If you are using
OpenTelemetry, extract the span context before creating the
request and pass it through:

```python nocheck
with tracer.start_as_current_span("agent.request") as span:
    ctx = trace.get_current_span().get_span_context()
    request = AgentLoopRequest(
        run_context=RunContext(
            trace_id=format(ctx.trace_id, "032x"),
            span_id=format(ctx.span_id, "016x"),
        ),
        # ...
    )
```

The trace context propagates through the entire execution and
appears in every log record. This lets you connect agent
executions to your broader service traces -- seeing the agent
run as a span within a larger request.

## Key Mental Models

**Everything correlates through RunContext.** Logs, events,
transcripts, and bundles all carry the same IDs. If you have a
run_id, you can find everything about that execution.

**Transcripts are the ground truth for conversation.** Logs tell
you about framework events. Transcripts tell you what the model
actually said and did. Both are useful; they serve different
questions.

**Bundles are the unit of investigation.** When something goes
wrong, you do not grep through log files. You open a bundle that
contains everything, already correlated and queryable.

**Observability is not debugging.** Debugging is reactive: something
broke, you investigate. Observability is the infrastructure that
makes debugging possible. Good observability means that when
something does break, the information you need is already captured,
correlated, and accessible.

## Next Steps

- [Debugging](debugging.md): Tools and commands for investigating
  agent behavior
- [Query](query.md): SQL-based analysis of debug bundles
- [Sessions](sessions.md): Understanding session state structure

# Troubleshooting

Common issues you'll hit when getting started, and how to fix them.

## "PromptValidationError: placeholder not found"

Your template uses `${foo}` but no bound dataclass has a `foo` field.

**Fix**: Check that your dataclass field names match placeholder names exactly.
Placeholders are case-sensitive.

```python
# Wrong: placeholder is ${query}, field is question
@dataclass
class Params:
    question: str  # Should be 'query'

# Right
@dataclass
class Params:
    query: str
```

## "Tool handler returned None"

Tool handlers must return `ToolResult`, not `None`.

**Fix**: Always return a `ToolResult`, even for failures:

```python
def handler(params, *, context):
    if something_wrong:
        return ToolResult.error("Failed")
    return ToolResult.ok(result, message="OK")
```

## "OutputParseError: missing required field"

The model's JSON response doesn't match your output dataclass.

**Fix**: Check that your dataclass fields match what the model returns. Add
clear instructions in your prompt about the expected JSON structure. Use
`allow_extra_keys=True` on the template if you want to ignore extra fields.

```python
# Be explicit in your prompt about the expected structure
template = "Return JSON with these exact fields: summary (string), items (list of strings)"
```

## Model Doesn't Call Tools

The model sees tools but chooses not to use them.

**Fixes**:

1. Make instructions clearer: "Use the search tool to find information before
   answering"
2. Add tool examples to show correct usage
3. Check that the tool description accurately describes what it does
4. Verify the tool is attached to an enabled section

```python
# Add explicit instruction
template = (
    "You MUST use the search tool to find relevant information "
    "before providing your answer."
)
```

## "DeadlineExceededError"

The agent ran past its deadline.

**Fixes**:

1. Increase the deadline
2. Reduce prompt size (use progressive disclosure)
3. Check for tool handlers that hang or take too long
4. Add timeouts to external API calls in tool handlers

## Session State Not Persisting

State changes aren't visible in subsequent queries.

**Fix**: Make sure you're using the same session instance, and that you've
registered reducers for your event types:

```python
session[Plan].register(AddStep, my_reducer)
session.dispatch(AddStep(step="do thing"))
```

Also verify you're querying the right slice:

```python
# This queries Plan slice
plans = session[Plan].all()

# Not this (different type, different slice)
plans = session[AgentPlan].all()  # Different type!
```

## Overrides Not Applying

Your override file exists but the prompt renders with the original text.

**Possible causes**:

1. **Hash mismatch**: The code changed but the override has the old hash
2. **Wrong tag**: You're using a different tag than the override was saved with
3. **Wrong path**: The override file isn't where the store expects it

**Fix**: Check the override file's `expected_hash` matches the current section
hash. Re-seed the override if the code changed:

```python
store.seed(prompt, tag="v2")  # Generate new override with current hash
```

## Tool Calls Failing Silently

The model calls a tool, but nothing happens.

**Check**:

1. Is the tool attached to an enabled section?
2. Is the handler raising an exception that gets swallowed?
3. Are you looking at the right session slice for results?

Enable debug logging to see what's happening:

```python
from weakincentives.runtime import configure_logging
configure_logging(level="DEBUG")
```

## Memory Usage Growing

Session state keeps growing over time.

**Fix**: Make sure you're using appropriate slice policies:

- Use `SlicePolicy.LOG` for append-only history
- Use `SlicePolicy.STATE` for working state that gets replaced
- Call `session.reset()` between unrelated requests if needed

## Import Errors for Extras

`ModuleNotFoundError` when importing adapters or tools.

**Fix**: Install the appropriate extra:

```bash
pip install "weakincentives[openai]"           # For OpenAIAdapter
pip install "weakincentives[litellm]"          # For LiteLLMAdapter
pip install "weakincentives[claude-agent-sdk]" # For ClaudeAgentSDKAdapter
pip install "weakincentives[asteval]"          # For AstevalSection
pip install "weakincentives[podman]"           # For PodmanSandboxSection
pip install "weakincentives[wink]"             # For debug CLI
```

## Debugging Prompts

To see exactly what's being sent to the model:

```python
rendered = prompt.render(session=session)
print(rendered.text)  # Full prompt markdown
print([t.name for t in rendered.tools])  # Tool names
```

For full session inspection, use the debug UI:

```bash
pip install "weakincentives[wink]"
wink debug snapshots/session.jsonl
```

## Getting Help

If you're still stuck:

1. Check the relevant spec document (see the guide index in README.md)
2. Look at `code_reviewer_example.py` for a working reference
3. Enable debug logging to see what's happening
4. Dump a session snapshot and inspect it

## Next Steps

- [Debugging](debugging.md): Deep dive into observability
- [Testing](testing.md): Catch issues before they hit production
- [Code Review Agent](code-review-agent.md): A working example to compare against

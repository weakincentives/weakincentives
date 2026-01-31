# Wink Experiment Command Specification

## Purpose

The `wink experiment` command provides a CLI interface for coding agents and
developers to create, configure, and manage experiments and their associated
prompt overrides. It enables programmatic editing of sections, tools, and
examples without manually constructing JSON files.

**Status:** Proposed (not yet implemented)

## Design Goals

1. **Agent-friendly** - Structured subcommands that coding agents can invoke
   reliably without constructing complex JSON payloads
1. **Safe editing** - Hash-based conflict detection prevents stale overrides
1. **Atomic operations** - Each edit command performs one logical change
1. **Discoverable** - Introspection commands reveal available override targets
1. **Idempotent** - Repeated identical edits produce the same result

## CLI Structure

```
wink experiment <subcommand> [options]
```

### Subcommand Overview

| Subcommand | Purpose |
|------------|---------|
| `setup` | Create a new experiment with initial configuration |
| `show` | Display experiment details and current overrides |
| `list` | List all experiments or available override targets |
| `edit-section` | Override a prompt section's content |
| `edit-tool` | Override a tool's description or parameters |
| `add-tool-example` | Append a new example to a tool |
| `edit-tool-example` | Modify an existing tool example |
| `remove-tool-example` | Remove a tool example |
| `add-task-example` | Append a new task example |
| `edit-task-example` | Modify an existing task example |
| `remove-task-example` | Remove a task example |
| `add-task-step` | Add a step to a task example |
| `remove-task-step` | Remove a step from a task example |
| `set-flag` | Set or update an experiment flag |
| `unset-flag` | Remove an experiment flag |
| `diff` | Show differences between experiment and baseline |
| `validate` | Check all overrides for staleness |
| `export` | Export experiment configuration as JSON |

______________________________________________________________________

## Subcommand Reference

### setup

Create a new experiment with optional initial configuration.

```
wink experiment setup <NAME> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Unique experiment identifier (alphanumeric + hyphens) |

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--tag` | string | Override tag name (default: experiment name) |
| `--owner` | string | Owner identifier for attribution |
| `--description` | string | Human-readable description |
| `--base` | string | Clone configuration from existing experiment |
| `--prompt` | string | Target prompt in `ns:key` format |

**Output:** JSON confirmation with experiment metadata.

**Example:**

```bash
# Create a new experiment
wink experiment setup improved-reasoning \
  --owner "agent-v2" \
  --description "Testing enhanced chain-of-thought prompting" \
  --prompt "core:assistant"

# Clone from existing experiment
wink experiment setup improved-reasoning-v2 \
  --base improved-reasoning
```

______________________________________________________________________

### show

Display experiment configuration and active overrides.

```
wink experiment show <NAME> [options]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--overrides` | flag | Include full override details |
| `--json` | flag | Output as JSON instead of formatted text |
| `--prompt` | string | Filter to specific prompt (`ns:key`) |

**Output:** Experiment metadata, flags, and override summary.

**Example:**

```bash
wink experiment show improved-reasoning --overrides
```

```
Experiment: improved-reasoning
Tag: improved-reasoning
Owner: agent-v2
Description: Testing enhanced chain-of-thought prompting

Flags:
  (none)

Overrides for core:assistant:
  Sections: 2 modified
    - system.instructions (hash: a1b2c3...)
    - system.constraints (hash: d4e5f6...)
  Tools: 1 modified
    - search_code: description overridden
  Tool Examples: 1 added
    - search_code[2]: new example appended
  Task Examples: 0 modified
```

______________________________________________________________________

### list

List experiments or enumerate available override targets.

```
wink experiment list [options]
wink experiment list targets <PROMPT> [options]
```

**Modes:**

| Mode | Description |
|------|-------------|
| (default) | List all defined experiments |
| `targets <PROMPT>` | List overridable sections, tools, examples for a prompt |

**Options for `list targets`:**

| Option | Type | Description |
|--------|------|-------------|
| `--sections` | flag | List only sections |
| `--tools` | flag | List only tools |
| `--examples` | flag | List only examples (tool + task) |
| `--json` | flag | Output as JSON |

**Example:**

```bash
# List all experiments
wink experiment list

# List override targets for a prompt
wink experiment list targets core:assistant --tools
```

```
Tools for core:assistant:
  search_code
    Description: "Search codebase using grep patterns"
    Params: pattern (str), path (str?), case_sensitive (bool)
    Examples: 2
    Contract hash: 7f8e9d...

  read_file
    Description: "Read file contents"
    Params: path (str), offset (int?), limit (int?)
    Examples: 1
    Contract hash: 3c4d5e...
```

______________________________________________________________________

### edit-section

Override a prompt section's body, summary, or visibility.

```
wink experiment edit-section <NAME> <SECTION_PATH> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Experiment name |
| `SECTION_PATH` | Yes | Dot-separated section path (e.g., `system.instructions`) |

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--prompt` | string | Target prompt (`ns:key`), required if ambiguous |
| `--body` | string | New section body content |
| `--body-file` | path | Read body from file |
| `--summary` | string | Override summary text |
| `--visibility` | choice | Set visibility: `full` or `summary` |
| `--hash` | string | Expected content hash (auto-detected if omitted) |

**Behavior:**

1. Loads current prompt and locates section by path
1. Computes content hash if `--hash` not provided
1. Creates or updates `SectionOverride` in experiment's override file
1. Validates the override applies cleanly

**Example:**

```bash
# Override section body from stdin
echo "New instructions here..." | \
  wink experiment edit-section improved-reasoning system.instructions \
    --prompt core:assistant \
    --body-file -

# Set visibility only
wink experiment edit-section improved-reasoning examples.advanced \
    --prompt core:assistant \
    --visibility summary
```

______________________________________________________________________

### edit-tool

Override a tool's description or parameter descriptions.

```
wink experiment edit-tool <NAME> <TOOL_NAME> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Experiment name |
| `TOOL_NAME` | Yes | Tool identifier (e.g., `search_code`) |

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--prompt` | string | Target prompt (`ns:key`) |
| `--description` | string | New tool description (1-200 chars) |
| `--param` | key=value | Override parameter description (repeatable) |
| `--hash` | string | Expected contract hash |

**Example:**

```bash
wink experiment edit-tool improved-reasoning search_code \
  --prompt core:assistant \
  --description "Search codebase with regex patterns" \
  --param pattern="Regular expression to match" \
  --param path="Directory to search (default: cwd)"
```

______________________________________________________________________

### add-tool-example

Append a new example to a tool.

```
wink experiment add-tool-example <NAME> <TOOL_NAME> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Experiment name |
| `TOOL_NAME` | Yes | Tool to add example to |

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--prompt` | string | Target prompt (`ns:key`) |
| `--description` | string | Example description (required, 1-200 chars) |
| `--input` | json | JSON object with input parameters |
| `--input-file` | path | Read input JSON from file |
| `--output` | json | JSON object with expected output |
| `--output-file` | path | Read output JSON from file |

**Example:**

```bash
wink experiment add-tool-example improved-reasoning search_code \
  --prompt core:assistant \
  --description "Find all Python test files" \
  --input '{"pattern": "test_.*\\.py$", "path": "tests/"}' \
  --output '{"files": ["tests/test_foo.py", "tests/test_bar.py"]}'
```

______________________________________________________________________

### edit-tool-example

Modify an existing tool example.

```
wink experiment edit-tool-example <NAME> <TOOL_NAME> <INDEX> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Experiment name |
| `TOOL_NAME` | Yes | Tool containing the example |
| `INDEX` | Yes | Zero-based example index |

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--prompt` | string | Target prompt (`ns:key`) |
| `--description` | string | New description |
| `--input` | json | New input JSON |
| `--output` | json | New output JSON |
| `--hash` | string | Expected example hash |

**Example:**

```bash
wink experiment edit-tool-example improved-reasoning search_code 0 \
  --prompt core:assistant \
  --description "Search for class definitions"
```

______________________________________________________________________

### remove-tool-example

Remove a tool example by index.

```
wink experiment remove-tool-example <NAME> <TOOL_NAME> <INDEX> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Experiment name |
| `TOOL_NAME` | Yes | Tool containing the example |
| `INDEX` | Yes | Zero-based example index |

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--prompt` | string | Target prompt (`ns:key`) |
| `--hash` | string | Expected example hash |

**Example:**

```bash
wink experiment remove-tool-example improved-reasoning search_code 1 \
  --prompt core:assistant
```

______________________________________________________________________

### add-task-example

Append a new task example to a TaskExamplesSection.

```
wink experiment add-task-example <NAME> <SECTION_PATH> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Experiment name |
| `SECTION_PATH` | Yes | Path to TaskExamplesSection |

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--prompt` | string | Target prompt (`ns:key`) |
| `--objective` | string | Task objective (required, 1-500 chars) |
| `--outcome` | string | Expected outcome (string or JSON) |
| `--outcome-file` | path | Read outcome from file |

**Notes:**

- Creates an empty task example; use `add-task-step` to populate steps
- At least one step must be added before the override is valid

**Example:**

```bash
wink experiment add-task-example improved-reasoning examples.tasks \
  --prompt core:assistant \
  --objective "Refactor a function to use async/await" \
  --outcome "Function converted to async with all callers updated"
```

______________________________________________________________________

### edit-task-example

Modify an existing task example's metadata.

```
wink experiment edit-task-example <NAME> <SECTION_PATH> <INDEX> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Experiment name |
| `SECTION_PATH` | Yes | Path to TaskExamplesSection |
| `INDEX` | Yes | Zero-based task example index |

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--prompt` | string | Target prompt (`ns:key`) |
| `--objective` | string | New objective |
| `--outcome` | string | New outcome |
| `--hash` | string | Expected task example hash |

**Example:**

```bash
wink experiment edit-task-example improved-reasoning examples.tasks 0 \
  --prompt core:assistant \
  --objective "Refactor function to modern async syntax"
```

______________________________________________________________________

### remove-task-example

Remove a task example by index.

```
wink experiment remove-task-example <NAME> <SECTION_PATH> <INDEX> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Experiment name |
| `SECTION_PATH` | Yes | Path to TaskExamplesSection |
| `INDEX` | Yes | Zero-based task example index |

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--prompt` | string | Target prompt (`ns:key`) |
| `--hash` | string | Expected task example hash |

______________________________________________________________________

### add-task-step

Add a step to a task example (new or existing).

```
wink experiment add-task-step <NAME> <SECTION_PATH> <TASK_INDEX> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Experiment name |
| `SECTION_PATH` | Yes | Path to TaskExamplesSection |
| `TASK_INDEX` | Yes | Index of task example to modify |

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--prompt` | string | Target prompt (`ns:key`) |
| `--tool` | string | Tool name for this step (required) |
| `--description` | string | Step description (required) |
| `--input` | json | Input parameters JSON |
| `--output` | json | Expected output JSON |
| `--position` | int | Insert position (default: append) |

**Example:**

```bash
# Add first step to task example
wink experiment add-task-step improved-reasoning examples.tasks 0 \
  --prompt core:assistant \
  --tool read_file \
  --description "Read the target function" \
  --input '{"path": "src/utils.py"}' \
  --output '{"content": "def old_function(): ..."}'

# Add second step
wink experiment add-task-step improved-reasoning examples.tasks 0 \
  --prompt core:assistant \
  --tool edit_file \
  --description "Convert to async syntax" \
  --input '{"path": "src/utils.py", "edit": "..."}' \
  --output '{"success": true}'
```

______________________________________________________________________

### remove-task-step

Remove a step from a task example.

```
wink experiment remove-task-step <NAME> <SECTION_PATH> <TASK_INDEX> <STEP_INDEX> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Experiment name |
| `SECTION_PATH` | Yes | Path to TaskExamplesSection |
| `TASK_INDEX` | Yes | Index of task example |
| `STEP_INDEX` | Yes | Index of step to remove |

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--prompt` | string | Target prompt (`ns:key`) |

______________________________________________________________________

### set-flag

Set or update an experiment flag.

```
wink experiment set-flag <NAME> <KEY> <VALUE>
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `NAME` | Yes | Experiment name |
| `KEY` | Yes | Flag key |
| `VALUE` | Yes | Flag value (parsed as JSON if valid, else string) |

**Example:**

```bash
wink experiment set-flag improved-reasoning temperature 0.7
wink experiment set-flag improved-reasoning features '["cot", "reflection"]'
```

______________________________________________________________________

### unset-flag

Remove an experiment flag.

```
wink experiment unset-flag <NAME> <KEY>
```

______________________________________________________________________

### diff

Show differences between experiment overrides and baseline.

```
wink experiment diff <NAME> [options]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--prompt` | string | Filter to specific prompt |
| `--baseline` | string | Compare against another experiment (default: no overrides) |
| `--format` | choice | Output format: `text`, `json`, `unified` |

**Example:**

```bash
wink experiment diff improved-reasoning --prompt core:assistant --format unified
```

______________________________________________________________________

### validate

Check all overrides for staleness against current prompt definitions.

```
wink experiment validate <NAME> [options]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--prompt` | string | Validate only specific prompt |
| `--fix` | flag | Attempt to update hashes for unchanged content |
| `--json` | flag | Output validation results as JSON |

**Output:** List of valid, stale, and invalid overrides.

**Example:**

```bash
wink experiment validate improved-reasoning

# Output:
# Validation Results for improved-reasoning:
#
# Valid (3):
#   - section: system.instructions
#   - tool: search_code
#   - tool-example: search_code[2]
#
# Stale (1):
#   - section: system.constraints
#     Expected hash: a1b2c3...
#     Current hash:  d4e5f6...
#     Content changed, override may need review
#
# Invalid (0)
```

______________________________________________________________________

### export

Export experiment configuration to JSON.

```
wink experiment export <NAME> [options]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--prompt` | string | Export only overrides for specific prompt |
| `--output` | path | Write to file (default: stdout) |
| `--pretty` | flag | Pretty-print JSON |

______________________________________________________________________

## Storage Layout

Experiments and overrides are stored in the `.weakincentives/` directory:

```
.weakincentives/
├── experiments/
│   └── {name}.json              # Experiment metadata
└── prompts/
    └── overrides/
        └── {ns}/
            └── {key}/
                └── {tag}.json   # PromptOverride for experiment
```

**Experiment File Schema:**

```json
{
  "name": "improved-reasoning",
  "overrides_tag": "improved-reasoning",
  "owner": "agent-v2",
  "description": "Testing enhanced chain-of-thought prompting",
  "flags": {
    "temperature": 0.7
  },
  "prompts": ["core:assistant"]
}
```

______________________________________________________________________

## Hash-Based Conflict Detection

All override types include content hashes to detect drift:

| Override Type | Hash Source |
|---------------|-------------|
| `SectionOverride` | SHA-256 of section body |
| `ToolOverride` | SHA-256 of tool contract (description + schemas) |
| `ToolExampleOverride` | SHA-256 of serialized example |
| `TaskExampleOverride` | SHA-256 of objective + outcome + steps |

**Workflow:**

1. When creating an override, the current hash is captured automatically
1. On load, overrides are validated against current content hashes
1. Stale overrides (hash mismatch) are reported but not applied
1. Use `validate --fix` to update hashes when content is unchanged

______________________________________________________________________

## Agent Workflow Examples

### Creating a New Experiment

```bash
# 1. Setup experiment
wink experiment setup my-experiment \
  --prompt core:assistant \
  --description "Testing improved error handling"

# 2. View available targets
wink experiment list targets core:assistant

# 3. Edit a section
wink experiment edit-section my-experiment system.error-handling \
  --prompt core:assistant \
  --body "When errors occur, always provide actionable suggestions..."

# 4. Validate
wink experiment validate my-experiment
```

### Adding Tool Examples

```bash
# 1. List current examples
wink experiment list targets core:assistant --tools

# 2. Add a new example
wink experiment add-tool-example my-experiment search_code \
  --prompt core:assistant \
  --description "Find TODO comments" \
  --input '{"pattern": "TODO:", "path": "."}' \
  --output '{"matches": [{"file": "main.py", "line": 42}]}'

# 3. Verify
wink experiment show my-experiment --overrides
```

### Creating Task Examples

```bash
# 1. Add task example shell
wink experiment add-task-example my-experiment examples.workflows \
  --prompt core:assistant \
  --objective "Fix a failing test" \
  --outcome "Test passes with correct behavior"

# 2. Add steps sequentially
wink experiment add-task-step my-experiment examples.workflows 0 \
  --tool run_tests \
  --description "Run tests to see failure" \
  --input '{"path": "tests/"}' \
  --output '{"failed": ["test_foo"]}'

wink experiment add-task-step my-experiment examples.workflows 0 \
  --tool read_file \
  --description "Read the failing test" \
  --input '{"path": "tests/test_foo.py"}' \
  --output '{"content": "..."}'

wink experiment add-task-step my-experiment examples.workflows 0 \
  --tool edit_file \
  --description "Fix the implementation" \
  --input '{"path": "src/foo.py", "edit": "..."}' \
  --output '{"success": true}'

# 3. Validate the complete task
wink experiment validate my-experiment
```

______________________________________________________________________

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `ExperimentNotFound` | Named experiment doesn't exist | Use `setup` to create |
| `PromptNotFound` | Specified prompt ns:key not registered | Check prompt registry |
| `SectionNotFound` | Section path doesn't exist in prompt | Use `list targets` to discover |
| `ToolNotFound` | Tool not attached to any section | Verify tool registration |
| `IndexOutOfRange` | Example/step index invalid | Use `list targets` for counts |
| `HashMismatch` | Content changed since override created | Use `validate` to review |
| `InvalidOverride` | Override violates constraints | Check field requirements |

All errors include actionable messages and exit with non-zero status.

______________________________________________________________________

## Integration with Evaluation

Experiments integrate with the evaluation framework:

```python
from weakincentives.experiment import Experiment
from weakincentives.evals import EvalRequest, EvalLoop

# Load experiment created via CLI
experiment = Experiment.load("my-experiment")

# Create evaluation requests
requests = [
    EvalRequest(sample=sample, experiment=experiment)
    for sample in samples
]

# Run evaluation with experiment's overrides applied
results = eval_loop.run(requests)
```

The `overrides_tag` on the Experiment automatically resolves to the correct
`PromptOverride` files for each prompt used during evaluation.

______________________________________________________________________

## Limitations

1. **Single prompt per command** - Each edit command operates on one prompt;
   use multiple commands for multi-prompt experiments
1. **No undo** - Edits are immediately persisted; use version control for
   recovery
1. **Sequential steps only** - Task steps must be added one at a time in order
1. **JSON input required** - Complex inputs/outputs must be valid JSON

______________________________________________________________________

## Related Specifications

- `specs/EXPERIMENTS.md` - Experiment data model and A/B testing workflow
- `specs/PROMPTS.md` - Prompt structure and override application
- `specs/EXAMPLES.md` - Tool and task example formats
- `specs/TOOLS.md` - Tool definition and contract hashing
- `specs/WINK_QUERY.md` - Similar CLI subcommand pattern
- `specs/EVALS.md` - Evaluation framework integration

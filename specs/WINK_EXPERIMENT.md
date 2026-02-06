# Wink Experiment Command Specification

## Purpose

The `wink experiment` command enables coding agents to create and modify
experiments by editing prompt overrides through simple CLI invocations.

**Status:** Proposed (not yet implemented)

## Core Insight

An agent iterating on prompts needs exactly three capabilities:

1. **Create** an experiment to hold changes
1. **Edit** prompt elements (sections, tools, examples)
1. **Inspect** what's been changed

Everything else is secondary.

## CLI Structure

```
wink experiment <subcommand> [arguments] [options]
```

### Command Summary

| Command | Purpose |
|---------|---------|
| `new` | Create an experiment bound to a prompt |
| `edit-section` | Override section body/visibility |
| `edit-tool` | Override tool description/params |
| `edit-tool-example` | Add, modify, or remove a tool example |
| `edit-task-example` | Add, modify, or remove a task example |
| `edit-flag` | Set experiment flags |
| `show` | Display experiment state |
| `ls` | List experiments or targets |
| `rm` | Delete an experiment |

______________________________________________________________________

## Creating Experiments

### new

```
wink experiment new <NAME> <PROMPT>
```

Creates an experiment bound to a specific prompt. All subsequent edit commands
operate on this prompt without needing to specify it.

| Argument | Description |
|----------|-------------|
| `NAME` | Experiment identifier (`[a-z0-9-]+`) |
| `PROMPT` | Target prompt as `namespace:key` |

| Option | Description |
|--------|-------------|
| `--from <EXP>` | Clone overrides from existing experiment |
| `--description` | Human-readable description |

```bash
wink experiment new better-reasoning core:assistant
wink experiment new variant-b core:assistant --from better-reasoning
```

Output:

```
Created experiment 'better-reasoning' for core:assistant
```

______________________________________________________________________

## Editing Overrides

All edit commands follow the pattern:

```
wink experiment edit-<type> <EXPERIMENT> <TARGET> [options]
```

### edit-section

Override a section's body, summary, or visibility.

```
wink experiment edit-section <EXP> <PATH> [options]
```

| Option | Description |
|--------|-------------|
| `--body <TEXT>` | New section body (or `-` for stdin) |
| `--body-file <PATH>` | Read body from file |
| `--summary <TEXT>` | Summary for collapsed view |
| `--visibility <full\|summary>` | Default visibility |
| `--clear` | Remove override, restore original |

```bash
# Override body
wink experiment edit-section better-reasoning system.instructions \
  --body "Think step by step before answering."

# From file
wink experiment edit-section better-reasoning system.constraints \
  --body-file ./new-constraints.md

# Change visibility only
wink experiment edit-section better-reasoning examples.advanced \
  --visibility summary

# Remove override
wink experiment edit-section better-reasoning system.instructions --clear
```

### edit-tool

Override a tool's description or parameter descriptions.

```
wink experiment edit-tool <EXP> <TOOL> [options]
```

| Option | Description |
|--------|-------------|
| `--description <TEXT>` | New tool description |
| `--param <NAME>=<DESC>` | Override param description (repeatable) |
| `--clear` | Remove override |

```bash
wink experiment edit-tool better-reasoning search_code \
  --description "Search using ripgrep patterns" \
  --param pattern="Regex pattern (Rust syntax)"
```

### edit-tool-example

Add, modify, or remove a tool example.

```
wink experiment edit-tool-example <EXP> <TOOL> [options]
```

| Option | Description |
|--------|-------------|
| `--add` | Append new example |
| `--at <N>` | Target existing example at index N |
| `--remove` | Remove example (requires `--at`) |
| `--description <TEXT>` | Example description |
| `--input <JSON>` | Input parameters |
| `--output <JSON>` | Expected output |

```bash
# Add new example
wink experiment edit-tool-example better-reasoning search_code --add \
  --description "Find test files" \
  --input '{"pattern": "test_.*\\.py$"}' \
  --output '{"files": ["test_main.py"]}'

# Modify existing
wink experiment edit-tool-example better-reasoning search_code --at 0 \
  --description "Updated description"

# Remove
wink experiment edit-tool-example better-reasoning search_code --at 1 --remove
```

### edit-task-example

Add, modify, or remove a task example.

```
wink experiment edit-task-example <EXP> <SECTION> [options]
```

| Option | Description |
|--------|-------------|
| `--add` | Append new task example |
| `--at <N>` | Target existing task at index N |
| `--remove` | Remove task (requires `--at`) |
| `--objective <TEXT>` | Task objective |
| `--outcome <TEXT\|JSON>` | Expected outcome |
| `--step <JSON>` | Add step (repeatable, order preserved) |
| `--clear-steps` | Remove all existing steps |

Each `--step` is a JSON object: `{"tool": "...", "description": "...", "input": {...}, "output": {...}}`

```bash
# Add complete task example
wink experiment edit-task-example better-reasoning examples.workflows --add \
  --objective "Fix a failing test" \
  --outcome "Test passes" \
  --step '{"tool": "run_tests", "description": "See failure", "input": {}, "output": {"failed": ["test_x"]}}' \
  --step '{"tool": "read_file", "description": "Read test", "input": {"path": "test_x.py"}, "output": {"content": "..."}}' \
  --step '{"tool": "edit_file", "description": "Fix code", "input": {"path": "x.py"}, "output": {"ok": true}}'

# Modify existing
wink experiment edit-task-example better-reasoning examples.workflows --at 0 \
  --objective "Fix a flaky test"

# Add step to existing task
wink experiment edit-task-example better-reasoning examples.workflows --at 0 \
  --step '{"tool": "run_tests", "description": "Verify fix", "input": {}, "output": {"passed": true}}'

# Remove
wink experiment edit-task-example better-reasoning examples.workflows --at 0 --remove
```

### edit-flag

Set or remove experiment flags.

```
wink experiment edit-flag <EXP> <KEY> [VALUE]
```

If VALUE is omitted, the flag is removed.

```bash
wink experiment edit-flag better-reasoning temperature 0.7
wink experiment edit-flag better-reasoning enable_cot true
wink experiment edit-flag better-reasoning temperature  # removes flag
```

______________________________________________________________________

## Inspecting State

### show

Display experiment configuration and overrides.

```
wink experiment show <EXP> [options]
```

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--diff` | Show unified diff against baseline |

```bash
wink experiment show better-reasoning
```

```
Experiment: better-reasoning
Prompt: core:assistant
Description: Testing chain-of-thought prompting

Flags:
  temperature: 0.7

Overrides:
  section system.instructions: body overridden (32 chars)
  tool search_code: description overridden
  tool-example search_code[+0]: added
  task-example examples.workflows[+0]: added (3 steps)
```

### ls

List experiments or available override targets.

```
wink experiment ls [EXP]
```

Without argument: list all experiments.
With argument: list overridable targets in that experiment's prompt.

```bash
# List experiments
wink experiment ls

# List targets
wink experiment ls better-reasoning
```

```
Targets for core:assistant:

Sections:
  system.instructions     "You are a helpful assistant..."
  system.constraints      "Follow these rules..."
  examples.basic          [3 examples]
  examples.advanced       [2 examples]

Tools:
  search_code             2 examples, hash: 7f8e9d...
  read_file               1 example,  hash: 3c4d5e...
  edit_file               1 example,  hash: a1b2c3...

Task Examples:
  examples.workflows      [0 tasks]
```

### rm

Delete an experiment and its overrides.

```
wink experiment rm <EXP>
```

______________________________________________________________________

## Storage

```
.weakincentives/
├── experiments.json                    # Experiment registry
└── prompts/overrides/{ns}/{key}/
    └── {experiment-name}.json          # Overrides for this experiment
```

**experiments.json:**

```json
{
  "better-reasoning": {
    "prompt": "core:assistant",
    "description": "Testing chain-of-thought prompting",
    "flags": {"temperature": 0.7}
  }
}
```

______________________________________________________________________

## Conflict Detection

Overrides include content hashes. When the underlying prompt changes:

1. `show` marks stale overrides with `[STALE]`
1. `edit-*` commands warn but proceed, capturing the new hash
1. Use `--force` to silence warnings

```
$ wink experiment show better-reasoning
...
  section system.instructions: body overridden [STALE - prompt changed]
```

______________________________________________________________________

## Agent Workflow

Typical iteration loop:

```bash
# 1. Create experiment
wink experiment new test-v1 core:assistant

# 2. Discover what's available
wink experiment ls test-v1

# 3. Make changes
wink experiment edit-section test-v1 system.instructions \
  --body "Think carefully before responding."

wink experiment edit-tool-example test-v1 search_code --add \
  --description "Search for imports" \
  --input '{"pattern": "^import"}' \
  --output '{"matches": 42}'

# 4. Review
wink experiment show test-v1 --diff

# 5. Run eval (separate command)
wink eval run --experiment test-v1 dataset.jsonl
```

______________________________________________________________________

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `ExperimentExists` | Name already taken | Use different name or `rm` first |
| `ExperimentNotFound` | Unknown experiment | Check `ls` output |
| `TargetNotFound` | Section/tool doesn't exist | Check `ls <exp>` output |
| `InvalidIndex` | Example index out of range | Use valid index from `ls` |
| `InvalidJSON` | Malformed JSON in `--input`/`--output`/`--step` | Fix JSON syntax |

All commands exit non-zero on error with actionable message.

______________________________________________________________________

## Design Decisions

**Why bind experiment to one prompt?**
Simplifies CLI (no `--prompt` everywhere) and matches how agents actually work:
iterate on one prompt at a time. For multi-prompt changes, create multiple
experiments.

**Why unified `edit-tool-example` instead of add/edit/remove?**
Reduces command count. The `--add`/`--at`/`--remove` flags clearly express
intent while keeping related functionality together.

**Why `--step` as repeatable JSON instead of separate add-step command?**
Task examples are conceptually atomic. Adding steps one-by-one creates invalid
intermediate states. Passing all steps at once ensures validity.

**Why no `validate` command?**
Staleness is shown in `show` output. Forcing separate validation adds friction.
Agents should iterate, not validate.

______________________________________________________________________

## Related Specifications

- `specs/EXPERIMENTS.md` - Experiment data model
- `specs/PROMPTS.md` - Override types and application
- `specs/EVALS.md` - Running evaluations with experiments

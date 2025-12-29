# Code Reviewer Example

A Textual-based code review agent demonstrating MainLoop and EvalLoop.

```bash
OPENAI_API_KEY=sk-... uv run python code_reviewer_example.py
```

## Overview

The example provides a TUI with two main features:

1. **Interactive Review** - Submit requests, see responses and plan updates
1. **Evaluation** - Run EvalLoop on a dataset with progress tracking

## TUI Layout

- **Left panel**: Review response (markdown)
- **Right panel**: Current plan snapshot
- **Bottom**: Input field and submit button
- **Eval panel**: Toggle with `Ctrl+E`, shows progress and results

### Keyboard Shortcuts

| Key | Action |
| --------- | --------------------- |
| `Ctrl+Q` | Quit |
| `Ctrl+E` | Toggle eval panel |
| `Ctrl+R` | Run evaluation |

## Architecture

### ReviewLoop

Extends `MainLoop[ReviewRequest, ReviewResponse]`:

```python
class ReviewLoop(MainLoop[ReviewRequest, ReviewResponse]):
    def prepare(self, request: ReviewRequest) -> tuple[Prompt[ReviewResponse], Session]:
        # Auto-optimizes workspace digest on first request
        # Returns bound prompt and persistent session
```

### ReviewApp

Textual app that:

- Creates mailboxes for request/response
- Runs ReviewLoop in a background thread
- Posts messages to update UI on completion
- Dumps session to `snapshots/` on exit

## Data Types

```python
@dataclass(slots=True, frozen=True)
class ReviewRequest:
    request: str

@dataclass(slots=True, frozen=True)
class ReviewResponse:
    summary: str
    issues: list[str]
    next_steps: list[str]

@dataclass(slots=True, frozen=True)
class ExpectedResponse:
    min_issues: int = 0
    min_next_steps: int = 0
```

## Evaluation

Press `Ctrl+R` or click "Run Evaluation" to:

1. Load dataset from `datasets/reviews.jsonl` (creates sample if missing)
1. Create fresh ReviewLoop for isolation
1. Wrap with EvalLoop using `evaluate_response`
1. Show progress and results in real-time
1. Display summary statistics

### Evaluator

```python
def evaluate_response(output: ReviewResponse, expected: ExpectedResponse) -> Score:
    has_issues = len(output.issues) >= expected.min_issues
    has_steps = len(output.next_steps) >= expected.min_next_steps
    passed = has_issues and has_steps
    return Score(value=..., passed=passed, reason=...)
```

## Prompt Structure

Sections in order:

1. **Instructions** - Static guidance for the agent
1. **Workspace Digest** - Auto-populated on first request
1. **Reference** - Progressive disclosure (starts summarized)
1. **Planning Tools** - PLAN_ACT_REFLECT strategy
1. **VFS Tools** - File operations on sunfish/ mount
1. **Asteval** - Safe expression evaluation
1. **Request** - User's review request

## Environment Variables

| Variable | Required | Default | Description |
| ------------------------ | -------- | --------- | ----------- |
| `OPENAI_API_KEY` | Yes | - | API key |
| `OPENAI_MODEL` | No | `gpt-4.1` | Model |
| `CODE_REVIEW_PROMPT_TAG` | No | `latest` | Overrides |

## Key Files

| File | Purpose |
| ---------------------------- | ------------------------- |
| `code_reviewer_example.py` | Main TUI script |
| `test-repositories/sunfish/` | Mounted repository |
| `snapshots/` | Session dumps |
| `datasets/reviews.jsonl` | Evaluation dataset |

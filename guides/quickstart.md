# Quickstart

Get from zero to a working agent in minutes using the WINK starter project.

## Clone the Starter Project

The fastest way to learn WINK is through the
[starter project](https://github.com/weakincentives/starter)—a complete,
working agent you can run immediately and customize.

```bash
git clone https://github.com/weakincentives/starter.git
cd starter
```

## Install and Run

```bash
make install                      # Install dependencies
make redis                        # Start Redis (required for queue)
export ANTHROPIC_API_KEY=your-key # Set your API key
make agent                        # Start the agent
```

In another terminal, send a question:

```bash
make dispatch QUESTION="What is the secret number?"
# Returns: 42
```

The starter implements a "secret trivia game" where the agent knows hidden
answers that players must discover. It demonstrates all of WINK's core features
in a minimal, understandable package.

## Project Structure

```
starter/
├── workspace/              # Agent persona and behavior
├── skills/secret-trivia/   # Domain knowledge (secret answers)
├── src/trivia_agent/
│   ├── tools.py            # Custom tools (hint_lookup, dice)
│   ├── sections.py         # Progressive disclosure of rules
│   ├── feedback.py         # Soft guidance during execution
│   └── evaluators.py       # Scoring functions for testing
├── tests/                  # Unit tests (100% coverage)
└── AGENTS.md               # Project context for Claude
```

## Core Concepts in Action

The starter project demonstrates these WINK patterns:

### Skills: Domain Knowledge

Skills are markdown files that get injected into the agent's context. The
trivia agent's secrets live in `skills/secret-trivia/`:

```
skills/secret-trivia/
├── secret-number.md   # Contains: 42
├── secret-word.md     # Contains: banana
└── secret-color.md    # Contains: purple
```

### Tools: Agent Capabilities

Tools give agents the ability to perform actions. See `src/trivia_agent/tools.py`:

```python nocheck
@dataclass(slots=True, frozen=True)
class HintParams:
    topic: str

def hint_handler(params: HintParams, *, context: ToolContext) -> ToolResult[str]:
    # Look up hints for the requested topic
    hints = {"number": "It's the answer to everything."}
    hint = hints.get(params.topic, "No hint available.")
    return ToolResult.ok(hint, message=f"Hint for {params.topic}")

hint_tool = Tool[HintParams, str](
    name="hint_lookup",
    description="Get a hint about a secret topic.",
    handler=hint_handler,
)
```

### Tool Policies: Enforcing Constraints

The dice tools demonstrate policies—rules that constrain tool usage:

```python nocheck
# The agent must pick up dice before throwing them
pick_up_dice = Tool[Empty, str](
    name="pick_up_dice",
    description="Pick up the dice.",
    handler=pick_up_handler,
)

throw_dice = Tool[Empty, int](
    name="throw_dice",
    description="Throw the dice. Must pick up first.",
    handler=throw_handler,
    policy=requires_tool_called("pick_up_dice"),  # Enforced constraint
)
```

### Progressive Disclosure: Token Management

Sections can start collapsed and expand when relevant. The game rules are
hidden until the agent needs them:

```python nocheck
rules_section = MarkdownSection(
    title="Game Rules",
    key="rules",
    template=FULL_RULES,
    disclosure=collapsed(summary="Rules available on request."),
)
```

### Feedback: Soft Guidance

Feedback observes agent behavior and provides course correction without
aborting. See `src/trivia_agent/feedback.py`:

```python nocheck
def remind_about_secrets(session: Session) -> str | None:
    """Remind the agent to use its secret knowledge."""
    # Check if agent is struggling to answer
    if should_remind(session):
        return "Remember: you have access to secret knowledge in your skills."
    return None
```

### Evaluators: Testing Agent Behavior

Evaluators score agent outputs for testing. See `src/trivia_agent/evaluators.py`:

```python nocheck
def answer_accuracy(expected: str) -> Evaluator:
    """Check if the agent's answer matches expected."""
    def evaluate(response: Response) -> EvalResult:
        if expected.lower() in response.text.lower():
            return EvalResult.pass_("Correct answer")
        return EvalResult.fail("Wrong answer")
    return evaluate
```

Run evaluations:

```bash
make dispatch-eval QUESTION="What is the secret number?" EXPECTED="42"
make dispatch-eval QUESTION="What is the secret word?" EXPECTED="banana"
```

## Development Workflow

```bash
make check              # Format, lint, typecheck, test (run before commits)
make test               # Unit tests with coverage
make integration-test   # Full integration tests (requires Redis + API key)
```

## Customize the Agent

1. **Change the secrets**: Edit files in `skills/secret-trivia/`
1. **Add new tools**: Create handlers in `src/trivia_agent/tools.py`
1. **Modify behavior**: Update `workspace/` persona files
1. **Add evaluations**: Extend `src/trivia_agent/evaluators.py`

Write a spec in `specs/` describing what you want, then implement it
maintaining 100% test coverage.

## Debug Agent Behavior

WINK saves debug bundles for each execution. Inspect them:

```bash
uv run wink query debug_bundles/*.zip
```

This opens an interactive query interface showing the full execution trace:
prompts sent, tool calls made, and responses received.

## Installing WINK Directly

If you prefer to start from scratch rather than the starter project:

```bash
pip install weakincentives
```

Extras for specific adapters and tools:

```bash
pip install "weakincentives[claude-agent-sdk]" # Claude Agent SDK adapter
pip install "weakincentives[wink]"             # Debug UI (wink CLI)
```

**Python requirement**: 3.12+

## What's Next

- [Prompts](prompts.md): How prompt templates and sections work
- [Tools](tools.md): Tool contracts, context, and policies
- [Sessions](sessions.md): State management with reducers
- [Skills Authoring](skills-authoring.md): Creating domain knowledge
- [Debugging](debugging.md): Inspecting agent behavior

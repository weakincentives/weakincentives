# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example with session state: multi-turn conversations with planning.

Building on 02_with_tools.py, this example adds session state that persists
across multiple evaluation turns. It demonstrates:
- Creating and reusing a Session across turns
- Using PlanningToolsSection for structured task planning
- The LLM creating and updating a plan
- Querying session state between turns

This is the key WINK pattern: Redux-style state management where the
session holds an immutable ledger of events and state slices.

Run with: uv run python examples/progressive/03_with_session.py
Requires: OPENAI_API_KEY environment variable
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from weakincentives import MarkdownSection, Prompt
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.contrib.tools import (
    Plan,
    PlanningStrategy,
    PlanningToolsSection,
)
from weakincentives.prompt import PromptTemplate
from weakincentives.runtime import Session

# --- Structured Output ---


@dataclass(slots=True, frozen=True)
class TurnResponse:
    """Response from each conversation turn."""

    message: str = field(
        metadata={"description": "What the assistant wants to communicate."}
    )
    plan_updated: bool = field(
        metadata={"description": "Whether the plan was modified this turn."}
    )
    ready_to_proceed: bool = field(
        metadata={"description": "Whether the assistant is ready for the next step."}
    )


# --- Prompt Parameters ---


@dataclass(slots=True, frozen=True)
class ConversationParams:
    """Parameters for each conversation turn."""

    user_message: str = field(
        metadata={"description": "The user's input for this turn."}
    )


# --- Prompt Template Builder ---


def build_template(session: Session) -> PromptTemplate[TurnResponse]:
    """Build the prompt template with planning tools.

    The session must be passed to PlanningToolsSection so it can
    install the Plan state slice and register reducers.
    """
    return PromptTemplate[TurnResponse](
        ns="examples/progressive",
        key="planning-conversation",
        name="planning_assistant",
        sections=(
            MarkdownSection[ConversationParams](
                title="Instructions",
                template="""
You are a helpful planning assistant. When given a goal or task:

1. Create a plan using `planning_setup_plan` with an objective and initial steps
2. Update step statuses as work progresses with `planning_update_step`
3. Add new steps if needed with `planning_add_step`
4. Use `planning_read_plan` to review the current plan

User message: ${user_message}

Respond with JSON containing:
- message: Your response to the user
- plan_updated: True if you modified the plan this turn
- ready_to_proceed: True if you're ready for the next user input
                """,
                key="instructions",
            ),
            # PlanningToolsSection adds planning tools and installs Plan state
            PlanningToolsSection(
                session=session,
                strategy=PlanningStrategy.PLAN_ACT_REFLECT,
            ),
        ),
    )


def render_plan(session: Session) -> str:
    """Render the current plan state for display."""
    plan = session[Plan].latest()
    if plan is None or not plan.objective:
        return "No plan created yet."

    lines = [f"Objective: {plan.objective}", f"Status: {plan.status}", "Steps:"]
    if not plan.steps:
        lines.append("  (no steps)")
    else:
        for step in plan.steps:
            marker = "[x]" if step.status == "done" else "[ ]"
            progress = f" ({step.status})" if step.status == "in_progress" else ""
            lines.append(f"  {marker} {step.step_id}. {step.title}{progress}")
    return "\n".join(lines)


def main() -> None:
    """Run a multi-turn planning conversation."""
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY to run this example.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    adapter = OpenAIAdapter(model=model)

    # Create a single session that persists across all turns
    session = Session()

    # Build template with the session
    template = build_template(session)

    # Simulate a multi-turn conversation
    conversation = [
        "Help me plan a birthday party for next Saturday.",
        "I've booked the venue. What's next?",
        "The invitations are sent. Mark that as done.",
        "Show me the current plan status.",
    ]

    print("=" * 60)
    print("Multi-turn Planning Conversation")
    print("=" * 60)

    for turn_num, user_message in enumerate(conversation, 1):
        print(f"\n--- Turn {turn_num} ---")
        print(f"User: {user_message}")

        # Create a new Prompt each turn, but reuse the same session
        prompt = Prompt(template).bind(ConversationParams(user_message=user_message))
        response = adapter.evaluate(prompt, session=session)

        if response.output is not None:
            print(f"\nAssistant: {response.output.message}")
            if response.output.plan_updated:
                print("\n[Plan was updated]")
        else:
            print(f"\nAssistant (raw): {response.text or '(no response)'}")

        # Show the plan state after each turn
        print("\n--- Current Plan ---")
        print(render_plan(session))

    print("\n" + "=" * 60)
    print("Conversation complete. Final plan state preserved in session.")
    print("=" * 60)


if __name__ == "__main__":
    main()

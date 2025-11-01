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

"""Interactive demo showcasing the planning tool suite with OpenAI."""

from __future__ import annotations

import os
from dataclasses import dataclass

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    import sys
    from pathlib import Path

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.events import InProcessEventBus
from weakincentives.prompt import MarkdownSection, Prompt
from weakincentives.session import Session, select_latest
from weakincentives.tools import Plan, PlanningToolsSection


@dataclass(slots=True)
class PlanningParams:
    """Prompt parameters for each agent turn."""

    objective: str
    request: str


def build_prompt(session: Session) -> Prompt[str]:
    """Construct the planning demo prompt."""

    return Prompt[str](
        ns="examples/planning",
        key="planning-demo",
        name="planning-demo",
        sections=[
            MarkdownSection[PlanningParams](
                title="Task",
                key="task",
                template=(
                    "Objective: ${objective}\n"
                    "Current request: ${request}\n"
                    "Use planning tools to manage multi-step work."
                ),
            ),
            PlanningToolsSection(session=session),
        ],
    )


def main() -> None:
    """Run the interactive planning demo."""

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    bus = InProcessEventBus()
    session = Session(bus=bus)
    prompt = build_prompt(session)
    adapter = OpenAIAdapter(model=model)

    objective = input("Objective for the agent: ").strip()
    if not objective:
        raise SystemExit("Objective is required.")

    print("Enter messages for the agent. Type 'plan' to show the current plan.")
    print("Press Enter on an empty line to exit.")

    while True:
        try:
            request = input("Message: ").strip()
        except EOFError:  # pragma: no cover - interactive convenience
            break
        if not request:
            break
        if request.lower() == "plan":
            _print_plan(session)
            continue
        response = adapter.evaluate(
            prompt,
            PlanningParams(objective=objective, request=request),
            bus=bus,
        )
        print(f"Agent: {response.text}\n")

    _print_plan(session)


def _print_plan(session: Session) -> None:
    plan = select_latest(session, Plan)
    if plan is None:
        print("No plan recorded yet.\n")
        return
    print("Current plan:")
    print(f"Objective: {plan.objective}")
    print(f"Status: {plan.status}")
    for step in plan.steps:
        print(f"- {step.step_id} [{step.status}]: {step.title}")
        if step.details:
            print(f"    details: {step.details}")
        if step.notes:
            print(f"    notes: {'; '.join(step.notes)}")
    print()


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

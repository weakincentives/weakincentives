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

"""Plan rendering utilities shared by runnable demos."""

from __future__ import annotations

from weakincentives.runtime import Session, select_latest
from weakincentives.tools import Plan

__all__ = [
    "render_plan_snapshot",
]


def render_plan_snapshot(session: Session) -> str:
    """Render the current plan state as a human-readable string.

    Args:
        session: The session containing plan state.

    Returns:
        A formatted string representation of the active plan, or a message
        indicating no active plan exists.
    """
    plan = select_latest(session, Plan)
    if plan is None:
        return "No active plan."

    lines = [f"Objective: {plan.objective} (status: {plan.status})"]
    lines.extend(
        f"- {step.step_id} [{step.status}] {step.title}" for step in plan.steps
    )
    return "\n".join(lines)

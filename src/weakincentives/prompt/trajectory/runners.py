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

"""Observer execution helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .types import Assessment, ObserverConfig, ObserverContext, ObserverTrigger

if TYPE_CHECKING:
    from ...runtime.session.protocols import SessionProtocol

__all__ = ["run_observers"]


def _utcnow() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)


def _should_trigger(trigger: ObserverTrigger, context: ObserverContext) -> bool:
    """Check if any trigger condition is met.

    Trigger conditions are OR'd: if either is satisfied, the observer runs.
    """
    if (
        trigger.every_n_calls is not None
        and context.tool_calls_since_last_assessment() >= trigger.every_n_calls
    ):
        return True

    if trigger.every_n_seconds is not None:
        last = context.last_assessment
        if last is None:
            # No previous assessment - trigger immediately
            return True
        elapsed = (_utcnow() - last.timestamp).total_seconds()
        if elapsed >= trigger.every_n_seconds:
            return True

    return False


def run_observers(
    *,
    observers: Sequence[ObserverConfig],
    context: ObserverContext,
    session: SessionProtocol,
) -> str | None:
    """Run observers and return rendered assessment if triggered.

    This function iterates through the configured observers, checking trigger
    conditions and observer readiness. The first observer that triggers and
    produces an assessment is used; subsequent observers are not evaluated.

    The assessment is stored in the session's Assessment slice for history
    tracking, then rendered to text for injection into the agent's context.

    Args:
        observers: Sequence of observer configurations from the prompt.
        context: Observer context with session state and resources.
        session: Session for dispatching the assessment event.

    Returns:
        Rendered assessment text if an observer produced output, None otherwise.
    """
    for config in observers:
        if _should_trigger(config.trigger, context) and config.observer.should_run(
            session, context=context
        ):
            assessment = config.observer.observe(session, context=context)
            # Record the call index for future trigger calculations
            assessment = replace(assessment, call_index=context.tool_call_count)
            # Store assessment in session slice for history
            session[Assessment].append(assessment)
            return assessment.render()

    return None

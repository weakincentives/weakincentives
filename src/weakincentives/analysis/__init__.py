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

"""Analysis loop for automated debug bundle investigation.

The best way to debug a complex agent is with another agent. This package
provides AnalysisLoop, a specialized AgentLoop that receives debug bundles,
analyzes them via ``wink query``, and produces structured reports.

Architecture
------------

Three components, loosely coupled:

**AgentLoop / EvalLoop** emit CompletionNotification when work completes.
They don't know about analysis—they just announce completion.

**AnalysisForwarder** consumes notifications, decides what to analyze
(sampling, budget), and sends requests to AnalysisLoop.

**AnalysisLoop** is a standard AgentLoop with its own mailbox. It receives
requests, analyzes bundles, and produces analysis bundles.

Quick Start
-----------

Wire up with the helper function::

    from weakincentives.analysis import connect_analysis

    forwarder, analysis = connect_analysis(
        notifications=notifications_mailbox,
        objective="Identify patterns in failing samples",
        sample_rate=0.1,
    )

    group = LoopGroup(loops=[agent_loop, eval_loop, forwarder, analysis])
    group.run()

On-Demand Analysis
------------------

Skip the notification machinery for one-off analysis::

    from weakincentives.analysis import AnalysisLoop, AnalysisRequest

    requests = InMemoryMailbox(name="analysis-requests")
    analysis = AnalysisLoop(adapter=adapter, requests=requests)

    requests.send(AgentLoopRequest(request=AnalysisRequest(
        objective="Why did this specific execution fail?",
        bundles=(Path("./debug/bundle-001.zip"),),
    )))

    analysis.run(max_iterations=1)

Exports
-------

**Data Types:**
    - :class:`CompletionNotification` - Emitted by loops on completion
    - :class:`AnalysisRequest` - Request to analyze bundles
    - :class:`AnalysisBundle` - Structured analysis output
    - :class:`EvalContext` - Optional evaluation metadata

**Configuration:**
    - :class:`AnalysisForwarderConfig` - Forwarder sampling and budget
    - :class:`AnalysisBudget` - Rate limiting for forwarding
    - :class:`AnalysisLoopConfig` - Loop output and prompt config
    - :class:`AnalysisPromptOverrides` - Prompt section overrides

**Components:**
    - :class:`AnalysisForwarder` - Notification-to-request bridge
    - :class:`AnalysisLoop` - Debug bundle analysis agent

**Helpers:**
    - :func:`connect_analysis` - Wire forwarder and loop together

**Prompt:**
    - :func:`build_analysis_prompt` - Build analysis prompt for a request
    - :func:`build_analysis_template` - Build reusable prompt template
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..runtime.agent_loop import AgentLoopConfig, AgentLoopRequest, AgentLoopResult
from ..runtime.mailbox import InMemoryMailbox
from ._forwarder import AnalysisForwarder
from ._loop import AnalysisLoop
from ._prompt import build_analysis_prompt, build_analysis_template
from ._types import (
    AnalysisBudget,
    AnalysisBundle,
    AnalysisForwarderConfig,
    AnalysisLoopConfig,
    AnalysisPromptOverrides,
    AnalysisRequest,
    CompletionNotification,
    EvalContext,
)

if TYPE_CHECKING:
    from ..adapters.core import ProviderAdapter
    from ..runtime.mailbox import Mailbox


def connect_analysis(  # noqa: PLR0913
    *,
    notifications: Mailbox[CompletionNotification, None],
    adapter: ProviderAdapter[AnalysisBundle],
    objective: str,
    sample_rate: float = 0.1,
    always_forward_failures: bool = True,
    budget: AnalysisBudget | None = None,
    analysis_config: AnalysisLoopConfig | None = None,
    agent_loop_config: AgentLoopConfig | None = None,
) -> tuple[AnalysisForwarder, AnalysisLoop]:
    """Wire up an AnalysisForwarder and AnalysisLoop together.

    Creates an in-memory mailbox for analysis requests and connects
    the forwarder to the analysis loop. Both components are returned
    for inclusion in a LoopGroup.

    Args:
        notifications: Mailbox receiving CompletionNotification messages.
        adapter: Provider adapter for the analysis agent.
        objective: Research question for analysis requests.
        sample_rate: Fraction of successful notifications to analyze.
        always_forward_failures: Whether to always analyze failures.
        budget: Optional rate limiting configuration.
        analysis_config: Optional AnalysisLoop configuration.
        agent_loop_config: Optional AgentLoop configuration.

    Returns:
        Tuple of (forwarder, analysis_loop) ready for LoopGroup.

    Example::

        forwarder, analysis = connect_analysis(
            notifications=notifications_mailbox,
            adapter=adapter,
            objective="Identify patterns in failing samples",
            sample_rate=0.1,
        )

        group = LoopGroup(loops=[agent_loop, eval_loop, forwarder, analysis])
        group.run()
    """
    analysis_requests: InMemoryMailbox[
        AgentLoopRequest[AnalysisRequest], AgentLoopResult[AnalysisBundle]
    ] = InMemoryMailbox(name="analysis-requests")

    # Separate mailbox for the forwarder to send raw AnalysisRequest messages
    forwarder_target: InMemoryMailbox[AnalysisRequest, None] = InMemoryMailbox(
        name="analysis-forwarder-target"
    )

    forwarder_config = AnalysisForwarderConfig(
        objective=objective,
        sample_rate=sample_rate,
        always_forward_failures=always_forward_failures,
        budget=budget if budget is not None else AnalysisBudget(),
    )

    forwarder = AnalysisForwarder(
        notifications=notifications,
        analysis_requests=forwarder_target,
        config=forwarder_config,
    )

    analysis = AnalysisLoop(
        adapter=adapter,
        requests=analysis_requests,
        config=analysis_config,
        agent_loop_config=agent_loop_config,
    )

    return forwarder, analysis


__all__ = [
    "AnalysisBudget",
    "AnalysisBundle",
    "AnalysisForwarder",
    "AnalysisForwarderConfig",
    "AnalysisLoop",
    "AnalysisLoopConfig",
    "AnalysisPromptOverrides",
    "AnalysisRequest",
    "CompletionNotification",
    "EvalContext",
    "build_analysis_prompt",
    "build_analysis_template",
    "connect_analysis",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})  # pragma: no cover - convenience shim

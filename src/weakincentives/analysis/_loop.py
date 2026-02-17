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

"""AnalysisLoop: specialized AgentLoop for analyzing debug bundles.

Receives AnalysisRequest messages and produces AnalysisBundle outputs
using a prebuilt analysis prompt with ``wink query`` as its primary tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from ..prompt import Prompt
from ..runtime.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopRequest,
    AgentLoopResult,
)
from ..runtime.mailbox import Mailbox
from ..runtime.session import Session
from ._prompt import build_analysis_prompt
from ._types import AnalysisBundle, AnalysisLoopConfig, AnalysisRequest

if TYPE_CHECKING:
    from ..adapters.core import ProviderAdapter
    from ..experiment import Experiment


class AnalysisLoop(AgentLoop[AnalysisRequest, AnalysisBundle]):
    """Specialized AgentLoop that analyzes debug bundles.

    Receives AnalysisRequest messages on its mailbox and produces
    AnalysisBundle outputs containing markdown reports with findings
    and recommendations.

    Uses a prebuilt analysis prompt that can be customized via
    AnalysisPromptOverrides in the config.

    Example::

        requests = InMemoryMailbox(name="analysis-requests")
        analysis = AnalysisLoop(adapter=adapter, requests=requests)

        requests.send(AnalysisRequest(
            objective="Why did this specific execution fail?",
            bundles=(Path("./debug/bundle-001.zip"),),
        ))

        analysis.run(max_iterations=1)
    """

    _analysis_config: AnalysisLoopConfig

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[AnalysisBundle],
        requests: Mailbox[
            AgentLoopRequest[AnalysisRequest], AgentLoopResult[AnalysisBundle]
        ],
        config: AnalysisLoopConfig | None = None,
        agent_loop_config: AgentLoopConfig | None = None,
        worker_id: str | None = None,
    ) -> None:
        """Initialize the AnalysisLoop.

        Args:
            adapter: Provider adapter for prompt evaluation.
            requests: Mailbox to receive AnalysisRequest messages from.
            config: Analysis-specific configuration (output dir, overrides).
            agent_loop_config: Optional AgentLoop configuration for budget/resources.
            worker_id: Optional worker identifier.
        """
        super().__init__(
            adapter=adapter,
            requests=requests,
            config=agent_loop_config,
            worker_id=worker_id,
        )
        self._analysis_config = config if config is not None else AnalysisLoopConfig()

    @override
    def prepare(
        self,
        request: AnalysisRequest,
        *,
        experiment: Experiment | None = None,
    ) -> tuple[Prompt[AnalysisBundle], Session]:
        """Build analysis prompt and session for the given request.

        Constructs a prompt using the prebuilt analysis template with
        optional section overrides from config.

        Args:
            request: The analysis request with objective and bundle paths.
            experiment: Optional experiment (unused for analysis).

        Returns:
            A tuple of (prompt, session) ready for evaluation.
        """
        _ = experiment
        prompt = build_analysis_prompt(
            request,
            overrides=self._analysis_config.overrides,
        )
        session = Session(tags={"loop": "analysis", "objective": request.objective})
        return prompt, session


__all__ = [
    "AnalysisLoop",
]

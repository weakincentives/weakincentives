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

"""Protocol defining the prompt optimizer interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from ..prompt import Prompt
    from ..runtime.session.protocols import SessionProtocol

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)


class PromptOptimizer(Protocol[InputT, OutputT]):
    """Protocol for prompt optimization algorithms.

    The protocol is parameterized by:

    - ``InputT`` - the prompt's output type, allowing optimizers to constrain
      which prompts they accept.
    - ``OutputT`` - the result type returned by the optimizer, enabling type-safe
      consumption of optimization artifacts.
    """

    def optimize(
        self,
        prompt: Prompt[InputT],
        *,
        session: SessionProtocol,
    ) -> OutputT:
        """Apply the optimization algorithm to the given prompt.

        Args:
            prompt: The prompt to optimize.
            session: The caller's session for state queries and result
                persistence. Optimizers must not mutate this session
                during internal evaluation; use isolated sessions instead.

        Returns:
            An optimization result whose shape depends on the algorithm.
        """
        ...


__all__ = ["PromptOptimizer"]

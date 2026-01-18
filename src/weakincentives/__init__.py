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

"""Public surface for :mod:`weakincentives`."""

from __future__ import annotations

from .adapters import PromptResponse
from .budget import Budget, BudgetExceededError, BudgetTracker
from .clock import (
    SYSTEM_CLOCK,
    Clock,
    FakeClock,
    MonotonicClock,
    Sleeper,
    SystemClock,
    WallClock,
)
from .threading import (
    SYSTEM_EXECUTOR,
    BackgroundWorker,
    CallbackRegistry,
    CancelledException,
    Checkpoint,
    Executor,
    FakeCheckpoint,
    FakeExecutor,
    FakeGate,
    FakeLatch,
    FakeScheduler,
    FifoScheduler,
    Future,
    Gate,
    Latch,
    Scheduler,
    SimpleCancellationToken,
    SystemCheckpoint,
    SystemExecutor,
    SystemGate,
)
from .dataclasses import FrozenDataclass
from .deadlines import Deadline
from .errors import DeadlineExceededError, ToolValidationError, WinkError
from .prompt import (
    MarkdownSection,
    Prompt,
    Tool,
    ToolContext,
    ToolHandler,
    ToolResult,
    parse_structured_output,
)
from .runtime import StructuredLogger, configure_logging, get_logger
from .skills import (
    Skill,
    SkillConfig,
    SkillError,
    SkillMount,
    SkillMountError,
    SkillNotFoundError,
    SkillValidationError,
)
from .types import JSONValue, SupportsDataclass

__all__ = [
    # Clock
    "SYSTEM_CLOCK",
    "Clock",
    "FakeClock",
    "MonotonicClock",
    "Sleeper",
    "SystemClock",
    "WallClock",
    # Threading
    "SYSTEM_EXECUTOR",
    "BackgroundWorker",
    "CallbackRegistry",
    "CancelledException",
    "Checkpoint",
    "Executor",
    "FakeCheckpoint",
    "FakeExecutor",
    "FakeGate",
    "FakeLatch",
    "FakeScheduler",
    "FifoScheduler",
    "Future",
    "Gate",
    "Latch",
    "Scheduler",
    "SimpleCancellationToken",
    "SystemCheckpoint",
    "SystemExecutor",
    "SystemGate",
    # Budget
    "Budget",
    "BudgetExceededError",
    "BudgetTracker",
    # Deadlines
    "Deadline",
    "DeadlineExceededError",
    # Dataclasses
    "FrozenDataclass",
    # Prompt
    "MarkdownSection",
    "Prompt",
    "PromptResponse",
    "Tool",
    "ToolContext",
    "ToolHandler",
    "ToolResult",
    "parse_structured_output",
    # Skills
    "Skill",
    "SkillConfig",
    "SkillError",
    "SkillMount",
    "SkillMountError",
    "SkillNotFoundError",
    "SkillValidationError",
    # Logging
    "StructuredLogger",
    "configure_logging",
    "get_logger",
    # Types
    "JSONValue",
    "SupportsDataclass",
    # Errors
    "ToolValidationError",
    "WinkError",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})

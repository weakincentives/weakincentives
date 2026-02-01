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

"""WINK: A framework for building reliable, testable AI agents.

The ``weakincentives`` package (also known as WINK) provides a comprehensive
toolkit for developing AI agents with composable prompts, tool handling,
session management, and evaluation infrastructure.

Subpackages
-----------

adapters
    Integration adapters for LLM providers (OpenAI, LiteLLM, Claude Agent SDK).
    Provides ``ProviderAdapter`` protocol and configuration classes for
    connecting to different model backends.

cli
    Command-line interface via the ``wink`` executable. Includes commands for
    debug bundle inspection (``wink debug``), documentation access
    (``wink docs``), and SQL queries on bundles (``wink query``).

contrib
    Contributed utilities extending core primitives. Contains subpackages for:

    - ``contrib.mailbox``: Redis-backed mailbox implementation
    - ``contrib.optimizers``: Workspace digest optimizer
    - ``contrib.tools``: Workspace digest tools

dataclasses
    Enhanced frozen dataclass decorator (``FrozenDataclass``) with copy helpers
    (``update()``, ``merge()``, ``map()``) and ``__pre_init__`` support.

dbc
    Design-by-contract decorators for runtime validation:

    - ``@require``: Validate preconditions before function execution
    - ``@ensure``: Validate postconditions after function returns
    - ``@invariant``: Enforce class invariants on public methods
    - ``@pure``: Assert function has no side effects

debug
    Debug bundle utilities for capturing and inspecting execution state.
    Provides ``BundleWriter`` for creating bundles and ``DebugBundle`` for
    loading and analyzing them.

evals
    Evaluation framework for agent testing. Includes dataset loading, scoring
    primitives (``exact_match``, ``contains``), session-aware evaluators
    (``tool_called``, ``all_tools_succeeded``), and LLM-as-judge support.

filesystem
    Core filesystem protocol abstracting over storage backends. Used by file
    operation tools.

formal
    Formal specification support via TLA+. Decorators attach specification
    metadata that can be extracted and validated by pytest plugins.

prompt
    Prompt authoring primitives:

    - ``Prompt`` / ``PromptTemplate``: Composable prompt builders
    - ``Section`` / ``MarkdownSection``: Content sections with visibility control
    - ``Tool`` / ``ToolHandler`` / ``ToolResult``: Tool definition and execution
    - ``Feedback`` / ``FeedbackProvider``: Dynamic prompt augmentation
    - ``ToolPolicy``: Sequencing and validation policies for tools

resources
    Dependency injection with scoped lifecycles (``SINGLETON``, ``TOOL_CALL``,
    ``PROTOTYPE``). Provides ``ResourceRegistry`` and ``Binding`` for wiring.

runtime
    Runtime primitives including:

    - ``Session``: Event-sourced state container with typed slices
    - ``AgentLoop``: Core orchestration loop for prompt evaluation
    - ``Mailbox``: Message queue abstraction for request/response
    - ``Dispatcher``: Event distribution and telemetry
    - ``Watchdog`` / ``Heartbeat``: Health monitoring utilities

serde
    Stdlib dataclass serialization utilities. Use ``parse(cls, data)`` for
    deserialization and ``dump(obj)`` for serialization. Supports constraints
    via ``Annotated`` and polymorphic unions via ``__type__`` field.

skills
    Agent skills following the Agent Skills specification. Skills are folders
    of instructions and resources that agents can discover and use.

types
    Shared typing helpers including ``JSONValue``, ``SupportsDataclass``,
    adapter name constants, and contract result types.

Core Exports
------------

Time and Clocks
~~~~~~~~~~~~~~~

- ``Clock``: Unified clock protocol (monotonic + wall-clock + sleep)
- ``MonotonicClock``: Protocol for elapsed time measurement
- ``WallClock``: Protocol for UTC datetime operations
- ``Sleeper``: Protocol for delay operations
- ``SystemClock``: Production clock using system time
- ``FakeClock``: Controllable clock for deterministic testing
- ``SYSTEM_CLOCK``: Default system clock singleton

Deadlines and Budgets
~~~~~~~~~~~~~~~~~~~~~

- ``Deadline``: Immutable wall-clock expiration with remaining/elapsed time
- ``Budget``: Resource envelope combining time and token limits
- ``BudgetTracker``: Thread-safe cumulative token usage tracker
- ``BudgetExceededError``: Raised when budget limits are breached
- ``DeadlineExceededError``: Raised when tool execution exceeds deadline

Prompt and Tools
~~~~~~~~~~~~~~~~

- ``Prompt``: Composable prompt builder with sections and tools
- ``MarkdownSection``: Content section rendered as markdown
- ``Tool``: Tool definition with handler, parameters, and validation
- ``ToolHandler``: Protocol for tool implementation functions
- ``ToolContext``: Context passed to tool handlers with session access
- ``ToolResult``: Structured result container (success or error)
- ``parse_structured_output``: Parse model output into typed dataclass

Skills
~~~~~~

- ``Skill``: Core skill representation
- ``SkillMount``: Configuration for mounting a skill to a section
- ``SkillError``, ``SkillValidationError``, ``SkillNotFoundError``,
  ``SkillMountError``: Skill-related exceptions

Errors
~~~~~~

- ``WinkError``: Base class for all weakincentives exceptions
- ``ToolValidationError``: Tool parameter validation failure

Logging
~~~~~~~

- ``configure_logging``: Set up structured logging configuration
- ``get_logger``: Get a logger instance by name
- ``StructuredLogger``: Logger wrapper with structured event support

Dataclasses and Types
~~~~~~~~~~~~~~~~~~~~~

- ``FrozenDataclass``: Decorator for frozen, slotted dataclasses with helpers
- ``PromptResponse``: Response from adapter prompt evaluation
- ``JSONValue``: Union type for JSON-compatible values
- ``SupportsDataclass``: Protocol for dataclass-like objects

Example Usage
-------------

Basic prompt with tool::

    from weakincentives import Prompt, MarkdownSection, Tool, ToolResult

    def greet(params, *, context):
        name = params.get("name", "World")
        return ToolResult.ok(f"Hello, {name}!")

    prompt = Prompt(
        ns="demo",
        key="greeting",
        sections=(
            MarkdownSection(key="intro", content="You are a helpful assistant."),
        ),
        tools=(
            Tool(
                name="greet",
                description="Greet someone by name",
                handler=greet,
            ),
        ),
    )

Using clocks for testable time-dependent code::

    from weakincentives import SYSTEM_CLOCK, FakeClock, Deadline
    from datetime import UTC, datetime, timedelta

    # Production: use system clock
    deadline = Deadline(
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        clock=SYSTEM_CLOCK,
    )

    # Testing: use fake clock for deterministic behavior
    clock = FakeClock()
    clock.set_wall(datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC))
    deadline = Deadline(
        expires_at=datetime(2024, 6, 1, 13, 0, 0, tzinfo=UTC),
        clock=clock,
    )
    clock.advance(1800)  # 30 minutes, no real delay
    assert deadline.remaining() == timedelta(minutes=30)

Budget enforcement::

    from weakincentives import Budget, BudgetTracker, Deadline
    from datetime import UTC, datetime, timedelta

    budget = Budget(
        deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1)),
        max_total_tokens=100000,
    )
    tracker = BudgetTracker(budget=budget)

    # Record usage and check limits
    tracker.record_cumulative("eval-1", usage)
    tracker.check()  # Raises BudgetExceededError if over limit

See Also
--------

- ``weakincentives.runtime``: Session management and main loop orchestration
- ``weakincentives.evals``: Evaluation framework for testing agents
- ``weakincentives.dbc``: Design-by-contract decorators
- ``weakincentives.resources``: Dependency injection framework
"""

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
    SkillError,
    SkillMount,
    SkillMountError,
    SkillNotFoundError,
    SkillValidationError,
)
from .types import JSONValue, SupportsDataclass

__all__ = [
    "SYSTEM_CLOCK",
    "Budget",
    "BudgetExceededError",
    "BudgetTracker",
    "Clock",
    "Deadline",
    "DeadlineExceededError",
    "FakeClock",
    "FrozenDataclass",
    "JSONValue",
    "MarkdownSection",
    "MonotonicClock",
    "Prompt",
    "PromptResponse",
    "Skill",
    "SkillError",
    "SkillMount",
    "SkillMountError",
    "SkillNotFoundError",
    "SkillValidationError",
    "Sleeper",
    "StructuredLogger",
    "SupportsDataclass",
    "SystemClock",
    "Tool",
    "ToolContext",
    "ToolHandler",
    "ToolResult",
    "ToolValidationError",
    "WallClock",
    "WinkError",
    "configure_logging",
    "get_logger",
    "parse_structured_output",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})

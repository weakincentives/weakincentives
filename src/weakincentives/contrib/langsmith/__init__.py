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

"""LangSmith integration for WINK telemetry and prompt management.

This module provides:

- **Auto-instrumentation**: Single-call configuration via :func:`configure_wink`
- **Telemetry**: Event-based tracing via :class:`LangSmithTelemetryHandler`
- **Prompt Hub**: Bidirectional prompt management via :class:`LangSmithPromptOverridesStore`

Example usage::

    from weakincentives.contrib.langsmith import configure_wink

    # Enable automatic tracing at application start
    configure_wink()

    # All WINK evaluations are now traced to LangSmith
    response = adapter.evaluate(prompt, session=session)

For manual control::

    from weakincentives.contrib.langsmith import (
        LangSmithConfig,
        LangSmithTelemetryHandler,
    )

    config = LangSmithConfig(project="my-agent")
    telemetry = LangSmithTelemetryHandler(config)
    telemetry.attach(bus)
"""

from __future__ import annotations

from ._config import LangSmithConfig
from ._configure import configure_wink
from ._context import TraceContext, get_current_run_tree
from ._events import (
    LangSmithTraceCompleted,
    LangSmithTraceStarted,
    LangSmithUploadFailed,
)
from ._overrides import LangSmithPromptOverridesStore
from ._telemetry import LangSmithTelemetryHandler

__all__ = [
    "LangSmithConfig",
    "LangSmithPromptOverridesStore",
    "LangSmithTelemetryHandler",
    "LangSmithTraceCompleted",
    "LangSmithTraceStarted",
    "LangSmithUploadFailed",
    "TraceContext",
    "configure_wink",
    "get_current_run_tree",
]

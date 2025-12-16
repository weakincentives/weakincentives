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

"""Runtime primitives for :mod:`weakincentives`.

This module exports essential runtime types. Advanced types are available
in submodules:

- :mod:`weakincentives.runtime.session` — Session internals and protocols
- :mod:`weakincentives.runtime.session.reducers` — Reducer helper functions
- :mod:`weakincentives.runtime.events` — Event types and bus implementations
"""

from __future__ import annotations

from .events import (
    EventBus,
    InProcessEventBus,
)
from .logging import StructuredLogger, configure_logging, get_logger
from .main_loop import (
    MainLoop,
    MainLoopCompleted,
    MainLoopConfig,
    MainLoopFailed,
    MainLoopRequest,
)
from .session import Session

__all__ = [
    "EventBus",
    "InProcessEventBus",
    "MainLoop",
    "MainLoopCompleted",
    "MainLoopConfig",
    "MainLoopFailed",
    "MainLoopRequest",
    "Session",
    "StructuredLogger",
    "configure_logging",
    "get_logger",
]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})  # pragma: no cover - convenience shim

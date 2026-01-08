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

"""Trajectory observation for ongoing agent progress assessment.

This module provides types and utilities for trajectory observers, which analyze
agent behavior over time and inject feedback into the agent's context. Unlike
tool policies that gate individual calls, observers produce guidance that the
agent can choose to act upon.

Key components:

- :class:`TrajectoryObserver`: Protocol for implementing observers
- :class:`Assessment`: Structured feedback from an observation
- :class:`ObserverContext`: Context provided to observers (mirrors ToolContext)
- :class:`ObserverConfig`: Configuration binding observer to trigger conditions
- :class:`DeadlineObserver`: Built-in observer for time remaining feedback
"""

from __future__ import annotations

from .observers import DeadlineObserver
from .runners import run_observers
from .types import (
    Assessment,
    Observation,
    ObserverConfig,
    ObserverContext,
    ObserverTrigger,
    TrajectoryObserver,
)

__all__ = [
    "Assessment",
    "DeadlineObserver",
    "Observation",
    "ObserverConfig",
    "ObserverContext",
    "ObserverTrigger",
    "TrajectoryObserver",
    "run_observers",
]

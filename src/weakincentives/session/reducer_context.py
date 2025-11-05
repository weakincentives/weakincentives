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

# pyright: reportImportCycles=false

"""Runtime context threaded into session reducer invocations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..events import EventBus

if TYPE_CHECKING:  # pragma: no cover - circular import safe-guard
    from .session import Session


@dataclass(slots=True, frozen=True)
class ReducerContext:
    """Shared session references available to reducers."""

    session: Session
    event_bus: EventBus


__all__ = ["ReducerContext"]

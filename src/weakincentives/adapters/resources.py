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

"""Resource registry building utilities for provider adapters."""

from __future__ import annotations

from ..budget import BudgetTracker
from ..filesystem import Filesystem
from ..resources import ResourceRegistry


def build_resources(  # pragma: no cover - simple builder, tested transitively
    *,
    filesystem: Filesystem | None,
    budget_tracker: BudgetTracker | None,
) -> ResourceRegistry:
    """Build a ResourceRegistry with the given resources.

    Resources are keyed by their protocol type (e.g., Filesystem) rather than
    their concrete type (e.g., InMemoryFilesystem) to enable protocol-based
    lookup in tool handlers.
    """
    entries: dict[type[object], object] = {}
    if filesystem is not None:
        entries[Filesystem] = filesystem
    if budget_tracker is not None:
        entries[BudgetTracker] = budget_tracker
    if not entries:
        return ResourceRegistry()
    return ResourceRegistry.build(entries)


__all__ = [
    "build_resources",
]

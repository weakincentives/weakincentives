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

"""Protocols describing Session behavior exposed to other modules."""

from __future__ import annotations

from typing import Protocol

from .snapshots import Snapshot

type SnapshotProtocol = Snapshot


class SessionProtocol(Protocol):
    """Structural protocol implemented by session state containers."""

    def snapshot(self) -> SnapshotProtocol: ...

    def rollback(self, snapshot: SnapshotProtocol) -> None: ...

    def reset(self) -> None: ...


__all__ = ["SessionProtocol", "SnapshotProtocol"]

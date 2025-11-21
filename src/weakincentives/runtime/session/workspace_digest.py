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

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, cast

from ...prompt._normalization import normalize_component_key
from ...prompt._types import SupportsDataclass
from .reducers import upsert_by


class WorkspaceDigestHost(Protocol):
    def register_reducer(
        self,
        data_type: type[SupportsDataclass],
        reducer: object,
        *,
        slice_type: type[SupportsDataclass] | None = None,
    ) -> None: ...

    def select_all(
        self, slice_type: type[SupportsDataclass]
    ) -> tuple[SupportsDataclass, ...]: ...

    def seed_slice(
        self, slice_type: type[SupportsDataclass], values: Iterable[SupportsDataclass]
    ) -> None: ...


@dataclass(slots=True, frozen=True)
class WorkspaceDigest(SupportsDataclass):
    """Digest entry persisted within a :class:`Session` slice."""

    section_key: str
    body: str


def _digest_key(digest: WorkspaceDigest) -> str:
    return digest.section_key


class WorkspaceDigestSlice:
    """Manage workspace digest entries stored on a session."""

    def __init__(self, session: WorkspaceDigestHost) -> None:
        super().__init__()
        self._session = session
        session.register_reducer(WorkspaceDigest, upsert_by(_digest_key))
        if not session.select_all(WorkspaceDigest):
            session.seed_slice(WorkspaceDigest, ())

    def set(self, section_key: str, body: str) -> WorkspaceDigest:
        """Persist a digest for the provided section key."""

        normalized_key = normalize_component_key(section_key, owner="WorkspaceDigest")
        entry = WorkspaceDigest(section_key=normalized_key, body=body.strip())
        digests = cast(
            tuple[WorkspaceDigest, ...], self._session.select_all(WorkspaceDigest)
        )
        existing = [
            digest for digest in digests if digest.section_key != normalized_key
        ]
        existing.append(entry)
        self._session.seed_slice(WorkspaceDigest, tuple(existing))
        return entry

    def latest(self, section_key: str) -> WorkspaceDigest | None:
        """Return the most recent digest for ``section_key`` when present."""

        normalized_key = normalize_component_key(section_key, owner="WorkspaceDigest")
        digests = cast(
            tuple[WorkspaceDigest, ...], self._session.select_all(WorkspaceDigest)
        )
        for entry in reversed(digests):
            if entry.section_key == normalized_key:
                return entry
        return None


__all__ = ["WorkspaceDigest", "WorkspaceDigestHost", "WorkspaceDigestSlice"]

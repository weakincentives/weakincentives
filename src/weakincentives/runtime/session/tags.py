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

"""Utilities for validating and enriching runtime tags."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from uuid import UUID


def normalize_tags(
    tags: Mapping[object, object] | None,
    *,
    error_cls: type[Exception],
    session_id: UUID | str | None = None,
    parent_session_id: UUID | str | None = None,
) -> Mapping[str, str]:
    """Validate tag keys/values and inject session metadata when provided."""

    normalized: dict[str, str] = {}

    if tags is not None:
        for key, value in tags.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise error_cls("Tags must be string key/value pairs.")
            normalized[key] = value

    if session_id is not None:
        normalized["session_id"] = str(session_id)

    if parent_session_id is not None:
        normalized.setdefault("parent_session_id", str(parent_session_id))

    return MappingProxyType(normalized)

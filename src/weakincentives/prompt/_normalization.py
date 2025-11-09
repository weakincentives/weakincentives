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

"""Shared utilities for normalizing prompt component identifiers."""

from __future__ import annotations

import re
from typing import Final

COMPONENT_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9][a-z0-9._-]{0,63}$"
)


def normalize_component_key(key: str, *, owner: str) -> str:
    """Normalize component keys across prompt primitives.

    Args:
        key: The user supplied identifier that should be normalized.
        owner: Human-friendly label for the component requesting normalization.

    Returns:
        The sanitized key that downstream consumers may safely use.

    Raises:
        ValueError: When ``key`` is empty or fails to match
            :data:`COMPONENT_KEY_PATTERN`.
    """

    normalized = key.strip().lower()
    if not normalized:
        raise ValueError(f"{owner} key must be a non-empty string.")
    if not COMPONENT_KEY_PATTERN.match(normalized):
        raise ValueError(f"{owner} key must match {COMPONENT_KEY_PATTERN.pattern}.")
    return normalized


__all__ = ["COMPONENT_KEY_PATTERN", "normalize_component_key"]

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

"""Session and override utilities shared by runnable demos."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import cast

from weakincentives.runtime import Session

from .logging import attach_logging_subscribers

__all__ = [
    "build_logged_session",
    "resolve_override_tag",
]


def build_logged_session(
    *,
    parent: Session | None = None,
    tags: Mapping[str, str] | None = None,
) -> Session:
    """Create a session with logging subscribers attached.

    Args:
        parent: Optional parent session for hierarchical session trees.
        tags: Optional mapping of string tags to attach to the session.

    Returns:
        A new Session instance with logging subscribers attached to its dispatcher.
    """
    session_tags: dict[str, str] = {}
    if tags:
        session_tags.update(tags)

    session = Session(parent=parent, tags=cast(Mapping[object, object], session_tags))
    attach_logging_subscribers(session.dispatcher)
    return session


def resolve_override_tag(
    tag: str | None,
    *,
    env_var: str | None = None,
    default: str = "latest",
) -> str:
    """Resolve an override tag with optional environment variable fallback.

    Args:
        tag: Explicit tag value. If provided and non-empty, this takes precedence.
        env_var: Optional environment variable name to check as fallback.
        default: Default value when neither tag nor env_var provides a value.

    Returns:
        The resolved tag string.
    """
    if tag is not None:
        normalized = tag.strip()
        if normalized:
            return normalized

    if env_var is not None:
        env_candidate = os.getenv(env_var, "").strip()
        if env_candidate:
            return env_candidate

    return default

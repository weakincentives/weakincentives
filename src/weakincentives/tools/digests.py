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

"""Workspace digest helpers and section definition."""

from __future__ import annotations

import textwrap
from typing import override

from ..dataclasses import FrozenDataclass
from ..prompt._types import SupportsDataclass
from ..prompt._visibility import SectionVisibility
from ..prompt.section import Section
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.session import Session
from ..runtime.session.protocols import SessionProtocol


@FrozenDataclass()
class WorkspaceDigest(SupportsDataclass):
    """Digest entry persisted within a :class:`Session` slice.

    The digest has two versions:
    - ``summary``: A concise overview for use when overrides are active.
    - ``body``: A detailed version for full context when no override exists.
    """

    section_key: str
    summary: str
    body: str


def _normalized_key(section_key: str) -> str:
    from ..prompt._normalization import normalize_component_key

    return normalize_component_key(section_key, owner="WorkspaceDigest")


def set_workspace_digest(
    session: SessionProtocol,
    section_key: str,
    summary: str,
    body: str,
) -> WorkspaceDigest:
    """Persist the workspace digest for ``section_key``.

    Args:
        session: The session to store the digest in.
        section_key: The key identifying which digest section this belongs to.
        summary: A concise overview rendered when overrides are active.
        body: A detailed version rendered when no override exists.

    Returns:
        The persisted WorkspaceDigest entry.
    """
    normalized_key = _normalized_key(section_key)
    entry = WorkspaceDigest(
        section_key=normalized_key,
        summary=summary.strip(),
        body=body.strip(),
    )
    existing = tuple(
        digest
        for digest in session.query(WorkspaceDigest).all()
        if getattr(digest, "section_key", None) != normalized_key
    )
    session.mutate(WorkspaceDigest).seed((*existing, entry))
    return entry


def clear_workspace_digest(session: SessionProtocol, section_key: str) -> None:
    """Remove cached digests for ``section_key``."""

    normalized_key = _normalized_key(section_key)
    session.mutate(WorkspaceDigest).clear(
        lambda digest: getattr(digest, "section_key", None) == normalized_key
    )


def latest_workspace_digest(
    session: SessionProtocol,
    section_key: str,
) -> WorkspaceDigest | None:
    """Return the freshest digest for ``section_key`` when present."""

    normalized_key = _normalized_key(section_key)

    entries = session.query(WorkspaceDigest).all()
    for digest in reversed(entries):
        if getattr(digest, "section_key", None) == normalized_key:
            return digest
    return None


_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "tools.workspace_digest"}
)
_DEFAULT_PLACEHOLDER = textwrap.dedent(
    """
    Workspace digest unavailable. Explore the repository (README, docs, build and
    test commands) and rerun the optimize workflow to populate this section.
    """
).strip()


class WorkspaceDigestSection(Section[SupportsDataclass]):
    """Render a cached workspace digest sourced from the active session."""

    def __init__(
        self,
        *,
        session: Session,
        title: str = "Workspace Digest",
        key: str = "workspace-digest",
        placeholder: str = _DEFAULT_PLACEHOLDER,
    ) -> None:
        self._session = session
        self._placeholder = placeholder.strip()
        super().__init__(title=title, key=key, accepts_overrides=True)
        self.params_type = None
        self.param_type = None

    @override
    def original_body_template(self) -> str:
        return self._placeholder

    @override
    def render(
        self,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        *,
        path: tuple[str, ...] = (),
        visibility: SectionVisibility | None = None,
    ) -> str:
        del params, visibility
        body = self._resolve_body()
        return self._render_block(body, depth, number, path)

    def render_with_template(
        self,
        template_text: str,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        path: tuple[str, ...] = (),
    ) -> str:
        del params
        body = self._resolve_body(override_body=template_text)
        return self._render_block(body, depth, number, path)

    @override
    def clone(self, **kwargs: object) -> WorkspaceDigestSection:
        session_obj = kwargs.get("session")
        if not isinstance(session_obj, Session):
            msg = "session is required to clone WorkspaceDigestSection."
            raise TypeError(msg)
        return WorkspaceDigestSection(
            session=session_obj,
            title=self.title,
            key=self.key,
            placeholder=self._placeholder,
        )

    def _resolve_body(self, override_body: str | None = None) -> str:
        digest = latest_workspace_digest(self._session, self.key)
        has_override = override_body is not None and override_body.strip()

        if digest is not None:
            # When an override exists, render the summary for brevity.
            # Otherwise, render the full body for complete context.
            if has_override:
                if digest.summary.strip():
                    return digest.summary.strip()
            elif digest.body.strip():
                return digest.body.strip()
            # Fall back to summary if body is empty
            if digest.summary.strip():
                return digest.summary.strip()

        if has_override:
            return textwrap.dedent(override_body).strip()  # type: ignore[arg-type]
        _LOGGER.warning(
            "Workspace digest missing; returning placeholder.",
            event="workspace_digest_missing",
            context={"section_key": self.key},
        )
        return self._placeholder

    def _render_block(
        self, body: str, depth: int, number: str, path: tuple[str, ...] = ()
    ) -> str:
        heading_level = "#" * (depth + 2)
        normalized_number = number.rstrip(".")
        path_str = ".".join(path) if path else ""
        title_with_path = (
            f"{self.title.strip()} ({path_str})" if path_str else self.title.strip()
        )
        heading = f"{heading_level} {normalized_number}. {title_with_path}"
        if body:
            return f"{heading}\n\n{body.strip()}"
        return heading


__all__ = [
    "WorkspaceDigest",
    "WorkspaceDigestSection",
    "clear_workspace_digest",
    "latest_workspace_digest",
    "set_workspace_digest",
]

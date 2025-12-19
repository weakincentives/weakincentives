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

from ...dataclasses import FrozenDataclass
from ...prompt._visibility import SectionVisibility
from ...prompt.section import Section
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session import Session
from ...runtime.session.protocols import SessionProtocol
from ...types.dataclass import SupportsDataclass


@FrozenDataclass()
class WorkspaceDigest(SupportsDataclass):
    """Digest entry persisted within a :class:`Session` slice.

    The digest contains a short summary and the full body content. By default,
    sections render the summary, and models can use the ``read_section`` tool
    to access the full body when needed.
    """

    section_key: str
    summary: str
    body: str


def _normalized_key(section_key: str) -> str:
    from ...prompt._normalization import normalize_component_key

    return normalize_component_key(section_key, owner="WorkspaceDigest")


def set_workspace_digest(
    session: SessionProtocol,
    section_key: str,
    body: str,
    *,
    summary: str | None = None,
) -> WorkspaceDigest:
    """Persist the workspace digest for ``section_key``.

    Args:
        session: The session to persist the digest in.
        section_key: The key of the workspace digest section.
        body: The full digest content.
        summary: Optional short summary (1 paragraph). If not provided,
            a default summary is generated from the body.

    Returns:
        The persisted WorkspaceDigest entry.
    """
    normalized_key = _normalized_key(section_key)
    effective_summary = summary.strip() if summary else _default_summary(body)
    entry = WorkspaceDigest(
        section_key=normalized_key,
        summary=effective_summary,
        body=body.strip(),
    )
    existing = tuple(
        digest
        for digest in session[WorkspaceDigest].all()
        if getattr(digest, "section_key", None) != normalized_key
    )
    session[WorkspaceDigest].seed((*existing, entry))
    return entry


def _default_summary(body: str) -> str:
    """Generate a default summary from the body content."""
    return "Workspace digest available. Use read_section to view the full details."


def clear_workspace_digest(session: SessionProtocol, section_key: str) -> None:
    """Remove cached digests for ``section_key``."""

    normalized_key = _normalized_key(section_key)
    session[WorkspaceDigest].clear(
        lambda digest: getattr(digest, "section_key", None) == normalized_key
    )


def latest_workspace_digest(
    session: SessionProtocol,
    section_key: str,
) -> WorkspaceDigest | None:
    """Return the freshest digest for ``section_key`` when present."""

    normalized_key = _normalized_key(section_key)

    entries = session[WorkspaceDigest].all()
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
_DEFAULT_SUMMARY_PLACEHOLDER = (
    "Workspace digest unavailable. Run the optimize workflow to populate."
)


class WorkspaceDigestSection(Section[SupportsDataclass]):
    """Render a cached workspace digest sourced from the active session.

    By default, this section renders with SUMMARY visibility, showing a short
    summary of the workspace. Models can use the ``read_section`` tool to
    access the full digest content when needed.
    """

    def __init__(
        self,
        *,
        session: Session,
        title: str = "Workspace Digest",
        key: str = "workspace-digest",
        placeholder: str = _DEFAULT_PLACEHOLDER,
        summary_placeholder: str = _DEFAULT_SUMMARY_PLACEHOLDER,
    ) -> None:
        self._session = session
        self._placeholder = placeholder.strip()
        self._summary_placeholder = summary_placeholder.strip()
        # Dynamically compute the summary for the parent constructor
        digest = latest_workspace_digest(session, key)
        initial_summary = (
            digest.summary if digest is not None else self._summary_placeholder
        )
        super().__init__(
            title=title,
            key=key,
            accepts_overrides=True,
            summary=initial_summary,
            visibility=SectionVisibility.SUMMARY,
        )
        self.params_type = None

    @override
    def original_body_template(self) -> str:
        return self._placeholder

    @override
    def original_summary_template(self) -> str | None:
        return self._summary_placeholder

    @override
    def render_body(
        self,
        params: SupportsDataclass | None,
        *,
        visibility: SectionVisibility | None = None,
        path: tuple[str, ...] = (),
        session: object = None,
    ) -> str:
        del params, path, session
        effective = (
            visibility
            if visibility is not None
            else self.effective_visibility(override=None, params=None)
        )
        if effective == SectionVisibility.SUMMARY:
            return self._resolve_summary()
        return self._resolve_body()

    @override
    def render_override(
        self,
        override_body: str,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        path: tuple[str, ...] = (),
    ) -> str:
        del params
        heading = self.format_heading(depth, number, path)
        body = self._resolve_body(override_body=override_body)
        if body:
            return f"{heading}\n\n{body.strip()}"
        return heading

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
            summary_placeholder=self._summary_placeholder,
        )

    def _resolve_summary(self) -> str:
        """Resolve the summary content from session or placeholder."""
        digest = latest_workspace_digest(self._session, self.key)
        if digest is not None and digest.summary.strip():
            return digest.summary.strip()
        _LOGGER.warning(
            "Workspace digest missing; returning summary placeholder.",
            event="workspace_digest_summary_missing",
            context={"section_key": self.key},
        )
        return self._summary_placeholder

    def _resolve_body(self, override_body: str | None = None) -> str:
        """Resolve the full body content from session, override, or placeholder."""
        digest = latest_workspace_digest(self._session, self.key)
        if digest is not None and digest.body.strip():
            return digest.body.strip()
        if override_body is not None and override_body.strip():
            return textwrap.dedent(override_body).strip()
        _LOGGER.warning(
            "Workspace digest missing; returning placeholder.",
            event="workspace_digest_missing",
            context={"section_key": self.key},
        )
        return self._placeholder


__all__ = [
    "WorkspaceDigest",
    "WorkspaceDigestSection",
    "clear_workspace_digest",
    "latest_workspace_digest",
    "set_workspace_digest",
]

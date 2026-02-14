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
from dataclasses import field
from typing import override

from ...dataclasses import FrozenDataclass
from ...prompt import normalize_component_key
from ...prompt.section import Section, SectionVisibility
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

    section_key: str = field(metadata={"description": "Key identifying the section."})
    summary: str = field(metadata={"description": "Short summary of the digest."})
    body: str = field(metadata={"description": "Full body content of the digest."})


def _normalized_key(section_key: str) -> str:
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


class WorkspaceDigestSection(Section[SupportsDataclass]):
    """Render a cached workspace digest sourced from the active session.

    This section renders content from a WorkspaceDigest stored in session state.
    The visibility is determined dynamically based on whether a digest exists:

    - When a digest exists: SUMMARY visibility shows the digest summary, FULL
      visibility shows the complete digest body
    - When no digest exists: FULL visibility shows a placeholder message

    Models can use the ``read_section`` tool to access the full digest content
    when the section is rendered with SUMMARY visibility.
    """

    params_type: type[SupportsDataclass] | None

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
        self._key = key  # Store key before super().__init__ for _get_current_summary
        super().__init__(
            title=title,
            key=key,
            accepts_overrides=True,
            # Summary and visibility are computed dynamically via property/selector
            summary=self._get_current_summary(),
            visibility=self._visibility_selector,
        )
        self.params_type = None

    def _visibility_selector(
        self, params: SupportsDataclass | None = None
    ) -> SectionVisibility:
        """Return SUMMARY if digest exists, FULL otherwise."""
        del params  # Not used - visibility based on session state
        digest = latest_workspace_digest(self._session, self._key)
        if digest is not None:
            return SectionVisibility.SUMMARY
        return SectionVisibility.FULL

    def _get_current_summary(self) -> str | None:
        """Return the current summary if a digest exists, None otherwise."""
        digest = latest_workspace_digest(self._session, self._key)
        if digest is not None:
            return digest.summary
        return None

    @property
    def summary(  # pyright: ignore[reportImplicitOverride]
        self,
    ) -> str | None:
        """Return the current summary from session state, or None if no digest."""
        return self._get_current_summary()

    @summary.setter
    def summary(self, value: str | None) -> None:
        """Ignore setter - summary is computed dynamically from session state."""
        # Parent's __init__ sets self.summary, but we compute it dynamically
        pass

    def _get_content_for_visibility(self, visibility: SectionVisibility | None) -> str:
        """Return appropriate content based on visibility.

        This is the central method for determining what content to render.
        Both render_body and render_override delegate to this method.

        Args:
            visibility: The effective visibility to use. If None, computes
                effective visibility from the section's selector.

        Returns:
            The content to render: summary for SUMMARY visibility,
            full body for FULL visibility, or placeholder if no digest.
        """
        digest = latest_workspace_digest(self._session, self._key)

        # If no digest exists, always return the placeholder
        if digest is None:
            _LOGGER.warning(
                "Workspace digest missing; returning placeholder.",
                event="workspace_digest_missing",
                context={"section_key": self._key},
            )
            return self._placeholder

        # Digest exists - determine effective visibility
        effective = (
            visibility
            if visibility is not None
            else self.effective_visibility(override=None, params=None)
        )

        if effective == SectionVisibility.SUMMARY:
            return digest.summary.strip()
        return digest.body.strip()

    @override
    def original_body_template(self) -> str:
        return self._placeholder

    @override
    def original_summary_template(self) -> str | None:
        # Return a placeholder for hashing purposes; actual summary is dynamic
        return "Workspace digest summary."

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
        return self._get_content_for_visibility(visibility)

    @override
    def render_override(
        self,
        override_body: str,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        path: tuple[str, ...] = (),
    ) -> str:
        """Render the section with an override body.

        When a digest exists in session state, the content is sourced from the
        session digest (respecting visibility - summary for SUMMARY, full body
        for FULL). The override_body is ignored in this case.

        When NO digest exists, the override_body is used as fallback content,
        allowing pre-seeded overrides to populate the section before the first
        optimization run.
        """
        del params
        heading = self.format_heading(depth, number, path)

        digest = latest_workspace_digest(self._session, self._key)
        if digest is not None:
            # Digest exists - use session content, respecting visibility
            effective = self.effective_visibility(
                override=None, params=None, session=self._session, path=path
            )
            if effective == SectionVisibility.SUMMARY:
                body = digest.summary.strip()
            else:
                body = digest.body.strip()
        elif override_body and override_body.strip():
            # No digest - use override body as fallback
            body = textwrap.dedent(override_body).strip()
        else:
            # No digest, no override - use placeholder
            body = self._placeholder

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
        )


__all__ = [
    "WorkspaceDigest",
    "WorkspaceDigestSection",
    "clear_workspace_digest",
    "latest_workspace_digest",
    "set_workspace_digest",
]

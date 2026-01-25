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
from ...prompt._normalization import normalize_component_key
from ...prompt._visibility import SectionVisibility
from ...prompt.section import Section
from ...runtime.logging import StructuredLogger, get_logger
from ...runtime.session import Session
from ...runtime.session.protocols import SessionProtocol
from ...types.dataclass import SupportsDataclass


@FrozenDataclass()
class WorkspaceDigest(SupportsDataclass):
    """Immutable digest entry persisted within a Session slice.

    A workspace digest captures documentation about a workspace (repository,
    project, etc.) in two forms: a short summary for quick context in prompts,
    and a full body for detailed reference via the ``read_section`` tool.

    Attributes:
        section_key: Normalized key identifying the section this digest belongs
            to. Must match pattern ``^[a-z0-9][a-z0-9._-]{0,63}$``.
        summary: Short summary (typically 1 paragraph) shown by default when
            the section renders with SUMMARY visibility.
        body: Full digest content accessible via ``read_section`` tool or when
            rendered with FULL visibility.

    Example:
        >>> digest = WorkspaceDigest(
        ...     section_key="workspace-digest",
        ...     summary="Python library for prompt composition.",
        ...     body="Full documentation including setup, usage, and API reference.",
        ... )
    """

    section_key: str
    summary: str
    body: str


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
    """Remove all cached digests for the specified section key.

    Clears any WorkspaceDigest entries matching the given section key from
    session state. Safe to call even if no digest exists for the key.

    Args:
        session: The session containing the digest slice to clear from.
        section_key: The key identifying which digest(s) to remove. Will be
            normalized before matching against stored entries.

    Example:
        >>> clear_workspace_digest(session, "workspace-digest")
    """
    normalized_key = _normalized_key(section_key)
    session[WorkspaceDigest].clear(
        lambda digest: getattr(digest, "section_key", None) == normalized_key
    )


def latest_workspace_digest(
    session: SessionProtocol,
    section_key: str,
) -> WorkspaceDigest | None:
    """Retrieve the most recent workspace digest for a given section key.

    Searches session state for WorkspaceDigest entries matching the specified
    key and returns the most recently added one. Useful for checking if a
    digest exists before rendering or for accessing digest content directly.

    Args:
        session: The session to search for digest entries.
        section_key: The key identifying the digest to retrieve. Will be
            normalized before matching against stored entries.

    Returns:
        The most recent WorkspaceDigest for the key, or None if no digest
        exists for that key in the session.

    Example:
        >>> digest = latest_workspace_digest(session, "workspace-digest")
        >>> if digest:
        ...     print(digest.summary)
    """
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
    """Section that renders workspace digest content from session state.

    This section dynamically sources its content from a WorkspaceDigest stored
    in the session, providing a two-tier content model:

    - **SUMMARY visibility**: Shows the digest's short summary, suitable for
      including in prompts without overwhelming context.
    - **FULL visibility**: Shows the complete digest body with all details.

    When no digest exists in session state, a configurable placeholder message
    is shown instead.

    The section integrates with the ``read_section`` tool, allowing models to
    request the full digest body when only the summary is rendered initially.

    Example:
        >>> from weakincentives.runtime.session import Session
        >>> session = Session()
        >>> section = WorkspaceDigestSection(session=session, title="Project Info")
        >>> # Initially renders placeholder until digest is set
        >>> set_workspace_digest(
        ...     session,
        ...     section.key,
        ...     body="Full project documentation...",
        ...     summary="A Python library for X.",
        ... )
        >>> # Now renders the digest content
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
        """Initialize a workspace digest section.

        Args:
            session: The session instance to read digest content from. The
                section will look up WorkspaceDigest entries by key from this
                session's state.
            title: Display title for the section heading. Defaults to
                "Workspace Digest".
            key: Unique section key used to identify and look up the digest.
                Must match pattern ``^[a-z0-9][a-z0-9._-]{0,63}$``. Defaults
                to "workspace-digest".
            placeholder: Message shown when no digest exists in session state.
                Defaults to a prompt suggesting the user explore the repository
                and run optimization.
        """
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
        """The current digest summary, or None if no digest exists.

        This property is computed dynamically from session state on each access,
        ensuring it always reflects the latest digest content. Returns None when
        no WorkspaceDigest exists for this section's key.
        """
        return self._get_current_summary()

    @summary.setter
    def summary(self, value: str | None) -> None:
        """No-op setter; summary is computed dynamically from session state."""
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
        """Return the placeholder text as the original body template.

        Used for template hashing and comparison. Returns the configured
        placeholder since actual content is sourced dynamically from session.
        """
        return self._placeholder

    @override
    def original_summary_template(self) -> str | None:
        """Return a static placeholder string for template hashing.

        Returns a fixed string rather than the dynamic summary, since the
        actual summary is computed from session state at render time.
        """
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
        """Render the section body content based on visibility level.

        Returns content from the session digest when available, respecting the
        visibility setting (summary for SUMMARY, full body for FULL). Returns
        the placeholder message when no digest exists.

        Args:
            params: Unused; included for API compatibility.
            visibility: Visibility level controlling content detail. If None,
                effective visibility is computed from the section's selector.
            path: Unused; included for API compatibility.
            session: Unused; included for API compatibility.

        Returns:
            The rendered body content: digest summary, full body, or placeholder.
        """
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
        """Render the section with heading and content, using override as fallback.

        Content priority:
        1. Session digest (if exists) - visibility determines summary vs full body
        2. Override body (if no digest but override provided)
        3. Placeholder message (if neither digest nor override exists)

        This allows pre-seeded overrides to populate the section before the first
        optimization run populates the digest.

        Args:
            override_body: Fallback content used when no digest exists in session.
            params: Unused; included for API compatibility.
            depth: Heading depth level (e.g., 1 for ``#``, 2 for ``##``).
            number: Section number string (e.g., "1.2.3").
            path: Section path for context in visibility resolution.

        Returns:
            Formatted section with heading and body content.
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
        """Create a copy of this section bound to a different session.

        Creates a new WorkspaceDigestSection with the same title, key, and
        placeholder, but bound to a new session instance. This is useful when
        forking sessions or creating isolated section instances.

        Args:
            **kwargs: Must include ``session`` with a Session instance.

        Returns:
            A new WorkspaceDigestSection bound to the provided session.

        Raises:
            TypeError: If ``session`` is not provided or is not a Session instance.
        """
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

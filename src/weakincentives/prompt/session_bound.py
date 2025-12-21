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

"""Session-bound section base classes with simplified cloning.

This module provides base classes that reduce the "subclass tax" for creating
tool sections that depend on a session. Instead of implementing the full
clone() method in each subclass, sections only need to implement a simpler
_rebind_to_session() method.

Example::

    class MyToolSection(SessionBoundMarkdownSection[MyParams]):
        def __init__(self, *, session: Session, config: MyConfig) -> None:
            self._session = session
            self._config = config
            # ... rest of initialization

        def _rebind_to_session(
            self, session: Session, **kwargs: object
        ) -> MyToolSection:
            return MyToolSection(session=session, config=self._config)

The base class handles:
- Session validation in clone()
- Bus consistency validation
- Calling _rebind_to_session() with the validated session
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar, override

from ..runtime.session import Session
from ..types.dataclass import SupportsDataclass
from .markdown import MarkdownSection
from .section import Section

SessionBoundParamsT = TypeVar(
    "SessionBoundParamsT",
    bound=SupportsDataclass,
    covariant=True,
)


def validate_clone_session(
    kwargs: dict[str, object],
    section_class_name: str,
) -> Session:
    """Validate and extract session from clone kwargs.

    Args:
        kwargs: The kwargs passed to clone().
        section_class_name: The name of the section class (for error messages).

    Returns:
        The validated Session instance.

    Raises:
        TypeError: If session is missing or not a Session instance.
        TypeError: If bus is provided but doesn't match session.event_bus.
    """
    from ..runtime.session import Session

    session = kwargs.get("session")
    if not isinstance(session, Session):
        msg = f"session is required to clone {section_class_name}."
        raise TypeError(msg)
    provided_bus = kwargs.get("bus")
    if provided_bus is not None and provided_bus is not session.event_bus:
        msg = "Provided bus must match the target session's event bus."
        raise TypeError(msg)
    return session


class SessionBoundMarkdownSection(MarkdownSection[SessionBoundParamsT]):
    """MarkdownSection subclass with session binding and simplified cloning.

    This base class reduces boilerplate for sections that need a session
    reference and follow the "session + config" pattern. Instead of
    implementing clone(), subclasses implement the simpler _rebind_to_session().

    Subclasses must:
    1. Store session as ``self._session``
    2. Store config as ``self._config`` (any FrozenDataclass)
    3. Implement ``_rebind_to_session()`` to create a new instance

    Example::

        class MyToolSection(SessionBoundMarkdownSection[MyParams]):
            def __init__(self, *, session: Session, config: MyConfig) -> None:
                self._session = session
                self._config = config
                super().__init__(...)

            @property
            def session(self) -> Session:
                return self._session

            def _rebind_to_session(
                self, session: Session, **kwargs: object
            ) -> MyToolSection:
                return MyToolSection(session=session, config=self._config)
    """

    _session: Session

    @property
    def session(self) -> Session:
        """Return the session this section is bound to."""
        return self._session

    @abstractmethod
    def _rebind_to_session(
        self,
        session: Session,
        **kwargs: object,
    ) -> SessionBoundMarkdownSection[SessionBoundParamsT]:
        """Create a new instance bound to the given session.

        This method is called by clone() after validating the session.
        Subclasses should:
        - Pass ``session=session``
        - Pass ``config=self._config``
        - Extract any shared resources from kwargs (e.g., filesystem)

        Args:
            session: The validated Session instance to bind to.
            **kwargs: Additional clone arguments (may contain shared resources).

        Returns:
            A new section instance bound to the given session.
        """

    @override
    def clone(
        self,
        **kwargs: object,
    ) -> SessionBoundMarkdownSection[SessionBoundParamsT]:
        """Clone this section to a new session.

        Validates that a session is provided in kwargs, then delegates
        to _rebind_to_session() for the actual cloning logic.

        Args:
            **kwargs: Must include 'session'. May include 'bus' for validation
                and other resources like 'filesystem'.

        Returns:
            A new section instance bound to the provided session.

        Raises:
            TypeError: If session is missing or not a Session instance.
            TypeError: If bus is provided but doesn't match session.event_bus.
        """
        session = validate_clone_session(kwargs, type(self).__name__)
        # Remove session from kwargs to avoid duplicate argument
        remaining_kwargs = {k: v for k, v in kwargs.items() if k != "session"}
        return self._rebind_to_session(session, **remaining_kwargs)


class SessionBoundSection(Section[SessionBoundParamsT]):
    """Section subclass with session binding and simplified cloning.

    Similar to SessionBoundMarkdownSection, but for sections that don't
    use MarkdownSection as a base. Use this for custom section implementations
    that still need session binding.

    Subclasses must:
    1. Store session as ``self._session``
    2. Implement ``_rebind_to_session()`` to create a new instance

    Example::

        class MySection(SessionBoundSection[MyParams]):
            def __init__(self, *, session: Session, title: str) -> None:
                self._session = session
                self._title = title
                super().__init__(title=title, ...)

            @property
            def session(self) -> Session:
                return self._session

            def _rebind_to_session(
                self, session: Session, **kwargs: object
            ) -> MySection:
                return MySection(session=session, title=self._title)
    """

    _session: Session

    @property
    def session(self) -> Session:
        """Return the session this section is bound to."""
        return self._session

    @abstractmethod
    def _rebind_to_session(
        self,
        session: Session,
        **kwargs: object,
    ) -> SessionBoundSection[SessionBoundParamsT]:
        """Create a new instance bound to the given session.

        This method is called by clone() after validating the session.
        Subclasses should construct a new instance with the given session
        and any stored configuration.

        Args:
            session: The validated Session instance to bind to.
            **kwargs: Additional clone arguments.

        Returns:
            A new section instance bound to the given session.
        """

    @override
    def clone(
        self,
        **kwargs: object,
    ) -> SessionBoundSection[SessionBoundParamsT]:
        """Clone this section to a new session.

        Validates that a session is provided in kwargs, then delegates
        to _rebind_to_session() for the actual cloning logic.

        Args:
            **kwargs: Must include 'session'. May include 'bus' for validation.

        Returns:
            A new section instance bound to the provided session.

        Raises:
            TypeError: If session is missing or not a Session instance.
            TypeError: If bus is provided but doesn't match session.event_bus.
        """
        session = validate_clone_session(kwargs, type(self).__name__)
        # Remove session from kwargs to avoid duplicate argument
        remaining_kwargs = {k: v for k, v in kwargs.items() if k != "session"}
        return self._rebind_to_session(session, **remaining_kwargs)


__all__ = [
    "SessionBoundMarkdownSection",
    "SessionBoundSection",
    "validate_clone_session",
]

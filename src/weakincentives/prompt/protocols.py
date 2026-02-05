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

"""Structural typing primitives shared across prompt tooling.

This module provides protocols for prompt-related types:

- **PromptProtocol**: Interface for bound prompts
- **PromptTemplateProtocol**: Interface for prompt templates
- **ProviderAdapterProtocol**: Interface for adapters
- **RenderedPromptProtocol**: Interface for rendered prompt snapshots
- **ToolSuiteSection**: Protocol for sections exposing tool suites
- **WorkspaceSection**: Protocol for workspace sections managing a filesystem
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, override, runtime_checkable

from ..deadlines import Deadline
from ._overrides_protocols import PromptOverridesStore
from ._types import SupportsDataclass

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..budget import Budget, BudgetTracker
    from ..filesystem import Filesystem
    from ..resources import ResourceRegistry
    from ..runtime.session import Session
    from ..runtime.session.protocols import SessionProtocol
    from ._prompt_resources import PromptResources
    from ._structured_output_config import StructuredOutputConfig
    from .overrides import PromptDescriptor


class PromptResponseProtocol[AdapterOutputT](Protocol):
    prompt_name: str
    text: str | None
    output: AdapterOutputT | None


class RenderedPromptProtocol[RenderedOutputT](Protocol):
    """Interface satisfied by rendered prompt snapshots."""

    @property
    def text(self) -> str: ...

    @property
    def output_type(self) -> type[Any] | None: ...

    @property
    def container(self) -> Literal["object", "array"] | None: ...

    @property
    def allow_extra_keys(self) -> bool | None: ...

    @property
    def deadline(self) -> Deadline | None: ...

    @property
    def tools(self) -> tuple[object, ...]: ...

    @property
    def tool_param_descriptions(self) -> Mapping[str, Mapping[str, str]]: ...

    @property
    def descriptor(self) -> PromptDescriptor | None: ...

    @property
    def structured_output(
        self,
    ) -> StructuredOutputConfig[SupportsDataclass] | None: ...


class PromptTemplateProtocol[TemplateOutputT](Protocol):
    """Interface describing the subset of prompt template state exposed to tools.

    Note: PromptTemplate does NOT have a render() method. Rendering is performed
    by creating a Prompt wrapper around the template and calling Prompt.render().
    """

    ns: str
    key: str
    name: str | None

    @property
    def sections(self) -> tuple[Any, ...]: ...

    @property
    def structured_output(
        self,
    ) -> StructuredOutputConfig[SupportsDataclass] | None: ...


class PromptProtocol[PromptOutputT](Protocol):
    """Interface describing the bound prompt wrapper used at runtime.

    Resource lifecycle is managed via ``prompt.resources``:
    - Use ``with prompt.resources:`` to manage lifecycle
    - Access resources via ``prompt.resources.get(Protocol)``
    """

    template: PromptTemplateProtocol[PromptOutputT]
    overrides_store: PromptOverridesStore | None
    overrides_tag: str

    ns: str
    key: str
    name: str | None

    @property
    def sections(self) -> tuple[Any, ...]: ...

    @property
    def params(self) -> tuple[SupportsDataclass, ...]: ...

    @property
    def structured_output(
        self,
    ) -> StructuredOutputConfig[SupportsDataclass] | None: ...

    @property
    def resources(self) -> PromptResources:
        """Resource accessor for lifecycle management and dependency resolution."""
        ...

    @property
    def feedback_providers(self) -> tuple[Any, ...]:
        """Return feedback providers configured on this prompt."""
        ...

    def bind(
        self,
        *params: SupportsDataclass,
        resources: ResourceRegistry | None = None,
    ) -> PromptProtocol[PromptOutputT]: ...

    def render(self) -> RenderedPromptProtocol[PromptOutputT]: ...


class ProviderAdapterProtocol[AdapterOutputT](Protocol):
    """Interface describing the subset of adapter behaviour required by tools.

    Telemetry is dispatched via session.dispatcher. Resources are accessed
    via the prompt's resource context (prompt.resources).
    """

    def evaluate(
        self,
        prompt: PromptProtocol[AdapterOutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> PromptResponseProtocol[AdapterOutputT]: ...


@runtime_checkable
class ToolSuiteSection(Protocol):
    """Protocol for sections that expose tool suites.

    Tool suite sections should implement this protocol to enable consistent
    handling of session binding and cloning across all capability sections.

    The protocol requires:

    - **session**: Property returning the associated Session
    - **accepts_overrides**: Property indicating if overrides are accepted
    - **clone()**: Method for creating copies with a new session

    Example::

        class MyToolSection:
            def __init__(self, session: Session) -> None:
                self._session = session

            @property
            def session(self) -> Session:
                return self._session

            @property
            def accepts_overrides(self) -> bool:
                return False

            def clone(self, **kwargs) -> MyToolSection:
                return MyToolSection(session=kwargs["session"])
    """

    @property
    def session(self) -> Session:
        """Return the session associated with this tool suite section."""
        ...

    @property
    def accepts_overrides(self) -> bool:
        """Return True if this section accepts parameter overrides."""
        ...

    def clone(self, **kwargs: object) -> ToolSuiteSection:
        """Clone the section with new session.

        Args:
            **kwargs: Must include ``session`` with the new Session instance.

        Returns:
            A new section instance bound to the provided session.
        """
        ...


@runtime_checkable
class WorkspaceSection(ToolSuiteSection, Protocol):
    """Protocol for workspace sections that manage a filesystem.

    Extends ToolSuiteSection with a filesystem property. Workspace sections
    (e.g., ClaudeAgentWorkspaceSection) should implement this protocol.
    This enables tools and optimizers to identify valid workspace sections
    without importing adapter-specific code.

    Additional requirement beyond ToolSuiteSection:

    - **filesystem**: Property returning the Filesystem managed by this section
    """

    @property
    def filesystem(self) -> Filesystem:
        """Return the filesystem managed by this workspace section."""
        ...

    @override
    def clone(self, **kwargs: object) -> WorkspaceSection:
        """Clone the section with new session/dispatcher."""
        ...


__all__ = [
    "PromptProtocol",
    "PromptResponseProtocol",
    "PromptTemplateProtocol",
    "ProviderAdapterProtocol",
    "RenderedPromptProtocol",
    "ToolSuiteSection",
    "WorkspaceSection",
]

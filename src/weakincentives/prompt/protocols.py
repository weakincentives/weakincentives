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
    """Response returned by a provider adapter after evaluating a prompt.

    Contains the raw text response and optionally a parsed structured output.

    Attributes:
        prompt_name: Fully qualified name of the evaluated prompt (ns.key format).
        text: Raw text response from the model, or None if parsing failed.
        output: Parsed structured output matching the prompt's output type,
            or None if no structured output was requested or parsing failed.
    """

    prompt_name: str
    text: str | None
    output: AdapterOutputT | None


class RenderedPromptProtocol[RenderedOutputT](Protocol):
    """Immutable snapshot of a prompt after rendering all sections.

    Created by calling ``prompt.render()`` on a bound prompt. Contains all
    information needed by a provider adapter to execute the prompt, including
    the fully rendered text, tool definitions, and output configuration.

    This snapshot is immutable and captures the prompt state at render time,
    making it safe to pass to adapters or store for later analysis.
    """

    @property
    def text(self) -> str:
        """Return the fully rendered prompt text with all sections expanded."""
        ...

    @property
    def output_type(self) -> type[Any] | None:
        """Return the expected output type for structured responses, or None."""
        ...

    @property
    def container(self) -> Literal["object", "array"] | None:
        """Return the container type for structured output parsing.

        Returns "object" for single object responses, "array" for list responses,
        or None if no structured output is configured.
        """
        ...

    @property
    def allow_extra_keys(self) -> bool | None:
        """Return whether extra keys are allowed in structured output parsing."""
        ...

    @property
    def deadline(self) -> Deadline | None:
        """Return the deadline for prompt execution, or None for no timeout."""
        ...

    @property
    def tools(self) -> tuple[object, ...]:
        """Return all tools available to this prompt from all sections."""
        ...

    @property
    def tool_param_descriptions(self) -> Mapping[str, Mapping[str, str]]:
        """Return parameter descriptions for tools, keyed by tool name then param."""
        ...

    @property
    def descriptor(self) -> PromptDescriptor | None:
        """Return the prompt descriptor for override resolution, or None."""
        ...

    @property
    def structured_output(
        self,
    ) -> StructuredOutputConfig[SupportsDataclass] | None:
        """Return structured output configuration, or None if not configured."""
        ...


class PromptTemplateProtocol[TemplateOutputT](Protocol):
    """Interface for prompt templates that define reusable prompt structures.

    A prompt template is a blueprint for creating prompts. It defines sections,
    namespace, key, and optionally structured output configuration. Templates
    are not directly renderable - wrap with ``Prompt`` and call ``render()``.

    Attributes:
        ns: Namespace for the prompt (e.g., "myapp.prompts").
        key: Unique key within the namespace (e.g., "summarize").
        name: Optional human-readable name for display/logging.

    Example::

        template = PromptTemplate(
            ns="myapp",
            key="greet",
            sections=(SystemSection("You are helpful."),),
        )
        prompt = Prompt(template).bind(UserParams(name="Alice"))
        rendered = prompt.render()
    """

    ns: str
    key: str
    name: str | None

    @property
    def sections(self) -> tuple[Any, ...]:
        """Return all sections that comprise this template."""
        ...

    @property
    def structured_output(
        self,
    ) -> StructuredOutputConfig[SupportsDataclass] | None:
        """Return structured output configuration, or None if not configured."""
        ...


class PromptProtocol[PromptOutputT](Protocol):
    """Interface for bound prompts ready for rendering and execution.

    A bound prompt wraps a template with parameters and manages resource lifecycle.
    This is the main interface used at runtime for prompt execution.

    Resource lifecycle is managed via ``prompt.resources``:

    - Use ``with prompt.resources:`` to manage lifecycle
    - Access resources via ``prompt.resources.get(Protocol)``

    Attributes:
        template: The underlying prompt template.
        overrides_store: Store for prompt overrides, or None if disabled.
        overrides_tag: Tag for selecting override variants (e.g., "default").
        ns: Namespace inherited from template.
        key: Key inherited from template.
        name: Optional display name inherited from template.

    Example::

        prompt = Prompt(template).bind(params)
        with prompt.resources:
            rendered = prompt.render()
            response = adapter.evaluate(prompt, session=session)
    """

    template: PromptTemplateProtocol[PromptOutputT]
    overrides_store: PromptOverridesStore | None
    overrides_tag: str

    ns: str
    key: str
    name: str | None

    @property
    def sections(self) -> tuple[Any, ...]:
        """Return all sections from the underlying template."""
        ...

    @property
    def params(self) -> tuple[SupportsDataclass, ...]:
        """Return all parameter objects bound to this prompt."""
        ...

    @property
    def structured_output(
        self,
    ) -> StructuredOutputConfig[SupportsDataclass] | None:
        """Return structured output configuration, or None if not configured."""
        ...

    @property
    def resources(self) -> PromptResources:
        """Return the resource accessor for lifecycle management.

        Use as a context manager to ensure proper resource cleanup::

            with prompt.resources:
                db = prompt.resources.get(DatabaseProtocol)
                # ... use resources ...
        """
        ...

    @property
    def feedback_providers(self) -> tuple[Any, ...]:
        """Return feedback providers configured on this prompt."""
        ...

    def bind(
        self,
        *params: SupportsDataclass,
        resources: ResourceRegistry | None = None,
    ) -> PromptProtocol[PromptOutputT]:
        """Create a new prompt with additional parameters bound.

        Parameters can be bound incrementally across multiple calls. Each call
        returns a new prompt instance; the original is unchanged.

        Args:
            *params: Dataclass instances to bind as section parameters.
            resources: Optional resource registry to use for dependency injection.

        Returns:
            A new prompt with the additional parameters bound.
        """
        ...

    def render(self) -> RenderedPromptProtocol[PromptOutputT]:
        """Render the prompt into an immutable snapshot ready for execution.

        Expands all sections with bound parameters and collects tools.
        The rendered prompt can be passed to a provider adapter for execution.

        Returns:
            An immutable snapshot containing rendered text, tools, and config.

        Raises:
            ValueError: If required parameters are missing.
        """
        ...


class ProviderAdapterProtocol[AdapterOutputT](Protocol):
    """Interface for provider adapters that execute prompts against LLM backends.

    Provider adapters translate prompts into provider-specific API calls and
    handle response parsing. Implementations exist for OpenAI, Anthropic,
    LiteLLM, and other providers.

    Telemetry events are dispatched via session.dispatcher. Resources needed
    during execution are accessed via the prompt's resource context.

    Example::

        adapter = OpenAIAdapter(model="gpt-4")
        response = adapter.evaluate(prompt, session=session)
        if response.output is not None:
            print(response.output)  # Parsed structured output
    """

    def evaluate(
        self,
        prompt: PromptProtocol[AdapterOutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> PromptResponseProtocol[AdapterOutputT]:
        """Execute the prompt and return the model response.

        Renders the prompt, sends it to the provider, and parses the response.
        Tool calls are executed automatically if tools are configured.

        Args:
            prompt: Bound prompt to evaluate.
            session: Session for dispatching events and accessing state.
            deadline: Optional deadline for the entire operation.
            budget: Optional token/cost budget limits.
            budget_tracker: Optional tracker for budget consumption.

        Returns:
            Response containing raw text and optionally parsed structured output.

        Raises:
            DeadlineExceeded: If the deadline is reached before completion.
            BudgetExhausted: If budget limits are exceeded.
        """
        ...


@runtime_checkable
class ToolSuiteSection(Protocol):
    """Protocol for sections that expose tool suites.

    All tool suite sections (VfsToolsSection, AstevalSection,
    PlanningToolsSection, PodmanSandboxSection, WorkspaceDigestSection)
    should implement this protocol. This enables consistent handling
    of session binding and cloning across all capability sections.

    The protocol requires:

    - **session**: Property returning the associated Session
    - **accepts_overrides**: Property indicating if overrides are accepted
    - **clone()**: Method for creating copies with a new session

    Example::

        from weakincentives.contrib.tools import VfsToolsSection, VfsConfig

        vfs = VfsToolsSection(session=session)
        assert vfs.accepts_overrides is False  # Default
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

    Extends ToolSuiteSection with a filesystem property. All workspace
    sections (VfsToolsSection, PodmanSandboxSection, ClaudeAgentWorkspaceSection)
    should implement this protocol. This enables the WorkspaceDigestOptimizer
    to identify valid workspace sections without importing adapter-specific code.

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

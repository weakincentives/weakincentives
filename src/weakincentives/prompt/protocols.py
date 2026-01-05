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

"""Structural typing primitives shared across prompt tooling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ..deadlines import Deadline
from ._overrides_protocols import PromptOverridesStore
from ._types import SupportsDataclass

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..budget import Budget, BudgetTracker
    from ..resources import ResourceRegistry
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


__all__ = [
    "PromptProtocol",
    "PromptResponseProtocol",
    "PromptTemplateProtocol",
    "ProviderAdapterProtocol",
    "RenderedPromptProtocol",
]

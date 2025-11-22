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
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

from ..deadlines import Deadline
from ._overrides_protocols import PromptOverridesStoreProtocol
from ._types import SupportsDataclass

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..runtime.events._types import EventBus
    from ..runtime.session.protocols import SessionProtocol
    from ._structured_output_config import StructuredOutputConfig
    from .overrides import PromptDescriptor

PromptOutputT = TypeVar("PromptOutputT", covariant=True)
RenderedOutputT = TypeVar("RenderedOutputT", covariant=True)
AdapterOutputT = TypeVar("AdapterOutputT")


class PromptResponseProtocol(Protocol[AdapterOutputT]):
    prompt_name: str
    text: str | None
    output: AdapterOutputT | None
    tool_results: tuple[object, ...]
    provider_payload: Mapping[str, Any] | None


class RenderedPromptProtocol(Protocol[RenderedOutputT]):
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


class PromptProtocol(Protocol[PromptOutputT]):
    """Interface describing the subset of Prompt state exposed to tools."""

    ns: str
    key: str
    name: str | None

    def render(
        self,
        *params: SupportsDataclass,
        overrides_store: PromptOverridesStoreProtocol | None = None,
        tag: str = "latest",
        inject_output_instructions: bool | None = None,
    ) -> RenderedPromptProtocol[PromptOutputT]: ...


class ProviderAdapterProtocol(Protocol[AdapterOutputT]):
    """Interface describing the subset of adapter behaviour required by tools."""

    def evaluate(
        self,
        prompt: PromptProtocol[AdapterOutputT],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBus,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        overrides_store: PromptOverridesStoreProtocol | None = None,
        overrides_tag: str = "latest",
    ) -> PromptResponseProtocol[AdapterOutputT]: ...


__all__ = [
    "PromptProtocol",
    "PromptResponseProtocol",
    "ProviderAdapterProtocol",
    "RenderedPromptProtocol",
]

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

"""Optional OpenAI adapter utilities."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import timedelta
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from ..deadlines import Deadline
from ..prompt import MarkdownSection
from ..prompt._types import SupportsDataclass
from ..prompt.overrides import PromptLike
from ..prompt.prompt import Prompt
from ..prompt.section import Section
from ..prompt.workspace_digest import WorkspaceDigestSection
from ..runtime.events import EventBus
from ..runtime.logging import StructuredLogger, get_logger
from ..tools.podman import PodmanSandboxSection
from ..tools.vfs import VfsToolsSection
from . import shared as _shared
from ._provider_protocols import ProviderChoice, ProviderCompletionResponse
from ._tool_messages import serialize_tool_message
from .core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_RESPONSE,
    OptimizationResult,
    OptimizationScope,
    PromptEvaluationError,
    PromptResponse,
    SessionProtocol,
)
from .shared import (
    OPENAI_ADAPTER_NAME,
    ToolChoice,
    build_json_schema_response_format,
    deadline_provider_payload,
    first_choice,
    format_publish_failures,
    parse_tool_arguments,
    run_conversation,
)

if TYPE_CHECKING:
    from ..adapters.core import ProviderAdapter
    from ..prompt.overrides import PromptOverridesStore

_ERROR_MESSAGE: Final[str] = (
    "OpenAI support requires the optional 'openai' dependency. "
    "Install it with `uv sync --extra openai` or `pip install weakincentives[openai]`."
)


class _CompletionsAPI(Protocol):
    def create(self, *args: object, **kwargs: object) -> ProviderCompletionResponse: ...


class _ChatAPI(Protocol):
    completions: _CompletionsAPI


class _OpenAIProtocol(Protocol):
    """Structural type for the OpenAI client."""

    chat: _ChatAPI


class _OpenAIClientFactory(Protocol):
    def __call__(self, **kwargs: object) -> _OpenAIProtocol: ...


OpenAIProtocol = _OpenAIProtocol


class _OpenAIModule(Protocol):
    OpenAI: _OpenAIClientFactory


def _load_openai_module() -> _OpenAIModule:
    try:
        module = import_module("openai")
    except ModuleNotFoundError as exc:
        raise RuntimeError(_ERROR_MESSAGE) from exc
    return cast(_OpenAIModule, module)


def create_openai_client(**kwargs: object) -> _OpenAIProtocol:
    """Create an OpenAI client, raising a helpful error if the extra is missing."""

    openai_module = _load_openai_module()
    return openai_module.OpenAI(**kwargs)


logger: StructuredLogger = get_logger(__name__, context={"component": "adapter.openai"})


class OpenAIAdapter:
    """Adapter that evaluates prompts against OpenAI's Responses API."""

    def __init__(
        self,
        *,
        model: str,
        tool_choice: ToolChoice = "auto",
        use_native_response_format: bool = True,
        client: _OpenAIProtocol | None = None,
        client_factory: _OpenAIClientFactory | None = None,
        client_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        if client is not None:
            if client_factory is not None:
                raise ValueError(
                    "client_factory cannot be provided when an explicit client is supplied.",
                )
            if client_kwargs:
                raise ValueError(
                    "client_kwargs cannot be provided when an explicit client is supplied.",
                )
        else:
            factory = client_factory or create_openai_client
            client = factory(**dict(client_kwargs or {}))

        self._client = client
        self._model = model
        self._tool_choice: ToolChoice = tool_choice
        self._use_native_response_format = use_native_response_format

    def evaluate[OutputT](
        self,
        prompt: Prompt[OutputT],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBus,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        overrides_store: PromptOverridesStore | None = None,
        overrides_tag: str = "latest",
    ) -> PromptResponse[OutputT]:
        prompt_name = prompt.name or prompt.__class__.__name__
        render_inputs: tuple[SupportsDataclass, ...] = tuple(params)

        if deadline is not None and deadline.remaining() <= timedelta(0):
            raise PromptEvaluationError(
                "Deadline expired before evaluation started.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
                provider_payload=deadline_provider_payload(deadline),
            )

        has_structured_output = prompt.structured_output is not None
        should_disable_instructions = (
            parse_output
            and has_structured_output
            and self._use_native_response_format
            and getattr(prompt, "inject_output_instructions", False)
        )

        if should_disable_instructions:
            rendered = prompt.render(
                *params,
                overrides_store=overrides_store,
                tag=overrides_tag,
                inject_output_instructions=False,
            )
        else:
            rendered = prompt.render(
                *params,
                overrides_store=overrides_store,
                tag=overrides_tag,
            )
        if deadline is not None:
            rendered = replace(rendered, deadline=deadline)
        response_format: dict[str, Any] | None = None
        should_parse_structured_output = (
            parse_output
            and rendered.output_type is not None
            and rendered.container is not None
        )
        if should_parse_structured_output and self._use_native_response_format:
            response_format = build_json_schema_response_format(rendered, prompt_name)

        def _call_provider(
            messages: list[dict[str, Any]],
            tool_specs: Sequence[Mapping[str, Any]],
            tool_choice_directive: ToolChoice | None,
            response_format_payload: Mapping[str, Any] | None,
        ) -> object:
            request_payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
            }
            if tool_specs:
                request_payload["tools"] = list(tool_specs)
                if tool_choice_directive is not None:
                    request_payload["tool_choice"] = tool_choice_directive
            if response_format_payload is not None:
                request_payload["response_format"] = response_format_payload

            try:
                return self._client.chat.completions.create(**request_payload)
            except Exception as error:  # pragma: no cover - network/SDK failure
                raise PromptEvaluationError(
                    "OpenAI request failed.",
                    prompt_name=prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_REQUEST,
                ) from error

        def _select_choice(response: object) -> ProviderChoice:
            return cast(
                ProviderChoice,
                first_choice(response, prompt_name=prompt_name),
            )

        return run_conversation(
            adapter_name=OPENAI_ADAPTER_NAME,
            adapter=cast("ProviderAdapter[OutputT]", self),
            prompt=prompt,
            prompt_name=prompt_name,
            rendered=rendered,
            render_inputs=render_inputs,
            initial_messages=[{"role": "system", "content": rendered.text}],
            parse_output=parse_output,
            bus=bus,
            session=session,
            tool_choice=self._tool_choice,
            response_format=response_format,
            require_structured_output_text=False,
            call_provider=_call_provider,
            select_choice=_select_choice,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=format_publish_failures,
            parse_arguments=parse_tool_arguments,
            logger_override=logger,
            deadline=deadline,
        )

    def optimize[
        OutputT
    ](  # pragma: no cover - integration path exercised in adapter integration tests
        self,
        prompt: Prompt[OutputT],
        *,
        store_scope: OptimizationScope = OptimizationScope.SESSION,
        overrides_store: PromptOverridesStore | None = None,
        overrides_tag: str | None = None,
        session: SessionProtocol,
    ) -> OptimizationResult:
        prompt_name = prompt.name or prompt.key
        digest_section = self._resolve_digest_section(prompt, prompt_name=prompt_name)
        workspace_section = self._resolve_workspace_section(prompt, prompt_name)

        safe_workspace = self._clone_section_for_session(
            workspace_section, session=session
        )
        safe_digest = self._clone_section_for_session(digest_section, session=session)

        optimization_prompt = Prompt[str](
            ns=f"{prompt.ns}:optimization",
            key=f"{prompt.key}-workspace-digest",
            name=(f"{prompt.name}_workspace_digest" if prompt.name else None),
            sections=(
                MarkdownSection[SupportsDataclass](
                    title="Optimization Goal",
                    template=(
                        "Summarize the workspace so future prompts can rely on a cached digest."
                    ),
                    key="optimization-goal",
                ),
                MarkdownSection[SupportsDataclass](
                    title="Expectations",
                    template=textwrap.dedent(
                        """
                        Explore README/docs/workflow files first. Capture build/test commands,
                        dependency managers, and watchouts. Keep the digest task agnostic.
                        """
                    ).strip(),
                    key="optimization-expectations",
                ),
                safe_workspace,
                safe_digest,
            ),
        )

        response = self.evaluate(
            optimization_prompt,
            parse_output=True,
            bus=session.event_bus,
            session=session,
            overrides_store=overrides_store,
            overrides_tag=overrides_tag or "latest",
        )

        digest = self._extract_digest(response=response, prompt_name=prompt_name)
        _ = session.workspace_digest.set(digest_section.key, digest)

        if store_scope is OptimizationScope.GLOBAL:
            if overrides_store is None or overrides_tag is None:
                message = "Global scope requires overrides_store and overrides_tag."
                raise PromptEvaluationError(
                    message,
                    prompt_name=prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_REQUEST,
                )
            section_path = self._find_section_path(prompt, digest_section.key)
            _ = overrides_store.set_section_override(
                cast(PromptLike, prompt),
                tag=overrides_tag,
                path=section_path,
                body=digest,
            )

        return OptimizationResult(
            response=response,
            digest=digest,
            scope=store_scope,
            section_key=digest_section.key,
        )

    @staticmethod
    def _clone_section_for_session(
        section: Section[SupportsDataclass], *, session: SessionProtocol
    ) -> Section[SupportsDataclass]:  # pragma: no cover - helper for optimize
        clone = getattr(section, "clone", None)
        if callable(clone):
            return cast(
                Section[SupportsDataclass],
                clone(session=session, bus=session.event_bus),
            )
        return section

    def _resolve_workspace_section(
        self, prompt: Prompt[object], prompt_name: str
    ) -> Section[SupportsDataclass]:  # pragma: no cover - helper for optimize
        candidates: tuple[object, ...] = (
            PodmanSandboxSection,
            VfsToolsSection,
            "podman.shell",
            "vfs.tools",
        )
        try:
            return prompt.find_section(candidates)
        except KeyError as error:  # pragma: no cover - defensive
            raise PromptEvaluationError(
                "Workspace section required for optimization.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ) from error

    def _resolve_digest_section(
        self, prompt: Prompt[object], *, prompt_name: str
    ) -> WorkspaceDigestSection:  # pragma: no cover - helper for optimize
        try:
            section = prompt.find_section((WorkspaceDigestSection, "workspace-digest"))
        except KeyError as error:  # pragma: no cover - defensive
            raise PromptEvaluationError(
                "Workspace digest section required for optimization.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            ) from error
        if not isinstance(section, WorkspaceDigestSection):
            message = "Workspace digest section has unexpected type."
            raise PromptEvaluationError(
                message,
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            )
        return section

    def _find_section_path(
        self, prompt: Prompt[object], section_key: str
    ) -> tuple[str, ...]:  # pragma: no cover - helper for optimize
        for node in prompt.sections:
            if node.section.key == section_key:
                return node.path
        message = f"Section path not found for key: {section_key}"
        raise PromptEvaluationError(
            message,
            prompt_name=prompt.name or prompt.key,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
        )

    def _extract_digest(
        self, *, response: PromptResponse[Any], prompt_name: str
    ) -> str:  # pragma: no cover - helper for optimize
        digest: str | None = None
        if isinstance(response.output, str):
            digest = response.output
        elif response.output is not None:
            candidate = getattr(response.output, "digest", None)
            if isinstance(candidate, str):
                digest = candidate
        if digest is None and response.text:
            digest = response.text
        if digest is None:
            raise PromptEvaluationError(
                "Optimization did not return digest content.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_RESPONSE,
            )
        return digest.strip()


__all__ = [
    "OpenAIAdapter",
    "OpenAIProtocol",
    "extract_parsed_content",
    "message_text_content",
    "parse_schema_constrained_payload",
]


message_text_content = _shared.message_text_content
extract_parsed_content = _shared.extract_parsed_content
parse_schema_constrained_payload = _shared.parse_schema_constrained_payload

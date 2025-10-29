"""Optional OpenAI adapter utilities."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Protocol, cast

_ERROR_MESSAGE = (
    "OpenAI support requires the optional 'openai' dependency. "
    "Install it with `uv sync --extra openai` or `pip install weakincentives[openai]`."
)


class _CompletionFunctionCall(Protocol):
    name: str
    arguments: str | None


class _ToolCall(Protocol):
    id: str
    function: _CompletionFunctionCall


class _Message(Protocol):
    content: str | None
    tool_calls: Sequence[_ToolCall] | None


class _CompletionChoice(Protocol):
    message: _Message


class _CompletionResponse(Protocol):
    choices: Sequence[_CompletionChoice]


class _CompletionsAPI(Protocol):
    def create(self, *args: object, **kwargs: object) -> _CompletionResponse: ...


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


__all__ = ["create_openai_client", "OpenAIProtocol"]

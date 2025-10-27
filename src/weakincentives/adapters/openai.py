"""Optional OpenAI adapter utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast

_ERROR_MESSAGE = (
    "OpenAI support requires the optional 'openai' dependency. "
    "Install it with `uv sync --extra openai` or `pip install weakincentives[openai]`."
)


class _OpenAIProtocol(Protocol):
    """Structural type for the OpenAI client."""


class _OpenAIClientFactory(Protocol):
    def __call__(self, **kwargs: object) -> _OpenAIProtocol: ...


class _OpenAIModule(Protocol):
    OpenAI: _OpenAIClientFactory


# Structural alias used for both runtime and type checking when the optional extra is absent.
OpenAI = _OpenAIProtocol


def _load_openai_module() -> _OpenAIModule:
    try:
        module = import_module("openai")
    except ModuleNotFoundError as exc:
        raise RuntimeError(_ERROR_MESSAGE) from exc
    return cast(_OpenAIModule, module)


def create_openai_client(**kwargs: object) -> OpenAI:
    """Create an OpenAI client, raising a helpful error if the extra is missing."""

    openai_module = _load_openai_module()
    return openai_module.OpenAI(**kwargs)

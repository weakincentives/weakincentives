import importlib
import sys
import types
from importlib import import_module as std_import_module
from typing import cast

import pytest

MODULE_PATH = "weakincentives.adapters.openai"


def _reload_module():
    return importlib.reload(std_import_module(MODULE_PATH))


def test_create_openai_client_requires_optional_dependency(monkeypatch):
    module = _reload_module()

    def fail_import(name: str, package: str | None = None):
        if name == "openai":
            raise ModuleNotFoundError("No module named 'openai'")
        return std_import_module(name, package)

    monkeypatch.setattr(module, "import_module", fail_import)

    with pytest.raises(RuntimeError) as err:
        module.create_openai_client()

    message = str(err.value)
    assert "uv sync --extra openai" in message
    assert "pip install weakincentives[openai]" in message


def test_create_openai_client_returns_openai_instance(monkeypatch):
    module = _reload_module()

    class DummyOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    dummy_module = cast(module._OpenAIModule, types.ModuleType("openai"))
    dummy_module.OpenAI = DummyOpenAI

    monkeypatch.setitem(sys.modules, "openai", dummy_module)

    client = module.create_openai_client(api_key="secret-key")

    assert isinstance(client, DummyOpenAI)
    assert client.kwargs == {"api_key": "secret-key"}

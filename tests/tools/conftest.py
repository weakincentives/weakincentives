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

"""Shared fixtures for tool suites."""

from __future__ import annotations

import ast
import sys
from importlib import import_module
from types import CodeType, ModuleType
from typing import cast

import pytest

from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session
from weakincentives.runtime.clock import FakeClock


def _compile_stub_segments(source: str) -> tuple[CodeType, CodeType | None]:
    parsed = ast.parse(source, filename="<asteval-stub>", mode="exec")
    statements = list(parsed.body)
    expr_code = None
    if statements and isinstance(statements[-1], ast.Expr):
        expr_stmt = cast(ast.Expr, statements.pop())
        expression = ast.Expression(expr_stmt.value)
        ast.fix_missing_locations(expression)
        expr_code = compile(expression, filename="<asteval-stub>", mode="eval")
    module = ast.Module(body=statements, type_ignores=[])
    ast.fix_missing_locations(module)
    exec_code = compile(module, filename="<asteval-stub>", mode="exec")
    return exec_code, expr_code


class _StubInterpreter:
    """Minimal interpreter used when the optional asteval dependency is absent."""

    def __init__(self, *, use_numpy: bool, minimal: bool) -> None:
        del use_numpy, minimal
        self.symtable: dict[str, object] = {}
        self.node_handlers: dict[str, object] = {}
        self.error: list[Exception] = []

    def eval(self, expression: str) -> object:
        exec_code, expr_code = _compile_stub_segments(expression)
        globals_dict = self.symtable
        globals_dict.setdefault("__builtins__", {})
        try:
            exec(exec_code, globals_dict, globals_dict)
            if expr_code is None:
                return None
            return eval(expr_code, globals_dict, globals_dict)
        except Exception as error:  # pragma: no cover - error is surfaced via stderr
            self.error.append(error)
            return None


def _build_stub_module() -> ModuleType:
    module = ModuleType("asteval")
    module.Interpreter = _StubInterpreter  # type: ignore[attr-defined]
    module.ALL_DISALLOWED = ()  # type: ignore[attr-defined]
    return module


@pytest.fixture(scope="session")
def _asteval_stub_module() -> ModuleType | None:
    try:
        import_module("asteval")
    except ModuleNotFoundError:
        return _build_stub_module()
    return None


@pytest.fixture(autouse=True)
def _install_asteval_stub(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    _asteval_stub_module: ModuleType | None,
) -> None:
    if _asteval_stub_module is None:
        return
    module = getattr(request.node, "module", None)
    if module is None or module.__name__ != "tests.tools.test_asteval_tool":
        return
    if request.node.name == "test_missing_dependency_instructs_extra_install":
        return
    monkeypatch.setitem(sys.modules, "asteval", _asteval_stub_module)


@pytest.fixture()
def session_and_dispatcher(clock: FakeClock) -> tuple[Session, InProcessDispatcher]:
    dispatcher = InProcessDispatcher()
    return Session(dispatcher=dispatcher, clock=clock), dispatcher

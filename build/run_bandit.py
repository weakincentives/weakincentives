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

"""Bandit runner with Python 3.14 AST compatibility shims."""

from __future__ import annotations

import ast
import importlib
import sys
from collections.abc import Callable
from typing import cast

BanditMain = Callable[[], int | None]


def _patch_ast() -> None:
    """Restore AST nodes removed in Python 3.14 that Bandit still expects."""

    constant = getattr(ast, "Constant", None)
    if constant is None:
        return

    deprecated_nodes = ("Num", "Str", "Bytes", "NameConstant", "Ellipsis")
    ast_members = ast.__dict__
    for deprecated in deprecated_nodes:
        if deprecated not in ast_members:
            setattr(ast, deprecated, constant)  # type: ignore[attr-defined]

    def _make_property() -> property:
        return property(lambda self: self.value)

    constant_members = constant.__dict__
    for attr_name in ("n", "s", "b"):
        if attr_name not in constant_members:
            setattr(constant, attr_name, _make_property())


def _load_bandit_main() -> BanditMain:
    module = importlib.import_module("bandit.__main__")
    main_attr = module.main
    if not callable(main_attr):  # pragma: no cover - defensive guard
        raise TypeError("bandit.__main__.main is not callable")
    return cast(BanditMain, main_attr)


def main() -> int:
    _patch_ast()
    bandit_main = _load_bandit_main()
    result = bandit_main()
    return 0 if result is None else int(result)


if __name__ == "__main__":
    sys.exit(main())

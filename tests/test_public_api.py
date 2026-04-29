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

"""Invariants on the public spine API."""

from __future__ import annotations

import importlib
import pkgutil

import weakincentives
from weakincentives import core


def test_top_level_re_exports_match_core() -> None:
    assert set(weakincentives.__all__) == set(core.__all__)


def test_every_advertised_symbol_is_resolvable() -> None:
    for name in weakincentives.__all__:
        assert hasattr(weakincentives, name)
        assert hasattr(core, name)


def test_core_imports_only_stdlib_and_self() -> None:
    """The spine must not depend on anything outside ``weakincentives.core``."""
    forbidden_prefixes = (
        "weakincentives.adapters",
        "weakincentives.cli",
        "weakincentives.contrib",
        "weakincentives.evals",
        "weakincentives.runtime",
    )
    for module_info in pkgutil.iter_modules(core.__path__):
        module = importlib.import_module(f"weakincentives.core.{module_info.name}")
        for attr in dir(module):
            value = getattr(module, attr)
            module_name = getattr(value, "__module__", "")
            if isinstance(module_name, str):
                for prefix in forbidden_prefixes:
                    assert not module_name.startswith(prefix), (
                        f"{module.__name__}.{attr} from forbidden {module_name}"
                    )

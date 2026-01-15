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

"""Contributed utilities for specific agent styles.

This package contains domain-specific tools and optimizers that extend
the core primitives. These are useful batteries for building agents but
are not part of the minimal core library.

Subpackages:

- ``contrib.mailbox``: Redis-backed mailbox implementation
- ``contrib.optimizers``: Workspace digest optimizer
- ``contrib.overrides``: Filesystem-backed prompt overrides implementation
- ``contrib.tools``: Planning tools, VFS, Podman sandbox, asteval, workspace digest

Example usage::

    from weakincentives.contrib.overrides import LocalPromptOverridesStore
    from weakincentives.contrib.tools import (
        PlanningToolsSection,
        VfsToolsSection,
        AstevalSection,
        PodmanSandboxSection,
    )

    from weakincentives.contrib.mailbox import RedisMailbox
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from . import mailbox, optimizers, overrides, tools

__all__ = ["mailbox", "optimizers", "overrides", "tools"]


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})


def __getattr__(name: str) -> ModuleType:
    """Lazy import submodules to avoid circular dependency issues.

    The optimizers submodule imports from tools, so eager import would fail
    if done in the wrong order. Lazy loading sidesteps this entirely.
    """
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

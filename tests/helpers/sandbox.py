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

"""Sandbox helpers for tests.

Builds lightweight sandboxes over an in-memory filesystem so tests can
exercise sandbox-aware code paths (tool contexts, transactions, checkers)
without touching the host disk or git snapshot plumbing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

from weakincentives.filesystem import Filesystem
from weakincentives.sandbox import LocalSandbox, LocalShell

__all__ = ["make_memory_sandbox"]


def make_memory_sandbox(filesystem: Filesystem | None = None) -> LocalSandbox:
    """Build a sandbox whose filesystem facet is in-memory.

    The root path is never created on disk, so ``close()`` is a no-op
    beyond bookkeeping and tests need no cleanup.
    """
    fs = filesystem if filesystem is not None else Filesystem.in_memory()
    root = Path(tempfile.gettempdir()) / f"wink-test-sandbox-{uuid4().hex}"
    return LocalSandbox(root=root, filesystem=fs, shell=LocalShell(root))

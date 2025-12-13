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

- ``contrib.tools``: Planning tools, VFS, Podman sandbox, asteval, workspace digest
- ``contrib.optimizers``: Workspace digest optimizer

Example usage::

    from weakincentives.contrib.tools import (
        PlanningToolsSection,
        VfsToolsSection,
        AstevalSection,
        PodmanSandboxSection,
    )
"""

from __future__ import annotations

from . import optimizers, tools

__all__ = ["optimizers", "tools"]

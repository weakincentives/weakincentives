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

"""Contributed optimizers for specific agent styles.

This package provides domain-specific optimizers that work with the
contributed tool suites.

Example usage::

    from weakincentives.contrib.optimizers import WorkspaceDigestOptimizer
"""

from __future__ import annotations

from .workspace_digest import WorkspaceDigestOptimizer

__all__ = ["WorkspaceDigestOptimizer"]

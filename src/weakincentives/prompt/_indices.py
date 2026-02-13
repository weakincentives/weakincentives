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

# pyright: reportImportCycles=false
"""Index-building for :mod:`weakincentives.prompt.registry`."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..types.dataclass import SupportsDataclass
from .errors import SectionPath

if TYPE_CHECKING:
    from .registry import SectionNode


def build_registry_indices(
    sections: tuple[SectionNode[SupportsDataclass], ...],
) -> tuple[
    Mapping[SectionPath, SectionNode[SupportsDataclass]],
    Mapping[SectionPath, tuple[str, ...]],
    Mapping[SectionPath, bool],
    Mapping[SectionPath, bool],
]:
    """Build precomputed indices for O(1) lookups in registry operations.

    Computes in a single O(n) pass:
    - node_by_path: Maps section path to its node
    - children_by_path: Maps section path to tuple of direct child keys
    - subtree_has_tools: Maps section path to whether section or descendants have tools
    - subtree_has_skills: Maps section path to whether section or descendants have skills

    Returns:
        Tuple of (node_by_path, children_by_path, subtree_has_tools, subtree_has_skills)
        as frozen mappings.
    """
    # Build node_by_path in O(n)
    node_by_path: dict[SectionPath, SectionNode[SupportsDataclass]] = {
        node.path: node for node in sections
    }

    # Build children_by_path in O(n)
    children_by_path: dict[SectionPath, list[str]] = {
        node.path: [] for node in sections
    }
    for node in sections:
        parent_path = node.path[:-1]
        if parent_path in children_by_path:
            children_by_path[parent_path].append(node.section.key)

    # Build subtree_has_tools and subtree_has_skills using reverse traversal in O(n)
    # Traverse in reverse DFS order to compute children before parents
    subtree_has_tools: dict[SectionPath, bool] = {}
    subtree_has_skills: dict[SectionPath, bool] = {}
    for node in reversed(sections):
        # Check if this section has tools
        has_tools = bool(node.section.tools())
        if has_tools:
            subtree_has_tools[node.path] = True
        else:
            # Check if any child subtree has tools
            child_has_tools = any(
                subtree_has_tools.get((*node.path, child_key), False)
                for child_key in children_by_path[node.path]
            )
            subtree_has_tools[node.path] = child_has_tools

        # Check if this section has skills
        has_skills = bool(node.section.skills())
        if has_skills:
            subtree_has_skills[node.path] = True
        else:
            # Check if any child subtree has skills
            child_has_skills = any(
                subtree_has_skills.get((*node.path, child_key), False)
                for child_key in children_by_path[node.path]
            )
            subtree_has_skills[node.path] = child_has_skills

    # Freeze the mappings
    frozen_children: dict[SectionPath, tuple[str, ...]] = {
        path: tuple(children) for path, children in children_by_path.items()
    }

    return (
        MappingProxyType(node_by_path),
        MappingProxyType(frozen_children),
        MappingProxyType(subtree_has_tools),
        MappingProxyType(subtree_has_skills),
    )

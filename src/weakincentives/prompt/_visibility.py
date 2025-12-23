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

"""Section visibility control for :mod:`weakincentives.prompt`.

This module provides the SectionVisibility enum. Visibility selectors,
normalization utilities, and predicates are consolidated in _section_guards.
"""

from __future__ import annotations

from enum import Enum


class SectionVisibility(Enum):
    """Controls how a section is rendered in a prompt.

    When a section has both a full template and a summary template,
    the visibility determines which one is used during rendering.
    """

    FULL = "full"
    """Render the full section content."""

    SUMMARY = "summary"
    """Render only the summary content."""


__all__ = [
    "SectionVisibility",
]

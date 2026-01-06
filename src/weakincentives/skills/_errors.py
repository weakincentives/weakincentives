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

"""Skill-related error types.

These exceptions are raised during skill validation, resolution, and mounting
operations. All skill errors inherit from :class:`SkillError`, which in turn
inherits from :class:`~weakincentives.errors.WinkError`.
"""

from __future__ import annotations

from ..errors import WinkError

__all__ = [
    "SkillError",
    "SkillMountError",
    "SkillNotFoundError",
    "SkillValidationError",
]


class SkillError(WinkError):
    """Base class for all skill-related exceptions.

    This allows callers to catch all skill errors with a single handler::

        try:
            # skill operations
        except SkillError:
            # handle any skill error
    """


class SkillValidationError(SkillError):
    """Raised when skill validation fails.

    This error is raised when:

    - A skill directory is missing the required SKILL.md file
    - A skill file does not have the .md extension
    - A skill file exceeds the maximum size limit
    """


class SkillNotFoundError(SkillError):
    """Raised when a skill source path does not exist.

    This error is raised during skill resolution when the specified
    source path cannot be found on the filesystem.
    """


class SkillMountError(SkillError):
    """Raised when skill mounting fails.

    This error is raised when:

    - A skill name contains invalid characters (path traversal)
    - Duplicate skill names are detected in a configuration
    - An I/O error occurs during skill copying
    - A skill exceeds the total size limit during mounting
    """

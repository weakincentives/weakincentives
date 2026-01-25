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

Exception hierarchy::

    WinkError
    └── SkillError (base for all skill errors)
        ├── SkillValidationError (invalid skill structure/content)
        ├── SkillNotFoundError (skill source path missing)
        └── SkillMountError (mounting operation failed)

Example::

    from weakincentives.skills import SkillError, validate_skill

    try:
        validate_skill(skill_path)
    except SkillError as e:
        print(f"Skill error: {e}")
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

    Inherit from this class when defining new skill error types. This allows
    callers to catch all skill errors with a single handler while still being
    able to handle specific error types when needed.

    Example::

        from weakincentives.skills import (
            SkillError,
            SkillValidationError,
            SkillNotFoundError,
        )

        try:
            mount_skills(config, workspace)
        except SkillNotFoundError as e:
            # Handle missing skill specifically
            print(f"Skill not found: {e}")
        except SkillValidationError as e:
            # Handle validation errors specifically
            print(f"Invalid skill: {e}")
        except SkillError as e:
            # Catch-all for any other skill error
            print(f"Skill error: {e}")
    """


class SkillValidationError(SkillError):
    """Raised when skill validation fails.

    This error indicates that a skill's structure or content does not conform
    to the Agent Skills specification. The error message contains details about
    what validation check failed.

    Common causes:

    - A skill directory is missing the required SKILL.md file
    - A skill file does not have the .md extension
    - A skill file exceeds :data:`MAX_SKILL_FILE_BYTES` (1 MiB)
    - SKILL.md is missing required YAML frontmatter
    - Frontmatter is missing required fields (``name``, ``description``)
    - Frontmatter field values are invalid (wrong type, too long, bad format)
    - For directory skills, the frontmatter ``name`` doesn't match directory name

    Example::

        from pathlib import Path
        from weakincentives.skills import validate_skill, SkillValidationError

        try:
            validate_skill(Path("./my-skill"))
        except SkillValidationError as e:
            print(f"Validation failed: {e}")
            # Fix the skill structure and try again

    See Also:
        :func:`validate_skill`: The validation function that raises this error.
    """


class SkillNotFoundError(SkillError):
    """Raised when a skill source path does not exist.

    This error is raised during skill resolution when the specified
    source path cannot be found on the filesystem. The error message
    includes the path that was not found.

    Common causes:

    - Typo in the skill path
    - Skill directory was moved or deleted
    - Relative path resolved incorrectly (check working directory)
    - Skill not yet created or installed

    Example::

        from pathlib import Path
        from weakincentives.skills import SkillMount, SkillNotFoundError

        mount = SkillMount(Path("./skills/nonexistent"))
        try:
            resolve_and_mount(mount)
        except SkillNotFoundError as e:
            print(f"Skill not found: {e}")
            # Check if the path exists: mount.source.exists()
    """


class SkillMountError(SkillError):
    """Raised when skill mounting fails.

    This error indicates that the skill mounting operation could not complete.
    Unlike :class:`SkillValidationError` which indicates invalid skill content,
    this error indicates problems with the mounting process itself.

    Common causes:

    - Skill name is invalid (empty, too long, bad characters, path traversal)
    - Duplicate skill names detected in a :class:`SkillConfig`
    - An I/O error occurred during skill copying (permissions, disk full)
    - A skill directory exceeds :data:`MAX_SKILL_TOTAL_BYTES` (10 MiB)

    Example::

        from weakincentives.skills import SkillConfig, SkillMountError

        try:
            mount_skills(config, workspace)
        except SkillMountError as e:
            print(f"Mount failed: {e}")
            # Check for duplicate names or invalid skill names

    See Also:
        :func:`validate_skill_name`: Validation rules for skill names.
    """

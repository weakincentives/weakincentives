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

"""Skill validation utilities.

This module provides functions for validating skill structure and resolving
skill names according to the Agent Skills specification.
"""

from __future__ import annotations

from pathlib import Path

from ._errors import SkillMountError, SkillValidationError
from ._types import MAX_SKILL_FILE_BYTES, SkillMount

__all__ = [
    "resolve_skill_name",
    "validate_skill",
    "validate_skill_name",
]


def resolve_skill_name(mount: SkillMount) -> str:
    """Resolve the effective skill name from a mount.

    The skill name is determined in the following order:

    1. Explicit name from ``mount.name`` if provided
    2. Directory name if source is a directory
    3. File stem (without .md extension) if source is a file

    Args:
        mount: The skill mount to resolve a name for.

    Returns:
        The skill name to use for the destination directory.

    Example::

        from pathlib import Path
        from weakincentives.skills import SkillMount, resolve_skill_name

        # Directory: uses directory name
        mount = SkillMount(Path("./skills/code-review"))
        assert resolve_skill_name(mount) == "code-review"

        # File: strips .md extension
        mount = SkillMount(Path("./my-skill.md"))
        assert resolve_skill_name(mount) == "my-skill"

        # Explicit name override
        mount = SkillMount(Path("./v2/review"), name="code-review")
        assert resolve_skill_name(mount) == "code-review"
    """
    if mount.name is not None:
        return mount.name
    if mount.source.is_dir():
        return mount.source.name
    # File: strip .md extension
    return mount.source.stem


def validate_skill_name(name: str) -> None:
    """Validate that a skill name is safe for filesystem use.

    Skill names must be valid directory names that cannot be used for
    path traversal attacks.

    Args:
        name: The skill name to validate.

    Raises:
        SkillMountError: If the name contains path traversal characters
            (``/``, ``\\``, or ``..``) or is empty/invalid.

    Example::

        from weakincentives.skills import validate_skill_name

        validate_skill_name("code-review")  # OK
        validate_skill_name("my_skill")     # OK

        validate_skill_name("../escape")    # Raises SkillMountError
        validate_skill_name("a/b")          # Raises SkillMountError
    """
    if "/" in name or "\\" in name or ".." in name:
        msg = f"Skill name contains invalid characters: {name}"
        raise SkillMountError(msg)
    if not name or name in {".", ".."}:
        msg = f"Invalid skill name: {name}"
        raise SkillMountError(msg)


def validate_skill(source: Path) -> None:
    """Validate skill structure before mounting.

    This function checks that a skill path meets the requirements of the
    Agent Skills specification:

    - **Directory skills** must contain a SKILL.md file at the root
    - **File skills** must have a .md extension and not exceed size limits

    Args:
        source: Path to the skill file or directory.

    Raises:
        SkillValidationError: If the skill structure is invalid.

    Example::

        from pathlib import Path
        from weakincentives.skills import validate_skill

        # Validates a directory skill has SKILL.md
        validate_skill(Path("./skills/code-review"))

        # Validates a file skill is markdown and within size limits
        validate_skill(Path("./my-skill.md"))
    """
    if source.is_dir():
        # Directory skill must contain SKILL.md
        skill_file = source / "SKILL.md"
        if not skill_file.is_file():
            msg = f"Skill directory missing SKILL.md: {source}"
            raise SkillValidationError(msg)
    else:
        # File skill must be markdown
        if source.suffix.lower() != ".md":
            msg = f"Skill file must be markdown (.md): {source}"
            raise SkillValidationError(msg)
        # Check file size
        size = source.stat().st_size
        if size > MAX_SKILL_FILE_BYTES:
            msg = f"Skill file exceeds size limit ({size} > {MAX_SKILL_FILE_BYTES}): {source}"
            raise SkillValidationError(msg)

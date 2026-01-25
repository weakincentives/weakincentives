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

"""Core skill types and constants.

This module defines the data structures for representing skills following
the Agent Skills specification (https://agentskills.io). Skills are folders
of instructions, scripts, and resources that agents can discover and use
to perform tasks more accurately and efficiently.

The SKILL.md file specification is the core format standard, originally
developed by Anthropic and released as an open standard.

Typical usage::

    from pathlib import Path
    from weakincentives.skills import SkillConfig, SkillMount

    config = SkillConfig(
        skills=(
            SkillMount(Path("./skills/code-review")),
            SkillMount(Path("./skills/testing")),
        )
    )

See Also:
    - :mod:`weakincentives.skills._validation` for validation utilities
    - :mod:`weakincentives.skills._errors` for exception types
"""

from __future__ import annotations

from pathlib import Path

from ..dataclasses import FrozenDataclass

__all__ = [
    "MAX_SKILL_FILE_BYTES",
    "MAX_SKILL_TOTAL_BYTES",
    "Skill",
    "SkillConfig",
    "SkillMount",
]


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

#: Maximum size for an individual skill file in bytes (1 MiB).
#: Files exceeding this limit will cause :class:`SkillValidationError`.
MAX_SKILL_FILE_BYTES: int = 1024 * 1024

#: Maximum total size for a skill directory including all files (10 MiB).
#: Skills exceeding this limit will cause :class:`SkillMountError` during mounting.
MAX_SKILL_TOTAL_BYTES: int = 10 * 1024 * 1024


# -----------------------------------------------------------------------------
# Core Types
# -----------------------------------------------------------------------------


@FrozenDataclass()
class Skill:
    """Represents a loaded skill definition.

    A skill is a collection of instructions, scripts, and resources that
    agents can discover and use. Skills follow the Agent Skills specification
    (https://agentskills.io) and use SKILL.md as the core format standard.

    Skills can be:

    - **Directory skills**: A directory containing a SKILL.md file and optional
      supporting resources (examples, templates, scripts)
    - **File skills**: A single markdown file that serves as the skill definition

    This class is immutable (frozen dataclass) and represents a skill that has
    been resolved and optionally loaded from disk.

    Attributes:
        name: The skill's identifier. Used for discovery and deduplication.
            Must follow the Agent Skills naming convention (lowercase
            alphanumeric with hyphens, 1-64 characters).
        source: Path to the skill file or directory on the host filesystem.
            For directory skills, this points to the directory containing
            SKILL.md. For file skills, this points to the .md file itself.
        content: The content of the SKILL.md file, or None if not yet loaded.
            Populated when the skill is read from disk.

    Example::

        # Directory skill structure
        my-skill/
        ├── SKILL.md          # Required: skill definition
        ├── examples/         # Optional: example files
        └── templates/        # Optional: template files

        # File skill
        my-skill.md           # Single markdown file

    Note:
        Skill instances are typically created by the skill loading machinery
        rather than constructed directly. Use :class:`SkillMount` and
        :class:`SkillConfig` to configure skills for an agent.
    """

    name: str
    source: Path
    content: str | None = None


@FrozenDataclass()
class SkillMount:
    """Configuration for mounting a skill into an agent environment.

    A skill mount specifies how to copy a skill from the host filesystem
    into an agent's accessible workspace. This enables declarative skill
    composition without modifying prompts or requiring agent internals.

    Use :class:`SkillConfig` to group multiple skill mounts together.

    Attributes:
        source: Path to a skill file (.md) or skill directory on the
            host filesystem. For directory skills, the directory must contain
            a SKILL.md file. Relative paths are resolved against the current
            working directory at mount time.
        name: Optional skill name override. If None, the name is derived from
            the source path using :func:`resolve_skill_name`: directory name
            for directory skills, or filename without .md extension for file
            skills. Must follow Agent Skills naming convention if provided.
        enabled: Whether the skill is active. Set to False to temporarily
            disable a skill without removing it from configuration. Disabled
            skills are skipped during mounting. Defaults to True.

    Example::

        from pathlib import Path
        from weakincentives.skills import SkillMount, SkillConfig

        # Mount a directory skill
        review_skill = SkillMount(Path("./skills/code-review"))

        # Mount with custom name (useful for versioning or renaming)
        custom_skill = SkillMount(
            source=Path("./internal/review-v2"),
            name="code-review",
        )

        # Conditionally disable a skill
        experimental = SkillMount(
            source=Path("./skills/experimental"),
            enabled=False,
        )

    See Also:
        :func:`resolve_skill_name`: Logic for deriving skill names.
        :func:`validate_skill_name`: Validation rules for skill names.
    """

    source: Path
    name: str | None = None
    enabled: bool = True


@FrozenDataclass()
class SkillConfig:
    """Collection of skills to install in an agent environment.

    This configuration specifies which skills to mount and validation settings.
    Skills are processed in order, with disabled skills skipped. Duplicate
    skill names (after resolution) raise :class:`SkillMountError`.

    Attributes:
        skills: Tuple of skill mounts to copy into the workspace. Order is
            preserved during mounting. An empty tuple means no skills will
            be installed.
        validate_on_mount: If True (default), validate skill structure before
            copying using :func:`validate_skill`. Validation checks:

            - Directory skills have a SKILL.md file
            - File skills have .md extension and valid size
            - SKILL.md frontmatter is valid with required fields

            Set to False during development to skip validation, but ensure
            skills are valid before deployment.

    Example::

        from pathlib import Path
        from weakincentives.skills import SkillConfig, SkillMount

        # Basic configuration with multiple skills
        config = SkillConfig(
            skills=(
                SkillMount(Path("./skills/code-review")),
                SkillMount(Path("./skills/testing")),
            )
        )

        # Disable validation during development
        dev_config = SkillConfig(
            skills=(SkillMount(Path("./wip-skill")),),
            validate_on_mount=False,
        )

    Raises:
        SkillMountError: During mounting if duplicate skill names are detected.
        SkillValidationError: During mounting if validation is enabled and
            a skill has invalid structure.
        SkillNotFoundError: During mounting if a skill source path does not exist.

    See Also:
        :class:`SkillMount`: Individual skill mount configuration.
        :func:`validate_skill`: Validation function used when validate_on_mount is True.
    """

    skills: tuple[SkillMount, ...] = ()
    validate_on_mount: bool = True

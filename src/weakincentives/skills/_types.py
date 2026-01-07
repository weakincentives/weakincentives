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

#: Maximum size for an individual skill file (1 MiB).
MAX_SKILL_FILE_BYTES: int = 1024 * 1024

#: Maximum total size for a skill including all files (10 MiB).
MAX_SKILL_TOTAL_BYTES: int = 10 * 1024 * 1024


# -----------------------------------------------------------------------------
# Core Types
# -----------------------------------------------------------------------------


@FrozenDataclass()
class Skill:
    """Represents a skill definition.

    A skill is a collection of instructions, scripts, and resources that
    agents can discover and use. Skills follow the Agent Skills specification
    (https://agentskills.io) and use SKILL.md as the core format standard.

    Skills can be:

    - **Directory skills**: A directory containing a SKILL.md file and optional
      supporting resources (examples, templates, scripts)
    - **File skills**: A single markdown file that serves as the skill definition

    Attributes:
        name: The skill's identifier. Used for discovery and deduplication.
        source: Path to the skill file or directory on the host filesystem.
        content: The content of the SKILL.md file (if loaded).

    Example::

        # Directory skill structure
        my-skill/
        ├── SKILL.md          # Required: skill definition
        ├── examples/         # Optional: example files
        └── templates/        # Optional: template files

        # File skill
        my-skill.md           # Single markdown file
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

    Attributes:
        source: Path to a skill file (SKILL.md) or skill directory on the
            host filesystem. Relative paths are resolved against the current
            working directory.
        name: Optional skill name override. If None, derived from the source
            path (directory name or filename without extension).
        enabled: Whether the skill is active. Disabled skills are not copied.
            Defaults to True.

    Example::

        from pathlib import Path
        from weakincentives.skills import SkillMount, SkillConfig

        # Mount a directory skill
        review_skill = SkillMount(Path("./skills/code-review"))

        # Mount with custom name
        custom_skill = SkillMount(
            source=Path("./internal/review-v2"),
            name="code-review",
        )

        # Conditionally disable a skill
        experimental = SkillMount(
            source=Path("./skills/experimental"),
            enabled=False,
        )
    """

    source: Path
    name: str | None = None
    enabled: bool = True


@FrozenDataclass()
class SkillConfig:
    """Collection of skills to install in an agent environment.

    This configuration specifies which skills to mount and validation settings.
    Skills are processed in order, and duplicate names raise an error.

    Attributes:
        skills: Tuple of skill mounts to copy into the workspace.
        validate_on_mount: If True, validate skill structure before copying.
            Validation checks for required SKILL.md file in directories and
            proper file extension for file skills. Defaults to True.

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
    """

    skills: tuple[SkillMount, ...] = ()
    validate_on_mount: bool = True

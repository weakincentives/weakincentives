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

from dataclasses import dataclass
from pathlib import Path

from ..dataclasses import FrozenDataclassMixin

__all__ = [
    "MAX_SKILL_FILE_BYTES",
    "MAX_SKILL_TOTAL_BYTES",
    "Skill",
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


@dataclass(slots=True, frozen=True)
class Skill(FrozenDataclassMixin):
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


@dataclass(slots=True, frozen=True)
class SkillMount(FrozenDataclassMixin):
    """Configuration for mounting a skill into an agent environment.

    A skill mount specifies how to copy a skill from the host filesystem
    into an agent's accessible workspace. Skills are attached to prompt
    sections and follow the same visibility rules as tools.

    Conditional skill activation is handled via section visibility:
    skills attached to sections with SUMMARY visibility are not collected
    until the section is expanded.

    Attributes:
        source: Path to a skill file (SKILL.md) or skill directory on the
            host filesystem. Relative paths are resolved against the current
            working directory.
        name: Optional skill name override. If None, derived from the source
            path (directory name or filename without extension).

    Example::

        from pathlib import Path
        from weakincentives.prompt import MarkdownSection
        from weakincentives.skills import SkillMount

        # Attach skills to a section
        section = MarkdownSection(
            title="Code Review",
            key="code-review",
            template="Review the code.",
            skills=(
                SkillMount(Path("./skills/code-review")),
                SkillMount(Path("./skills/security-audit")),
            ),
        )

        # Mount with custom name
        custom_skill = SkillMount(
            source=Path("./internal/review-v2"),
            name="code-review",
        )
    """

    source: Path
    name: str | None = None

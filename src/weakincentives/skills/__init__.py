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

"""Core skill types and validation for agent skill composition.

This module provides the data structures for representing skills following
the Agent Skills specification (https://agentskills.io). Skills are folders
of instructions, scripts, and resources that agents can discover and use
to perform tasks more accurately and efficiently.

The SKILL.md file specification is the core format standard, originally
developed by Anthropic and released as an open standard.

Example usage::

    from pathlib import Path
    from weakincentives.skills import (
        Skill,
        SkillConfig,
        SkillMount,
        resolve_skill_name,
        validate_skill,
    )

    # Create a skill configuration
    config = SkillConfig(
        skills=(
            SkillMount(Path("./skills/code-review")),
            SkillMount(Path("./skills/testing")),
        )
    )

    # Resolve and validate each skill
    for mount in config.skills:
        if mount.enabled:
            name = resolve_skill_name(mount)
            validate_skill(mount.source.resolve())
            print(f"Skill '{name}' validated")

Skill Types:

- :class:`Skill`: Core representation of a skill definition
- :class:`SkillMount`: Configuration for mounting a skill
- :class:`SkillConfig`: Collection of skills to install

Error Types:

- :class:`SkillError`: Base class for all skill errors
- :class:`SkillValidationError`: Invalid skill structure
- :class:`SkillNotFoundError`: Skill source path not found
- :class:`SkillMountError`: Mounting operation failed

Validation Functions:

- :func:`validate_skill`: Validate skill structure
- :func:`validate_skill_name`: Validate skill name is safe
- :func:`resolve_skill_name`: Derive name from mount configuration

Constants:

- :data:`MAX_SKILL_FILE_BYTES`: Maximum size for an individual file (1 MiB)
- :data:`MAX_SKILL_TOTAL_BYTES`: Maximum total size for a skill (10 MiB)
"""

from __future__ import annotations

from ._errors import (
    SkillError,
    SkillMountError,
    SkillNotFoundError,
    SkillValidationError,
)
from ._types import (
    MAX_SKILL_FILE_BYTES,
    MAX_SKILL_TOTAL_BYTES,
    Skill,
    SkillConfig,
    SkillMount,
)
from ._validation import (
    resolve_skill_name,
    validate_skill,
    validate_skill_name,
)

__all__ = [
    "MAX_SKILL_FILE_BYTES",
    "MAX_SKILL_TOTAL_BYTES",
    "Skill",
    "SkillConfig",
    "SkillError",
    "SkillMount",
    "SkillMountError",
    "SkillNotFoundError",
    "SkillValidationError",
    "resolve_skill_name",
    "validate_skill",
    "validate_skill_name",
]

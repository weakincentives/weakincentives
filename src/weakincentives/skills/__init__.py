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

"""Skills package for agent skill composition and validation.

This module provides the data structures and validation utilities for
representing skills following the Agent Skills specification
(https://agentskills.io). Skills are folders of instructions, scripts, and
resources that agents can discover and use to perform tasks more accurately
and efficiently.

The SKILL.md file specification is the core format standard, originally
developed by Anthropic and released as an open standard.

What are Skills?
----------------

Skills are reusable packages of domain knowledge that can be composed into
agent environments. Each skill is either:

- A **directory skill**: A folder containing a ``SKILL.md`` file plus optional
  supporting resources like examples, templates, and scripts.
- A **file skill**: A single markdown file with a ``.md`` extension.

Skills are attached to prompt sections via the ``skills`` parameter, following
the same pattern as tools. They follow the same visibility rules: skills
attached to sections with SUMMARY visibility are not collected until the
section is expanded.

SKILL.md Format
---------------

Every skill requires a ``SKILL.md`` file with YAML frontmatter containing
metadata about the skill. The format is::

    ---
    name: my-skill
    description: A brief description of what this skill does
    license: MIT
    compatibility: claude-3-opus, claude-3-sonnet
    metadata:
      version: "1.0.0"
      author: "Your Name"
    allowed-tools: "Bash, Read, Write"
    ---

    # My Skill

    Instructions and documentation for the skill...

Frontmatter Fields:

- ``name`` (required): Skill identifier, 1-64 characters, lowercase
  alphanumeric with hyphens. Must match directory name for directory skills.
- ``description`` (required): Brief description, 1-1024 characters.
- ``license`` (optional): SPDX license identifier (e.g., "MIT", "Apache-2.0").
- ``compatibility`` (optional): Comma-separated list of compatible models or
  agents, up to 500 characters.
- ``metadata`` (optional): Key-value pairs for custom metadata, both keys and
  values must be strings.
- ``allowed-tools`` (optional): Comma-separated list of tools the skill can use.

Skill Name Conventions
----------------------

Skill names must follow these rules per the Agent Skills specification:

- Length: 1-64 characters
- Characters: lowercase letters (a-z), numbers (0-9), and hyphens (-)
- Cannot start or end with a hyphen
- Cannot contain consecutive hyphens

Valid examples: ``code-review``, ``my-skill-2``, ``testing``
Invalid examples: ``My-Skill``, ``-skill``, ``skill--v2``, ``../escape``

Basic Usage
-----------

Attaching skills to prompt sections::

    from pathlib import Path
    from weakincentives.prompt import MarkdownSection, PromptTemplate
    from weakincentives.skills import SkillMount

    # Attach skills to a section
    section = MarkdownSection(
        title="Code Review",
        key="code-review",
        template="Review the code for issues.",
        skills=(
            SkillMount(Path("./skills/code-review")),
            SkillMount(Path("./skills/testing")),
        ),
    )

    # Create prompt with the section
    template = PromptTemplate(
        ns="demo",
        key="reviewer",
        sections=[section],
    )

Working with Skill Mounts
-------------------------

SkillMount provides flexible configuration options::

    from pathlib import Path
    from weakincentives.skills import SkillMount

    # Mount a directory skill (name derived from directory)
    review_skill = SkillMount(Path("./skills/code-review"))

    # Mount with a custom name override
    custom_skill = SkillMount(
        source=Path("./internal/review-v2"),
        name="code-review",
    )

Conditional Skills via Section Visibility
-----------------------------------------

Skills follow section visibility rules. Attach skills to sections with
``visibility=SUMMARY`` to defer skill loading until the section is expanded::

    from weakincentives.prompt import MarkdownSection, SectionVisibility
    from weakincentives.skills import SkillMount

    # Skills only loaded when section is expanded
    advanced_section = MarkdownSection(
        title="Advanced Analysis",
        key="advanced",
        template="Perform deep code analysis.",
        summary="Advanced analysis capabilities available on request.",
        visibility=SectionVisibility.SUMMARY,
        skills=(
            SkillMount(Path("./skills/deep-analysis")),
            SkillMount(Path("./skills/performance-profiling")),
        ),
    )

Skill Name Resolution
---------------------

The :func:`resolve_skill_name` function determines the skill name using this
priority order::

    from pathlib import Path
    from weakincentives.skills import SkillMount, resolve_skill_name

    # 1. Explicit name takes priority
    mount = SkillMount(Path("./v2/review"), name="code-review")
    assert resolve_skill_name(mount) == "code-review"

    # 2. Directory name for directory skills
    mount = SkillMount(Path("./skills/code-review"))
    assert resolve_skill_name(mount) == "code-review"

    # 3. File stem for file skills (strips .md extension)
    mount = SkillMount(Path("./my-skill.md"))
    assert resolve_skill_name(mount) == "my-skill"

Skill Validation
----------------

The :func:`validate_skill` function performs comprehensive validation::

    from pathlib import Path
    from weakincentives.skills import validate_skill, SkillValidationError

    try:
        # Validates directory has SKILL.md with valid frontmatter
        validate_skill(Path("./skills/code-review"))

        # Validates file is markdown with valid frontmatter
        validate_skill(Path("./my-skill.md"))
    except SkillValidationError as e:
        print(f"Validation failed: {e}")

Validation checks include:

- Directory skills must contain a ``SKILL.md`` file at the root
- File skills must have a ``.md`` extension
- File skills must not exceed :data:`MAX_SKILL_FILE_BYTES` (1 MiB)
- SKILL.md must have valid YAML frontmatter
- Required fields (``name``, ``description``) must be present
- Field values must meet type and length constraints
- For directory skills, ``name`` must match the directory name

The :func:`validate_skill_name` function validates just the name format::

    from weakincentives.skills import validate_skill_name, SkillMountError

    validate_skill_name("code-review")  # OK
    validate_skill_name("my-skill-2")   # OK

    try:
        validate_skill_name("../escape")  # Raises SkillMountError
    except SkillMountError as e:
        print(f"Invalid name: {e}")

Error Handling
--------------

All skill errors inherit from :class:`SkillError`, allowing unified handling::

    from weakincentives.skills import (
        SkillError,
        SkillValidationError,
        SkillNotFoundError,
        SkillMountError,
    )

    try:
        # Perform skill operations
        validate_skill(path)
    except SkillValidationError:
        # Handle invalid skill structure
        pass
    except SkillNotFoundError:
        # Handle missing skill source
        pass
    except SkillMountError:
        # Handle mounting failures
        pass
    except SkillError:
        # Catch any other skill error
        pass

Dependencies
------------

Skill validation requires ``pyyaml`` for parsing SKILL.md frontmatter.
Install with::

    pip install 'weakincentives[skills]'

Exports
-------

Core Types:

- :class:`Skill`: Core representation of a skill definition with name, source,
  and optional content.
- :class:`SkillMount`: Configuration for mounting a skill to a prompt section.

Error Types:

- :class:`SkillError`: Base class for all skill-related exceptions.
- :class:`SkillValidationError`: Raised when skill structure validation fails.
- :class:`SkillNotFoundError`: Raised when a skill source path does not exist.
- :class:`SkillMountError`: Raised when skill mounting or name validation fails.

Validation Functions:

- :func:`validate_skill`: Validate skill structure and frontmatter.
- :func:`validate_skill_name`: Validate a skill name follows the specification.
- :func:`resolve_skill_name`: Derive the effective skill name from a mount.

Constants:

- :data:`MAX_SKILL_FILE_BYTES`: Maximum size for an individual skill file (1 MiB).
- :data:`MAX_SKILL_TOTAL_BYTES`: Maximum total size for a skill (10 MiB).
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
    "SkillError",
    "SkillMount",
    "SkillMountError",
    "SkillNotFoundError",
    "SkillValidationError",
    "resolve_skill_name",
    "validate_skill",
    "validate_skill_name",
]

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

import re
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from ._errors import SkillMountError, SkillValidationError
from ._types import MAX_SKILL_FILE_BYTES, SkillMount


class _YAMLModule(Protocol):
    """Protocol for the yaml module interface we use."""

    YAMLError: type[Exception]

    def safe_load(self, stream: str) -> object: ...


_ERROR_MESSAGE = (
    "pyyaml is required for skill validation. "
    "Install it with: pip install 'weakincentives[skills]'"
)


def _load_yaml_module() -> _YAMLModule:
    """Lazily load the yaml module.

    Returns:
        The yaml module.

    Raises:
        RuntimeError: If pyyaml is not installed.
    """
    try:
        module = import_module("yaml")
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(_ERROR_MESSAGE) from exc
    return module  # type: ignore[return-value]


__all__ = [
    "resolve_skill_name",
    "validate_skill",
    "validate_skill_name",
]

# Pattern for valid skill names according to Agent Skills spec
# Must start/end with alphanumeric, no consecutive hyphens
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# Skill name and description length limits from Agent Skills spec
_MAX_SKILL_NAME_LENGTH = 64
_MAX_SKILL_DESCRIPTION_LENGTH = 1024
_MAX_COMPATIBILITY_LENGTH = 500


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
    """Validate that a skill name follows Agent Skills specification.

    According to the Agent Skills spec, skill names must:

    - Be 1-64 characters long
    - Contain only lowercase letters, numbers, and hyphens
    - Not start or end with a hyphen
    - Not contain consecutive hyphens

    Args:
        name: The skill name to validate.

    Raises:
        SkillMountError: If the name doesn't meet the specification.

    Example::

        from weakincentives.skills import validate_skill_name

        validate_skill_name("code-review")  # OK
        validate_skill_name("my-skill-2")   # OK

        validate_skill_name("My-Skill")     # Raises (uppercase)
        validate_skill_name("-skill")       # Raises (starts with hyphen)
        validate_skill_name("skill--v2")    # Raises (consecutive hyphens)
        validate_skill_name("../escape")    # Raises (invalid characters)
    """
    if not name:
        msg = "Skill name cannot be empty"
        raise SkillMountError(msg)

    if len(name) > _MAX_SKILL_NAME_LENGTH:
        msg = f"Skill name exceeds {_MAX_SKILL_NAME_LENGTH} characters: {name}"
        raise SkillMountError(msg)

    if not _SKILL_NAME_PATTERN.match(name):
        msg = (
            f"Invalid skill name: {name}. "
            "Must contain only lowercase letters, numbers, and hyphens. "
            "Cannot start or end with hyphen or contain consecutive hyphens."
        )
        raise SkillMountError(msg)


def _parse_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML frontmatter from SKILL.md content.

    Args:
        content: The full SKILL.md file content.

    Returns:
        Dictionary of frontmatter fields.

    Raises:
        SkillValidationError: If frontmatter is missing or invalid.
        RuntimeError: If pyyaml is not installed.
    """
    if not content.startswith("---\n"):
        msg = "SKILL.md must start with YAML frontmatter (---)"
        raise SkillValidationError(msg)

    # Find the closing ---
    end_marker = "\n---\n"
    end_pos = content.find(end_marker, 4)
    if end_pos == -1:
        msg = "SKILL.md frontmatter must end with ---"
        raise SkillValidationError(msg)

    yaml_content = content[4:end_pos]
    yaml_module = _load_yaml_module()

    try:
        parsed = yaml_module.safe_load(yaml_content)
    except yaml_module.YAMLError as e:
        msg = f"Invalid YAML in SKILL.md frontmatter: {e}"
        raise SkillValidationError(msg) from e

    if not isinstance(parsed, dict):
        msg = "SKILL.md frontmatter must be a mapping"
        raise SkillValidationError(msg)

    # Type-checked: we've verified it's a dict
    # pyright doesn't know yaml.safe_load's return type, but we validate it's a dict
    frontmatter: dict[str, Any] = parsed  # type: ignore[assignment]
    return frontmatter


def _validate_name_field(frontmatter: dict[str, Any], dir_name: str | None) -> None:
    """Validate the name field in frontmatter."""
    if "name" not in frontmatter:
        msg = "SKILL.md frontmatter missing required field: name"
        raise SkillValidationError(msg)

    name = frontmatter["name"]
    if not isinstance(name, str):
        msg = f"SKILL.md frontmatter 'name' must be a string, got {type(name).__name__}"
        raise SkillValidationError(msg)

    if not name:
        msg = "SKILL.md frontmatter 'name' cannot be empty"
        raise SkillValidationError(msg)

    if len(name) > _MAX_SKILL_NAME_LENGTH:
        msg = f"SKILL.md frontmatter 'name' exceeds {_MAX_SKILL_NAME_LENGTH} characters: {name}"
        raise SkillValidationError(msg)

    if not _SKILL_NAME_PATTERN.match(name):
        msg = (
            f"SKILL.md frontmatter 'name' is invalid: {name}. "
            "Must contain only lowercase letters, numbers, and hyphens. "
            "Cannot start or end with hyphen or contain consecutive hyphens."
        )
        raise SkillValidationError(msg)

    # For directory skills, name must match directory name
    if dir_name is not None and name != dir_name:
        msg = (
            f"SKILL.md frontmatter 'name' ({name}) must match "
            f"directory name ({dir_name})"
        )
        raise SkillValidationError(msg)


def _validate_description_field(frontmatter: dict[str, Any]) -> None:
    """Validate the description field in frontmatter."""
    if "description" not in frontmatter:
        msg = "SKILL.md frontmatter missing required field: description"
        raise SkillValidationError(msg)

    description = frontmatter["description"]
    if not isinstance(description, str):
        msg = (
            f"SKILL.md frontmatter 'description' must be a string, "
            f"got {type(description).__name__}"
        )
        raise SkillValidationError(msg)

    if not description or len(description) < 1:
        msg = "SKILL.md frontmatter 'description' cannot be empty"
        raise SkillValidationError(msg)

    if len(description) > _MAX_SKILL_DESCRIPTION_LENGTH:
        msg = (
            "SKILL.md frontmatter 'description' exceeds "
            f"{_MAX_SKILL_DESCRIPTION_LENGTH} characters"
        )
        raise SkillValidationError(msg)


def _validate_license_field(frontmatter: dict[str, Any]) -> None:
    """Validate the optional license field."""
    if "license" in frontmatter:
        license_val = frontmatter["license"]
        if not isinstance(license_val, str):
            msg = (
                f"SKILL.md frontmatter 'license' must be a string, "
                f"got {type(license_val).__name__}"
            )
            raise SkillValidationError(msg)


def _validate_compatibility_field(frontmatter: dict[str, Any]) -> None:
    """Validate the optional compatibility field."""
    if "compatibility" in frontmatter:
        compat = frontmatter["compatibility"]
        if not isinstance(compat, str):
            msg = (
                f"SKILL.md frontmatter 'compatibility' must be a string, "
                f"got {type(compat).__name__}"
            )
            raise SkillValidationError(msg)

        if len(compat) > _MAX_COMPATIBILITY_LENGTH:
            msg = (
                "SKILL.md frontmatter 'compatibility' exceeds "
                f"{_MAX_COMPATIBILITY_LENGTH} characters"
            )
            raise SkillValidationError(msg)


def _validate_metadata_field(frontmatter: dict[str, Any]) -> None:
    """Validate the optional metadata field."""
    if "metadata" in frontmatter:
        metadata_value = frontmatter["metadata"]
        if not isinstance(metadata_value, dict):
            msg = (
                f"SKILL.md frontmatter 'metadata' must be a mapping, "
                f"got {type(metadata_value).__name__}"
            )
            raise SkillValidationError(msg)

        # All keys and values must be strings
        # Type-checked: we've verified metadata_value is a dict
        # pyright doesn't know the dict types from YAML, but we validate at runtime
        metadata_dict: dict[Any, Any] = metadata_value  # pyright: ignore[reportUnknownVariableType]
        for key, value in metadata_dict.items():
            if not isinstance(key, str):
                msg = "SKILL.md frontmatter 'metadata' keys must be strings"
                raise SkillValidationError(msg)
            if not isinstance(value, str):
                msg = "SKILL.md frontmatter 'metadata' values must be strings"
                raise SkillValidationError(msg)


def _validate_allowed_tools_field(frontmatter: dict[str, Any]) -> None:
    """Validate the optional allowed-tools field."""
    if "allowed-tools" in frontmatter:
        tools = frontmatter["allowed-tools"]
        if not isinstance(tools, str):
            msg = (
                f"SKILL.md frontmatter 'allowed-tools' must be a string, "
                f"got {type(tools).__name__}"
            )
            raise SkillValidationError(msg)


def _validate_optional_fields(frontmatter: dict[str, Any]) -> None:
    """Validate optional fields in frontmatter."""
    _validate_license_field(frontmatter)
    _validate_compatibility_field(frontmatter)
    _validate_metadata_field(frontmatter)
    _validate_allowed_tools_field(frontmatter)


def _validate_frontmatter(frontmatter: dict[str, Any], dir_name: str | None) -> None:
    """Validate SKILL.md frontmatter fields.

    Args:
        frontmatter: Parsed frontmatter dictionary.
        dir_name: Directory name for directory skills (to validate name match),
            or None for file skills.

    Raises:
        SkillValidationError: If any required field is missing or invalid.
    """
    _validate_name_field(frontmatter, dir_name)
    _validate_description_field(frontmatter)
    _validate_optional_fields(frontmatter)


def validate_skill(source: Path) -> None:
    """Validate skill structure before mounting.

    This function checks that a skill path meets the requirements of the
    Agent Skills specification:

    - **Directory skills** must contain a SKILL.md file at the root
    - **File skills** must have a .md extension and not exceed size limits
    - **SKILL.md** must have valid YAML frontmatter with required fields

    The frontmatter is validated for:

    - Required fields: ``name`` and ``description``
    - Optional fields: ``license``, ``compatibility``, ``metadata``, ``allowed-tools``
    - Field constraints (length, format, types)
    - For directory skills, ``name`` must match the directory name

    Args:
        source: Path to the skill file or directory.

    Raises:
        SkillValidationError: If the skill structure is invalid.

    Example::

        from pathlib import Path
        from weakincentives.skills import validate_skill

        # Validates a directory skill has SKILL.md with valid frontmatter
        validate_skill(Path("./skills/code-review"))

        # Validates a file skill is markdown with valid frontmatter
        validate_skill(Path("./my-skill.md"))
    """
    if source.is_dir():
        # Directory skill must contain SKILL.md
        skill_file = source / "SKILL.md"
        if not skill_file.is_file():
            msg = f"Skill directory missing SKILL.md: {source}"
            raise SkillValidationError(msg)

        # Validate frontmatter
        content = skill_file.read_text(encoding="utf-8")
        frontmatter = _parse_frontmatter(content)
        _validate_frontmatter(frontmatter, dir_name=source.name)
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

        # Validate frontmatter
        content = source.read_text(encoding="utf-8")
        frontmatter = _parse_frontmatter(content)
        # File skills don't require name to match anything
        _validate_frontmatter(frontmatter, dir_name=None)

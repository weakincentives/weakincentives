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

The validation functions check that skills conform to the SKILL.md format
standard, including YAML frontmatter requirements and file/directory
structure constraints.

Public API:
    - :func:`resolve_skill_name`: Derive skill name from a mount configuration
    - :func:`validate_skill_name`: Check that a skill name is valid
    - :func:`validate_skill`: Validate skill structure and frontmatter

Note:
    Validation requires the ``pyyaml`` package. Install it with::

        pip install 'weakincentives[skills]'
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
    """Resolve the effective skill name from a mount configuration.

    The skill name is determined in priority order:

    1. **Explicit name**: Use ``mount.name`` if provided
    2. **Directory name**: Use the directory name if source is a directory
    3. **File stem**: Use filename without .md extension if source is a file

    This function does NOT validate that the resolved name follows the Agent
    Skills naming convention. Use :func:`validate_skill_name` to check validity.

    Args:
        mount: The skill mount configuration to resolve a name for. The source
            path should exist, though this function does not verify existence.

    Returns:
        The skill name string to use for the destination directory. This is
        used as the skill's identifier for deduplication and discovery.

    Example::

        from pathlib import Path
        from weakincentives.skills import SkillMount, resolve_skill_name

        # Directory: uses directory name
        mount = SkillMount(Path("./skills/code-review"))
        assert resolve_skill_name(mount) == "code-review"

        # File: strips .md extension
        mount = SkillMount(Path("./my-skill.md"))
        assert resolve_skill_name(mount) == "my-skill"

        # Explicit name override (useful for versioning)
        mount = SkillMount(Path("./v2/review"), name="code-review")
        assert resolve_skill_name(mount) == "code-review"

    Note:
        The resolved name should be validated with :func:`validate_skill_name`
        before use in mounting operations to ensure it follows the naming rules.
    """
    if mount.name is not None:
        return mount.name
    if mount.source.is_dir():
        return mount.source.name
    # File: strip .md extension
    return mount.source.stem


def validate_skill_name(name: str) -> None:
    """Validate that a skill name follows Agent Skills specification.

    This function checks that a skill name conforms to the naming rules
    defined in the Agent Skills specification. Call this after
    :func:`resolve_skill_name` to ensure the resolved name is valid.

    Naming rules (per Agent Skills spec):

    - Length: 1-64 characters
    - Characters: lowercase letters (a-z), numbers (0-9), and hyphens (-)
    - Start/end: must start and end with alphanumeric character
    - No consecutive hyphens: ``skill--name`` is invalid

    These rules prevent path traversal attacks and ensure consistent naming
    across different filesystems and platforms.

    Args:
        name: The skill name to validate. Typically obtained from
            :func:`resolve_skill_name` or from user input.

    Raises:
        SkillMountError: If the name is empty, too long, or contains
            invalid characters/patterns. The error message describes
            what rule was violated.

    Example::

        from weakincentives.skills import validate_skill_name, SkillMountError

        # Valid names
        validate_skill_name("code-review")  # OK
        validate_skill_name("my-skill-2")   # OK
        validate_skill_name("a")            # OK (minimum length)

        # Invalid names raise SkillMountError
        validate_skill_name("My-Skill")     # Raises: uppercase not allowed
        validate_skill_name("-skill")       # Raises: starts with hyphen
        validate_skill_name("skill--v2")    # Raises: consecutive hyphens
        validate_skill_name("../escape")    # Raises: invalid characters
        validate_skill_name("")             # Raises: empty name
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
    """Validate skill structure and content before mounting.

    This function performs comprehensive validation of a skill according to
    the Agent Skills specification. It checks both the file/directory structure
    and the SKILL.md frontmatter content.

    Structure validation:

    - **Directory skills**: Must contain a SKILL.md file at the root
    - **File skills**: Must have .md extension and size <= 1 MiB

    Frontmatter validation (in SKILL.md):

    - Must start with YAML frontmatter delimited by ``---``
    - Required fields: ``name`` (str, 1-64 chars) and ``description`` (str, 1-1024 chars)
    - Optional fields: ``license`` (str), ``compatibility`` (str, max 500 chars),
      ``metadata`` (dict[str, str]), ``allowed-tools`` (str)
    - For directory skills, ``name`` must exactly match the directory name

    This function is called automatically during mounting when
    :attr:`SkillConfig.validate_on_mount` is True (the default).

    Args:
        source: Path to the skill file or directory. Must exist on the
            filesystem. Use :meth:`Path.resolve` to get absolute paths
            if needed.

    Raises:
        SkillValidationError: If any validation check fails. The error
            message describes which check failed and why.
        RuntimeError: If pyyaml is not installed (required for frontmatter parsing).

    Example::

        from pathlib import Path
        from weakincentives.skills import validate_skill, SkillValidationError

        # Validate a directory skill
        try:
            validate_skill(Path("./skills/code-review"))
            print("Skill is valid")
        except SkillValidationError as e:
            print(f"Invalid skill: {e}")

        # Validate a file skill
        validate_skill(Path("./my-skill.md"))

    Note:
        This function reads the SKILL.md file content to parse frontmatter.
        For large skill directories, only the SKILL.md file is read during
        validation; other files are checked during mounting.

    See Also:
        :class:`SkillValidationError`: Exception raised on validation failure.
        :data:`MAX_SKILL_FILE_BYTES`: Maximum file size for file skills.
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

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

"""Authoring and loading of agent skills.

A skill is a small markdown document with TOML frontmatter declaring its
metadata. The format is deliberately simple so the loader can rely only on
the standard library:

```
+++
name = "code-review"
description = "Review a Python diff for issues"
+++
You are a careful code reviewer. ...
```

Use :func:`load_skill` to read a single ``.md`` file and :func:`load_skills`
to read every skill in a directory.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, final

from weakincentives.core import WinkError

__all__ = [
    "Skill",
    "SkillError",
    "SkillMount",
    "SkillNotFoundError",
    "SkillValidationError",
    "load_skill",
    "load_skills",
    "render_skill",
]

_FENCE = "+++"


class SkillError(WinkError):
    """Base class for skill failures."""


class SkillValidationError(SkillError):
    """Raised when a skill file is malformed."""


class SkillNotFoundError(SkillError):
    """Raised when a skill cannot be located."""


def _empty_metadata() -> Mapping[str, object]:
    return {}


@final
@dataclass(frozen=True, slots=True)
class Skill:
    """A loaded skill: metadata plus rendered instructions."""

    name: str
    description: str
    instructions: str
    source: Path | None = None
    metadata: Mapping[str, object] = field(default_factory=_empty_metadata)


@final
@dataclass(frozen=True, slots=True)
class SkillMount:
    """Configuration for mounting a :class:`Skill` onto a section.

    Higher-level integrations (for example a future ``SkillsSection``)
    consume :class:`SkillMount` values; the spine itself never reaches
    inside a skill.
    """

    skill: Skill
    enabled: bool = True


# =============================================================================
# Loading
# =============================================================================


def load_skill(path: Path | str) -> Skill:
    """Read a single ``.md`` file as a :class:`Skill`."""
    target = Path(path)
    if not target.exists():
        raise SkillNotFoundError(f"skill not found: {target}")
    raw = target.read_text(encoding="utf-8")
    metadata, body = _split_frontmatter(raw, source=target)
    name = _required_str(metadata, "name", source=target)
    description = _required_str(metadata, "description", source=target)
    return Skill(
        name=name,
        description=description,
        instructions=body.strip(),
        source=target,
        metadata=metadata,
    )


def load_skills(directory: Path | str) -> tuple[Skill, ...]:
    """Load every ``*.md`` skill file directly under ``directory``."""
    base = Path(directory)
    if not base.is_dir():
        raise SkillNotFoundError(f"skills directory not found: {base}")
    return tuple(
        load_skill(child)
        for child in sorted(base.iterdir())
        if child.is_file() and child.suffix == ".md"
    )


def render_skill(skill: Skill) -> str:
    """Render a skill as a plain markdown section heading + body."""
    return f"## {skill.name}\n\n{skill.description}\n\n{skill.instructions}"


# =============================================================================
# Internals
# =============================================================================


def _split_frontmatter(raw: str, *, source: Path) -> tuple[Mapping[str, object], str]:
    if not raw.startswith(_FENCE):
        raise SkillValidationError(
            f"{source}: missing TOML frontmatter (expected '{_FENCE}' fence)"
        )
    try:
        end = raw.index(f"\n{_FENCE}", len(_FENCE))
    except ValueError as error:
        raise SkillValidationError(
            f"{source}: unterminated TOML frontmatter"
        ) from error
    header = raw[len(_FENCE) : end].strip()
    body = raw[end + len(_FENCE) + 1 :].lstrip("\n")
    try:
        parsed = tomllib.loads(header)
    except tomllib.TOMLDecodeError as error:
        raise SkillValidationError(
            f"{source}: invalid TOML frontmatter: {error}"
        ) from error
    return cast("Mapping[str, object]", parsed), body


def _required_str(metadata: Mapping[str, object], key: str, *, source: Path) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise SkillValidationError(
            f"{source}: required field {key!r} must be a non-empty string"
        )
    return value

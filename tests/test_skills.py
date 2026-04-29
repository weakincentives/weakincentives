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

"""Tests for ``weakincentives.skills``."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.skills import (
    Skill,
    SkillMount,
    SkillNotFoundError,
    SkillValidationError,
    load_skill,
    load_skills,
    render_skill,
)


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_load_skill_parses_frontmatter_and_body(tmp_path: Path) -> None:
    skill_file = _write(
        tmp_path / "review.md",
        '+++\nname = "code-review"\ndescription = "Review code"\n+++\n'
        "You are a careful reviewer.\n",
    )
    skill = load_skill(skill_file)
    assert skill.name == "code-review"
    assert skill.description == "Review code"
    assert skill.instructions == "You are a careful reviewer."
    assert skill.source == skill_file


def test_load_skill_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(SkillNotFoundError, match="skill not found"):
        load_skill(tmp_path / "missing.md")


def test_load_skill_rejects_missing_fence(tmp_path: Path) -> None:
    skill_file = _write(tmp_path / "broken.md", "no fence here")
    with pytest.raises(SkillValidationError, match="missing TOML frontmatter"):
        load_skill(skill_file)


def test_load_skill_rejects_unterminated_fence(tmp_path: Path) -> None:
    skill_file = _write(tmp_path / "broken.md", '+++\nname = "x"\n')
    with pytest.raises(SkillValidationError, match="unterminated"):
        load_skill(skill_file)


def test_load_skill_rejects_invalid_toml(tmp_path: Path) -> None:
    skill_file = _write(tmp_path / "broken.md", "+++\n=garbage\n+++\n")
    with pytest.raises(SkillValidationError, match="invalid TOML"):
        load_skill(skill_file)


def test_load_skill_rejects_missing_required_fields(tmp_path: Path) -> None:
    skill_file = _write(tmp_path / "broken.md", '+++\nname = "x"\n+++\nbody\n')
    with pytest.raises(SkillValidationError, match="description"):
        load_skill(skill_file)


def test_load_skill_rejects_non_string_required_fields(tmp_path: Path) -> None:
    skill_file = _write(
        tmp_path / "broken.md",
        '+++\nname = 5\ndescription = "x"\n+++\nbody\n',
    )
    with pytest.raises(SkillValidationError, match="name"):
        load_skill(skill_file)


def test_load_skills_returns_sorted_tuple(tmp_path: Path) -> None:
    _write(
        tmp_path / "b.md",
        '+++\nname = "b"\ndescription = "second"\n+++\nb body\n',
    )
    _write(
        tmp_path / "a.md",
        '+++\nname = "a"\ndescription = "first"\n+++\na body\n',
    )
    _write(tmp_path / "ignore.txt", "not a skill")
    skills = load_skills(tmp_path)
    names = [skill.name for skill in skills]
    assert names == ["a", "b"]


def test_load_skills_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(SkillNotFoundError, match="skills directory"):
        load_skills(tmp_path / "nope")


def test_render_skill_returns_markdown() -> None:
    skill = Skill(name="x", description="d", instructions="body")
    text = render_skill(skill)
    assert "## x" in text
    assert "d" in text
    assert "body" in text


def test_skill_mount_holds_skill_and_enabled_default() -> None:
    skill = Skill(name="x", description="d", instructions="body")
    mount = SkillMount(skill=skill)
    assert mount.skill is skill
    assert mount.enabled is True

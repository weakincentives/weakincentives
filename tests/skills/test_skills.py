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

"""Tests for the core skills module."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.skills import (
    MAX_SKILL_FILE_BYTES,
    MAX_SKILL_TOTAL_BYTES,
    Skill,
    SkillConfig,
    SkillError,
    SkillMount,
    SkillMountError,
    SkillNotFoundError,
    SkillValidationError,
    resolve_skill_name,
    validate_skill,
    validate_skill_name,
)


class TestSkill:
    """Tests for the Skill dataclass."""

    def test_basic_construction(self) -> None:
        skill = Skill(name="test-skill", source=Path("/path/to/skill"))
        assert skill.name == "test-skill"
        assert skill.source == Path("/path/to/skill")
        assert skill.content is None

    def test_with_content(self) -> None:
        skill = Skill(
            name="test-skill",
            source=Path("/path/to/skill"),
            content="# Test Skill\n\nDoes testing.",
        )
        assert skill.content == "# Test Skill\n\nDoes testing."

    def test_is_frozen(self) -> None:
        skill = Skill(name="test", source=Path("/test"))
        with pytest.raises(AttributeError):
            skill.name = "new-name"  # type: ignore[misc]


class TestSkillMount:
    """Tests for the SkillMount dataclass."""

    def test_basic_construction(self) -> None:
        mount = SkillMount(source=Path("/path/to/skill"))
        assert mount.source == Path("/path/to/skill")
        assert mount.name is None
        assert mount.enabled is True

    def test_with_name(self) -> None:
        mount = SkillMount(source=Path("/path/to/skill"), name="custom-name")
        assert mount.name == "custom-name"

    def test_disabled(self) -> None:
        mount = SkillMount(source=Path("/path/to/skill"), enabled=False)
        assert mount.enabled is False

    def test_is_frozen(self) -> None:
        mount = SkillMount(source=Path("/test"))
        with pytest.raises(AttributeError):
            mount.enabled = False  # type: ignore[misc]


class TestSkillConfig:
    """Tests for the SkillConfig dataclass."""

    def test_defaults(self) -> None:
        config = SkillConfig()
        assert config.skills == ()
        assert config.validate_on_mount is True

    def test_with_skills(self) -> None:
        mounts = (
            SkillMount(source=Path("/skill1")),
            SkillMount(source=Path("/skill2")),
        )
        config = SkillConfig(skills=mounts)
        assert len(config.skills) == 2

    def test_validation_disabled(self) -> None:
        config = SkillConfig(validate_on_mount=False)
        assert config.validate_on_mount is False

    def test_is_frozen(self) -> None:
        config = SkillConfig()
        with pytest.raises(AttributeError):
            config.validate_on_mount = False  # type: ignore[misc]


class TestSkillErrors:
    """Tests for skill error hierarchy."""

    def test_skill_error_is_base(self) -> None:
        assert issubclass(SkillValidationError, SkillError)
        assert issubclass(SkillNotFoundError, SkillError)
        assert issubclass(SkillMountError, SkillError)

    def test_can_catch_all_with_skill_error(self) -> None:
        for exc_class in (SkillValidationError, SkillNotFoundError, SkillMountError):
            try:
                raise exc_class("test")
            except SkillError:
                pass  # Expected


class TestResolveSkillName:
    """Tests for resolve_skill_name function."""

    def test_explicit_name(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        mount = SkillMount(source=skill_dir, name="custom-name")
        assert resolve_skill_name(mount) == "custom-name"

    def test_directory_name(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "code-review"
        skill_dir.mkdir()
        mount = SkillMount(source=skill_dir)
        assert resolve_skill_name(mount) == "code-review"

    def test_file_stem(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "testing.md"
        skill_file.write_text("# Testing")
        mount = SkillMount(source=skill_file)
        assert resolve_skill_name(mount) == "testing"


class TestValidateSkillName:
    """Tests for validate_skill_name function."""

    def test_valid_names(self) -> None:
        validate_skill_name("my-skill")
        validate_skill_name("skill_v2")
        validate_skill_name("123-test")

    def test_rejects_forward_slash(self) -> None:
        with pytest.raises(SkillMountError, match="invalid characters"):
            validate_skill_name("path/traversal")

    def test_rejects_backslash(self) -> None:
        with pytest.raises(SkillMountError, match="invalid characters"):
            validate_skill_name("path\\traversal")

    def test_rejects_double_dot(self) -> None:
        with pytest.raises(SkillMountError, match="invalid characters"):
            validate_skill_name("..evil")

    def test_rejects_empty(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("")

    def test_rejects_single_dot(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name(".")


class TestValidateSkill:
    """Tests for validate_skill function."""

    def test_valid_directory_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test Skill")
        validate_skill(skill_dir)  # Should not raise

    def test_directory_missing_skill_md(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        with pytest.raises(SkillValidationError, match=r"missing SKILL\.md"):
            validate_skill(skill_dir)

    def test_valid_file_skill(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "test-skill.md"
        skill_file.write_text("# Test Skill")
        validate_skill(skill_file)  # Should not raise

    def test_file_wrong_extension(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "test-skill.txt"
        skill_file.write_text("Not markdown")
        with pytest.raises(SkillValidationError, match="must be markdown"):
            validate_skill(skill_file)

    def test_file_exceeds_size_limit(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "large-skill.md"
        skill_file.write_text("x" * (MAX_SKILL_FILE_BYTES + 1))
        with pytest.raises(SkillValidationError, match="exceeds size limit"):
            validate_skill(skill_file)


class TestConstants:
    """Tests for skill constants."""

    def test_max_file_bytes(self) -> None:
        assert MAX_SKILL_FILE_BYTES == 1024 * 1024  # 1 MiB

    def test_max_total_bytes(self) -> None:
        assert MAX_SKILL_TOTAL_BYTES == 10 * 1024 * 1024  # 10 MiB

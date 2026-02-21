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

"""Tests for skill mounting, validation, and copying."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from weakincentives.adapters.claude_agent_sdk._ephemeral_home import (
    EphemeralHome,
    _copy_skill,
)
from weakincentives.adapters.claude_agent_sdk.isolation import (
    IsolationConfig,
)
from weakincentives.skills import (
    SkillMount,
    SkillMountError,
    SkillNotFoundError,
    SkillValidationError,
    resolve_skill_name,
    validate_skill,
    validate_skill_name,
)


class TestSkillMount:
    def test_defaults(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill"
        source.mkdir()
        mount = SkillMount(source=source)
        assert mount.source == source
        assert mount.name is None

    def test_with_name(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill"
        source.mkdir()
        mount = SkillMount(source=source, name="custom-name")
        assert mount.name == "custom-name"


class TestResolveSkillName:
    def test_explicit_name(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill"
        source.mkdir()
        mount = SkillMount(source=source, name="explicit")
        assert resolve_skill_name(mount) == "explicit"

    def test_directory_name(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill-dir"
        source.mkdir()
        mount = SkillMount(source=source)
        assert resolve_skill_name(mount) == "my-skill-dir"

    def test_file_name_strips_extension(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill.md"
        source.write_text("# Test")
        mount = SkillMount(source=source)
        assert resolve_skill_name(mount) == "my-skill"


class TestValidateSkillName:
    def test_valid_name(self) -> None:
        validate_skill_name("my-skill")
        validate_skill_name("skill2")
        validate_skill_name("123-test")

    def test_rejects_forward_slash(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("path/traversal")

    def test_rejects_backslash(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("path\\traversal")

    def test_rejects_double_dot(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("..evil")

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(SkillMountError, match="cannot be empty"):
            validate_skill_name("")

    def test_rejects_dot(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name(".")


class TestValidateSkill:
    def test_valid_directory_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: test-skill\n"
            "description: A test skill for validation\n"
            "---\n"
            "\n"
            "# Test Skill\n\nContent"
        )
        validate_skill(skill_dir)  # Should not raise

    def test_directory_missing_skill_md(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        with pytest.raises(SkillValidationError, match=r"missing SKILL\.md"):
            validate_skill(skill_dir)

    def test_valid_file_skill(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "test-skill.md"
        skill_file.write_text(
            "---\n"
            "name: test-skill\n"
            "description: A test skill for validation\n"
            "---\n"
            "\n"
            "# Test Skill\n\nContent"
        )
        validate_skill(skill_file)  # Should not raise

    def test_file_wrong_extension(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "test-skill.txt"
        skill_file.write_text("# Test Skill\n\nContent")
        with pytest.raises(SkillValidationError, match="must be markdown"):
            validate_skill(skill_file)

    def test_file_too_large(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "huge-skill.md"
        # Create a file larger than 1 MiB
        skill_file.write_text("x" * (1024 * 1024 + 1))
        with pytest.raises(SkillValidationError, match="exceeds size limit"):
            validate_skill(skill_file)


class TestCopySkill:
    def test_copy_directory_skill(self, tmp_path: Path) -> None:
        # Create source skill directory
        source = tmp_path / "source-skill"
        source.mkdir()
        skill_content = (
            "---\n"
            "name: source-skill\n"
            "description: A test skill for copying\n"
            "---\n"
            "\n"
            "# Test Skill"
        )
        (source / "SKILL.md").write_text(skill_content)
        (source / "examples").mkdir()
        (source / "examples" / "example.py").write_text("print('hello')")

        dest = tmp_path / "dest-skill"
        bytes_copied = _copy_skill(source, dest)

        assert dest.is_dir()
        assert (dest / "SKILL.md").read_text() == skill_content
        assert (dest / "examples" / "example.py").read_text() == "print('hello')"
        assert bytes_copied > 0

    def test_copy_file_skill_wraps_in_directory(self, tmp_path: Path) -> None:
        # Create source skill file
        source = tmp_path / "skill.md"
        skill_content = (
            "---\n"
            "name: skill\n"
            "description: A single file test skill\n"
            "---\n"
            "\n"
            "# Single File Skill"
        )
        source.write_text(skill_content)

        dest = tmp_path / "dest-skill"
        bytes_copied = _copy_skill(source, dest)

        assert dest.is_dir()
        assert (dest / "SKILL.md").read_text() == skill_content
        assert bytes_copied > 0

    def test_copy_directory_exceeds_size_limit(self, tmp_path: Path) -> None:
        # Create large skill directory
        source = tmp_path / "large-skill"
        source.mkdir()
        (source / "SKILL.md").write_text(
            "---\n"
            "name: large-skill\n"
            "description: A large test skill\n"
            "---\n"
            "\n"
            "# Large Skill"
        )
        (source / "big_file.txt").write_text("x" * 100)

        dest = tmp_path / "dest-skill"
        # Use a very small limit to trigger the error
        with pytest.raises(SkillMountError, match="exceeds total size limit"):
            _copy_skill(source, dest, max_total_bytes=10)

    def test_copy_file_exceeds_size_limit(self, tmp_path: Path) -> None:
        # Create large single-file skill
        source = tmp_path / "large-skill.md"
        source.write_text(
            "---\n"
            "name: large-skill\n"
            "description: A large single file skill\n"
            "---\n"
            "\n"
            "# Large Skill\n" + "x" * 100
        )

        dest = tmp_path / "dest-skill"
        # Use a very small limit to trigger the error
        with pytest.raises(SkillMountError, match="exceeds total size limit"):
            _copy_skill(source, dest, max_total_bytes=10)

    def test_copy_ignores_symlinks_by_default(self, tmp_path: Path) -> None:
        # Create source skill directory with symlink
        source = tmp_path / "source-skill"
        source.mkdir()
        (source / "SKILL.md").write_text(
            "---\n"
            "name: source-skill\n"
            "description: A test skill with symlinks\n"
            "---\n"
            "\n"
            "# Test Skill"
        )
        external_file = tmp_path / "external.txt"
        external_file.write_text("external content")
        (source / "link.txt").symlink_to(external_file)

        dest = tmp_path / "dest-skill"
        _copy_skill(source, dest, follow_symlinks=False)

        assert dest.is_dir()
        assert (dest / "SKILL.md").exists()
        assert not (dest / "link.txt").exists()  # Symlink should be skipped

    def test_copy_raises_on_io_error(self, tmp_path: Path) -> None:
        # Create a source file
        source = tmp_path / "skill.md"
        source.write_text(
            "---\n"
            "name: skill\n"
            "description: A test skill for error handling\n"
            "---\n"
            "\n"
            "# Test Skill"
        )

        dest = tmp_path / "dest-skill"

        # Mock shutil.copy2 to raise OSError
        with mock.patch("shutil.copy2", side_effect=OSError("Disk full")):
            with pytest.raises(SkillMountError, match="Failed to copy skill"):
                _copy_skill(source, dest)


class TestEphemeralHomeMountSkills:
    """Tests for EphemeralHome.mount_skills() method."""

    def test_mounts_directory_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: my-skill\n"
            "description: A test skill for mounting\n"
            "---\n"
            "\n"
            "# My Skill\n"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=skill_dir),))
            assert home.skills_dir.is_dir()
            skill_dest = home.skills_dir / "my-skill"
            assert skill_dest.is_dir()
            content = (skill_dest / "SKILL.md").read_text()
            assert "# My Skill" in content

    def test_mounts_file_skill(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "my-skill.md"
        skill_file.write_text(
            "---\n"
            "name: my-skill\n"
            "description: A file-based test skill\n"
            "---\n"
            "\n"
            "# File Skill\n"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=skill_file),))
            skill_dest = home.skills_dir / "my-skill"
            assert skill_dest.is_dir()
            content = (skill_dest / "SKILL.md").read_text()
            assert "# File Skill" in content

    def test_mounts_skill_with_custom_name(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "original-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: original-name\n"
            "description: A skill with a custom mount name\n"
            "---\n"
            "\n"
            "# Custom Named\n"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=skill_dir, name="custom-name"),))
            assert (home.skills_dir / "custom-name").is_dir()
            assert not (home.skills_dir / "original-name").exists()

    def test_mounts_multiple_skills(self, tmp_path: Path) -> None:
        # Create two skills
        skill1 = tmp_path / "skill-one"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text(
            "---\nname: skill-one\ndescription: First test skill\n---\n\n# Skill One\n"
        )

        skill2 = tmp_path / "skill-two"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text(
            "---\nname: skill-two\ndescription: Second test skill\n---\n\n# Skill Two\n"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home.mount_skills(
                (
                    SkillMount(source=skill1),
                    SkillMount(source=skill2),
                )
            )
            assert (home.skills_dir / "skill-one").is_dir()
            assert (home.skills_dir / "skill-two").is_dir()

    def test_rejects_duplicate_skill_names(self, tmp_path: Path) -> None:
        skill1 = tmp_path / "skill-a"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: First skill\n---\n\n# Skill A\n"
        )

        skill2 = tmp_path / "skill-b"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text(
            "---\nname: skill-b\ndescription: Second skill\n---\n\n# Skill B\n"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            with pytest.raises(SkillMountError, match="Duplicate skill name"):
                home.mount_skills(
                    (
                        SkillMount(source=skill1, name="same-name"),
                        SkillMount(source=skill2, name="same-name"),
                    )
                )

    def test_raises_on_missing_skill_source(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does-not-exist"
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            with pytest.raises(SkillNotFoundError, match="Skill not found"):
                home.mount_skills((SkillMount(source=nonexistent),))

    def test_validates_skill_when_enabled(self, tmp_path: Path) -> None:
        # Directory without SKILL.md
        invalid_skill = tmp_path / "invalid-skill"
        invalid_skill.mkdir()

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            with pytest.raises(SkillValidationError, match=r"missing SKILL\.md"):
                home.mount_skills((SkillMount(source=invalid_skill),), validate=True)

    def test_skips_validation_when_disabled(self, tmp_path: Path) -> None:
        # Directory without SKILL.md (would fail validation)
        invalid_skill = tmp_path / "invalid-skill"
        invalid_skill.mkdir()
        # Create some content to copy
        (invalid_skill / "README.md").write_text("# Not a skill")

        config = IsolationConfig()
        # Should not raise because validation is disabled
        with EphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=invalid_skill),), validate=False)
            assert (home.skills_dir / "invalid-skill").is_dir()

    def test_no_skills_directory_when_no_skills_mounted(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            # skills_dir property should return path but dir shouldn't exist
            assert home.skills_dir == home.claude_dir / "skills"
            assert not home.skills_dir.exists()

    def test_empty_skills_tuple_does_nothing(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home.mount_skills(())  # Empty tuple
            assert home.skills_dir == home.claude_dir / "skills"
            assert not home.skills_dir.exists()

    def test_skills_dir_property(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=skill_dir),))
            assert home.skills_dir == home.claude_dir / "skills"
            assert home.skills_dir.is_dir()

    def test_mount_skills_rejects_second_call(self, tmp_path: Path) -> None:
        """mount_skills() can only be called once per EphemeralHome instance."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test"
        )

        config = IsolationConfig()
        with EphemeralHome(config) as home:
            # First call succeeds
            home.mount_skills((SkillMount(source=skill_dir),))

            # Second call raises
            with pytest.raises(SkillMountError, match="Skills already mounted"):
                home.mount_skills((SkillMount(source=skill_dir),))

    def test_mount_skills_rejects_second_call_even_with_empty_first(self) -> None:
        """mount_skills() can only be called once even if first call was empty."""
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            # First call with empty tuple succeeds
            home.mount_skills(())

            # Second call raises even though first was empty
            with pytest.raises(SkillMountError, match="Skills already mounted"):
                home.mount_skills(())

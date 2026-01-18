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
from unittest import mock

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
from weakincentives.skills._validation import _load_yaml_module


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
        # Valid according to Agent Skills spec
        validate_skill_name("my-skill")
        validate_skill_name("code-review")
        validate_skill_name("123-test")
        validate_skill_name("a")
        validate_skill_name("skill2")
        validate_skill_name("my-skill-v2")

    def test_rejects_uppercase(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("My-Skill")

    def test_rejects_underscore(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("skill_v2")

    def test_rejects_leading_hyphen(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("-skill")

    def test_rejects_trailing_hyphen(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("skill-")

    def test_rejects_consecutive_hyphens(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("skill--v2")

    def test_rejects_forward_slash(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("path/traversal")

    def test_rejects_backslash(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("path\\traversal")

    def test_rejects_double_dot(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            validate_skill_name("..evil")

    def test_rejects_empty(self) -> None:
        with pytest.raises(SkillMountError, match="cannot be empty"):
            validate_skill_name("")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(SkillMountError, match="exceeds 64 characters"):
            validate_skill_name("a" * 65)


class TestValidateSkill:
    """Tests for validate_skill function."""

    def test_valid_directory_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: test-skill\n"
            "description: A test skill for validation\n"
            "---\n"
            "\n"
            "# Test Skill\n"
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
            "# Test Skill\n"
        )
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

    def test_missing_frontmatter(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test Skill\n\nNo frontmatter")
        with pytest.raises(
            SkillValidationError, match="must start with YAML frontmatter"
        ):
            validate_skill(skill_dir)

    def test_unclosed_frontmatter(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test-skill\n")
        with pytest.raises(SkillValidationError, match="must end with ---"):
            validate_skill(skill_dir)

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ninvalid: yaml: syntax:\n---\n"
        )
        with pytest.raises(SkillValidationError, match="Invalid YAML"):
            validate_skill(skill_dir)

    def test_frontmatter_not_dict(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\n- item1\n- item2\n---\n")
        with pytest.raises(SkillValidationError, match="must be a mapping"):
            validate_skill(skill_dir)

    def test_missing_name_field(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: A test skill\n---\n")
        with pytest.raises(SkillValidationError, match="missing required field: name"):
            validate_skill(skill_dir)

    def test_missing_description_field(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test-skill\n---\n")
        with pytest.raises(
            SkillValidationError, match="missing required field: description"
        ):
            validate_skill(skill_dir)

    def test_name_not_string(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: 123\ndescription: A test skill\n---\n"
        )
        with pytest.raises(SkillValidationError, match="'name' must be a string"):
            validate_skill(skill_dir)

    def test_description_not_string(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: 123\n---\n"
        )
        with pytest.raises(
            SkillValidationError, match="'description' must be a string"
        ):
            validate_skill(skill_dir)

    def test_name_empty(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            '---\nname: ""\ndescription: A test skill\n---\n'
        )
        with pytest.raises(SkillValidationError, match="'name' cannot be empty"):
            validate_skill(skill_dir)

    def test_description_empty(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            '---\nname: test-skill\ndescription: ""\n---\n'
        )
        with pytest.raises(SkillValidationError, match="'description' cannot be empty"):
            validate_skill(skill_dir)

    def test_name_too_long(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        long_name = "a" * 65
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {long_name}\ndescription: A test skill\n---\n"
        )
        with pytest.raises(SkillValidationError, match="'name' exceeds 64 characters"):
            validate_skill(skill_dir)

    def test_description_too_long(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        long_desc = "a" * 1025
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: test-skill\ndescription: {long_desc}\n---\n"
        )
        with pytest.raises(
            SkillValidationError, match="'description' exceeds 1024 characters"
        ):
            validate_skill(skill_dir)

    def test_name_invalid_format(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: Invalid-Name\ndescription: A test skill\n---\n"
        )
        with pytest.raises(SkillValidationError, match="'name' is invalid"):
            validate_skill(skill_dir)

    def test_name_mismatch_directory(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: different-name\ndescription: A test skill\n---\n"
        )
        with pytest.raises(SkillValidationError, match="must match directory name"):
            validate_skill(skill_dir)

    def test_file_skill_name_no_match_required(self, tmp_path: Path) -> None:
        # File skills don't require name to match filename
        skill_file = tmp_path / "my-file.md"
        skill_file.write_text(
            "---\nname: different-name\ndescription: A test skill\n---\n"
        )
        validate_skill(skill_file)  # Should not raise

    def test_valid_with_all_optional_fields(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: test-skill\n"
            "description: A comprehensive test skill\n"
            "license: Apache-2.0\n"
            "compatibility: Requires git, docker, jq\n"
            "metadata:\n"
            "  author: test-author\n"
            "  version: '1.0'\n"
            "allowed-tools: Bash(git:*) Read\n"
            "---\n"
        )
        validate_skill(skill_dir)  # Should not raise

    def test_license_not_string(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\nlicense: 123\n---\n"
        )
        with pytest.raises(SkillValidationError, match="'license' must be a string"):
            validate_skill(skill_dir)

    def test_compatibility_not_string(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: test-skill\n"
            "description: A test skill\n"
            "compatibility: 123\n"
            "---\n"
        )
        with pytest.raises(
            SkillValidationError, match="'compatibility' must be a string"
        ):
            validate_skill(skill_dir)

    def test_compatibility_too_long(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        long_compat = "a" * 501
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: test-skill\n"
            "description: A test skill\n"
            f"compatibility: {long_compat}\n"
            "---\n"
        )
        with pytest.raises(
            SkillValidationError, match="'compatibility' exceeds 500 characters"
        ):
            validate_skill(skill_dir)

    def test_metadata_not_dict(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: test-skill\n"
            "description: A test skill\n"
            "metadata: not-a-dict\n"
            "---\n"
        )
        with pytest.raises(SkillValidationError, match="'metadata' must be a mapping"):
            validate_skill(skill_dir)

    def test_metadata_non_string_value(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: test-skill\n"
            "description: A test skill\n"
            "metadata:\n"
            "  version: 1.0\n"
            "---\n"
        )
        with pytest.raises(
            SkillValidationError, match="'metadata' values must be strings"
        ):
            validate_skill(skill_dir)

    def test_metadata_non_string_key(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        # YAML allows non-string keys
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: test-skill\n"
            "description: A test skill\n"
            "metadata:\n"
            "  123: value\n"
            "---\n"
        )
        with pytest.raises(
            SkillValidationError, match="'metadata' keys must be strings"
        ):
            validate_skill(skill_dir)

    def test_allowed_tools_not_string(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: test-skill\n"
            "description: A test skill\n"
            "allowed-tools: 123\n"
            "---\n"
        )
        with pytest.raises(
            SkillValidationError, match="'allowed-tools' must be a string"
        ):
            validate_skill(skill_dir)


class TestConstants:
    """Tests for skill constants."""

    def test_max_file_bytes(self) -> None:
        assert MAX_SKILL_FILE_BYTES == 1024 * 1024  # 1 MiB

    def test_max_total_bytes(self) -> None:
        assert MAX_SKILL_TOTAL_BYTES == 10 * 1024 * 1024  # 10 MiB


class TestDemoSkills:
    """Tests to validate all demo-skills."""

    def test_python_style_skill_is_valid(self) -> None:
        # Get path relative to repository root
        skill_path = (
            Path(__file__).parent.parent.parent / "demo-skills" / "python-style"
        )
        if not skill_path.exists():
            pytest.skip("demo-skills/python-style not found")
        validate_skill(skill_path)

    def test_ascii_art_skill_is_valid(self) -> None:
        skill_path = Path(__file__).parent.parent.parent / "demo-skills" / "ascii-art"
        if not skill_path.exists():
            pytest.skip("demo-skills/ascii-art not found")
        validate_skill(skill_path)

    def test_code_review_skill_is_valid(self) -> None:
        skill_path = Path(__file__).parent.parent.parent / "demo-skills" / "code-review"
        if not skill_path.exists():
            pytest.skip("demo-skills/code-review not found")
        validate_skill(skill_path)

    def test_all_demo_skills_have_valid_names(self) -> None:
        demo_skills_dir = Path(__file__).parent.parent.parent / "demo-skills"
        if not demo_skills_dir.exists():
            pytest.skip("demo-skills directory not found")

        # Find all subdirectories with SKILL.md files
        skill_dirs = [
            d
            for d in demo_skills_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        ]

        assert len(skill_dirs) > 0, "No demo skills found"

        for skill_dir in skill_dirs:
            # Validate the skill
            validate_skill(skill_dir)
            # Also validate the directory name
            validate_skill_name(skill_dir.name)


class TestLazyYamlImport:
    """Tests for lazy yaml import behavior."""

    def test_load_yaml_module_success(self) -> None:
        """Verify yaml module loads successfully when installed."""
        module = _load_yaml_module()
        # Should have safe_load function
        assert hasattr(module, "safe_load")
        assert hasattr(module, "YAMLError")

    def test_load_yaml_module_raises_when_missing(self) -> None:
        """Verify helpful error when yaml is not installed."""
        with mock.patch.dict("sys.modules", {"yaml": None}):
            # Clear cached import
            with mock.patch(
                "weakincentives.skills._validation.import_module",
                side_effect=ModuleNotFoundError("No module named 'yaml'"),
            ):
                with pytest.raises(RuntimeError, match="pyyaml is required"):
                    _load_yaml_module()

    def test_error_message_includes_install_hint(self) -> None:
        """Verify error message includes install instructions."""
        with mock.patch(
            "weakincentives.skills._validation.import_module",
            side_effect=ModuleNotFoundError("No module named 'yaml'"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _load_yaml_module()
            assert "pip install 'weakincentives[skills]'" in str(exc_info.value)

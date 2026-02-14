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

"""Tests for CodexEphemeralHome."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from weakincentives.adapters.codex_app_server._ephemeral_home import (
    CodexEphemeralHome,
    _copy_skill,
)
from weakincentives.skills import (
    SkillMount,
    SkillMountError,
    SkillNotFoundError,
    SkillValidationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SKILL_FRONTMATTER = "---\nname: {name}\ndescription: A test skill\n---\n\n# {name}\n"


def _make_dir_skill(base: Path, name: str) -> Path:
    """Create a minimal directory skill and return its path."""
    d = base / name
    d.mkdir()
    (d / "SKILL.md").write_text(_SKILL_FRONTMATTER.format(name=name))
    return d


def _make_file_skill(base: Path, name: str) -> Path:
    """Create a minimal single-file skill and return its path."""
    f = base / f"{name}.md"
    f.write_text(_SKILL_FRONTMATTER.format(name=name))
    return f


# ---------------------------------------------------------------------------
# _copy_skill
# ---------------------------------------------------------------------------


class TestCopySkill:
    def test_copies_directory_skill(self, tmp_path: Path) -> None:
        src = _make_dir_skill(tmp_path, "my-skill")
        dest = tmp_path / "dest" / "my-skill"
        total = _copy_skill(src, dest)
        assert total > 0
        assert (dest / "SKILL.md").is_file()

    def test_copies_file_skill_as_skill_md(self, tmp_path: Path) -> None:
        src = _make_file_skill(tmp_path, "my-skill")
        dest = tmp_path / "dest" / "my-skill"
        total = _copy_skill(src, dest)
        assert total > 0
        assert (dest / "SKILL.md").is_file()
        assert "# my-skill" in (dest / "SKILL.md").read_text()

    def test_preserves_directory_structure(self, tmp_path: Path) -> None:
        src = tmp_path / "deep-skill"
        src.mkdir()
        (src / "SKILL.md").write_text(_SKILL_FRONTMATTER.format(name="deep-skill"))
        sub = src / "examples"
        sub.mkdir()
        (sub / "example.txt").write_text("example content")

        dest = tmp_path / "dest" / "deep-skill"
        _copy_skill(src, dest)
        assert (dest / "examples" / "example.txt").read_text() == "example content"

    def test_enforces_size_limit(self, tmp_path: Path) -> None:
        src = _make_file_skill(tmp_path, "big-skill")
        dest = tmp_path / "dest" / "big-skill"
        with pytest.raises(SkillMountError, match="size limit"):
            _copy_skill(src, dest, max_total_bytes=10)

    def test_enforces_size_limit_directory(self, tmp_path: Path) -> None:
        src = _make_dir_skill(tmp_path, "big-dir")
        dest = tmp_path / "dest" / "big-dir"
        with pytest.raises(SkillMountError, match="size limit"):
            _copy_skill(src, dest, max_total_bytes=10)

    def test_skips_symlinks_by_default(self, tmp_path: Path) -> None:
        src = tmp_path / "link-skill"
        src.mkdir()
        (src / "SKILL.md").write_text(_SKILL_FRONTMATTER.format(name="link-skill"))
        target = tmp_path / "target.txt"
        target.write_text("target")
        (src / "link.txt").symlink_to(target)

        dest = tmp_path / "dest" / "link-skill"
        _copy_skill(src, dest, follow_symlinks=False)
        assert not (dest / "link.txt").exists()
        assert (dest / "SKILL.md").is_file()

    def test_os_error_wrapped(self, tmp_path: Path) -> None:
        src = tmp_path / "noexist"
        dest = tmp_path / "dest"
        with pytest.raises(SkillMountError, match="Failed to copy"):
            _copy_skill(src, dest)


# ---------------------------------------------------------------------------
# CodexEphemeralHome — lifecycle
# ---------------------------------------------------------------------------


class TestCodexEphemeralHomeLifecycle:
    def test_creates_temp_directory(self) -> None:
        with CodexEphemeralHome() as home:
            assert Path(home.home_path).is_dir()

    def test_cleanup_removes_directory(self) -> None:
        home = CodexEphemeralHome()
        path = home.home_path
        home.cleanup()
        assert not Path(path).exists()

    def test_cleanup_idempotent(self) -> None:
        home = CodexEphemeralHome()
        home.cleanup()
        home.cleanup()  # second call should not raise

    def test_context_manager_cleans_up(self) -> None:
        with CodexEphemeralHome() as home:
            path = home.home_path
            assert Path(path).is_dir()
        assert not Path(path).exists()

    def test_custom_prefix(self) -> None:
        with CodexEphemeralHome(temp_dir_prefix="custom-prefix-") as home:
            assert "custom-prefix-" in Path(home.home_path).name


# ---------------------------------------------------------------------------
# CodexEphemeralHome — mount_skills
# ---------------------------------------------------------------------------


class TestCodexEphemeralHomeMountSkills:
    def test_mounts_directory_skill(self, tmp_path: Path) -> None:
        src = _make_dir_skill(tmp_path, "my-skill")
        with CodexEphemeralHome() as home:
            home.mount_skills((SkillMount(source=src),))
            assert (home.skills_dir / "my-skill" / "SKILL.md").is_file()

    def test_mounts_file_skill(self, tmp_path: Path) -> None:
        src = _make_file_skill(tmp_path, "my-skill")
        with CodexEphemeralHome() as home:
            home.mount_skills((SkillMount(source=src),))
            dest = home.skills_dir / "my-skill"
            assert dest.is_dir()
            assert (dest / "SKILL.md").is_file()

    def test_mounts_skill_with_custom_name(self, tmp_path: Path) -> None:
        src = _make_dir_skill(tmp_path, "original-name")
        with CodexEphemeralHome() as home:
            home.mount_skills((SkillMount(source=src, name="custom-name"),))
            assert (home.skills_dir / "custom-name").is_dir()
            assert not (home.skills_dir / "original-name").exists()

    def test_mounts_multiple_skills(self, tmp_path: Path) -> None:
        src1 = _make_dir_skill(tmp_path, "skill-one")
        src2 = _make_dir_skill(tmp_path, "skill-two")
        with CodexEphemeralHome() as home:
            home.mount_skills((SkillMount(source=src1), SkillMount(source=src2)))
            assert (home.skills_dir / "skill-one").is_dir()
            assert (home.skills_dir / "skill-two").is_dir()

    def test_rejects_duplicate_skill_names(self, tmp_path: Path) -> None:
        src1 = _make_dir_skill(tmp_path, "skill-a")
        src2 = _make_dir_skill(tmp_path, "skill-b")
        with CodexEphemeralHome() as home:
            with pytest.raises(SkillMountError, match="Duplicate skill name"):
                home.mount_skills(
                    (
                        SkillMount(source=src1, name="same-name"),
                        SkillMount(source=src2, name="same-name"),
                    )
                )

    def test_raises_on_missing_source(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does-not-exist"
        with CodexEphemeralHome() as home:
            with pytest.raises(SkillNotFoundError, match="Skill not found"):
                home.mount_skills((SkillMount(source=nonexistent),))

    def test_double_mount_raises(self, tmp_path: Path) -> None:
        src = _make_dir_skill(tmp_path, "my-skill")
        with CodexEphemeralHome() as home:
            home.mount_skills((SkillMount(source=src),))
            with pytest.raises(SkillMountError, match="Skills already mounted"):
                home.mount_skills((SkillMount(source=src),))

    def test_double_mount_raises_even_with_empty_first(self) -> None:
        with CodexEphemeralHome() as home:
            home.mount_skills(())
            with pytest.raises(SkillMountError, match="Skills already mounted"):
                home.mount_skills(())

    def test_empty_tuple_does_nothing(self) -> None:
        with CodexEphemeralHome() as home:
            home.mount_skills(())
            assert not home.skills_dir.exists()

    def test_validates_skill_when_enabled(self, tmp_path: Path) -> None:
        invalid = tmp_path / "invalid-skill"
        invalid.mkdir()
        # no SKILL.md
        with CodexEphemeralHome() as home:
            with pytest.raises(SkillValidationError, match=r"missing SKILL\.md"):
                home.mount_skills((SkillMount(source=invalid),), validate=True)

    def test_skips_validation_when_disabled(self, tmp_path: Path) -> None:
        invalid = tmp_path / "invalid-skill"
        invalid.mkdir()
        (invalid / "README.md").write_text("# Not a skill")
        with CodexEphemeralHome() as home:
            home.mount_skills((SkillMount(source=invalid),), validate=False)
            assert (home.skills_dir / "invalid-skill").is_dir()

    def test_skills_dir_property(self, tmp_path: Path) -> None:
        src = _make_dir_skill(tmp_path, "test-skill")
        with CodexEphemeralHome() as home:
            home.mount_skills((SkillMount(source=src),))
            assert home.skills_dir == Path(home.home_path) / ".agents" / "skills"
            assert home.skills_dir.is_dir()


# ---------------------------------------------------------------------------
# CodexEphemeralHome — get_env
# ---------------------------------------------------------------------------


class TestCodexEphemeralHomeGetEnv:
    def test_returns_home_pointing_to_temp(self) -> None:
        with CodexEphemeralHome() as home:
            env = home.get_env()
            assert env["HOME"] == home.home_path

    def test_returns_codex_home_pointing_to_original(self) -> None:
        with patch.dict("os.environ", {"HOME": "/real/home"}):
            with CodexEphemeralHome() as home:
                env = home.get_env()
                assert env["CODEX_HOME"] == "/real/home/.codex"

    def test_no_codex_home_when_home_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with CodexEphemeralHome() as home:
                env = home.get_env()
                assert "CODEX_HOME" not in env
                assert "HOME" in env

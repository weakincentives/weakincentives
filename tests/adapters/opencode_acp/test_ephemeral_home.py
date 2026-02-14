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

"""Tests for the OpenCode ephemeral home directory."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from weakincentives.skills import SkillMount, SkillMountError, SkillNotFoundError


def _make_skill_dir(base: Path, name: str = "test-skill") -> Path:
    """Create a minimal valid skill directory."""
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {name}\ndescription: A test skill\n---\n\n# {name}\n\nTest content.\n"
    )
    return skill_dir


def _make_skill_file(base: Path, name: str = "test-skill") -> Path:
    """Create a minimal valid skill file."""
    skill_file = base / f"{name}.md"
    skill_file.write_text(
        f"---\nname: {name}\ndescription: A test skill\n---\n\n# {name}\n\nTest content.\n"
    )
    return skill_file


class TestOpenCodeEphemeralHome:
    def test_creation_and_cleanup(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        home = OpenCodeEphemeralHome(workspace_path=str(tmp_path))
        home_path = home.home_path
        assert Path(home_path).exists()
        assert (Path(home_path) / ".claude").is_dir()

        home.cleanup()
        assert not Path(home_path).exists()

    def test_context_manager(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        with OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home:
            home_path = home.home_path
            assert Path(home_path).exists()

        assert not Path(home_path).exists()

    def test_double_cleanup_safe(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        home = OpenCodeEphemeralHome(workspace_path=str(tmp_path))
        home.cleanup()
        # Should not raise
        home.cleanup()

    def test_home_path_property(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        with OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home:
            assert home.home_path.startswith("/")
            assert "opencode-agent-" in home.home_path

    def test_skills_dir_property(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        with OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home:
            assert home.skills_dir == Path(home.home_path) / ".claude" / "skills"

    def test_destructor_cleanup(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        home = OpenCodeEphemeralHome(workspace_path=str(tmp_path))
        home_path = home.home_path
        assert Path(home_path).exists()
        home.__del__()
        assert not Path(home_path).exists()


class TestAuthPassthrough:
    def test_copies_opencode_auth_when_present(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        opencode_dir = fake_home / ".local" / "share" / "opencode"
        opencode_dir.mkdir(parents=True)
        (opencode_dir / "auth.json").write_text('{"key": "value"}')

        with (
            patch.dict(os.environ, {"HOME": str(fake_home)}),
            OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home,
        ):
            copied = (
                Path(home.home_path) / ".local" / "share" / "opencode" / "auth.json"
            )
            assert copied.exists()
            assert copied.read_text() == '{"key": "value"}'

    def test_copies_aws_config_when_present(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        aws_dir = fake_home / ".aws"
        aws_dir.mkdir()
        (aws_dir / "credentials").write_text("[default]\naws_access_key_id=test\n")

        with (
            patch.dict(os.environ, {"HOME": str(fake_home)}),
            OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home,
        ):
            copied = Path(home.home_path) / ".aws" / "credentials"
            assert copied.exists()

    def test_graceful_skip_when_no_auth_dirs(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        fake_home = tmp_path / "emptyhome"
        fake_home.mkdir()

        with (
            patch.dict(os.environ, {"HOME": str(fake_home)}),
            OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home,
        ):
            assert not (Path(home.home_path) / ".local").exists()
            assert not (Path(home.home_path) / ".aws").exists()

    def test_graceful_skip_when_home_not_set(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        env = dict(os.environ)
        env.pop("HOME", None)
        with (
            patch.dict(os.environ, env, clear=True),
            OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home,
        ):
            assert Path(home.home_path).exists()

    def test_os_error_during_opencode_copy_warns(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        opencode_dir = fake_home / ".local" / "share" / "opencode"
        opencode_dir.mkdir(parents=True)
        (opencode_dir / "auth.json").write_text("data")

        with (
            patch.dict(os.environ, {"HOME": str(fake_home)}),
            patch("shutil.copytree", side_effect=OSError("permission denied")),
            OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home,
        ):
            # Should not raise, just warn
            assert Path(home.home_path).exists()

    def test_os_error_during_aws_copy_warns(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        aws_dir = fake_home / ".aws"
        aws_dir.mkdir()
        (aws_dir / "config").write_text("[default]")

        original_copytree = __import__("shutil").copytree

        def _fail_on_aws(src: object, dst: object, **kwargs: object) -> object:
            if ".aws" in str(src):
                raise OSError("permission denied")
            return original_copytree(src, dst, **kwargs)  # type: ignore[arg-type]

        with (
            patch.dict(os.environ, {"HOME": str(fake_home)}),
            patch("shutil.copytree", side_effect=_fail_on_aws),
            OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home,
        ):
            assert Path(home.home_path).exists()


class TestMountSkills:
    def test_valid_directory_skill(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        skill_dir = _make_skill_dir(tmp_path, "my-skill")

        with OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home:
            home.mount_skills((SkillMount(skill_dir),))
            installed = home.skills_dir / "my-skill" / "SKILL.md"
            assert installed.exists()
            assert "A test skill" in installed.read_text()

    def test_valid_file_skill(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        skill_file = _make_skill_file(tmp_path, "file-skill")

        with OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home:
            home.mount_skills((SkillMount(skill_file),))
            installed = home.skills_dir / "file-skill" / "SKILL.md"
            assert installed.exists()

    def test_duplicate_name_rejection(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        skill_dir = _make_skill_dir(tmp_path, "dupe-skill")

        with OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home:
            with pytest.raises(SkillMountError, match="Duplicate skill name"):
                home.mount_skills(
                    (
                        SkillMount(skill_dir),
                        SkillMount(skill_dir, name="dupe-skill"),
                    )
                )

    def test_missing_source_rejection(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        with OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home:
            with pytest.raises(SkillNotFoundError, match="Skill not found"):
                home.mount_skills((SkillMount(tmp_path / "nonexistent-skill"),))

    def test_skills_already_mounted_guard(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        skill_dir = _make_skill_dir(tmp_path, "once-skill")

        with OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home:
            home.mount_skills((SkillMount(skill_dir),))
            with pytest.raises(SkillMountError, match="already mounted"):
                home.mount_skills((SkillMount(skill_dir),))

    def test_mount_empty_skills_tuple(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        with OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home:
            home.mount_skills(())
            # skills_dir should not be created when no skills are passed
            assert not home.skills_dir.exists()

    def test_validation_disabled(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import (
            OpenCodeEphemeralHome,
        )

        # Create a skill dir without valid frontmatter
        skill_dir = tmp_path / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("No frontmatter here.")

        with OpenCodeEphemeralHome(workspace_path=str(tmp_path)) as home:
            # With validation disabled, it should copy without checking frontmatter
            home.mount_skills((SkillMount(skill_dir),), validate=False)
            assert (home.skills_dir / "bad-skill" / "SKILL.md").exists()


class TestCopySkill:
    def test_directory_copy(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import _copy_skill

        source = tmp_path / "src-skill"
        source.mkdir()
        (source / "SKILL.md").write_text("content")
        sub = source / "examples"
        sub.mkdir()
        (sub / "example.py").write_text("print('hello')")

        dest = tmp_path / "dest-skill"
        total = _copy_skill(source, dest)
        assert total > 0
        assert (dest / "SKILL.md").exists()
        assert (dest / "examples" / "example.py").exists()

    def test_single_file_as_skill_md(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import _copy_skill

        source = tmp_path / "skill.md"
        source.write_text("---\nname: skill\n---\nContent")

        dest = tmp_path / "dest-skill"
        total = _copy_skill(source, dest)
        assert total > 0
        assert (dest / "SKILL.md").exists()
        assert (dest / "SKILL.md").read_text() == source.read_text()

    def test_size_limit_enforcement(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import _copy_skill

        source = tmp_path / "big-skill"
        source.mkdir()
        (source / "SKILL.md").write_text("x" * 100)

        dest = tmp_path / "dest"
        with pytest.raises(SkillMountError, match="exceeds total size limit"):
            _copy_skill(source, dest, max_total_bytes=10)

    def test_size_limit_enforcement_single_file(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import _copy_skill

        source = tmp_path / "big.md"
        source.write_text("x" * 100)

        dest = tmp_path / "dest"
        with pytest.raises(SkillMountError, match="exceeds total size limit"):
            _copy_skill(source, dest, max_total_bytes=10)

    def test_symlink_skipping(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import _copy_skill

        source = tmp_path / "link-skill"
        source.mkdir()
        (source / "SKILL.md").write_text("content")

        # Create a symlink inside the source
        target = tmp_path / "target.txt"
        target.write_text("target content")
        (source / "linked.txt").symlink_to(target)

        dest = tmp_path / "dest-skill"
        _copy_skill(source, dest, follow_symlinks=False)
        assert (dest / "SKILL.md").exists()
        assert not (dest / "linked.txt").exists()

    def test_os_error_wrapping(self, tmp_path: Path) -> None:
        from weakincentives.adapters.opencode_acp._ephemeral_home import _copy_skill

        source = tmp_path / "err-skill"
        source.mkdir()
        (source / "SKILL.md").write_text("content")

        dest = tmp_path / "dest"
        with (
            patch("shutil.copy2", side_effect=OSError("disk full")),
            pytest.raises(SkillMountError, match="Failed to copy skill"),
        ):
            _copy_skill(source, dest)

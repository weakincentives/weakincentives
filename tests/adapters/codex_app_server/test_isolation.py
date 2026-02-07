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

"""Tests for Codex App Server hermetic home directory isolation."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from weakincentives.adapters.codex_app_server.isolation import (
    _CODEX_CREDENTIAL_VARS,
    _SENSITIVE_ENV_PREFIXES,
    CodexEphemeralHome,
    CodexHermeticHomeConfig,
    _copy_skill,
)
from weakincentives.skills import (
    SkillMount,
    SkillMountError,
    SkillNotFoundError,
    SkillValidationError,
)

# ---- Helpers ----


def _make_valid_skill_dir(base: Path, name: str) -> Path:
    """Create a valid directory skill with SKILL.md and frontmatter."""
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill {name}\n---\n\n# {name}\n"
    )
    return skill_dir


def _make_valid_skill_file(base: Path, name: str) -> Path:
    """Create a valid single-file skill."""
    skill_file = base / f"{name}.md"
    skill_file.write_text(
        f"---\nname: {name}\ndescription: File skill {name}\n---\n\n# {name}\n"
    )
    return skill_file


# ---- Config tests ----


class TestCodexHermeticHomeConfig:
    def test_defaults(self) -> None:
        config = CodexHermeticHomeConfig()
        assert config.copy_host_credentials is True
        assert config.include_host_env is False
        assert config.env is None

    def test_custom_values(self) -> None:
        config = CodexHermeticHomeConfig(
            copy_host_credentials=False,
            include_host_env=True,
            env={"CUSTOM_VAR": "value"},
        )
        assert config.copy_host_credentials is False
        assert config.include_host_env is True
        assert config.env is not None
        assert config.env["CUSTOM_VAR"] == "value"

    def test_frozen(self) -> None:
        config = CodexHermeticHomeConfig()
        with pytest.raises(AttributeError):
            config.copy_host_credentials = False  # type: ignore[misc]


# ---- _copy_skill tests ----


class TestCopySkill:
    def test_copies_directory_skill(self, tmp_path: Path) -> None:
        source = _make_valid_skill_dir(tmp_path, "my-skill")
        dest = tmp_path / "dest-skill"

        bytes_copied = _copy_skill(source, dest)

        assert (dest / "SKILL.md").exists()
        assert bytes_copied > 0

    def test_copies_file_skill(self, tmp_path: Path) -> None:
        source = _make_valid_skill_file(tmp_path, "my-skill")
        dest = tmp_path / "dest-skill"

        bytes_copied = _copy_skill(source, dest)

        assert (dest / "SKILL.md").exists()
        content = (dest / "SKILL.md").read_text()
        assert "# my-skill" in content
        assert bytes_copied > 0

    def test_copies_nested_directory_skill(self, tmp_path: Path) -> None:
        source = _make_valid_skill_dir(tmp_path, "nested-skill")
        subdir = source / "examples"
        subdir.mkdir()
        (subdir / "example.py").write_text("print('hello')")

        dest = tmp_path / "dest-skill"
        bytes_copied = _copy_skill(source, dest)

        assert (dest / "SKILL.md").exists()
        assert (dest / "examples" / "example.py").exists()
        assert bytes_copied > 0

    def test_rejects_oversized_file_skill(self, tmp_path: Path) -> None:
        source = tmp_path / "big-skill.md"
        source.write_text("x" * 100)
        dest = tmp_path / "dest-skill"

        with pytest.raises(SkillMountError, match="exceeds total size limit"):
            _copy_skill(source, dest, max_total_bytes=10)

    def test_rejects_oversized_directory_skill(self, tmp_path: Path) -> None:
        source = tmp_path / "big-dir"
        source.mkdir()
        (source / "large.txt").write_text("x" * 100)

        dest = tmp_path / "dest-skill"
        with pytest.raises(SkillMountError, match="exceeds total size limit"):
            _copy_skill(source, dest, max_total_bytes=10)

    def test_skips_symlinks_by_default(self, tmp_path: Path) -> None:
        source = tmp_path / "link-skill"
        source.mkdir()
        real_file = tmp_path / "real.txt"
        real_file.write_text("content")
        (source / "link.txt").symlink_to(real_file)
        (source / "normal.txt").write_text("normal")

        dest = tmp_path / "dest-skill"
        _copy_skill(source, dest)

        assert (dest / "normal.txt").exists()
        assert not (dest / "link.txt").exists()

    def test_os_error_raises_skill_mount_error(self, tmp_path: Path) -> None:
        source = _make_valid_skill_dir(tmp_path, "err-skill")
        dest = tmp_path / "dest-skill"

        with mock.patch("shutil.copy2", side_effect=OSError("Disk full")):
            with pytest.raises(SkillMountError, match="Failed to copy skill"):
                _copy_skill(source, dest)


# ---- EphemeralHome basic tests ----


class TestCodexEphemeralHome:
    def test_creates_temp_dir(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            assert Path(home.home_path).is_dir()

    def test_creates_codex_dir(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            assert home.codex_dir.is_dir()
            assert home.codex_dir == Path(home.home_path) / ".codex"

    def test_cleanup_removes_temp_dir(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        home = CodexEphemeralHome(config)
        temp_dir = home.home_path
        assert Path(temp_dir).is_dir()

        home.cleanup()
        assert not Path(temp_dir).is_dir()

    def test_cleanup_is_idempotent(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        home = CodexEphemeralHome(config)
        home.cleanup()
        home.cleanup()  # Should not raise

    def test_context_manager(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            temp_dir = home.home_path
            assert Path(temp_dir).is_dir()
        assert not Path(temp_dir).is_dir()

    def test_properties(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            assert home.codex_dir == Path(home.home_path) / ".codex"
            assert home.skills_dir == Path(home.home_path) / ".codex" / "skills"


# ---- Credential copying tests ----


class TestCopyHostCredentials:
    def test_copies_codex_dir_from_host(self, tmp_path: Path) -> None:
        # Set up a fake host home with .codex directory
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()
        host_codex = fake_home / ".codex"
        host_codex.mkdir()
        (host_codex / "auth.json").write_text('{"token": "secret"}')

        config = CodexHermeticHomeConfig(copy_host_credentials=True)
        with mock.patch.dict(os.environ, {"HOME": str(fake_home)}):
            with CodexEphemeralHome(config) as home:
                copied_auth = home.codex_dir / "auth.json"
                assert copied_auth.exists()
                assert '"token"' in copied_auth.read_text()

    def test_copies_nested_codex_content(self, tmp_path: Path) -> None:
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()
        host_codex = fake_home / ".codex"
        host_codex.mkdir()
        (host_codex / "config.json").write_text("{}")
        sub = host_codex / "sessions"
        sub.mkdir()
        (sub / "session.json").write_text("{}")

        config = CodexHermeticHomeConfig(copy_host_credentials=True)
        with mock.patch.dict(os.environ, {"HOME": str(fake_home)}):
            with CodexEphemeralHome(config) as home:
                assert (home.codex_dir / "config.json").exists()
                assert (home.codex_dir / "sessions" / "session.json").exists()

    def test_skips_when_home_not_set(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=True)
        with mock.patch.dict(os.environ, {}, clear=True):
            # Re-add PATH so the process doesn't break
            with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}):
                home = CodexEphemeralHome(config)
                try:
                    # Should not raise; .codex is created empty
                    assert home.codex_dir.is_dir()
                finally:
                    home.cleanup()

    def test_skips_when_codex_dir_not_found(self, tmp_path: Path) -> None:
        fake_home = tmp_path / "empty-home"
        fake_home.mkdir()
        # No .codex directory exists

        config = CodexHermeticHomeConfig(copy_host_credentials=True)
        with mock.patch.dict(os.environ, {"HOME": str(fake_home)}):
            with CodexEphemeralHome(config) as home:
                # .codex dir exists (created by init) but has no host content
                assert home.codex_dir.is_dir()
                assert not (home.codex_dir / "auth.json").exists()

    def test_skips_on_copy_error(self, tmp_path: Path) -> None:
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()
        host_codex = fake_home / ".codex"
        host_codex.mkdir()
        (host_codex / "config.json").write_text("{}")

        config = CodexHermeticHomeConfig(copy_host_credentials=True)
        with mock.patch.dict(os.environ, {"HOME": str(fake_home)}):
            with mock.patch(
                "shutil.copytree", side_effect=OSError("Permission denied")
            ):
                # Should not raise
                with CodexEphemeralHome(config) as home:
                    assert home.codex_dir.is_dir()

    def test_disabled_via_config(self, tmp_path: Path) -> None:
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()
        host_codex = fake_home / ".codex"
        host_codex.mkdir()
        (host_codex / "auth.json").write_text('{"token": "secret"}')

        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with mock.patch.dict(os.environ, {"HOME": str(fake_home)}):
            with CodexEphemeralHome(config) as home:
                # .codex dir exists but auth.json was NOT copied
                assert home.codex_dir.is_dir()
                assert not (home.codex_dir / "auth.json").exists()


# ---- Skill mounting tests ----


class TestMountSkills:
    def test_mounts_directory_skill(self, tmp_path: Path) -> None:
        skill_dir = _make_valid_skill_dir(tmp_path, "my-skill")

        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=skill_dir),))
            assert home.skills_dir.is_dir()
            skill_dest = home.skills_dir / "my-skill"
            assert skill_dest.is_dir()
            assert "# my-skill" in (skill_dest / "SKILL.md").read_text()

    def test_mounts_file_skill(self, tmp_path: Path) -> None:
        skill_file = _make_valid_skill_file(tmp_path, "my-skill")

        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=skill_file),))
            skill_dest = home.skills_dir / "my-skill"
            assert skill_dest.is_dir()
            assert "# my-skill" in (skill_dest / "SKILL.md").read_text()

    def test_mounts_skill_with_custom_name(self, tmp_path: Path) -> None:
        skill_dir = _make_valid_skill_dir(tmp_path, "original-name")

        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=skill_dir, name="custom-name"),))
            assert (home.skills_dir / "custom-name").is_dir()
            assert not (home.skills_dir / "original-name").exists()

    def test_mounts_multiple_skills(self, tmp_path: Path) -> None:
        skill1 = _make_valid_skill_dir(tmp_path, "skill-one")
        skill2 = _make_valid_skill_dir(tmp_path, "skill-two")

        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            home.mount_skills(
                (
                    SkillMount(source=skill1),
                    SkillMount(source=skill2),
                )
            )
            assert (home.skills_dir / "skill-one").is_dir()
            assert (home.skills_dir / "skill-two").is_dir()

    def test_rejects_duplicate_skill_names(self, tmp_path: Path) -> None:
        skill1 = _make_valid_skill_dir(tmp_path, "skill-a")
        skill2 = _make_valid_skill_dir(tmp_path, "skill-b")

        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            with pytest.raises(SkillMountError, match="Duplicate skill name"):
                home.mount_skills(
                    (
                        SkillMount(source=skill1, name="same-name"),
                        SkillMount(source=skill2, name="same-name"),
                    )
                )

    def test_raises_on_missing_source(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does-not-exist"

        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            with pytest.raises(SkillNotFoundError, match="Skill not found"):
                home.mount_skills((SkillMount(source=nonexistent),))

    def test_validates_when_enabled(self, tmp_path: Path) -> None:
        invalid_skill = tmp_path / "invalid-skill"
        invalid_skill.mkdir()
        # No SKILL.md → validation will fail

        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            with pytest.raises(SkillValidationError, match=r"missing SKILL\.md"):
                home.mount_skills((SkillMount(source=invalid_skill),), validate=True)

    def test_skips_validation_when_disabled(self, tmp_path: Path) -> None:
        invalid_skill = tmp_path / "invalid-skill"
        invalid_skill.mkdir()
        (invalid_skill / "README.md").write_text("# Not a skill")

        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=invalid_skill),), validate=False)
            assert (home.skills_dir / "invalid-skill").is_dir()

    def test_empty_skills_does_nothing(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            home.mount_skills(())
            assert not home.skills_dir.exists()

    def test_rejects_second_call(self, tmp_path: Path) -> None:
        skill_dir = _make_valid_skill_dir(tmp_path, "test-skill")

        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            home.mount_skills((SkillMount(source=skill_dir),))
            with pytest.raises(SkillMountError, match="Skills already mounted"):
                home.mount_skills((SkillMount(source=skill_dir),))

    def test_rejects_second_call_even_after_empty(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            home.mount_skills(())
            with pytest.raises(SkillMountError, match="Skills already mounted"):
                home.mount_skills(())

    def test_skills_dir_not_created_without_mount(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            assert home.skills_dir == home.codex_dir / "skills"
            assert not home.skills_dir.exists()


# ---- get_env tests ----


class TestGetEnv:
    def test_includes_home(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with CodexEphemeralHome(config) as home:
            env = home.get_env()
            assert env["HOME"] == home.home_path

    def test_includes_path(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin:/usr/local/bin"}):
            with CodexEphemeralHome(config) as home:
                env = home.get_env()
                assert env["PATH"] == "/usr/bin:/usr/local/bin"

    def test_passes_credential_vars(self) -> None:
        test_env = {
            "OPENAI_API_KEY": "sk-test-key",
            "OPENAI_ORG_ID": "org-123",
            "OPENAI_BASE_URL": "https://api.example.com",
            "CODEX_API_KEY": "codex-key",
            "PATH": "/usr/bin",
        }
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with mock.patch.dict(os.environ, test_env, clear=True):
            with CodexEphemeralHome(config) as home:
                env = home.get_env()
                for var in _CODEX_CREDENTIAL_VARS:
                    assert env[var] == test_env[var]

    def test_does_not_pass_missing_credential_vars(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
            with CodexEphemeralHome(config) as home:
                env = home.get_env()
                assert "OPENAI_API_KEY" not in env
                assert "CODEX_API_KEY" not in env

    def test_include_host_env(self) -> None:
        test_env = {
            "PATH": "/usr/bin",
            "EDITOR": "vim",
            "TERM": "xterm",
        }
        config = CodexHermeticHomeConfig(
            copy_host_credentials=False, include_host_env=True
        )
        with mock.patch.dict(os.environ, test_env, clear=True):
            with CodexEphemeralHome(config) as home:
                env = home.get_env()
                assert env["EDITOR"] == "vim"
                assert env["TERM"] == "xterm"

    def test_excludes_sensitive_env_when_inheriting(self) -> None:
        test_env = {
            "PATH": "/usr/bin",
            "HOME": "/original/home",
            "OPENAI_API_KEY": "should-not-inherit-this-way",
            "AWS_SECRET_KEY": "secret",
            "ANTHROPIC_API_KEY": "key",
            "EDITOR": "vim",
        }
        config = CodexHermeticHomeConfig(
            copy_host_credentials=False, include_host_env=True
        )
        with mock.patch.dict(os.environ, test_env, clear=True):
            with CodexEphemeralHome(config) as home:
                env = home.get_env()
                # Sensitive vars NOT inherited via include_host_env
                assert env["HOME"] == home.home_path  # Overridden
                assert "AWS_SECRET_KEY" not in env
                assert "ANTHROPIC_API_KEY" not in env
                # But OPENAI_API_KEY IS passed via credential passthrough
                assert env["OPENAI_API_KEY"] == "should-not-inherit-this-way"
                # Non-sensitive vars inherited
                assert env["EDITOR"] == "vim"

    def test_user_env_overrides(self) -> None:
        config = CodexHermeticHomeConfig(
            copy_host_credentials=False,
            env={"CUSTOM_VAR": "custom-value", "ANOTHER": "val"},
        )
        with CodexEphemeralHome(config) as home:
            env = home.get_env()
            assert env["CUSTOM_VAR"] == "custom-value"
            assert env["ANOTHER"] == "val"

    def test_user_env_highest_priority(self) -> None:
        test_env = {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "from-host",
        }
        config = CodexHermeticHomeConfig(
            copy_host_credentials=False,
            env={"OPENAI_API_KEY": "from-config"},
        )
        with mock.patch.dict(os.environ, test_env, clear=True):
            with CodexEphemeralHome(config) as home:
                env = home.get_env()
                # User env overrides credential passthrough
                assert env["OPENAI_API_KEY"] == "from-config"

    def test_without_path_in_env(self) -> None:
        config = CodexHermeticHomeConfig(copy_host_credentials=False)
        with mock.patch.dict(os.environ, {}, clear=True):
            with CodexEphemeralHome(config) as home:
                env = home.get_env()
                assert "PATH" not in env


# ---- Constants tests ----


class TestConstants:
    def test_sensitive_env_prefixes_cover_key_providers(self) -> None:
        expected = {
            "HOME",
            "OPENAI_",
            "CODEX_",
            "ANTHROPIC_",
            "AWS_",
            "GOOGLE_",
            "AZURE_",
        }
        assert set(_SENSITIVE_ENV_PREFIXES) == expected

    def test_credential_vars_cover_openai_and_codex(self) -> None:
        expected = {
            "OPENAI_API_KEY",
            "OPENAI_ORG_ID",
            "OPENAI_BASE_URL",
            "CODEX_API_KEY",
        }
        assert set(_CODEX_CREDENTIAL_VARS) == expected

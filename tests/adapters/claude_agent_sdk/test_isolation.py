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

"""Tests for Claude Agent SDK isolation module."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from weakincentives.adapters.claude_agent_sdk.isolation import (
    EphemeralHome,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
    SkillConfig,
    SkillMount,
    SkillMountError,
    SkillNotFoundError,
    SkillValidationError,
    _copy_skill,
    _validate_skill,
    _validate_skill_name,
    resolve_skill_name,
)


class TestNetworkPolicy:
    def test_defaults(self) -> None:
        policy = NetworkPolicy()
        assert policy.allowed_domains == ()

    def test_no_network_factory(self) -> None:
        policy = NetworkPolicy.no_network()
        assert policy.allowed_domains == ()

    def test_with_domains_factory(self) -> None:
        policy = NetworkPolicy.with_domains("api.github.com", "pypi.org")
        assert policy.allowed_domains == ("api.github.com", "pypi.org")


class TestSandboxConfig:
    def test_defaults(self) -> None:
        config = SandboxConfig()
        assert config.enabled is True
        assert config.writable_paths == ()
        assert config.readable_paths == ()
        assert config.excluded_commands == ()
        assert config.allow_unsandboxed_commands is False
        assert config.bash_auto_allow is True

    def test_with_paths(self) -> None:
        config = SandboxConfig(
            writable_paths=("/tmp/output",),
            readable_paths=("/data/readonly",),
        )
        assert config.writable_paths == ("/tmp/output",)
        assert config.readable_paths == ("/data/readonly",)

    def test_with_excluded_commands(self) -> None:
        config = SandboxConfig(
            excluded_commands=("docker", "podman"),
            allow_unsandboxed_commands=True,
        )
        assert config.excluded_commands == ("docker", "podman")
        assert config.allow_unsandboxed_commands is True

    def test_disabled_sandbox(self) -> None:
        config = SandboxConfig(enabled=False, bash_auto_allow=False)
        assert config.enabled is False
        assert config.bash_auto_allow is False


class TestIsolationConfig:
    def test_defaults(self) -> None:
        config = IsolationConfig()
        assert config.network_policy is None
        assert config.sandbox is None
        assert config.env is None
        assert config.api_key is None
        assert config.include_host_env is False

    def test_with_network_policy(self) -> None:
        policy = NetworkPolicy.no_network()
        config = IsolationConfig(network_policy=policy)
        assert config.network_policy is policy

    def test_with_sandbox(self) -> None:
        sandbox = SandboxConfig(enabled=True)
        config = IsolationConfig(sandbox=sandbox)
        assert config.sandbox is sandbox

    def test_with_env(self) -> None:
        config = IsolationConfig(env={"MY_VAR": "value"})
        assert config.env == {"MY_VAR": "value"}

    def test_with_api_key(self) -> None:
        config = IsolationConfig(api_key="sk-ant-test")
        assert config.api_key == "sk-ant-test"

    def test_with_include_host_env(self) -> None:
        config = IsolationConfig(include_host_env=True)
        assert config.include_host_env is True


class TestEphemeralHome:
    def test_creates_temp_directory(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            assert Path(home.home_path).is_dir()
            # Temp dir location is platform-dependent (e.g., /tmp on Linux, /var/folders on macOS)
            assert "claude-agent-" in home.home_path

    def test_creates_claude_directory(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            assert home.claude_dir.is_dir()
            assert home.claude_dir == Path(home.home_path) / ".claude"

    def test_creates_settings_json(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            assert home.settings_path.is_file()
            settings = json.loads(home.settings_path.read_text())
            assert "sandbox" in settings

    def test_cleanup_removes_directory(self) -> None:
        config = IsolationConfig()
        home = EphemeralHome(config)
        home_path = Path(home.home_path)
        assert home_path.is_dir()
        home.cleanup()
        assert not home_path.exists()

    def test_cleanup_is_idempotent(self) -> None:
        config = IsolationConfig()
        home = EphemeralHome(config)
        home.cleanup()
        home.cleanup()  # Should not raise

    def test_context_manager_cleanup(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            home_path = Path(home.home_path)
            assert home_path.is_dir()
        assert not home_path.exists()

    def test_get_setting_sources_returns_user(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            sources = home.get_setting_sources()
            # Returns ["user"] to load settings from ephemeral HOME
            assert sources == ["user"]


class TestEphemeralHomeSettingsGeneration:
    def test_default_sandbox_settings(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["enabled"] is True
            assert settings["sandbox"]["autoAllowBashIfSandboxed"] is True
            assert settings["sandbox"]["network"]["allowedDomains"] == []

    def test_network_policy_allowed_domains(self) -> None:
        config = IsolationConfig(
            network_policy=NetworkPolicy(
                allowed_domains=("api.anthropic.com", "api.github.com")
            )
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["network"]["allowedDomains"] == [
                "api.anthropic.com",
                "api.github.com",
            ]

    def test_sandbox_disabled(self) -> None:
        config = IsolationConfig(
            sandbox=SandboxConfig(enabled=False, bash_auto_allow=False)
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["enabled"] is False
            assert settings["sandbox"]["autoAllowBashIfSandboxed"] is False

    def test_sandbox_excluded_commands(self) -> None:
        config = IsolationConfig(
            sandbox=SandboxConfig(
                excluded_commands=("docker", "podman"),
                allow_unsandboxed_commands=True,
            )
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["excludedCommands"] == ["docker", "podman"]
            assert settings["sandbox"]["allowUnsandboxedCommands"] is True

    def test_sandbox_writable_paths(self) -> None:
        config = IsolationConfig(
            sandbox=SandboxConfig(writable_paths=("/tmp/output", "/var/log"))
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["writablePaths"] == ["/tmp/output", "/var/log"]

    def test_sandbox_readable_paths(self) -> None:
        config = IsolationConfig(
            sandbox=SandboxConfig(readable_paths=("/data/readonly",))
        )
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert settings["sandbox"]["readablePaths"] == ["/data/readonly"]

    def test_env_section_disables_bedrock(self) -> None:
        """Settings should include env section that explicitly disables Bedrock.

        This is critical for hermetic isolation - the host may have Claude
        configured for AWS Bedrock, and we must force Anthropic API usage.
        """
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            settings = json.loads(home.settings_path.read_text())
            assert "env" in settings
            assert settings["env"]["CLAUDE_CODE_USE_BEDROCK"] == "0"
            assert settings["env"]["CLAUDE_USE_BEDROCK"] == "0"
            assert settings["env"]["DISABLE_AUTOUPDATER"] == "1"


class TestEphemeralHomeEnv:
    def test_home_is_ephemeral_directory(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["HOME"] == home.home_path

    def test_api_key_from_config(self) -> None:
        config = IsolationConfig(api_key="sk-ant-test-key")
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["ANTHROPIC_API_KEY"] == "sk-ant-test-key"

    def test_api_key_from_environment(self) -> None:
        config = IsolationConfig()
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-from-env"}):
            with EphemeralHome(config) as home:
                env = home.get_env()
                assert env["ANTHROPIC_API_KEY"] == "sk-ant-from-env"

    def test_config_api_key_overrides_env(self) -> None:
        config = IsolationConfig(api_key="sk-ant-from-config")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-from-env"}):
            with EphemeralHome(config) as home:
                env = home.get_env()
                assert env["ANTHROPIC_API_KEY"] == "sk-ant-from-config"

    def test_no_api_key_in_config_or_environment(self) -> None:
        config = IsolationConfig()
        # Clear ANTHROPIC_API_KEY from environment
        with mock.patch.dict(os.environ, {}, clear=True):
            with EphemeralHome(config) as home:
                env = home.get_env()
                # Should have HOME and Bedrock-disabling vars but no ANTHROPIC_API_KEY
                assert "HOME" in env
                assert "ANTHROPIC_API_KEY" not in env
                # Should always have Bedrock-disabling vars for hermetic isolation
                assert env["CLAUDE_CODE_USE_BEDROCK"] == "0"
                assert env["CLAUDE_USE_BEDROCK"] == "0"
                assert env["DISABLE_AUTOUPDATER"] == "1"

    def test_custom_env_vars(self) -> None:
        config = IsolationConfig(env={"MY_CUSTOM_VAR": "custom_value"})
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["MY_CUSTOM_VAR"] == "custom_value"

    def test_custom_env_vars_override_generated(self) -> None:
        # Custom env should take precedence over generated values
        config = IsolationConfig(
            api_key="sk-ant-generated",
            env={"ANTHROPIC_API_KEY": "sk-ant-custom-override"},
        )
        with EphemeralHome(config) as home:
            env = home.get_env()
            assert env["ANTHROPIC_API_KEY"] == "sk-ant-custom-override"

    def test_include_host_env_false_excludes_all(self) -> None:
        config = IsolationConfig(include_host_env=False)
        with mock.patch.dict(
            os.environ,
            {"PATH": "/usr/bin", "MY_VAR": "value", "ANTHROPIC_API_KEY": "key"},
            clear=True,
        ):
            with EphemeralHome(config) as home:
                env = home.get_env()
                # Should only have HOME, ANTHROPIC_API_KEY (from environ),
                # and Bedrock-disabling vars (always added for hermetic isolation)
                assert "PATH" not in env
                assert "MY_VAR" not in env
                assert "HOME" in env
                assert "ANTHROPIC_API_KEY" in env
                assert env["CLAUDE_CODE_USE_BEDROCK"] == "0"
                assert env["CLAUDE_USE_BEDROCK"] == "0"
                assert env["DISABLE_AUTOUPDATER"] == "1"

    def test_include_host_env_true_copies_safe_vars(self) -> None:
        config = IsolationConfig(include_host_env=True)
        with mock.patch.dict(
            os.environ,
            {
                "PATH": "/usr/bin",
                "MY_VAR": "value",
                "ANTHROPIC_API_KEY": "key",
                "HOME": "/home/user",
                "CLAUDE_CONFIG": "something",
                "AWS_ACCESS_KEY": "secret",
            },
            clear=True,
        ):
            with EphemeralHome(config) as home:
                env = home.get_env()
                # Should include safe vars
                assert env["PATH"] == "/usr/bin"
                assert env["MY_VAR"] == "value"
                # Should NOT include sensitive vars from host
                # (but HOME is overridden and ANTHROPIC_API_KEY is copied)
                assert env["HOME"] == home.home_path  # Overridden
                assert env["ANTHROPIC_API_KEY"] == "key"  # Explicitly copied
                # Should exclude other sensitive prefixes
                assert "CLAUDE_CONFIG" not in env
                assert "AWS_ACCESS_KEY" not in env

    def test_sensitive_prefixes_excluded(self) -> None:
        config = IsolationConfig(include_host_env=True)
        sensitive_vars = {
            "HOME": "/home/user",
            "CLAUDE_CONFIG_DIR": "/claude",
            "CLAUDE_API_KEY": "key1",
            "ANTHROPIC_API_KEY": "key2",
            "ANTHROPIC_BASE_URL": "url",
            "AWS_SECRET_KEY": "secret",
            "AWS_ACCESS_KEY_ID": "id",
            "GOOGLE_APPLICATION_CREDENTIALS": "creds",
            "GOOGLE_API_KEY": "key",
            "AZURE_CLIENT_SECRET": "secret",
            "OPENAI_API_KEY": "key",
        }
        with mock.patch.dict(os.environ, sensitive_vars, clear=True):
            with EphemeralHome(config) as home:
                env = home.get_env()
                # HOME should be overridden to ephemeral
                assert env["HOME"] == home.home_path
                # ANTHROPIC_API_KEY is explicitly copied
                assert "ANTHROPIC_API_KEY" in env
                # Other sensitive vars should not be inherited
                for key in sensitive_vars:
                    if key == "HOME":
                        continue  # Overridden
                    if key == "ANTHROPIC_API_KEY":
                        continue  # Explicitly copied
                    assert key not in env, f"{key} should be excluded"


class TestEphemeralHomeCustomPrefix:
    def test_custom_prefix(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config, temp_dir_prefix="my-custom-") as home:
            assert "my-custom-" in home.home_path


class TestEphemeralHomeWorkspacePath:
    def test_workspace_path_stored(self) -> None:
        config = IsolationConfig()
        # The workspace_path is currently just stored for potential future use
        home = EphemeralHome(config, workspace_path="/my/workspace")
        try:
            assert home._workspace_path == "/my/workspace"
        finally:
            home.cleanup()


class TestSkillMount:
    def test_defaults(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill"
        source.mkdir()
        mount = SkillMount(source=source)
        assert mount.source == source
        assert mount.name is None
        assert mount.enabled is True

    def test_with_name(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill"
        source.mkdir()
        mount = SkillMount(source=source, name="custom-name")
        assert mount.name == "custom-name"

    def test_disabled(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill"
        source.mkdir()
        mount = SkillMount(source=source, enabled=False)
        assert mount.enabled is False


class TestSkillConfig:
    def test_defaults(self) -> None:
        config = SkillConfig()
        assert config.skills == ()
        assert config.validate_on_mount is True

    def test_with_skills(self, tmp_path: Path) -> None:
        source = tmp_path / "my-skill"
        source.mkdir()
        mount = SkillMount(source=source)
        config = SkillConfig(skills=(mount,))
        assert config.skills == (mount,)

    def test_validation_disabled(self) -> None:
        config = SkillConfig(validate_on_mount=False)
        assert config.validate_on_mount is False


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
        _validate_skill_name("my-skill")
        _validate_skill_name("skill_v2")
        _validate_skill_name("123-test")

    def test_rejects_forward_slash(self) -> None:
        with pytest.raises(SkillMountError, match="invalid characters"):
            _validate_skill_name("path/traversal")

    def test_rejects_backslash(self) -> None:
        with pytest.raises(SkillMountError, match="invalid characters"):
            _validate_skill_name("path\\traversal")

    def test_rejects_double_dot(self) -> None:
        with pytest.raises(SkillMountError, match="invalid characters"):
            _validate_skill_name("..evil")

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            _validate_skill_name("")

    def test_rejects_dot(self) -> None:
        with pytest.raises(SkillMountError, match="Invalid skill name"):
            _validate_skill_name(".")


class TestValidateSkill:
    def test_valid_directory_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test Skill\n\nContent")
        _validate_skill(skill_dir)  # Should not raise

    def test_directory_missing_skill_md(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        with pytest.raises(SkillValidationError, match=r"missing SKILL\.md"):
            _validate_skill(skill_dir)

    def test_valid_file_skill(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "test-skill.md"
        skill_file.write_text("# Test Skill\n\nContent")
        _validate_skill(skill_file)  # Should not raise

    def test_file_wrong_extension(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "test-skill.txt"
        skill_file.write_text("# Test Skill\n\nContent")
        with pytest.raises(SkillValidationError, match="must be markdown"):
            _validate_skill(skill_file)

    def test_file_too_large(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "huge-skill.md"
        # Create a file larger than 1 MiB
        skill_file.write_text("x" * (1024 * 1024 + 1))
        with pytest.raises(SkillValidationError, match="exceeds size limit"):
            _validate_skill(skill_file)


class TestCopySkill:
    def test_copy_directory_skill(self, tmp_path: Path) -> None:
        # Create source skill directory
        source = tmp_path / "source-skill"
        source.mkdir()
        (source / "SKILL.md").write_text("# Test Skill")
        (source / "examples").mkdir()
        (source / "examples" / "example.py").write_text("print('hello')")

        dest = tmp_path / "dest-skill"
        bytes_copied = _copy_skill(source, dest)

        assert dest.is_dir()
        assert (dest / "SKILL.md").read_text() == "# Test Skill"
        assert (dest / "examples" / "example.py").read_text() == "print('hello')"
        assert bytes_copied > 0

    def test_copy_file_skill_wraps_in_directory(self, tmp_path: Path) -> None:
        # Create source skill file
        source = tmp_path / "skill.md"
        source.write_text("# Single File Skill")

        dest = tmp_path / "dest-skill"
        bytes_copied = _copy_skill(source, dest)

        assert dest.is_dir()
        assert (dest / "SKILL.md").read_text() == "# Single File Skill"
        assert bytes_copied > 0

    def test_copy_directory_exceeds_size_limit(self, tmp_path: Path) -> None:
        # Create large skill directory
        source = tmp_path / "large-skill"
        source.mkdir()
        (source / "SKILL.md").write_text("# Large Skill")
        (source / "big_file.txt").write_text("x" * 100)

        dest = tmp_path / "dest-skill"
        # Use a very small limit to trigger the error
        with pytest.raises(SkillMountError, match="exceeds total size limit"):
            _copy_skill(source, dest, max_total_bytes=10)

    def test_copy_file_exceeds_size_limit(self, tmp_path: Path) -> None:
        # Create large single-file skill
        source = tmp_path / "large-skill.md"
        source.write_text("# Large Skill\n" + "x" * 100)

        dest = tmp_path / "dest-skill"
        # Use a very small limit to trigger the error
        with pytest.raises(SkillMountError, match="exceeds total size limit"):
            _copy_skill(source, dest, max_total_bytes=10)

    def test_copy_ignores_symlinks_by_default(self, tmp_path: Path) -> None:
        # Create source skill directory with symlink
        source = tmp_path / "source-skill"
        source.mkdir()
        (source / "SKILL.md").write_text("# Test Skill")
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
        source.write_text("# Test Skill")

        dest = tmp_path / "dest-skill"

        # Mock shutil.copy2 to raise OSError
        with mock.patch("shutil.copy2", side_effect=OSError("Disk full")):
            with pytest.raises(SkillMountError, match="Failed to copy skill"):
                _copy_skill(source, dest)


class TestEphemeralHomeSkillMounting:
    def test_mounts_directory_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill")

        config = IsolationConfig(
            skills=SkillConfig(skills=(SkillMount(source=skill_dir),))
        )
        with EphemeralHome(config) as home:
            assert home.skills_dir.is_dir()
            skill_dest = home.skills_dir / "my-skill"
            assert skill_dest.is_dir()
            assert (skill_dest / "SKILL.md").read_text() == "# My Skill"

    def test_mounts_file_skill(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "my-skill.md"
        skill_file.write_text("# File Skill")

        config = IsolationConfig(
            skills=SkillConfig(skills=(SkillMount(source=skill_file),))
        )
        with EphemeralHome(config) as home:
            skill_dest = home.skills_dir / "my-skill"
            assert skill_dest.is_dir()
            assert (skill_dest / "SKILL.md").read_text() == "# File Skill"

    def test_mounts_skill_with_custom_name(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "original-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Custom Named")

        config = IsolationConfig(
            skills=SkillConfig(
                skills=(SkillMount(source=skill_dir, name="custom-name"),)
            )
        )
        with EphemeralHome(config) as home:
            assert (home.skills_dir / "custom-name").is_dir()
            assert not (home.skills_dir / "original-name").exists()

    def test_skips_disabled_skills(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "disabled-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Disabled")

        config = IsolationConfig(
            skills=SkillConfig(skills=(SkillMount(source=skill_dir, enabled=False),))
        )
        with EphemeralHome(config) as home:
            assert not (home.skills_dir / "disabled-skill").exists()

    def test_mounts_multiple_skills(self, tmp_path: Path) -> None:
        # Create two skills
        skill1 = tmp_path / "skill-one"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text("# Skill One")

        skill2 = tmp_path / "skill-two"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text("# Skill Two")

        config = IsolationConfig(
            skills=SkillConfig(
                skills=(
                    SkillMount(source=skill1),
                    SkillMount(source=skill2),
                )
            )
        )
        with EphemeralHome(config) as home:
            assert (home.skills_dir / "skill-one").is_dir()
            assert (home.skills_dir / "skill-two").is_dir()

    def test_rejects_duplicate_skill_names(self, tmp_path: Path) -> None:
        skill1 = tmp_path / "skill-a"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text("# Skill A")

        skill2 = tmp_path / "skill-b"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text("# Skill B")

        config = IsolationConfig(
            skills=SkillConfig(
                skills=(
                    SkillMount(source=skill1, name="same-name"),
                    SkillMount(source=skill2, name="same-name"),
                )
            )
        )
        with pytest.raises(SkillMountError, match="Duplicate skill name"):
            EphemeralHome(config)

    def test_raises_on_missing_skill_source(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does-not-exist"
        config = IsolationConfig(
            skills=SkillConfig(skills=(SkillMount(source=nonexistent),))
        )
        with pytest.raises(SkillNotFoundError, match="Skill not found"):
            EphemeralHome(config)

    def test_validates_skill_when_enabled(self, tmp_path: Path) -> None:
        # Directory without SKILL.md
        invalid_skill = tmp_path / "invalid-skill"
        invalid_skill.mkdir()

        config = IsolationConfig(
            skills=SkillConfig(
                skills=(SkillMount(source=invalid_skill),),
                validate_on_mount=True,
            )
        )
        with pytest.raises(SkillValidationError, match=r"missing SKILL\.md"):
            EphemeralHome(config)

    def test_skips_validation_when_disabled(self, tmp_path: Path) -> None:
        # Directory without SKILL.md (would fail validation)
        invalid_skill = tmp_path / "invalid-skill"
        invalid_skill.mkdir()
        # Create some content to copy
        (invalid_skill / "README.md").write_text("# Not a skill")

        config = IsolationConfig(
            skills=SkillConfig(
                skills=(SkillMount(source=invalid_skill),),
                validate_on_mount=False,
            )
        )
        # Should not raise because validation is disabled
        with EphemeralHome(config) as home:
            assert (home.skills_dir / "invalid-skill").is_dir()

    def test_no_skills_directory_when_no_skills_configured(self) -> None:
        config = IsolationConfig()
        with EphemeralHome(config) as home:
            # skills_dir property should return path but dir shouldn't exist
            assert home.skills_dir == home.claude_dir / "skills"
            assert not home.skills_dir.exists()

    def test_skills_dir_property(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test")

        config = IsolationConfig(
            skills=SkillConfig(skills=(SkillMount(source=skill_dir),))
        )
        with EphemeralHome(config) as home:
            assert home.skills_dir == home.claude_dir / "skills"
            assert home.skills_dir.is_dir()


class TestIsolationConfigWithSkills:
    def test_isolation_config_accepts_skills(self, tmp_path: Path) -> None:
        source = tmp_path / "skill"
        source.mkdir()
        (source / "SKILL.md").write_text("# Test")

        skills = SkillConfig(skills=(SkillMount(source=source),))
        config = IsolationConfig(skills=skills)
        assert config.skills is skills

    def test_isolation_config_skills_default_none(self) -> None:
        config = IsolationConfig()
        assert config.skills is None

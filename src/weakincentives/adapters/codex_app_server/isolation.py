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

"""Hermetic home directory support for Codex App Server adapter.

This module provides configuration and runtime support for an isolated
ephemeral home directory that prevents the Codex subprocess from reading
or modifying the user's ``~/.codex`` configuration.

The ephemeral home:

- Copies host credentials from ``~/.codex/`` so the Codex CLI can
  authenticate without exposing the original config directory.
- Passes through credential environment variables (``OPENAI_API_KEY``,
  ``CODEX_API_KEY``, etc.) from the host environment.
- Mounts skills from the rendered prompt into ``$HOME/.codex/skills/``
  so the Codex CLI can discover and load them.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from ...dataclasses import FrozenDataclass
from ...skills import (
    MAX_SKILL_TOTAL_BYTES,
    SkillMount,
    SkillMountError,
    SkillNotFoundError,
    resolve_skill_name,
    validate_skill,
    validate_skill_name,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "CodexEphemeralHome",
    "CodexHermeticHomeConfig",
]

# Prefixes of environment variables that should never be inherited
# when ``include_host_env`` is ``True``.
_SENSITIVE_ENV_PREFIXES: tuple[str, ...] = (
    "HOME",
    "OPENAI_",
    "CODEX_",
    "ANTHROPIC_",
    "AWS_",
    "GOOGLE_",
    "AZURE_",
)

# Credential environment variables to pass through to the Codex subprocess
# regardless of ``include_host_env``.
_CODEX_CREDENTIAL_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_ORG_ID",
    "OPENAI_BASE_URL",
    "CODEX_API_KEY",
)


def _copy_skill(
    source: Path,
    dest_dir: Path,
    *,
    follow_symlinks: bool = False,
    max_total_bytes: int = MAX_SKILL_TOTAL_BYTES,
) -> int:
    """Copy a skill to the destination directory.

    Args:
        source: Path to the skill file or directory.
        dest_dir: Destination directory for the skill.
        follow_symlinks: Whether to follow symlinks during copy.
        max_total_bytes: Maximum total bytes to copy.

    Returns:
        Total bytes copied.

    Raises:
        SkillMountError: If copy fails or exceeds byte limit.
    """
    total_bytes = 0
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        if source.is_dir():
            for item in source.rglob("*"):
                if item.is_symlink() and not follow_symlinks:
                    continue
                if item.is_file():
                    rel_path = item.relative_to(source)
                    dest_file = dest_dir / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    size = item.stat().st_size
                    total_bytes += size
                    if total_bytes > max_total_bytes:
                        msg = f"Skill exceeds total size limit ({total_bytes} > {max_total_bytes})"
                        raise SkillMountError(msg)
                    _ = shutil.copy2(item, dest_file)
        else:
            dest_file = dest_dir / "SKILL.md"
            size = source.stat().st_size
            total_bytes = size
            if total_bytes > max_total_bytes:
                msg = f"Skill exceeds total size limit ({total_bytes} > {max_total_bytes})"
                raise SkillMountError(msg)
            _ = shutil.copy2(source, dest_file)
    except OSError as e:
        msg = f"Failed to copy skill: {e}"
        raise SkillMountError(msg) from e

    return total_bytes


@FrozenDataclass()
class CodexHermeticHomeConfig:
    """Configuration for hermetic home directory isolation.

    When provided to the Codex adapter via
    ``CodexAppServerClientConfig.hermetic_home``, creates an ephemeral home
    directory that isolates the Codex subprocess from the host's ``~/.codex``
    configuration, credentials, and session state.

    Attributes:
        copy_host_credentials: If True, copy ``~/.codex/`` from the host
            into the ephemeral home. This preserves authentication tokens
            and config so ``codex`` can authenticate without the original
            home directory.
        include_host_env: If True, inherit non-sensitive host environment
            variables.  Sensitive prefixes (HOME, OPENAI_*, CODEX_*,
            ANTHROPIC_*, AWS_*, GOOGLE_*, AZURE_*) are always excluded.
        env: Additional environment variables for the Codex subprocess.
            These take highest priority after credential passthrough.
    """

    copy_host_credentials: bool = True
    include_host_env: bool = False
    env: Mapping[str, str] | None = None


class CodexEphemeralHome:
    """Manages temporary home directory for hermetic Codex isolation.

    Creates and manages a temporary directory that serves as ``HOME`` for
    the Codex app-server subprocess.  This prevents the process from
    reading or modifying the user's ``~/.codex`` configuration.

    The ephemeral home contains:

    - ``.codex/`` directory (copied from host if ``copy_host_credentials``
      is ``True``)
    - ``.codex/skills/`` directory (skills mounted from rendered prompt)

    Example::

        config = CodexHermeticHomeConfig(copy_host_credentials=True)
        with CodexEphemeralHome(config) as ephemeral:
            env = ephemeral.get_env()
            # Pass env to Codex subprocess
    """

    def __init__(
        self,
        config: CodexHermeticHomeConfig,
        *,
        temp_dir_prefix: str = "wink-codex-home-",
    ) -> None:
        super().__init__()
        self._config = config
        self._temp_dir = tempfile.mkdtemp(prefix=temp_dir_prefix)
        self._codex_dir = Path(self._temp_dir) / ".codex"
        self._codex_dir.mkdir(parents=True, exist_ok=True)

        _logger.debug(
            "codex.ephemeral_home.init",
            extra={
                "ephemeral_home": self._temp_dir,
                "copy_host_credentials": config.copy_host_credentials,
                "include_host_env": config.include_host_env,
            },
        )

        if config.copy_host_credentials:
            self._copy_host_credentials()

        self._cleaned_up = False
        self._skills_mounted = False

    def _copy_host_credentials(self) -> None:
        """Copy ``~/.codex/`` from host into ephemeral home.

        Preserves authentication tokens and configuration from the host
        environment.  Fails silently if the host directory does not exist.
        """
        real_home = os.environ.get("HOME", "")
        if not real_home:
            _logger.debug(
                "codex.ephemeral_home.copy_credentials.skipped",
                extra={"reason": "HOME_not_set"},
            )
            return

        host_codex_dir = Path(real_home) / ".codex"
        if not host_codex_dir.exists():
            _logger.debug(
                "codex.ephemeral_home.copy_credentials.skipped",
                extra={
                    "reason": "codex_dir_not_found",
                    "path": str(host_codex_dir),
                },
            )
            return

        ephemeral_codex_dir = Path(self._temp_dir) / ".codex"
        try:
            _ = shutil.copytree(host_codex_dir, ephemeral_codex_dir, dirs_exist_ok=True)
            _logger.debug(
                "codex.ephemeral_home.copy_credentials.success",
                extra={
                    "source": str(host_codex_dir),
                    "dest": str(ephemeral_codex_dir),
                },
            )
        except OSError as e:
            _logger.warning(
                "codex.ephemeral_home.copy_credentials.failed",
                extra={"source": str(host_codex_dir), "error": str(e)},
            )

    def mount_skills(
        self,
        skills: tuple[SkillMount, ...],
        *,
        validate: bool = True,
    ) -> None:
        """Mount skills into the ephemeral home.

        Skills are typically collected from the rendered prompt and passed
        to this method by the adapter before spawning the Codex process.

        This method can only be called once per ``CodexEphemeralHome``
        instance.

        Args:
            skills: Tuple of SkillMount instances from
                ``RenderedPrompt.skills``.
            validate: If True, validate skill structure before copying.

        Raises:
            SkillMountError: If skills have already been mounted, if a
                skill name is duplicated, or if copy fails.
            SkillNotFoundError: If a skill source path does not exist.
            SkillValidationError: If validation is enabled and a skill
                is invalid.
        """
        if self._skills_mounted:
            raise SkillMountError("Skills already mounted on this ephemeral home")
        self._skills_mounted = True

        if not skills:
            return

        skills_dir = self._codex_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        seen_names: set[str] = set()
        for mount in skills:
            name = resolve_skill_name(mount)
            validate_skill_name(name)

            if name in seen_names:
                msg = f"Duplicate skill name: {name}"
                raise SkillMountError(msg)
            seen_names.add(name)

            source = Path(mount.source).resolve()
            if not source.exists():
                msg = f"Skill not found: {mount.source}"
                raise SkillNotFoundError(msg)

            if validate:
                validate_skill(source)

            dest = skills_dir / name
            _ = _copy_skill(source, dest)

    def get_env(self) -> dict[str, str]:
        """Build environment variables for the Codex subprocess.

        Returns a complete environment dictionary.  When used with
        ``replace_env=True`` on the client, this replaces the inherited
        host environment entirely.

        Returns:
            Dictionary of environment variables.
        """
        env: dict[str, str] = {}

        # 1. Inherit non-sensitive host env vars if configured
        if self._config.include_host_env:
            env.update(
                {
                    k: v
                    for k, v in os.environ.items()
                    if not any(k.startswith(p) for p in _SENSITIVE_ENV_PREFIXES)
                }
            )

        # 2. Always include PATH
        if "PATH" in os.environ:
            env["PATH"] = os.environ["PATH"]

        # 3. Set HOME to ephemeral directory
        env["HOME"] = self._temp_dir

        # 4. Pass through credential env vars
        for key in _CODEX_CREDENTIAL_VARS:
            if key in os.environ:
                env[key] = os.environ[key]

        # 5. Apply user env overrides (highest priority)
        if self._config.env:
            env.update(self._config.env)

        _logger.debug(
            "codex.ephemeral_home.get_env",
            extra={
                "ephemeral_home": self._temp_dir,
                "env_var_count": len(env),
                "env_keys": sorted(
                    k
                    for k in env
                    if "KEY" not in k and "SECRET" not in k and "TOKEN" not in k
                ),
            },
        )

        return env

    def cleanup(self) -> None:
        """Remove ephemeral home directory.

        Safe to call multiple times.
        """
        if not getattr(self, "_cleaned_up", True):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._cleaned_up = True

    @property
    def home_path(self) -> str:
        """Absolute path to the ephemeral home directory."""
        return self._temp_dir

    @property
    def codex_dir(self) -> Path:
        """Path to the ``.codex`` directory within ephemeral home."""
        return self._codex_dir

    @property
    def skills_dir(self) -> Path:
        """Path to the skills directory within ephemeral home."""
        return self._codex_dir / "skills"

    def __enter__(self) -> CodexEphemeralHome:
        """Context manager entry."""
        return self

    def __exit__(self, *_: object) -> None:
        """Context manager exit with automatic cleanup."""
        self.cleanup()

    def __del__(self) -> None:
        """Destructor that attempts cleanup if not already done."""
        self.cleanup()

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

"""Ephemeral home directory for OpenCode skill installation.

Manages a temporary HOME directory so that skills can be mounted at
``$HOME/.claude/skills/<name>/SKILL.md`` — the path OpenCode uses for
global skill discovery (Claude-compatible).

Unlike the Claude Agent SDK's :class:`EphemeralHome`, this class does
**not** generate a ``settings.json`` or apply sandbox/network policies.
It only handles skill mounting and auth passthrough to preserve provider
credentials when HOME is overridden.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

from ...skills import (
    MAX_SKILL_TOTAL_BYTES,
    SkillMount,
    SkillMountError,
    SkillNotFoundError,
    resolve_skill_name,
    validate_skill,
    validate_skill_name,
)

__all__ = ["OpenCodeEphemeralHome"]

_logger = logging.getLogger(__name__)


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
                    shutil.copy2(item, dest_file)
        else:
            dest_file = dest_dir / "SKILL.md"
            size = source.stat().st_size
            total_bytes = size
            if total_bytes > max_total_bytes:
                msg = f"Skill exceeds total size limit ({total_bytes} > {max_total_bytes})"
                raise SkillMountError(msg)
            shutil.copy2(source, dest_file)
    except OSError as e:
        msg = f"Failed to copy skill: {e}"
        raise SkillMountError(msg) from e

    return total_bytes


class OpenCodeEphemeralHome:
    """Manages a temporary home directory for OpenCode skill installation.

    Creates a temp directory that serves as HOME for the OpenCode subprocess
    so that skills can be discovered at ``$HOME/.claude/skills/<name>/SKILL.md``.

    Auth data is copied from the real HOME to preserve provider credentials:

    - ``~/.local/share/opencode/`` — stored API credentials (``auth.json``)
    - ``~/.aws/`` — AWS credentials for Bedrock provider support

    Both copies are best-effort (warn on failure, don't crash) since auth
    may come from environment variables instead.
    """

    def __init__(
        self,
        *,
        workspace_path: str | None = None,
    ) -> None:
        self._workspace_path = workspace_path
        self._temp_dir = tempfile.mkdtemp(prefix="opencode-agent-")
        self._claude_dir = Path(self._temp_dir) / ".claude"
        self._claude_dir.mkdir(parents=True, exist_ok=True)
        self._cleaned_up = False
        self._skills_mounted = False

        _logger.debug(
            "opencode.ephemeral_home.init",
            extra={"ephemeral_home": self._temp_dir},
        )

        self._copy_auth_data()

    def _copy_auth_data(self) -> None:
        """Copy auth data from real HOME to ephemeral home.

        Copies OpenCode stored credentials and AWS config so that provider
        authentication continues to work when HOME is overridden.
        """
        real_home = os.environ.get("HOME", "")
        if not real_home:
            _logger.debug(
                "opencode.ephemeral_home.auth.skip",
                extra={"reason": "HOME_not_set"},
            )
            return

        self._copy_opencode_auth(real_home)
        self._copy_aws_config(real_home)

    def _copy_opencode_auth(self, real_home: str) -> None:
        """Copy OpenCode auth data (``~/.local/share/opencode/``)."""
        source = Path(real_home) / ".local" / "share" / "opencode"
        if not source.exists():
            _logger.debug(
                "opencode.ephemeral_home.auth.opencode_skip",
                extra={"reason": "dir_not_found", "path": str(source)},
            )
            return

        dest = Path(self._temp_dir) / ".local" / "share" / "opencode"
        try:
            shutil.copytree(source, dest, symlinks=True, dirs_exist_ok=True)
            _logger.debug(
                "opencode.ephemeral_home.auth.opencode_copied",
                extra={"source": str(source), "dest": str(dest)},
            )
        except OSError as e:
            _logger.warning(
                "opencode.ephemeral_home.auth.opencode_failed",
                extra={"source": str(source), "error": str(e)},
            )

    def _copy_aws_config(self, real_home: str) -> None:
        """Copy AWS config directory (``~/.aws/``)."""
        source = Path(real_home) / ".aws"
        if not source.exists():
            _logger.debug(
                "opencode.ephemeral_home.auth.aws_skip",
                extra={"reason": "dir_not_found", "path": str(source)},
            )
            return

        dest = Path(self._temp_dir) / ".aws"
        try:
            shutil.copytree(source, dest, symlinks=True, dirs_exist_ok=True)
            _logger.debug(
                "opencode.ephemeral_home.auth.aws_copied",
                extra={"source": str(source), "dest": str(dest)},
            )
        except OSError as e:
            _logger.warning(
                "opencode.ephemeral_home.auth.aws_failed",
                extra={"source": str(source), "error": str(e)},
            )

    def mount_skills(
        self,
        skills: tuple[SkillMount, ...],
        *,
        validate: bool = True,
    ) -> None:
        """Mount skills into the ephemeral home.

        Skills are installed at ``$EPHEMERAL_HOME/.claude/skills/<name>/``
        following the Claude-compatible global discovery path that OpenCode
        natively searches.

        This method can only be called once per instance.

        Args:
            skills: Tuple of SkillMount instances from RenderedPrompt.skills.
            validate: If True, validate skill structure before copying.

        Raises:
            SkillMountError: If skills have already been mounted, if a skill
                name is duplicated, or if copy fails.
            SkillNotFoundError: If a skill source path does not exist.
            SkillValidationError: If validation is enabled and a skill is invalid.
        """
        if self._skills_mounted:
            raise SkillMountError("Skills already mounted on this ephemeral home")
        self._skills_mounted = True

        if not skills:
            return

        skills_dir = self._claude_dir / "skills"
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
            _copy_skill(source, dest)

    @property
    def home_path(self) -> str:
        """Absolute path to the ephemeral home directory."""
        return self._temp_dir

    @property
    def skills_dir(self) -> Path:
        """Path to the skills directory within ephemeral home."""
        return self._claude_dir / "skills"

    def cleanup(self) -> None:
        """Remove ephemeral home directory.

        Safe to call multiple times.
        """
        if not getattr(self, "_cleaned_up", True):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._cleaned_up = True

    def __enter__(self) -> OpenCodeEphemeralHome:
        """Context manager entry."""
        return self

    def __exit__(self, *_: object) -> None:
        """Context manager exit with automatic cleanup."""
        self.cleanup()

    def __del__(self) -> None:
        """Destructor that attempts cleanup if not already done."""
        self.cleanup()

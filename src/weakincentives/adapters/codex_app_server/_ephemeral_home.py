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

"""Ephemeral home directory for Codex skill installation.

Creates a temporary HOME directory so Codex discovers skills at
``$HOME/.agents/skills/``.  The real ``~/.codex`` path is preserved
via the ``CODEX_HOME`` environment variable so auth and config
remain accessible.
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


class CodexEphemeralHome:
    """Manages a temporary HOME directory for Codex skill discovery.

    Codex discovers user-scoped skills by scanning ``$HOME/.agents/skills/``.
    This class creates a temporary directory, mounts skills into it, and
    provides environment overrides so that:

    - ``HOME`` points to the ephemeral directory (skill discovery)
    - ``CODEX_HOME`` points to the original ``~/.codex`` (auth / config)

    Example::

        home = CodexEphemeralHome()
        try:
            home.mount_skills(rendered.skills)
            env = home.get_env()
            # pass env to CodexAppServerClient
        finally:
            home.cleanup()
    """

    def __init__(self, *, temp_dir_prefix: str = "wink-codex-home-") -> None:
        super().__init__()
        self._real_home = os.environ.get("HOME", "")
        self._temp_dir = tempfile.mkdtemp(prefix=temp_dir_prefix)
        self._cleaned_up = False
        self._skills_mounted = False

        _logger.debug(
            "codex.ephemeral_home.init",
            extra={"ephemeral_home": self._temp_dir},
        )

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    def mount_skills(
        self,
        skills: tuple[SkillMount, ...],
        *,
        validate: bool = True,
    ) -> None:
        """Mount skills into ``<ephemeral_home>/.agents/skills/``.

        This method can only be called once per instance.

        Args:
            skills: Tuple of SkillMount instances from RenderedPrompt.skills.
            validate: If True, validate skill structure before copying.

        Raises:
            SkillMountError: If already mounted, duplicate names, or copy fails.
            SkillNotFoundError: If a skill source path does not exist.
            SkillValidationError: If validation enabled and a skill is invalid.
        """
        if self._skills_mounted:
            raise SkillMountError("Skills already mounted on this ephemeral home")
        self._skills_mounted = True

        if not skills:
            return

        skills_dir = Path(self._temp_dir) / ".agents" / "skills"
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

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def get_env(self) -> dict[str, str]:
        """Return environment overrides for the Codex subprocess.

        Returns:
            ``HOME`` pointing to the ephemeral directory and ``CODEX_HOME``
            pointing to the original ``~/.codex``.
        """
        codex_home = str(Path(self._real_home) / ".codex") if self._real_home else ""
        env: dict[str, str] = {"HOME": self._temp_dir}
        if codex_home:
            env["CODEX_HOME"] = codex_home
        return env

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def home_path(self) -> str:
        """Absolute path to the ephemeral home directory."""
        return self._temp_dir

    @property
    def skills_dir(self) -> Path:
        """Path to the skills directory within the ephemeral home."""
        return Path(self._temp_dir) / ".agents" / "skills"

    def cleanup(self) -> None:
        """Remove the ephemeral home directory.  Idempotent."""
        if not self._cleaned_up:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._cleaned_up = True

    def __enter__(self) -> CodexEphemeralHome:
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        self.cleanup()

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

"""Data transfer objects for the asteval tool suite.

This module contains the dataclasses used by both the asteval section
and the asteval engine, enabling clean separation of concerns without
circular imports.
"""

from __future__ import annotations

from dataclasses import field
from typing import Literal

from ...dataclasses import FrozenDataclass
from .vfs_types import VfsPath, format_path as _format_vfs_path


def _str_dict_factory() -> dict[str, str]:
    return {}


@FrozenDataclass()
class EvalFileRead:
    """File that should be read from the virtual filesystem before execution."""

    path: VfsPath = field(
        metadata={
            "description": (
                "Relative VFS path to load. Contents are injected into "
                "`reads` for convenience."
            )
        }
    )

    def render(self) -> str:
        return f"read {_format_vfs_path(self.path)}"


@FrozenDataclass()
class EvalFileWrite:
    """File that should be written back to the virtual filesystem."""

    path: VfsPath = field(
        metadata={"description": "Relative VFS path to create or update."}
    )
    content: str = field(
        metadata={
            "description": (
                "ASCII text to persist after execution. Content longer than 48k "
                "characters is rejected."
            )
        }
    )
    mode: Literal["create", "overwrite", "append"] = field(
        default="create",
        metadata={
            "description": (
                "Write strategy for the file: create a new entry, overwrite the "
                "existing content, or append."
            )
        },
    )

    def render(self) -> str:
        size = len(self.content)
        return f"{self.mode} {_format_vfs_path(self.path)} ({size} chars)"


@FrozenDataclass()
class EvalParams:
    """Parameter payload passed to the Python evaluation tool."""

    code: str = field(
        metadata={"description": "Python script to execute (<=2,000 characters)."}
    )
    globals: dict[str, str] = field(
        default_factory=_str_dict_factory,
        metadata={
            "description": (
                "Mapping of global variable names to JSON-encoded strings. The "
                "payload is decoded before execution."
            )
        },
    )
    reads: tuple[EvalFileRead, ...] = field(
        default_factory=tuple,
        metadata={
            "description": (
                "Files to load into the VFS before execution. Each entry is "
                "available to helper utilities."
            )
        },
    )
    writes: tuple[EvalFileWrite, ...] = field(
        default_factory=tuple,
        metadata={
            "description": (
                "Files to write after execution completes. These mirror calls to "
                "`write_text`."
            )
        },
    )


@FrozenDataclass()
class EvalResult:
    """Structured result produced by the Python evaluation tool."""

    value_repr: str | None = field(
        metadata={
            "description": (
                "String representation of the final expression result. Null when "
                "no value was produced."
            )
        }
    )
    stdout: str = field(
        metadata={
            "description": (
                "Captured standard output stream, truncated to 4,096 characters."
            )
        }
    )
    stderr: str = field(
        metadata={
            "description": (
                "Captured standard error stream, truncated to 4,096 characters."
            )
        }
    )
    globals: dict[str, str] = field(
        metadata={
            "description": (
                "JSON-serialisable globals returned from the sandbox after execution."
            )
        }
    )
    reads: tuple[EvalFileRead, ...] = field(
        metadata={"description": "File read requests fulfilled during execution."}
    )
    writes: tuple[EvalFileWrite, ...] = field(
        metadata={"description": "File write operations requested by the code."}
    )

    def render(self) -> str:
        lines: list[str] = ["Python evaluation result:"]
        if self.value_repr is not None:
            lines.append(f"Result: {self.value_repr}")
        lines.append("STDOUT:")
        lines.append(self.stdout or "<empty>")
        lines.append("STDERR:")
        lines.append(self.stderr or "<empty>")
        if self.reads:
            lines.append("Reads:")
            lines.extend(f"- {read.render()}" for read in self.reads)
        if self.writes:
            lines.append("Writes:")
            lines.extend(f"- {write.render()}" for write in self.writes)
        if self.globals:
            lines.append("Globals:")
            lines.extend(f"- {key}={value}" for key, value in self.globals.items())
        return "\n".join(lines)


@FrozenDataclass()
class AstevalConfig:
    """Configuration for :class:`AstevalSection`.

    All constructor arguments for AstevalSection are consolidated here.
    This avoids accumulating long argument lists as the section evolves.

    Example::

        from weakincentives.contrib.tools import AstevalConfig, AstevalSection

        config = AstevalConfig(accepts_overrides=True)
        section = AstevalSection(session=session, config=config)
    """

    accepts_overrides: bool = field(
        default=False,
        metadata={"description": "Whether the section accepts parameter overrides."},
    )


__all__ = [
    "AstevalConfig",
    "EvalFileRead",
    "EvalFileWrite",
    "EvalParams",
    "EvalResult",
]

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

"""Sandboxed Python expression evaluation backed by :mod:`asteval`."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Final, override

from ...dataclasses import FrozenDataclass
from ...prompt.markdown import MarkdownSection
from ...prompt.tool import Tool, ToolExample
from ...runtime.session import Session
from .asteval_engine import (
    AstevalToolSuite,
    alias_for_path,
    normalize_reads,
    normalize_write,
    normalize_writes,
    parse_user_globals,
    summarize_writes,
)
from .asteval_types import (
    AstevalConfig,
    EvalFileRead,
    EvalFileWrite,
    EvalParams,
    EvalResult,
)
from .filesystem import Filesystem
from .filesystem_memory import InMemoryFilesystem
from .vfs_types import VfsPath

_EVAL_TEMPLATE: Final[str] = (
    "Use the Python evaluation tool for quick calculations and one-off scripts.\n"
    "- Scripts run in a sandbox with a narrow set of safe builtins (abs, len, max, min,\n"
    "  print, range, round, sum, str) plus math/statistics helpers. Import statements\n"
    "  and other blocked nodes are stripped, so networking and host filesystem access\n"
    "  are unavailable.\n"
    "- Keep code concise (<=2,000 characters) and avoid control characters other than\n"
    "  newlines or tabs.\n"
    "- Pre-load files via `reads`, or call `read_text(path)` inside code to fetch VFS\n"
    "  files. Paths must be relative, use <=16 segments of <=80 ASCII characters, and\n"
    "  may not target a read and write in the same call.\n"
    "- Stage edits with `write_text(path, content, mode)` or declare them in `writes`.\n"
    "  Content must be ASCII, <=48k characters, and choose from modes create,\n"
    "  overwrite, or append.\n"
    "- Globals accept JSON-encoded strings keyed by valid identifiers. Payloads are\n"
    "  parsed before execution; invalid JSON or names raise a validation error.\n"
    "- Execution stops after five seconds. Stdout/stderr are captured and truncated to\n"
    "  4,096 characters, and the repr of the final expression is returned when present.\n\n"
    "The tool executes multi-line scripts, captures stdout, and returns the repr of the final expression when present:\n"
    "```json\n"
    "{\n"
    '  "name": "evaluate_python",\n'
    '  "arguments": {\n'
    '    "code": "total = 0\\nfor value in range(5):\\n    total += value\\nprint(total)\\ntotal",\n'
    '    "globals": {},\n'
    '    "reads": [],\n'
    '    "writes": []\n'
    "  }\n"
    "}\n"
    "```"
)


@FrozenDataclass()
class _AstevalSectionParams:
    """Placeholder params container for the asteval section."""

    pass


# -----------------------------------------------------------------------------
# Public API Functions
# -----------------------------------------------------------------------------


def normalize_eval_reads(reads: Iterable[EvalFileRead]) -> tuple[EvalFileRead, ...]:
    return normalize_reads(reads)


def normalize_eval_writes(
    writes: Iterable[EvalFileWrite],
) -> tuple[EvalFileWrite, ...]:
    return normalize_writes(writes)


def normalize_eval_write(write: EvalFileWrite) -> EvalFileWrite:
    return normalize_write(write)


def parse_eval_globals(payload: Mapping[str, str]) -> dict[str, object]:
    return parse_user_globals(payload)


def alias_for_eval_path(path: VfsPath) -> str:
    return alias_for_path(path)


def summarize_eval_writes(writes: Sequence[EvalFileWrite]) -> str | None:
    return summarize_writes(writes)


# -----------------------------------------------------------------------------
# Section Class
# -----------------------------------------------------------------------------


class AstevalSection(MarkdownSection[_AstevalSectionParams]):
    """Prompt section exposing the :mod:`asteval` evaluation tool.

    Use :class:`AstevalConfig` to consolidate configuration::

        config = AstevalConfig(accepts_overrides=True)
        section = AstevalSection(session=session, config=config)

    Individual parameters are still accepted for backward compatibility,
    but config takes precedence when provided.
    """

    def __init__(
        self,
        *,
        session: Session,
        config: AstevalConfig | None = None,
        filesystem: Filesystem | None = None,
        accepts_overrides: bool = False,
    ) -> None:
        # Resolve config - explicit config takes precedence
        if config is not None:
            resolved_accepts_overrides = config.accepts_overrides
        else:
            resolved_accepts_overrides = accepts_overrides

        self._session = session
        # Use provided filesystem or create a new one
        self._filesystem = (
            filesystem if filesystem is not None else InMemoryFilesystem()
        )

        # Store config for cloning
        self._config = AstevalConfig(accepts_overrides=resolved_accepts_overrides)

        tool_suite = AstevalToolSuite(filesystem=self._filesystem)
        tool = Tool[EvalParams, EvalResult](
            name="evaluate_python",
            description=(
                "Run a short Python expression or script in a sandbox. Supports "
                "preloading VFS files, staging writes, and returning captured "
                "stdout, stderr, and result data."
            ),
            handler=tool_suite.run,
            accepts_overrides=resolved_accepts_overrides,
            examples=(
                ToolExample(
                    description=(
                        "Sum contents of a staged VFS file and capture stdout."
                    ),
                    input=EvalParams(
                        code=(
                            "total = sum(int(value) for value in "
                            "read_text('numbers.txt').split())\nprint(total)\n"
                            "total"
                        ),
                        globals={},
                        reads=(
                            EvalFileRead(
                                path=VfsPath(segments=("numbers.txt",)),
                            ),
                        ),
                        writes=(),
                    ),
                    output=EvalResult(
                        value_repr="6",
                        stdout="6\n",
                        stderr="",
                        globals={
                            "total": "6",
                            "vfs:numbers.txt": "1\n2\n3\n",
                        },
                        reads=(EvalFileRead(path=VfsPath(segments=("numbers.txt",))),),
                        writes=(),
                    ),
                ),
            ),
        )
        super().__init__(
            title="Python Evaluation Tool",
            key="tools.asteval",
            template=_EVAL_TEMPLATE,
            default_params=_AstevalSectionParams(),
            tools=(tool,),
            accepts_overrides=resolved_accepts_overrides,
        )

    @property
    def session(self) -> Session:
        return self._session

    @property
    def filesystem(self) -> Filesystem:
        """Return the filesystem managed by this section."""
        return self._filesystem

    @override
    def clone(self, **kwargs: object) -> AstevalSection:
        session = kwargs.get("session")
        if not isinstance(session, Session):
            msg = "session is required to clone AstevalSection."
            raise TypeError(msg)
        # Use provided filesystem if given, otherwise keep the current one.
        # This allows sharing a filesystem across sections (e.g., with VfsToolsSection).
        filesystem = kwargs.get("filesystem")
        if filesystem is not None and not isinstance(filesystem, Filesystem):
            msg = "filesystem must be a Filesystem instance."
            raise TypeError(msg)
        return AstevalSection(
            session=session,
            config=self._config,
            filesystem=filesystem if filesystem is not None else self._filesystem,
        )


__all__ = [
    "AstevalConfig",
    "AstevalSection",
    "EvalFileRead",
    "EvalFileWrite",
    "EvalParams",
    "EvalResult",
    "alias_for_eval_path",
    "normalize_eval_reads",
    "normalize_eval_write",
    "normalize_eval_writes",
    "parse_eval_globals",
    "summarize_eval_writes",
]

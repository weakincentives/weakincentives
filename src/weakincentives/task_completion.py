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

"""Task-completion checkers: programmatic goal verification.

A :class:`CompletionChecker` decides whether the current
:class:`Session` proves the agent has finished. The runtime layer can
poll one between iterations and, when satisfied, surface the
:class:`CompletionStatus` to the caller. The protocol is small enough
that adapters and tests can satisfy it inline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, final, runtime_checkable

from weakincentives.core import Session, ToolInvoked
from weakincentives.transcript import Transcript

__all__ = [
    "AllOfChecker",
    "AnyOfChecker",
    "CompletionChecker",
    "CompletionStatus",
    "ToolSucceededChecker",
    "callable_checker",
]


@final
@dataclass(frozen=True, slots=True)
class CompletionStatus:
    """Outcome of a single completion check."""

    is_complete: bool
    detail: str = ""


@runtime_checkable
class CompletionChecker(Protocol):
    """Decide whether a session represents a finished task."""

    def check(self, session: Session, transcript: Transcript) -> CompletionStatus: ...


# =============================================================================
# Built-in checkers
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class ToolSucceededChecker:
    """Pass once ``tool_name`` has produced a successful result."""

    tool_name: str

    def check(self, session: Session, transcript: Transcript) -> CompletionStatus:
        del session
        for entry in transcript.entries():
            if entry.kind != ToolInvoked.__name__:
                continue
            payload = entry.payload
            if payload.get("name") == self.tool_name and payload.get("success") is True:
                return CompletionStatus(
                    is_complete=True,
                    detail=f"{self.tool_name!r} succeeded",
                )
        return CompletionStatus(
            is_complete=False,
            detail=f"awaiting successful {self.tool_name!r}",
        )


@final
@dataclass(frozen=True, slots=True)
class AllOfChecker:
    """Pass when every supplied checker passes."""

    checkers: tuple[CompletionChecker, ...]

    def check(self, session: Session, transcript: Transcript) -> CompletionStatus:
        for checker in self.checkers:
            status = checker.check(session, transcript)
            if not status.is_complete:
                return status
        return CompletionStatus(is_complete=True, detail="all checks passed")


@final
@dataclass(frozen=True, slots=True)
class AnyOfChecker:
    """Pass when at least one supplied checker passes."""

    checkers: tuple[CompletionChecker, ...]

    def check(self, session: Session, transcript: Transcript) -> CompletionStatus:
        details: list[str] = []
        for checker in self.checkers:
            status = checker.check(session, transcript)
            if status.is_complete:
                return CompletionStatus(is_complete=True, detail=status.detail)
            details.append(status.detail)
        return CompletionStatus(is_complete=False, detail=" / ".join(details))


def callable_checker(
    predicate: Callable[[Session, Transcript], bool | CompletionStatus],
) -> CompletionChecker:
    """Wrap a plain callable so it satisfies :class:`CompletionChecker`."""

    @final
    @dataclass(frozen=True, slots=True)
    class _Wrapped:
        def check(self, session: Session, transcript: Transcript) -> CompletionStatus:
            outcome = predicate(session, transcript)
            if isinstance(outcome, CompletionStatus):
                return outcome
            return CompletionStatus(
                is_complete=bool(outcome),
                detail="callable checker satisfied"
                if outcome
                else "callable checker not satisfied",
            )

    return _Wrapped()

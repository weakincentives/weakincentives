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

"""Base exception hierarchy for :mod:`weakincentives`."""

from __future__ import annotations


class WinkError(Exception):
    """Base class for all weakincentives exceptions.

    This class serves as the root of the exception hierarchy, allowing callers
    to catch all library-specific exceptions with a single handler::

        try:
            # weakincentives operations
        except WinkError:
            # handle any library error
    """


class ToolValidationError(WinkError, ValueError):
    """Raised when tool parameters fail validation checks."""


class DeadlineExceededError(WinkError, RuntimeError):
    """Raised when tool execution cannot finish before the deadline."""


class SnapshotError(WinkError, RuntimeError):
    """Base class for snapshot-related errors."""


class SnapshotRestoreError(SnapshotError):
    """Raised when restoring from a snapshot fails."""


class TransactionError(WinkError, RuntimeError):
    """Base class for transaction-related errors."""


class RestoreFailedError(TransactionError):
    """Failed to restore from snapshot during transaction rollback."""


__all__ = [
    "DeadlineExceededError",
    "RestoreFailedError",
    "SnapshotError",
    "SnapshotRestoreError",
    "ToolValidationError",
    "TransactionError",
    "WinkError",
]

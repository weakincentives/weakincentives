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
    to catch all library-specific exceptions with a single handler while letting
    standard Python exceptions propagate normally.

    Example:
        Catch any weakincentives-specific error::

            try:
                session.dispatch(event)
            except WinkError as e:
                logger.error("Library error: %s", e)
                # Handle gracefully or re-raise

    Note:
        Subclasses may also inherit from standard exception types (e.g.,
        ``ValueError``, ``RuntimeError``) to enable more specific handling
        when needed.
    """


class ToolValidationError(WinkError, ValueError):
    """Raised when tool parameters fail validation checks.

    This exception indicates that a tool received input that does not conform
    to its declared parameter schema. Common causes include:

    - Missing required parameters
    - Parameters with invalid types
    - Values outside allowed ranges or patterns
    - Constraint violations (e.g., ``ge``, ``le``, ``pattern``)

    Example:
        Handling validation errors in tool execution::

            try:
                result = tool.execute(params)
            except ToolValidationError as e:
                return ToolResult.error(f"Invalid parameters: {e}")

    Note:
        This exception also inherits from ``ValueError``, so it can be caught
        by handlers expecting standard validation errors.
    """


class DeadlineExceededError(WinkError, RuntimeError):
    """Raised when an operation cannot complete before its deadline.

    Deadlines are wall-clock times (UTC) by which an operation must complete.
    When the current time exceeds the deadline, this exception is raised to
    prevent unbounded execution.

    Common scenarios:
        - Tool execution taking longer than allowed
        - Network requests timing out
        - Long-running computations exceeding their budget

    Example:
        Setting and handling deadlines::

            from datetime import datetime, UTC, timedelta

            deadline = datetime.now(UTC) + timedelta(seconds=30)
            try:
                result = execute_with_deadline(operation, deadline=deadline)
            except DeadlineExceededError:
                # Clean up and return partial result or error
                return ToolResult.error("Operation timed out")

    Note:
        Deadlines should always use timezone-aware datetimes with ``UTC``.
        This exception also inherits from ``RuntimeError``.
    """


class SnapshotError(WinkError, RuntimeError):
    """Base class for snapshot-related errors.

    Snapshots capture the state of resources implementing the ``Snapshotable``
    protocol, enabling rollback on failure. This exception hierarchy covers
    errors during snapshot creation, storage, and restoration.

    Example:
        Catching any snapshot-related error::

            try:
                snapshot = resource.snapshot()
                # ... perform operations ...
                resource.restore(snapshot)
            except SnapshotError as e:
                logger.error("Snapshot operation failed: %s", e)

    See Also:
        - :class:`SnapshotRestoreError`: Raised specifically during restore.
        - ``Snapshotable`` protocol in ``weakincentives.resources``.
    """


class SnapshotRestoreError(SnapshotError):
    """Raised when restoring from a snapshot fails.

    This exception indicates that a ``Snapshotable`` resource could not be
    returned to its previous state. This is a serious error that may leave
    the resource in an inconsistent state.

    Common causes:
        - Corrupted or invalid snapshot data
        - Resource state has changed incompatibly since snapshot
        - External dependencies (files, connections) no longer available
        - Snapshot from a different resource instance

    Example:
        Handling restore failures::

            try:
                resource.restore(snapshot)
            except SnapshotRestoreError as e:
                # Resource may be in inconsistent state
                logger.critical("Restore failed: %s", e)
                # Consider creating a fresh resource instance
                resource = create_new_resource()

    Warning:
        After this exception, the resource state is undefined. Callers should
        either discard the resource or reinitialize it completely.
    """


class TransactionError(WinkError, RuntimeError):
    """Base class for transaction-related errors.

    Transactions in weakincentives provide atomic execution of tool operations.
    When a tool fails, the transaction automatically rolls back any changes
    made to ``Snapshotable`` resources. This exception hierarchy covers errors
    that occur during transaction management.

    Example:
        Catching transaction errors::

            try:
                with transaction_context(resources):
                    tool.execute(params)
            except TransactionError as e:
                logger.error("Transaction failed: %s", e)
                # Resources may need manual inspection

    See Also:
        - :class:`RestoreFailedError`: Raised when rollback itself fails.
    """


class RestoreFailedError(TransactionError):
    """Raised when snapshot restoration fails during transaction rollback.

    This is a critical error indicating that a transaction could not be
    cleanly rolled back. The original operation failed, and the attempt to
    restore resources to their pre-transaction state also failed.

    This typically occurs when:
        - A :class:`SnapshotRestoreError` is raised during rollback
        - Multiple resources fail to restore (partial rollback)
        - System resources became unavailable during rollback

    Example:
        Handling double-failure scenarios::

            try:
                with transaction_context(resources):
                    tool.execute(params)
            except RestoreFailedError as e:
                # Both the operation AND rollback failed
                logger.critical("Rollback failed: %s", e)
                # Resources are in an undefined state
                # Consider alerting, manual intervention, or restart

    Warning:
        After this exception, affected resources are in an undefined state.
        The application should treat this as a critical failure requiring
        manual intervention or a full restart.
    """


__all__ = [
    "DeadlineExceededError",
    "RestoreFailedError",
    "SnapshotError",
    "SnapshotRestoreError",
    "ToolValidationError",
    "TransactionError",
    "WinkError",
]

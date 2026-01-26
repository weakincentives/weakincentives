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

"""Message handling utilities for AgentLoop.

This module provides helper functions for handling message lifecycle in AgentLoop,
including failure handling, dead-lettering, and reply/acknowledge operations.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .dlq import DeadLetter
from .logging import StructuredLogger, bind_run_context, get_logger
from .mailbox import Message, ReceiptHandleExpiredError, ReplyNotAvailableError
from .run_context import RunContext

if TYPE_CHECKING:
    from pathlib import Path

    from .agent_loop_types import AgentLoopRequest, AgentLoopResult
    from .dlq import DLQPolicy
    from .mailbox import Mailbox

_logger: StructuredLogger = get_logger(
    __name__, context={"component": "runtime.message_handlers"}
)


def handle_failure[UserRequestT, OutputT](  # noqa: PLR0913
    msg: Message[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]],
    error: Exception,
    *,
    run_context: RunContext,
    dlq: DLQPolicy[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]] | None,
    requests_mailbox: Mailbox[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]],
    result_class: type[AgentLoopResult[OutputT]],
    bundle_path: Path | None = None,
) -> None:
    """Handle message processing failure.

    When DLQ is configured:
    - Checks DLQ policy and either dead-letters or retries with backoff
    - Only sends error replies on terminal outcomes (DLQ)

    When DLQ is not configured:
    - Sends error reply and acknowledges (original behavior)

    Args:
        msg: The message that failed processing.
        error: The exception that caused the failure.
        run_context: Execution context for logging.
        dlq: Optional DLQ policy configuration.
        requests_mailbox: The source mailbox for dead letter metadata.
        result_class: The AgentLoopResult class to use for error responses.
        bundle_path: Optional path to debug bundle if one was created.
    """
    if dlq is None:
        # No DLQ configured - use original behavior: error reply + acknowledge
        result = result_class(
            request_id=msg.body.request_id,
            error=str(error),
            run_context=run_context,
            bundle_path=bundle_path,
        )
        reply_and_ack(msg, result)
        return

    # DLQ is configured - check if we should dead-letter
    if dlq.should_dead_letter(msg, error):
        dead_letter_message(
            msg,
            error,
            run_context=run_context,
            dlq=dlq,
            requests_mailbox=requests_mailbox,
            result_class=result_class,
            bundle_path=bundle_path,
        )
        return

    # Retry with backoff - do NOT send error reply here.
    # The message will be redelivered and may succeed on retry.
    # Only send error replies on terminal outcomes (DLQ or final failure).
    backoff = min(60 * msg.delivery_count, 900)
    with contextlib.suppress(ReceiptHandleExpiredError):
        msg.nack(visibility_timeout=backoff)


def dead_letter_message[UserRequestT, OutputT](  # noqa: PLR0913
    msg: Message[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]],
    error: Exception,
    *,
    run_context: RunContext,
    dlq: DLQPolicy[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]],
    requests_mailbox: Mailbox[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]],
    result_class: type[AgentLoopResult[OutputT]],
    bundle_path: Path | None = None,
) -> None:
    """Send message to dead letter queue.

    Args:
        msg: The message to dead-letter.
        error: The exception that caused the failure.
        run_context: Execution context for logging and correlation.
        dlq: The DLQ policy with target mailbox.
        requests_mailbox: The source mailbox for metadata.
        result_class: The AgentLoopResult class to use for error responses.
        bundle_path: Optional path to debug bundle if one was created.
    """
    # Create run-scoped logger for tracing
    log = bind_run_context(_logger, run_context)

    dead_letter = DeadLetter(
        message_id=msg.id,
        body=msg.body,
        source_mailbox=requests_mailbox.name,
        delivery_count=msg.delivery_count,
        last_error=str(error),
        last_error_type=f"{type(error).__module__}.{type(error).__qualname__}",
        dead_lettered_at=datetime.now(UTC),
        first_received_at=msg.enqueued_at,
        request_id=msg.body.request_id,
        reply_to=msg.reply_to.name if msg.reply_to else None,
        trace_id=run_context.trace_id,
    )

    # Send error reply - this is a terminal outcome. Failures here
    # should not prevent the dead letter from being sent.
    try:
        if msg.reply_to:
            _ = msg.reply(
                result_class(
                    request_id=msg.body.request_id,
                    error=f"Dead-lettered after {msg.delivery_count} attempts: {error}",
                    run_context=run_context,
                    bundle_path=bundle_path,
                )
            )
    except Exception as reply_error:  # nosec B110 - intentional: reply failure should not block dead-lettering
        log.debug(
            "Failed to send error reply during dead-lettering.",
            event="agent_loop.dead_letter_reply_failed",
            context={"message_id": msg.id, "error": str(reply_error)},
        )

    _ = dlq.mailbox.send(dead_letter)
    log.warning(
        "Message dead-lettered.",
        event="agent_loop.message_dead_lettered",
        context={
            "message_id": msg.id,
            "delivery_count": msg.delivery_count,
            "error_type": dead_letter.last_error_type,
        },
    )

    # Acknowledge to remove from source queue
    with contextlib.suppress(ReceiptHandleExpiredError):
        msg.acknowledge()


def reply_and_ack[UserRequestT, OutputT](
    msg: Message[AgentLoopRequest[UserRequestT], AgentLoopResult[OutputT]],
    result: AgentLoopResult[OutputT],
) -> None:
    """Reply with result and acknowledge message, handling expired handles gracefully.

    Args:
        msg: The message to reply to and acknowledge.
        result: The result to send as a reply.
    """
    # Create run-scoped logger if run_context available
    log = bind_run_context(_logger, result.run_context)
    try:
        _ = msg.reply(result)
        msg.acknowledge()
    except ReplyNotAvailableError:
        # No reply_to specified - log and acknowledge without reply
        log.warning(
            "No reply_to for message, acknowledging without reply.",
            event="agent_loop.no_reply_to",
            context={"message_id": msg.id},
        )
        with contextlib.suppress(ReceiptHandleExpiredError):
            msg.acknowledge()
    except ReceiptHandleExpiredError:
        # Handle expired during processing - message already requeued by reaper.
        # This is expected for long-running requests. The duplicate response
        # will be sent when the message is reprocessed.
        pass
    except Exception:
        # Reply send failed - nack so message is retried
        with contextlib.suppress(ReceiptHandleExpiredError):
            backoff = min(60 * msg.delivery_count, 900)
            msg.nack(visibility_timeout=backoff)


__all__ = [
    "dead_letter_message",
    "handle_failure",
    "reply_and_ack",
]

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

"""AWS SQS-backed mailbox implementation.

This module provides a durable message queue implementation using AWS SQS.
It directly maps to SQS API operations for maximum compatibility.

See ``specs/MAILBOX.md`` for the complete specification.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnusedCallResult=false
# pyright: reportArgumentType=false, reportGeneralTypeIssues=false
# pyright: reportUnknownLambdaType=false, reportReturnType=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from weakincentives.runtime.mailbox import (
    MailboxConnectionError,
    MailboxFullError,
    Message,
    ReceiptHandleExpiredError,
    SerializationError,
)
from weakincentives.serde import dump, parse

if TYPE_CHECKING:
    from mypy_boto3_sqs import SQSClient
    from mypy_boto3_sqs.type_defs import MessageTypeDef


# SQS limits
_MAX_DELAY_SECONDS = 900
_MAX_VISIBILITY_TIMEOUT = 43200  # 12 hours
_MAX_WAIT_TIME_SECONDS = 20
_MAX_MESSAGES = 10


@dataclass(slots=True)
class SQSMailbox[T]:
    """AWS SQS-backed mailbox with native SQS semantics.

    Direct mapping to SQS API. No additional data structures.

    SQS-specific considerations:

    - Create queue via AWS console, CloudFormation, or Terraform
    - Use Standard queues for throughput, FIFO for ordering
    - Configure dead-letter queue in AWS for failed messages
    - MessageAttributes map to ``Message.attributes``

    Example::

        import boto3
        from weakincentives.contrib.mailbox import SQSMailbox

        sqs = boto3.client("sqs")
        mailbox: SQSMailbox[MyEvent] = SQSMailbox(
            queue_url="https://sqs.us-east-1.amazonaws.com/123456789/my-queue",
            client=sqs,
        )

        try:
            mailbox.send(MyEvent(data="hello"))
            for msg in mailbox.receive(visibility_timeout=60):
                process(msg.body)
                msg.acknowledge()
        finally:
            mailbox.close()
    """

    queue_url: str
    """The SQS queue URL."""

    client: SQSClient
    """Boto3 SQS client instance."""

    body_type: type[T] | None = None
    """Optional type hint for message body deserialization."""

    _closed: bool = field(default=False, repr=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _serialize(self, body: T) -> str:
        """Serialize message body to JSON string."""
        try:
            if hasattr(body, "__dataclass_fields__"):
                return json.dumps(dump(body))
            return json.dumps(body)
        except Exception as e:
            raise SerializationError(f"Failed to serialize message body: {e}") from e

    def _deserialize(self, data: str) -> T:
        """Deserialize message body from JSON string."""
        try:
            json_data = json.loads(data)
            if self.body_type is not None:
                if hasattr(self.body_type, "__dataclass_fields__"):
                    return parse(self.body_type, json_data)
                body_type: Any = self.body_type
                return body_type(json_data)
            return cast(T, json_data)
        except Exception as e:
            raise SerializationError(f"Failed to deserialize message body: {e}") from e

    def send(self, body: T, *, delay_seconds: int = 0) -> str:
        """Enqueue a message, optionally delaying visibility.

        Args:
            body: Message payload (must be serializable via serde).
            delay_seconds: Seconds before message becomes visible (0-900).

        Returns:
            Message ID (unique within this queue).

        Raises:
            MailboxFullError: Queue capacity exceeded (120K in-flight for standard).
            SerializationError: Body cannot be serialized.
            MailboxConnectionError: Cannot connect to SQS.
        """
        delay_seconds = max(0, min(delay_seconds, _MAX_DELAY_SECONDS))
        serialized = self._serialize(body)

        try:
            response = self.client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=serialized,
                DelaySeconds=delay_seconds,
            )
            return response["MessageId"]

        except self.client.exceptions.OverLimit as e:
            raise MailboxFullError(f"Queue capacity exceeded: {e}") from e
        except SerializationError:
            raise
        except Exception as e:
            # Check for AWS error codes that indicate capacity issues
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if error_code == "OverLimit":
                raise MailboxFullError(f"Queue capacity exceeded: {e}") from e
            raise MailboxConnectionError(f"Failed to send message: {e}") from e

    def receive(
        self,
        *,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> Sequence[Message[T]]:
        """Receive messages from the queue.

        Args:
            max_messages: Maximum messages to receive (1-10).
            visibility_timeout: Seconds message remains invisible (0-43200).
            wait_time_seconds: Long poll duration (0-20). Zero returns
                immediately if no messages are available.

        Returns:
            Sequence of messages (may be empty). Returns empty if mailbox closed.

        Raises:
            MailboxConnectionError: Cannot connect to SQS.
        """
        if self._closed:
            return []

        max_messages = min(max(1, max_messages), _MAX_MESSAGES)
        visibility_timeout = max(0, min(visibility_timeout, _MAX_VISIBILITY_TIMEOUT))
        wait_time_seconds = max(0, min(wait_time_seconds, _MAX_WAIT_TIME_SECONDS))

        try:
            response = self.client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=max_messages,
                VisibilityTimeout=visibility_timeout,
                WaitTimeSeconds=wait_time_seconds,
                AttributeNames=["All"],
                MessageAttributeNames=["All"],
            )

            sqs_messages = response.get("Messages", [])
            messages: list[Message[T]] = []

            for sqs_msg in sqs_messages:
                msg = self._convert_message(sqs_msg)
                if msg is not None:
                    messages.append(msg)

            return messages

        except SerializationError:
            raise
        except Exception as e:
            if self._closed:
                return []
            raise MailboxConnectionError(f"Failed to receive messages: {e}") from e

    def _convert_message(self, sqs_msg: MessageTypeDef) -> Message[T] | None:
        """Convert an SQS message to a Message object."""
        # These fields are always present when receiving messages but TypedDict
        # marks them as NotRequired since they're optional when creating messages
        msg_id = sqs_msg.get("MessageId")
        receipt_handle = sqs_msg.get("ReceiptHandle")
        body_str = sqs_msg.get("Body")

        if not msg_id or not receipt_handle or not body_str:
            return None

        body = self._deserialize(body_str)

        # Extract attributes
        attrs = sqs_msg.get("Attributes", {})
        msg_attrs = sqs_msg.get("MessageAttributes", {})

        # Delivery count from ApproximateReceiveCount
        delivery_count = int(attrs.get("ApproximateReceiveCount", "1"))

        # Enqueued time from SentTimestamp (milliseconds since epoch)
        sent_ts = attrs.get("SentTimestamp")
        if sent_ts:
            enqueued_at = datetime.fromtimestamp(int(sent_ts) / 1000, tz=UTC)
        else:
            enqueued_at = datetime.now(UTC)

        # Convert MessageAttributes to simple string dict
        attributes: dict[str, str] = {}
        for key, value in msg_attrs.items():
            if value.get("DataType") == "String":
                attributes[key] = value.get("StringValue", "")

        return Message(
            id=msg_id,
            body=body,
            receipt_handle=receipt_handle,
            delivery_count=delivery_count,
            enqueued_at=enqueued_at,
            attributes=attributes,
            _acknowledge_fn=lambda rh=receipt_handle: self._acknowledge(rh),
            _nack_fn=lambda t, rh=receipt_handle: self._nack(rh, t),
            _extend_fn=lambda t, rh=receipt_handle: self._extend(rh, t),
        )

    def _acknowledge(self, receipt_handle: str) -> None:
        """Delete message from queue.

        Args:
            receipt_handle: The receipt handle for this delivery.

        Raises:
            ReceiptHandleExpiredError: If the receipt handle is invalid or expired.
        """
        try:
            self.client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle,
            )
        except Exception as e:
            # Check for receipt handle errors
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if error_code in {"ReceiptHandleIsInvalid", "InvalidParameterValue"}:
                raise ReceiptHandleExpiredError(
                    f"Receipt handle expired or invalid: {e}"
                ) from e
            raise MailboxConnectionError(f"Failed to acknowledge message: {e}") from e

    def _nack(self, receipt_handle: str, visibility_timeout: int) -> None:
        """Return message to queue by changing its visibility timeout.

        Args:
            receipt_handle: The receipt handle for this delivery.
            visibility_timeout: Seconds before message becomes visible again.

        Raises:
            ReceiptHandleExpiredError: If the receipt handle is invalid or expired.
        """
        visibility_timeout = max(0, min(visibility_timeout, _MAX_VISIBILITY_TIMEOUT))

        try:
            self.client.change_message_visibility(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=visibility_timeout,
            )
        except Exception as e:
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if error_code in {"ReceiptHandleIsInvalid", "InvalidParameterValue"}:
                raise ReceiptHandleExpiredError(
                    f"Receipt handle expired or invalid: {e}"
                ) from e
            raise MailboxConnectionError(f"Failed to nack message: {e}") from e

    def _extend(self, receipt_handle: str, timeout: int) -> None:
        """Extend visibility timeout.

        Args:
            receipt_handle: The receipt handle for this delivery.
            timeout: New visibility timeout in seconds from now.

        Raises:
            ReceiptHandleExpiredError: If the receipt handle is invalid or expired.
        """
        timeout = max(0, min(timeout, _MAX_VISIBILITY_TIMEOUT))

        try:
            self.client.change_message_visibility(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=timeout,
            )
        except Exception as e:
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if error_code in {"ReceiptHandleIsInvalid", "InvalidParameterValue"}:
                raise ReceiptHandleExpiredError(
                    f"Receipt handle expired or invalid: {e}"
                ) from e
            raise MailboxConnectionError(
                f"Failed to extend visibility timeout: {e}"
            ) from e

    def purge(self) -> int:
        """Delete all messages from the queue.

        Returns:
            Approximate count of messages deleted.

        Note:
            SQS enforces 60-second cooldown between purges.
            The returned count is approximate as SQS doesn't provide
            an exact count during purge.
        """
        try:
            # Get approximate count before purge
            count = self.approximate_count()
            self.client.purge_queue(QueueUrl=self.queue_url)
            return count
        except Exception as e:
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if error_code == "PurgeQueueInProgress":
                # Purge already in progress, this is not an error
                return 0
            raise MailboxConnectionError(f"Failed to purge queue: {e}") from e

    def approximate_count(self) -> int:
        """Return approximate number of messages in the queue.

        The count includes both visible and invisible messages.
        Value is eventually consistent (~1 minute lag for SQS).
        """
        try:
            response = self.client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=[
                    "ApproximateNumberOfMessages",
                    "ApproximateNumberOfMessagesNotVisible",
                ],
            )
            attrs = response.get("Attributes", {})
            visible = int(attrs.get("ApproximateNumberOfMessages", "0"))
            invisible = int(attrs.get("ApproximateNumberOfMessagesNotVisible", "0"))
            return visible + invisible
        except Exception as e:
            raise MailboxConnectionError(f"Failed to get queue count: {e}") from e

    def close(self) -> None:
        """Mark the mailbox as closed.

        After closing, receive() returns empty immediately.
        Does not close the boto3 client.
        """
        with self._lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        """Return True if mailbox has been closed."""
        return self._closed


__all__ = ["SQSMailbox"]

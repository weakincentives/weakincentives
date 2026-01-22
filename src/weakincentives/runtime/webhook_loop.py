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

"""Webhook forwarding loop for MainLoop results.

WebhookLoop provides a mailbox-based pattern for forwarding MainLoop
request/response pairs to HTTP endpoints. It receives WebhookDelivery
messages and POSTs them to configured webhook endpoints based on
routing rules.

Example::

    routes = (
        WebhookRoute(
            pattern=r"code-review",
            endpoint="https://api.example.com/webhooks/reviews",
            name="code-reviews",
        ),
        WebhookRoute(
            pattern=r".*",  # catch-all
            endpoint="https://api.example.com/webhooks/default",
            name="default",
        ),
    )
    config = WebhookLoopConfig(routes=routes)

    loop = WebhookLoop(
        http_client=http_client,
        requests=requests_mailbox,
        config=config,
    )
    loop.run()
"""

from __future__ import annotations

import contextlib
import re
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Self, cast

from ..serde import dump
from .dlq import DeadLetter, DLQPolicy
from .lease_extender import LeaseExtender, LeaseExtenderConfig
from .lifecycle import wait_until
from .logging import StructuredLogger, get_logger
from .mailbox import (
    Mailbox,
    Message,
    ReceiptHandleExpiredError,
    ReplyNotAvailableError,
)
from .watchdog import Heartbeat
from .webhook_loop_types import (
    HTTPClient,
    HTTPClientError,
    WebhookDelivery,
    WebhookLoopConfig,
    WebhookLoopResult,
    WebhookRoute,
)

# HTTP status code ranges
_HTTP_OK = 200
_HTTP_REDIRECT = 300
_HTTP_CLIENT_ERROR = 400
_HTTP_SERVER_ERROR = 500

_logger: StructuredLogger = get_logger(
    __name__, context={"component": "runtime.webhook_loop"}
)


class WebhookLoop[UserRequestT, OutputT]:
    """Forwards MainLoop request/result pairs to HTTP endpoints.

    WebhookLoop receives WebhookDelivery messages from a mailbox and POSTs
    them to configured HTTP endpoints based on routing rules. It supports
    retry with exponential backoff, configurable routes, and dead letter
    queue handling.

    Features:
        - Pattern-based routing to different endpoints
        - Retry with exponential backoff for transient failures
        - Dead letter queue for poison pill handling
        - Heartbeat-based lease extension during delivery
        - Graceful shutdown with in-flight completion

    Execution flow:
        1. Receive WebhookDelivery message from requests mailbox
        2. Match route_key against configured routes (first match wins)
        3. Build JSON payload from delivery
        4. POST to matched endpoint with retries
        5. Send WebhookLoopResult via msg.reply()
        6. Acknowledge the request message

    Error handling:
        - Transient HTTP errors: retry with exponential backoff
        - No matching route: skip delivery, reply with error
        - Permanent failures: dead-letter or reply with error

    Lifecycle:
        - Use ``run()`` to start processing messages
        - Call ``shutdown()`` to request graceful stop
        - In-flight deliveries complete before exit
        - Supports context manager protocol for automatic cleanup
    """

    _http_client: HTTPClient
    _requests: Mailbox[WebhookDelivery[UserRequestT, OutputT], WebhookLoopResult]
    _config: WebhookLoopConfig
    _dlq: DLQPolicy[WebhookDelivery[UserRequestT, OutputT], WebhookLoopResult] | None
    _shutdown_event: threading.Event
    _running: bool
    _lock: threading.Lock
    _heartbeat: Heartbeat
    _lease_extender: LeaseExtender
    _compiled_routes: list[tuple[re.Pattern[str], WebhookRoute]]

    def __init__(
        self,
        *,
        http_client: HTTPClient,
        requests: Mailbox[WebhookDelivery[UserRequestT, OutputT], WebhookLoopResult],
        config: WebhookLoopConfig,
        dlq: DLQPolicy[WebhookDelivery[UserRequestT, OutputT], WebhookLoopResult]
        | None = None,
    ) -> None:
        """Initialize the WebhookLoop.

        Args:
            http_client: HTTP client for posting webhook payloads.
            requests: Mailbox to receive WebhookDelivery messages from.
                Response routing derives from each message's reply_to field.
            config: Configuration with routes, retry settings, and headers.
            dlq: Optional dead letter queue policy. When configured, messages
                that fail repeatedly are sent to the DLQ mailbox instead of
                retrying indefinitely.
        """
        super().__init__()
        self._http_client = http_client
        self._requests = requests
        self._config = config
        self._dlq = dlq
        self._shutdown_event = threading.Event()
        self._running = False
        self._lock = threading.Lock()
        self._heartbeat = Heartbeat()
        # Initialize lease extender with config or defaults
        lease_config = (
            self._config.lease_extender
            if self._config.lease_extender is not None
            else LeaseExtenderConfig()
        )
        self._lease_extender = LeaseExtender(config=lease_config)
        # Pre-compile route patterns for efficiency
        self._compiled_routes = [
            (re.compile(route.pattern), route) for route in self._config.routes
        ]

    @property
    def heartbeat(self) -> Heartbeat:
        """Heartbeat tracker for watchdog monitoring.

        The loop beats after receiving messages and after processing each
        message, enabling the watchdog to detect stuck workers.
        """
        return self._heartbeat

    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> None:
        """Run the worker loop, processing messages from the requests mailbox.

        Polls the requests mailbox for messages and processes each one.
        Messages are acknowledged after successful processing or after
        sending an error response.

        The loop exits when:
        - max_iterations is reached
        - shutdown() is called
        - The requests mailbox is closed

        In-flight messages complete before exit. Unprocessed messages from
        the current batch are nacked for redelivery.

        Args:
            max_iterations: Maximum polling iterations. None for unlimited.
            visibility_timeout: Seconds messages remain invisible during processing.
                Should exceed maximum expected execution time.
            wait_time_seconds: Long poll duration for receiving messages.
        """
        with self._lock:
            self._running = True
            self._shutdown_event.clear()

        iterations = 0
        try:
            while max_iterations is None or iterations < max_iterations:
                # Check shutdown before blocking on receive
                if self._shutdown_event.is_set():
                    break

                # Exit if mailbox closed
                if self._requests.closed:
                    break

                messages = self._requests.receive(
                    visibility_timeout=visibility_timeout,
                    wait_time_seconds=wait_time_seconds,
                )

                # Beat after receive (proves we're not stuck waiting)
                self._heartbeat.beat()

                for msg in messages:
                    # Check shutdown between messages (defensive: default batch=1)
                    if self._shutdown_event.is_set():  # pragma: no cover
                        # Nack unprocessed message for redelivery
                        with contextlib.suppress(ReceiptHandleExpiredError):
                            msg.nack(visibility_timeout=0)
                        break

                    self._handle_message(msg)

                    # Beat after each message (proves processing completes)
                    self._heartbeat.beat()

                iterations += 1
        finally:
            with self._lock:
                self._running = False

    def shutdown(self, *, timeout: float = 30.0) -> bool:
        """Request graceful shutdown and wait for completion.

        Sets the shutdown flag. If the loop is running, waits up to timeout
        seconds for it to stop.

        Args:
            timeout: Maximum seconds to wait for the loop to stop.

        Returns:
            True if loop stopped cleanly, False if timeout expired.
        """
        self._shutdown_event.set()
        return wait_until(lambda: not self.running, timeout=timeout)

    @property
    def running(self) -> bool:
        """True if the loop is currently processing messages."""
        with self._lock:
            return self._running

    def __enter__(self) -> Self:
        """Context manager entry. Returns self for use in with statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit. Triggers shutdown and waits for completion."""
        _ = (exc_type, exc_val, exc_tb)
        _ = self.shutdown()

    def _handle_message(
        self,
        msg: Message[WebhookDelivery[UserRequestT, OutputT], WebhookLoopResult],
    ) -> None:
        """Process a single message from the requests mailbox."""
        delivery = msg.body

        _logger.debug(
            "Processing webhook delivery.",
            event="webhook_loop.message_received",
            context={
                "message_id": msg.id,
                "delivery_id": str(delivery.delivery_id),
                "route_key": delivery.route_key,
            },
        )

        # Attach lease extender to heartbeat for this message
        with self._lease_extender.attach(msg, self._heartbeat):
            try:
                result = self._deliver_webhook(delivery)
            except Exception as e:
                self._handle_failure(msg, e)
                return
            self._reply_and_ack(msg, result)

    def _match_route(self, route_key: str) -> WebhookRoute | None:
        """Find the first matching route for the given key."""
        for pattern, route in self._compiled_routes:
            if pattern.search(route_key):
                return route
        return None

    def _build_payload(
        self, delivery: WebhookDelivery[UserRequestT, OutputT]
    ) -> dict[str, object]:
        """Build JSON payload from delivery."""
        _ = self  # Instance method for API consistency
        # Serialize the MainLoopRequest and MainLoopResult
        request_data = dump(delivery.request)
        result_data = dump(delivery.result)

        payload: dict[str, object] = {
            "delivery_id": str(delivery.delivery_id),
            "route_key": delivery.route_key,
            "created_at": delivery.created_at.isoformat(),
            "request": request_data,
            "result": result_data,
        }

        if delivery.metadata:
            payload["metadata"] = dict(delivery.metadata)

        return payload

    def _build_headers(self, route: WebhookRoute) -> dict[str, str]:
        """Build headers combining defaults with route-specific overrides."""
        headers: dict[str, str] = {"Content-Type": "application/json"}

        # Apply default headers
        if self._config.default_headers:
            headers.update(self._config.default_headers)

        # Apply route-specific headers (override defaults)
        if route.headers:
            headers.update(route.headers)

        return headers

    def _deliver_webhook(
        self, delivery: WebhookDelivery[UserRequestT, OutputT]
    ) -> WebhookLoopResult:
        """Attempt to deliver the webhook with retries."""
        route = self._match_route(delivery.route_key)

        if route is None:
            _logger.warning(
                "No matching route for webhook delivery.",
                event="webhook_loop.no_route_match",
                context={
                    "delivery_id": str(delivery.delivery_id),
                    "route_key": delivery.route_key,
                },
            )
            return WebhookLoopResult(
                delivery_id=delivery.delivery_id,
                request_id=delivery.request.request_id,
                success=False,
                error=f"No matching route for route_key: {delivery.route_key}",
                attempts=0,
            )

        payload = self._build_payload(delivery)
        headers = self._build_headers(route)

        last_error: str | None = None
        attempts = 0

        for attempt in range(self._config.retry_attempts):
            attempts = attempt + 1
            try:
                response = self._http_client.post(
                    route.endpoint,
                    json=cast(Mapping[str, object], payload),
                    headers=headers,
                    timeout=route.timeout_seconds,
                )

                # Success: 2xx status codes
                if _HTTP_OK <= response.status_code < _HTTP_REDIRECT:
                    _logger.info(
                        "Webhook delivered successfully.",
                        event="webhook_loop.delivery_success",
                        context={
                            "delivery_id": str(delivery.delivery_id),
                            "route_name": route.name,
                            "endpoint": route.endpoint,
                            "status_code": response.status_code,
                            "attempts": attempts,
                        },
                    )
                    return WebhookLoopResult(
                        delivery_id=delivery.delivery_id,
                        request_id=delivery.request.request_id,
                        success=True,
                        route_name=route.name,
                        route_pattern=route.pattern,
                        status_code=response.status_code,
                        attempts=attempts,
                    )

                # Client errors (4xx): don't retry
                if _HTTP_CLIENT_ERROR <= response.status_code < _HTTP_SERVER_ERROR:
                    _logger.warning(
                        "Webhook delivery failed with client error.",
                        event="webhook_loop.delivery_client_error",
                        context={
                            "delivery_id": str(delivery.delivery_id),
                            "route_name": route.name,
                            "endpoint": route.endpoint,
                            "status_code": response.status_code,
                            "attempts": attempts,
                        },
                    )
                    return WebhookLoopResult(
                        delivery_id=delivery.delivery_id,
                        request_id=delivery.request.request_id,
                        success=False,
                        route_name=route.name,
                        route_pattern=route.pattern,
                        status_code=response.status_code,
                        error=f"Client error: HTTP {response.status_code}",
                        attempts=attempts,
                    )

                # Server errors (5xx): retry
                last_error = f"Server error: HTTP {response.status_code}"
                _logger.debug(
                    "Webhook delivery got server error, will retry.",
                    event="webhook_loop.delivery_server_error",
                    context={
                        "delivery_id": str(delivery.delivery_id),
                        "route_name": route.name,
                        "status_code": response.status_code,
                        "attempt": attempts,
                    },
                )

            except HTTPClientError as e:
                last_error = str(e)
                _logger.debug(
                    "Webhook delivery failed with HTTP error, will retry.",
                    event="webhook_loop.delivery_http_error",
                    context={
                        "delivery_id": str(delivery.delivery_id),
                        "route_name": route.name,
                        "error": str(e),
                        "attempt": attempts,
                    },
                )

            # Exponential backoff before retry (skip on last attempt)
            if attempt < self._config.retry_attempts - 1:
                delay = self._config.retry_base_delay_seconds * (2**attempt)
                time.sleep(delay)
                # Beat during retry wait to extend lease
                self._heartbeat.beat()

        # All retries exhausted
        _logger.warning(
            "Webhook delivery failed after all retries.",
            event="webhook_loop.delivery_exhausted",
            context={
                "delivery_id": str(delivery.delivery_id),
                "route_name": route.name,
                "endpoint": route.endpoint,
                "attempts": attempts,
                "last_error": last_error,
            },
        )
        return WebhookLoopResult(
            delivery_id=delivery.delivery_id,
            request_id=delivery.request.request_id,
            success=False,
            route_name=route.name,
            route_pattern=route.pattern,
            error=f"Delivery failed after {attempts} attempts: {last_error}",
            attempts=attempts,
        )

    def _reply_and_ack(
        self,
        msg: Message[WebhookDelivery[UserRequestT, OutputT], WebhookLoopResult],
        result: WebhookLoopResult,
    ) -> None:
        """Reply with result and acknowledge message, handling failures gracefully."""
        _ = self  # Instance method for API consistency
        try:
            _ = msg.reply(result)
            msg.acknowledge()
        except ReplyNotAvailableError:
            # No reply_to specified - log and acknowledge without reply
            _logger.debug(
                "No reply_to for message, acknowledging without reply.",
                event="webhook_loop.no_reply_to",
                context={"message_id": msg.id},
            )
            with contextlib.suppress(ReceiptHandleExpiredError):
                msg.acknowledge()
        except ReceiptHandleExpiredError:
            # Handle expired during processing - message already requeued
            pass
        except Exception:
            # Reply send failed - nack so message is retried
            with contextlib.suppress(ReceiptHandleExpiredError):
                backoff = min(60 * msg.delivery_count, 900)
                msg.nack(visibility_timeout=backoff)

    def _handle_failure(
        self,
        msg: Message[WebhookDelivery[UserRequestT, OutputT], WebhookLoopResult],
        error: Exception,
    ) -> None:
        """Handle message processing failure.

        When DLQ is configured:
        - Checks DLQ policy and either dead-letters or retries with backoff
        - Only sends error replies on terminal outcomes (DLQ)

        When DLQ is not configured:
        - Sends error reply and acknowledges (original behavior)
        """
        delivery = msg.body

        if self._dlq is None:
            # No DLQ configured - use original behavior: error reply + acknowledge
            result = WebhookLoopResult(
                delivery_id=delivery.delivery_id,
                request_id=delivery.request.request_id,
                success=False,
                error=str(error),
            )
            self._reply_and_ack(msg, result)
            return

        # DLQ is configured - check if we should dead-letter
        if self._dlq.should_dead_letter(msg, error):
            self._dead_letter(msg, error)
            return

        # Retry with backoff - do NOT send error reply here.
        backoff = min(60 * msg.delivery_count, 900)
        with contextlib.suppress(ReceiptHandleExpiredError):
            msg.nack(visibility_timeout=backoff)

    def _dead_letter(
        self,
        msg: Message[WebhookDelivery[UserRequestT, OutputT], WebhookLoopResult],
        error: Exception,
    ) -> None:
        """Send message to dead letter queue."""
        if self._dlq is None:  # pragma: no cover - defensive check
            return

        delivery = msg.body
        enqueued_at: datetime = msg.enqueued_at

        dead_letter: DeadLetter[WebhookDelivery[UserRequestT, OutputT]] = DeadLetter(
            message_id=msg.id,
            body=msg.body,
            source_mailbox=self._requests.name,
            delivery_count=msg.delivery_count,
            last_error=str(error),
            last_error_type=f"{type(error).__module__}.{type(error).__qualname__}",
            dead_lettered_at=datetime.now(UTC),
            first_received_at=enqueued_at,
            request_id=delivery.delivery_id,
            reply_to=msg.reply_to.name if msg.reply_to else None,
        )

        # Send error reply - this is a terminal outcome
        try:
            if msg.reply_to:
                _ = msg.reply(
                    WebhookLoopResult(
                        delivery_id=delivery.delivery_id,
                        request_id=delivery.request.request_id,
                        success=False,
                        error=f"Dead-lettered after {msg.delivery_count} attempts: {error}",
                    )
                )
        except Exception as reply_error:  # nosec B110 - reply failure should not block dead-lettering
            _logger.debug(
                "Failed to send error reply during dead-lettering.",
                event="webhook_loop.dead_letter_reply_failed",
                context={"message_id": msg.id, "error": str(reply_error)},
            )

        _ = self._dlq.mailbox.send(dead_letter)
        _logger.warning(
            "Message dead-lettered.",
            event="webhook_loop.message_dead_lettered",
            context={
                "message_id": msg.id,
                "delivery_id": str(delivery.delivery_id),
                "delivery_count": msg.delivery_count,
                "error_type": dead_letter.last_error_type,
            },
        )

        # Acknowledge to remove from source queue
        with contextlib.suppress(ReceiptHandleExpiredError):
            msg.acknowledge()


__all__ = [
    "WebhookLoop",
    "WebhookLoopConfig",
    "WebhookLoopResult",
]

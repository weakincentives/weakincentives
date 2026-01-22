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

"""Data types for WebhookLoop request and response handling.

This module provides the data classes used by WebhookLoop for forwarding
MainLoop request/response pairs to HTTP endpoints.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID, uuid4

from ..dataclasses import FrozenDataclass
from .lease_extender import LeaseExtenderConfig
from .main_loop_types import MainLoopRequest, MainLoopResult

if TYPE_CHECKING:
    pass


@FrozenDataclass()
class HTTPResponse:
    """Response from an HTTP request."""

    status_code: int
    """HTTP status code (e.g., 200, 404, 500)."""

    body: str
    """Response body as text."""

    headers: Mapping[str, str] = field(default_factory=lambda: dict[str, str]())
    """Response headers."""


@runtime_checkable
class HTTPClient(Protocol):
    """Protocol for HTTP client injection.

    Implementations must be thread-safe. The client is used by WebhookLoop
    to POST webhook payloads to configured endpoints.
    """

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
    ) -> HTTPResponse:
        """Send a POST request with JSON body.

        Args:
            url: The target URL.
            json: JSON-serializable payload.
            headers: HTTP headers to include.
            timeout: Request timeout in seconds.

        Returns:
            HTTPResponse with status code, body, and headers.

        Raises:
            HTTPClientError: On network or protocol errors.
        """
        ...


class HTTPClientError(Exception):
    """Raised when HTTP client encounters an error."""

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code


@FrozenDataclass()
class WebhookRoute:
    """A single webhook routing rule.

    Routes match against a string identifier (typically derived from the
    request type or a custom tag). When a delivery matches, it is POSTed
    to the configured endpoint.
    """

    pattern: str
    """Regex pattern to match against the route key."""

    endpoint: str
    """HTTP URL to POST to."""

    headers: Mapping[str, str] | None = None
    """Optional custom headers for this route."""

    timeout_seconds: float = 30.0
    """Request timeout in seconds."""

    name: str | None = None
    """Optional human-readable name for logging."""

    @classmethod
    def __pre_init__(
        cls,
        *,
        pattern: str,
        endpoint: str,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
        name: str | None = None,
    ) -> Mapping[str, object]:
        # Validate pattern is a valid regex
        try:
            _ = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e
        # Validate endpoint is a valid URL
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid endpoint URL '{endpoint}': must start with http:// or https://"
            )
        # Validate timeout is positive
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        return {
            "pattern": pattern,
            "endpoint": endpoint,
            "headers": headers,
            "timeout_seconds": timeout_seconds,
            "name": name,
        }


@FrozenDataclass()
class WebhookLoopConfig:
    """Configuration for WebhookLoop execution defaults.

    The ``routes`` field defines the webhook routing rules. Each incoming
    delivery is matched against routes in order; the first match wins.

    The ``lease_extender`` field controls automatic message visibility extension
    during webhook delivery. When enabled, heartbeats extend the message lease,
    preventing timeout during slow webhook endpoints.
    """

    routes: tuple[WebhookRoute, ...]
    """Ordered list of webhook routing rules."""

    default_headers: Mapping[str, str] | None = None
    """Headers applied to all webhook requests (route headers override)."""

    retry_attempts: int = 3
    """Number of retry attempts for failed webhook deliveries."""

    retry_base_delay_seconds: float = 1.0
    """Base delay for exponential backoff between retries."""

    lease_extender: LeaseExtenderConfig | None = None
    """Optional lease extension configuration."""

    @classmethod
    def __pre_init__(
        cls,
        *,
        routes: tuple[WebhookRoute, ...],
        default_headers: Mapping[str, str] | None = None,
        retry_attempts: int = 3,
        retry_base_delay_seconds: float = 1.0,
        lease_extender: LeaseExtenderConfig | None = None,
    ) -> Mapping[str, object]:
        if retry_attempts < 0:
            raise ValueError(
                f"retry_attempts must be non-negative, got {retry_attempts}"
            )
        if retry_base_delay_seconds < 0:
            raise ValueError(
                f"retry_base_delay_seconds must be non-negative, got {retry_base_delay_seconds}"
            )
        return {
            "routes": routes,
            "default_headers": default_headers,
            "retry_attempts": retry_attempts,
            "retry_base_delay_seconds": retry_base_delay_seconds,
            "lease_extender": lease_extender,
        }


@FrozenDataclass()
class WebhookDelivery[UserRequestT, OutputT]:
    """A MainLoop request/result pair to be delivered via webhook.

    This is the message type received by WebhookLoop. Callers are responsible
    for creating these after receiving MainLoopResults.
    """

    request: MainLoopRequest[UserRequestT]
    """The original MainLoop request."""

    result: MainLoopResult[OutputT]
    """The MainLoop result to forward."""

    route_key: str
    """String to match against route patterns (e.g., request type name)."""

    delivery_id: UUID = field(default_factory=uuid4)
    """Unique identifier for this delivery attempt."""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Timestamp when this delivery was created."""

    metadata: Mapping[str, str] | None = None
    """Optional metadata to include in webhook payload."""


@FrozenDataclass()
class WebhookLoopResult:
    """Result of a webhook delivery attempt.

    Returned via Message.reply() after processing a WebhookDelivery.
    """

    delivery_id: UUID
    """Correlates with WebhookDelivery.delivery_id."""

    request_id: UUID
    """Correlates with MainLoopRequest.request_id."""

    success: bool
    """True if webhook was delivered successfully."""

    route_name: str | None = None
    """Name of the matched route (if any)."""

    route_pattern: str | None = None
    """Pattern of the matched route (if any)."""

    status_code: int | None = None
    """HTTP status code from webhook endpoint (on success)."""

    error: str | None = None
    """Error message on failure."""

    attempts: int = 1
    """Number of delivery attempts made."""

    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Timestamp when processing completed."""


__all__ = [
    "HTTPClient",
    "HTTPClientError",
    "HTTPResponse",
    "WebhookDelivery",
    "WebhookLoopConfig",
    "WebhookLoopResult",
    "WebhookRoute",
]

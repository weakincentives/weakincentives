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

"""Tests for WebhookLoop orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC
from uuid import UUID

import pytest

from weakincentives.runtime.mailbox import (
    FakeMailbox,
    InMemoryMailbox,
    MailboxConnectionError,
)
from weakincentives.runtime.main_loop_types import MainLoopRequest, MainLoopResult
from weakincentives.runtime.webhook_loop import WebhookLoop
from weakincentives.runtime.webhook_loop_types import (
    HTTPClient,
    HTTPClientError,
    HTTPResponse,
    WebhookDelivery,
    WebhookLoopConfig,
    WebhookLoopResult,
    WebhookRoute,
)


@dataclass(slots=True, frozen=True)
class _Request:
    """Sample request type for testing."""

    message: str


@dataclass(slots=True, frozen=True)
class _Output:
    """Sample output type for testing."""

    result: str


# =============================================================================
# Mock HTTP Client
# =============================================================================


@dataclass
class _MockHTTPClient:
    """Mock HTTP client for testing WebhookLoop behavior."""

    responses: list[HTTPResponse] = field(default_factory=list)
    errors: list[HTTPClientError | None] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
    ) -> HTTPResponse:
        call = {
            "url": url,
            "json": dict(json),
            "headers": dict(headers),
            "timeout": timeout,
        }
        self.calls.append(call)

        # Return error if configured
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error

        # Return response if configured, otherwise default success
        if self.responses:
            return self.responses.pop(0)
        return HTTPResponse(status_code=200, body='{"status": "ok"}')


# Verify _MockHTTPClient is a valid HTTPClient
def _verify_http_client_protocol() -> HTTPClient:
    return _MockHTTPClient()


# =============================================================================
# WebhookRoute Tests
# =============================================================================


def test_route_default_values() -> None:
    """WebhookRoute has sensible defaults."""
    route = WebhookRoute(
        pattern=r"test-.*",
        endpoint="https://api.example.com/webhooks",
    )
    assert route.pattern == r"test-.*"
    assert route.endpoint == "https://api.example.com/webhooks"
    assert route.headers is None
    assert route.timeout_seconds == 30.0
    assert route.name is None


def test_route_custom_values() -> None:
    """WebhookRoute accepts custom values."""
    headers = {"X-Custom": "value"}
    route = WebhookRoute(
        pattern=r"code-review",
        endpoint="https://api.example.com/reviews",
        headers=headers,
        timeout_seconds=60.0,
        name="code-reviews",
    )
    assert route.pattern == r"code-review"
    assert route.endpoint == "https://api.example.com/reviews"
    assert route.headers == headers
    assert route.timeout_seconds == 60.0
    assert route.name == "code-reviews"


def test_route_invalid_pattern() -> None:
    """WebhookRoute rejects invalid regex patterns."""
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        WebhookRoute(pattern=r"[invalid", endpoint="https://example.com")


def test_route_invalid_endpoint() -> None:
    """WebhookRoute rejects invalid endpoint URLs."""
    with pytest.raises(ValueError, match="must start with http"):
        WebhookRoute(pattern=r".*", endpoint="ftp://example.com")


def test_route_invalid_timeout() -> None:
    """WebhookRoute rejects non-positive timeout."""
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        WebhookRoute(pattern=r".*", endpoint="https://example.com", timeout_seconds=0)


# =============================================================================
# WebhookLoopConfig Tests
# =============================================================================


def test_config_default_values() -> None:
    """WebhookLoopConfig has sensible defaults."""
    routes = (WebhookRoute(pattern=r".*", endpoint="https://example.com"),)
    config = WebhookLoopConfig(routes=routes)
    assert config.routes == routes
    assert config.default_headers is None
    assert config.retry_attempts == 3
    assert config.retry_base_delay_seconds == 1.0
    assert config.lease_extender is None


def test_config_custom_values() -> None:
    """WebhookLoopConfig accepts custom values."""
    routes = (WebhookRoute(pattern=r"test", endpoint="https://example.com"),)
    headers = {"Authorization": "Bearer token"}
    config = WebhookLoopConfig(
        routes=routes,
        default_headers=headers,
        retry_attempts=5,
        retry_base_delay_seconds=0.5,
    )
    assert config.routes == routes
    assert config.default_headers == headers
    assert config.retry_attempts == 5
    assert config.retry_base_delay_seconds == 0.5


def test_config_invalid_retry_attempts() -> None:
    """WebhookLoopConfig rejects negative retry attempts."""
    routes = (WebhookRoute(pattern=r".*", endpoint="https://example.com"),)
    with pytest.raises(ValueError, match="retry_attempts must be non-negative"):
        WebhookLoopConfig(routes=routes, retry_attempts=-1)


def test_config_invalid_retry_delay() -> None:
    """WebhookLoopConfig rejects negative retry delay."""
    routes = (WebhookRoute(pattern=r".*", endpoint="https://example.com"),)
    with pytest.raises(
        ValueError, match="retry_base_delay_seconds must be non-negative"
    ):
        WebhookLoopConfig(routes=routes, retry_base_delay_seconds=-1)


# =============================================================================
# WebhookDelivery Tests
# =============================================================================


def test_delivery_default_values() -> None:
    """WebhookDelivery has sensible defaults."""
    request = MainLoopRequest(request=_Request(message="hello"))
    result: MainLoopResult[_Output] = MainLoopResult(
        request_id=request.request_id,
        output=_Output(result="success"),
    )
    delivery = WebhookDelivery(
        request=request,
        result=result,
        route_key="test-route",
    )
    assert delivery.request == request
    assert delivery.result == result
    assert delivery.route_key == "test-route"
    assert isinstance(delivery.delivery_id, UUID)
    assert delivery.created_at.tzinfo == UTC
    assert delivery.metadata is None


def test_delivery_custom_values() -> None:
    """WebhookDelivery accepts custom values."""
    request = MainLoopRequest(request=_Request(message="hello"))
    result: MainLoopResult[_Output] = MainLoopResult(
        request_id=request.request_id,
        output=_Output(result="success"),
    )
    metadata = {"source": "test"}
    delivery = WebhookDelivery(
        request=request,
        result=result,
        route_key="custom-route",
        metadata=metadata,
    )
    assert delivery.route_key == "custom-route"
    assert delivery.metadata == metadata


# =============================================================================
# WebhookLoopResult Tests
# =============================================================================


def test_loop_result_success_case() -> None:
    """WebhookLoopResult represents successful delivery."""
    delivery_id = UUID("12345678-1234-5678-1234-567812345678")
    request_id = UUID("87654321-4321-8765-4321-876543218765")
    result = WebhookLoopResult(
        delivery_id=delivery_id,
        request_id=request_id,
        success=True,
        route_name="test-route",
        route_pattern=r"test-.*",
        status_code=200,
    )
    assert result.delivery_id == delivery_id
    assert result.request_id == request_id
    assert result.success is True
    assert result.route_name == "test-route"
    assert result.status_code == 200
    assert result.error is None
    assert result.completed_at.tzinfo == UTC


def test_loop_result_error_case() -> None:
    """WebhookLoopResult represents failed delivery."""
    delivery_id = UUID("12345678-1234-5678-1234-567812345678")
    request_id = UUID("87654321-4321-8765-4321-876543218765")
    result = WebhookLoopResult(
        delivery_id=delivery_id,
        request_id=request_id,
        success=False,
        error="Connection refused",
        attempts=3,
    )
    assert result.delivery_id == delivery_id
    assert result.success is False
    assert result.error == "Connection refused"
    assert result.attempts == 3


def test_loop_result_is_frozen() -> None:
    """WebhookLoopResult is immutable."""
    delivery_id = UUID("12345678-1234-5678-1234-567812345678")
    request_id = UUID("87654321-4321-8765-4321-876543218765")
    result = WebhookLoopResult(
        delivery_id=delivery_id,
        request_id=request_id,
        success=True,
    )
    with pytest.raises(AttributeError):
        result.success = False  # type: ignore[misc]


# =============================================================================
# HTTPResponse Tests
# =============================================================================


def test_http_response_default_headers() -> None:
    """HTTPResponse has empty headers by default."""
    response = HTTPResponse(status_code=200, body="OK")
    assert response.status_code == 200
    assert response.body == "OK"
    assert response.headers == {}


def test_http_response_with_headers() -> None:
    """HTTPResponse accepts headers."""
    headers = {"Content-Type": "application/json"}
    response = HTTPResponse(status_code=201, body='{"id": 1}', headers=headers)
    assert response.status_code == 201
    assert response.headers == headers


# =============================================================================
# HTTPClientError Tests
# =============================================================================


def test_http_client_error() -> None:
    """HTTPClientError carries URL and status code."""
    error = HTTPClientError(
        "Connection timeout",
        url="https://example.com",
        status_code=None,
    )
    assert str(error) == "Connection timeout"
    assert error.url == "https://example.com"
    assert error.status_code is None


# =============================================================================
# WebhookLoop Tests
# =============================================================================


def test_loop_processes_delivery() -> None:
    """WebhookLoop processes delivery from mailbox."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (
            WebhookRoute(pattern=r"test-.*", endpoint="https://api.example.com/hook"),
        )
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        # Create delivery
        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test-delivery",
        )
        requests.send(delivery, reply_to=results)

        # Run single iteration
        loop.run(max_iterations=1, wait_time_seconds=0)

        # Check response
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.delivery_id == delivery.delivery_id
        assert msgs[0].body.success is True
        assert msgs[0].body.status_code == 200
        msgs[0].acknowledge()

        # Check HTTP client was called
        assert len(http_client.calls) == 1
        assert http_client.calls[0]["url"] == "https://api.example.com/hook"
    finally:
        requests.close()
        results.close()


def test_loop_matches_route_by_pattern() -> None:
    """WebhookLoop matches route by pattern."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (
            WebhookRoute(
                pattern=r"code-review",
                endpoint="https://api.example.com/reviews",
                name="reviews",
            ),
            WebhookRoute(
                pattern=r"test-.*",
                endpoint="https://api.example.com/tests",
                name="tests",
            ),
        )
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        # Create delivery with route_key matching second route
        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test-something",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        assert msgs[0].body.route_name == "tests"
        assert msgs[0].body.route_pattern == r"test-.*"
        msgs[0].acknowledge()

        # Should have called the tests endpoint
        assert http_client.calls[0]["url"] == "https://api.example.com/tests"
    finally:
        requests.close()
        results.close()


def test_loop_no_matching_route() -> None:
    """WebhookLoop returns error when no route matches."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (
            WebhookRoute(
                pattern=r"specific-pattern", endpoint="https://api.example.com"
            ),
        )
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        # Create delivery with non-matching route_key
        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="no-match",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is False
        assert "No matching route" in (msgs[0].body.error or "")
        assert msgs[0].body.attempts == 0
        msgs[0].acknowledge()

        # HTTP client should not have been called
        assert len(http_client.calls) == 0
    finally:
        requests.close()
        results.close()


def test_loop_uses_default_headers() -> None:
    """WebhookLoop applies default headers to requests."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        default_headers = {"Authorization": "Bearer token", "X-Custom": "value"}
        config = WebhookLoopConfig(
            routes=routes,
            default_headers=default_headers,
            retry_attempts=1,
        )
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Check headers were applied
        assert "Authorization" in http_client.calls[0]["headers"]
        assert http_client.calls[0]["headers"]["Authorization"] == "Bearer token"
        assert http_client.calls[0]["headers"]["X-Custom"] == "value"
        # Content-Type should always be set
        assert http_client.calls[0]["headers"]["Content-Type"] == "application/json"
    finally:
        requests.close()
        results.close()


def test_loop_route_headers_override_defaults() -> None:
    """WebhookLoop route headers override default headers."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (
            WebhookRoute(
                pattern=r".*",
                endpoint="https://api.example.com",
                headers={"Authorization": "Bearer route-token"},
            ),
        )
        default_headers = {"Authorization": "Bearer default-token"}
        config = WebhookLoopConfig(
            routes=routes,
            default_headers=default_headers,
            retry_attempts=1,
        )
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Route header should override default
        assert http_client.calls[0]["headers"]["Authorization"] == "Bearer route-token"
    finally:
        requests.close()
        results.close()


def test_loop_retries_on_server_error() -> None:
    """WebhookLoop retries on 5xx server errors."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient(
            responses=[
                HTTPResponse(status_code=503, body="Service Unavailable"),
                HTTPResponse(status_code=200, body="OK"),
            ]
        )
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(
            routes=routes,
            retry_attempts=3,
            retry_base_delay_seconds=0.01,  # Fast for testing
        )
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        assert msgs[0].body.attempts == 2  # 1 failed + 1 success
        msgs[0].acknowledge()

        # Should have made 2 requests
        assert len(http_client.calls) == 2
    finally:
        requests.close()
        results.close()


def test_loop_no_retry_on_client_error() -> None:
    """WebhookLoop does not retry on 4xx client errors."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient(
            responses=[HTTPResponse(status_code=400, body="Bad Request")]
        )
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=3)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is False
        assert msgs[0].body.status_code == 400
        assert "Client error" in (msgs[0].body.error or "")
        assert msgs[0].body.attempts == 1  # No retry
        msgs[0].acknowledge()

        # Should have made only 1 request
        assert len(http_client.calls) == 1
    finally:
        requests.close()
        results.close()


def test_loop_retries_on_http_client_error() -> None:
    """WebhookLoop retries on HTTPClientError."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient(
            errors=[
                HTTPClientError("Connection refused"),
                None,  # No error on second try
            ],
            responses=[HTTPResponse(status_code=200, body="OK")],
        )
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(
            routes=routes,
            retry_attempts=3,
            retry_base_delay_seconds=0.01,
        )
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        assert msgs[0].body.attempts == 2
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_exhausts_retries() -> None:
    """WebhookLoop reports failure after exhausting retries."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient(
            responses=[
                HTTPResponse(status_code=500, body="Error"),
                HTTPResponse(status_code=500, body="Error"),
                HTTPResponse(status_code=500, body="Error"),
            ]
        )
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(
            routes=routes,
            retry_attempts=3,
            retry_base_delay_seconds=0.01,
        )
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is False
        assert msgs[0].body.attempts == 3
        assert "failed after 3 attempts" in (msgs[0].body.error or "")
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_uses_route_timeout() -> None:
    """WebhookLoop uses route-specific timeout."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (
            WebhookRoute(
                pattern=r".*",
                endpoint="https://api.example.com",
                timeout_seconds=45.0,
            ),
        )
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Check timeout was passed to HTTP client
        assert http_client.calls[0]["timeout"] == 45.0
    finally:
        requests.close()
        results.close()


def test_loop_includes_metadata_in_payload() -> None:
    """WebhookLoop includes delivery metadata in payload."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        metadata = {"source": "test", "version": "1.0"}
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
            metadata=metadata,
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Check metadata was included in payload
        payload = http_client.calls[0]["json"]
        assert "metadata" in payload
        assert payload["metadata"] == {"source": "test", "version": "1.0"}
    finally:
        requests.close()
        results.close()


def test_loop_acknowledges_request() -> None:
    """WebhookLoop acknowledges processed request."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Request should be acknowledged (gone from queue)
        assert requests.approximate_count() == 0
    finally:
        requests.close()
        results.close()


def test_loop_respects_max_iterations() -> None:
    """WebhookLoop respects max_iterations limit."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        for i in range(5):
            main_request = MainLoopRequest(request=_Request(message=f"msg-{i}"))
            main_result: MainLoopResult[_Output] = MainLoopResult(
                request_id=main_request.request_id,
                output=_Output(result="success"),
            )
            delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
                request=main_request,
                result=main_result,
                route_key="test",
            )
            requests.send(delivery, reply_to=results)

        # Only run 2 iterations
        loop.run(max_iterations=2, wait_time_seconds=0)

        # Some requests may still be pending
        assert results.approximate_count() >= 1
    finally:
        requests.close()
        results.close()


def test_loop_nacks_on_response_send_failure() -> None:
    """WebhookLoop nacks request when response send fails."""
    results: FakeMailbox[WebhookLoopResult, None] = FakeMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        requests.send(delivery, reply_to=results)

        # Make response send fail
        results.set_connection_error(MailboxConnectionError("connection lost"))

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Request should be nacked (still in queue for retry)
        assert requests.approximate_count() == 1
    finally:
        requests.close()


def test_loop_exits_when_mailbox_closed() -> None:
    """WebhookLoop.run() exits when requests mailbox is closed."""
    import threading

    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    http_client = _MockHTTPClient()
    routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
    config = WebhookLoopConfig(routes=routes, retry_attempts=1)
    loop: WebhookLoop[_Request, _Output] = WebhookLoop(
        http_client=http_client,
        requests=requests,
        config=config,
    )

    exited = []

    def run_loop() -> None:
        loop.run(max_iterations=None, wait_time_seconds=1)
        exited.append(True)

    thread = threading.Thread(target=run_loop)
    thread.start()

    # Close the mailbox - should cause loop to exit
    requests.close()
    results.close()

    # Thread should exit quickly
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(exited) == 1


def test_loop_heartbeat_property() -> None:
    """WebhookLoop exposes heartbeat property."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )
        assert loop.heartbeat is not None
    finally:
        requests.close()
        results.close()


def test_loop_running_property() -> None:
    """WebhookLoop exposes running property."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )
        assert loop.running is False

        # Run with no messages should still show not running after
        loop.run(max_iterations=1, wait_time_seconds=0)
        assert loop.running is False
    finally:
        requests.close()
        results.close()


def test_loop_context_manager() -> None:
    """WebhookLoop supports context manager protocol."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        with WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        ) as loop:
            assert loop.running is False
    finally:
        requests.close()
        results.close()


def test_loop_shutdown() -> None:
    """WebhookLoop shutdown method works correctly."""
    import threading

    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    http_client = _MockHTTPClient()
    routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
    config = WebhookLoopConfig(routes=routes, retry_attempts=1)
    loop: WebhookLoop[_Request, _Output] = WebhookLoop(
        http_client=http_client,
        requests=requests,
        config=config,
    )

    def run_loop() -> None:
        loop.run(max_iterations=None, wait_time_seconds=1)

    thread = threading.Thread(target=run_loop)
    thread.start()

    # Wait briefly for loop to start
    import time

    time.sleep(0.1)

    # Shutdown should stop the loop
    result = loop.shutdown(timeout=2.0)
    assert result is True
    assert loop.running is False

    thread.join(timeout=1.0)
    assert not thread.is_alive()

    requests.close()
    results.close()


def test_loop_payload_structure() -> None:
    """WebhookLoop creates correct payload structure."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test-route",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Check payload structure
        payload = http_client.calls[0]["json"]
        assert "delivery_id" in payload
        assert "route_key" in payload
        assert payload["route_key"] == "test-route"
        assert "created_at" in payload
        assert "request" in payload
        assert "result" in payload
        # Check nested request data
        assert "request" in payload["request"]  # MainLoopRequest has a request field
    finally:
        requests.close()
        results.close()


def test_loop_handles_no_reply_to() -> None:
    """WebhookLoop handles messages without reply_to."""
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()
        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        # Send without reply_to
        requests.send(delivery)

        # Should not raise
        loop.run(max_iterations=1, wait_time_seconds=0)

        # Request should still be acknowledged
        assert requests.approximate_count() == 0
    finally:
        requests.close()


# =============================================================================
# Shutdown and DLQ Tests
# =============================================================================


def test_loop_stops_on_shutdown_signal() -> None:
    """WebhookLoop stops cleanly when shutdown is signaled."""
    import threading

    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    http_client = _MockHTTPClient()
    routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
    config = WebhookLoopConfig(routes=routes, retry_attempts=1)
    loop: WebhookLoop[_Request, _Output] = WebhookLoop(
        http_client=http_client,
        requests=requests,
        config=config,
    )

    # Add a message
    main_request = MainLoopRequest(request=_Request(message="msg"))
    main_result: MainLoopResult[_Output] = MainLoopResult(
        request_id=main_request.request_id,
        output=_Output(result="success"),
    )
    delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
        request=main_request,
        result=main_result,
        route_key="test",
    )
    requests.send(delivery, reply_to=results)

    stopped = []

    def run_loop() -> None:
        loop.run(max_iterations=None, wait_time_seconds=1)
        stopped.append(True)

    thread = threading.Thread(target=run_loop)
    thread.start()

    # Wait briefly for loop to start
    import time

    time.sleep(0.1)

    # Trigger shutdown
    loop.shutdown(timeout=2.0)

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert len(stopped) == 1

    requests.close()
    results.close()


def test_loop_handle_failure_without_dlq() -> None:
    """WebhookLoop handles failures without DLQ configured."""
    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        # HTTP client that raises an exception that causes _handle_failure to be called
        http_client = _MockHTTPClient()

        # Make the post method raise a non-HTTPClientError exception
        def raising_post(
            url: str,
            *,
            json: Mapping[str, object],
            headers: Mapping[str, str],
            timeout: float,
        ) -> HTTPResponse:
            raise RuntimeError("Unexpected error")

        http_client.post = raising_post  # type: ignore[method-assign]

        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)
        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
            dlq=None,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Should have received an error response
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is False
        assert "Unexpected error" in (msgs[0].body.error or "")
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_handle_failure_with_dlq_retry() -> None:
    """WebhookLoop retries with backoff when DLQ says don't dead-letter yet."""
    from weakincentives.runtime.dlq import DLQPolicy

    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()

        def raising_post(
            url: str,
            *,
            json: Mapping[str, object],
            headers: Mapping[str, str],
            timeout: float,
        ) -> HTTPResponse:
            raise RuntimeError("Unexpected error")

        http_client.post = raising_post  # type: ignore[method-assign]

        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)

        # DLQ that always says don't dead-letter (retry)
        from weakincentives.runtime.dlq import DeadLetter

        dlq_mailbox: InMemoryMailbox[
            DeadLetter[WebhookDelivery[_Request, _Output]], None
        ] = InMemoryMailbox(name="dlq")
        dlq = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=10,  # High count so it always retries
        )

        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
            dlq=dlq,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # No error response (retry with backoff)
        assert results.approximate_count() == 0
        # Message should still be in queue for retry
        assert requests.approximate_count() == 1

        dlq_mailbox.close()
    finally:
        requests.close()
        results.close()


def test_loop_handle_failure_with_dlq_dead_letter() -> None:
    """WebhookLoop dead-letters when DLQ policy says so."""
    from weakincentives.runtime.dlq import DeadLetter, DLQPolicy

    results: InMemoryMailbox[WebhookLoopResult, None] = InMemoryMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()

        def raising_post(
            url: str,
            *,
            json: Mapping[str, object],
            headers: Mapping[str, str],
            timeout: float,
        ) -> HTTPResponse:
            raise RuntimeError("Unexpected error")

        http_client.post = raising_post  # type: ignore[method-assign]

        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)

        # DLQ that always says dead-letter immediately
        dlq_mailbox: InMemoryMailbox[
            DeadLetter[WebhookDelivery[_Request, _Output]], None
        ] = InMemoryMailbox(name="dlq")
        dlq = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=1,  # Immediately dead-letter
        )

        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
            dlq=dlq,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        requests.send(delivery, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Should have received an error response (dead-lettered)
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is False
        assert "Dead-lettered" in (msgs[0].body.error or "")
        msgs[0].acknowledge()

        # Message should be in DLQ
        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert len(dlq_msgs) == 1
        assert dlq_msgs[0].body.last_error == "Unexpected error"
        dlq_msgs[0].acknowledge()

        # Original request should be acknowledged
        assert requests.approximate_count() == 0

        dlq_mailbox.close()
    finally:
        requests.close()
        results.close()


def test_loop_dead_letter_without_reply_to() -> None:
    """WebhookLoop dead-letters correctly when message has no reply_to."""
    from weakincentives.runtime.dlq import DeadLetter, DLQPolicy

    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()

        def raising_post(
            url: str,
            *,
            json: Mapping[str, object],
            headers: Mapping[str, str],
            timeout: float,
        ) -> HTTPResponse:
            raise RuntimeError("Unexpected error")

        http_client.post = raising_post  # type: ignore[method-assign]

        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)

        dlq_mailbox: InMemoryMailbox[
            DeadLetter[WebhookDelivery[_Request, _Output]], None
        ] = InMemoryMailbox(name="dlq")
        dlq = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=1,
        )

        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
            dlq=dlq,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        # Send without reply_to
        requests.send(delivery)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Message should be in DLQ
        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert len(dlq_msgs) == 1
        assert dlq_msgs[0].body.reply_to is None
        dlq_msgs[0].acknowledge()

        dlq_mailbox.close()
    finally:
        requests.close()


def test_loop_dead_letter_reply_failure() -> None:
    """WebhookLoop continues dead-lettering even if reply fails."""
    from weakincentives.runtime.dlq import DeadLetter, DLQPolicy

    results: FakeMailbox[WebhookLoopResult, None] = FakeMailbox(name="results")
    requests: InMemoryMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        InMemoryMailbox(name="requests")
    )
    try:
        http_client = _MockHTTPClient()

        def raising_post(
            url: str,
            *,
            json: Mapping[str, object],
            headers: Mapping[str, str],
            timeout: float,
        ) -> HTTPResponse:
            raise RuntimeError("Unexpected error")

        http_client.post = raising_post  # type: ignore[method-assign]

        routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
        config = WebhookLoopConfig(routes=routes, retry_attempts=1)

        dlq_mailbox: InMemoryMailbox[
            DeadLetter[WebhookDelivery[_Request, _Output]], None
        ] = InMemoryMailbox(name="dlq")
        dlq = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=1,
        )

        loop: WebhookLoop[_Request, _Output] = WebhookLoop(
            http_client=http_client,
            requests=requests,
            config=config,
            dlq=dlq,
        )

        main_request = MainLoopRequest(request=_Request(message="hello"))
        main_result: MainLoopResult[_Output] = MainLoopResult(
            request_id=main_request.request_id,
            output=_Output(result="success"),
        )
        delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
            request=main_request,
            result=main_result,
            route_key="test",
        )
        requests.send(delivery, reply_to=results)

        # Make reply fail
        results.set_connection_error(MailboxConnectionError("connection lost"))

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Message should still be in DLQ despite reply failure
        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert len(dlq_msgs) == 1
        dlq_msgs[0].acknowledge()

        dlq_mailbox.close()
    finally:
        requests.close()


def test_loop_handles_expired_receipt_handle_on_ack() -> None:
    """WebhookLoop handles expired receipt handle during ack."""
    results: FakeMailbox[WebhookLoopResult, None] = FakeMailbox(name="results")
    requests: FakeMailbox[WebhookDelivery[_Request, _Output], WebhookLoopResult] = (
        FakeMailbox(name="requests")
    )

    http_client = _MockHTTPClient()
    routes = (WebhookRoute(pattern=r".*", endpoint="https://api.example.com"),)
    config = WebhookLoopConfig(routes=routes, retry_attempts=1)
    loop: WebhookLoop[_Request, _Output] = WebhookLoop(
        http_client=http_client,
        requests=requests,
        config=config,
    )

    main_request = MainLoopRequest(request=_Request(message="hello"))
    main_result: MainLoopResult[_Output] = MainLoopResult(
        request_id=main_request.request_id,
        output=_Output(result="success"),
    )
    delivery: WebhookDelivery[_Request, _Output] = WebhookDelivery(
        request=main_request,
        result=main_result,
        route_key="test",
    )
    requests.send(delivery, reply_to=results)

    # Receive and expire the handle
    msgs = requests.receive(max_messages=1)
    assert len(msgs) == 1
    msg = msgs[0]
    requests.expire_handle(msg.receipt_handle)

    # Create result and try to reply/ack
    result = WebhookLoopResult(
        delivery_id=delivery.delivery_id,
        request_id=delivery.request.request_id,
        success=True,
    )
    # This should not raise
    loop._reply_and_ack(msg, result)

    # Response should still be sent
    assert results.approximate_count() == 1

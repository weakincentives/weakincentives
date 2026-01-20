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

"""Shared retry-after parsing utilities for provider adapters.

This module provides utilities for extracting retry-after information from
provider errors. Both OpenAI and LiteLLM adapters use these functions to
handle rate limiting and other throttling scenarios.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any, cast


def coerce_retry_after(value: object) -> timedelta | None:
    """Coerce various retry-after value formats to timedelta.

    Args:
        value: A retry-after value that may be:
            - None (returns None)
            - timedelta (returned if positive)
            - int/float seconds (converted to timedelta if positive)
            - string digits (parsed as seconds)

    Returns:
        A timedelta if valid positive duration, None otherwise.
    """
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value if value > timedelta(0) else None
    if isinstance(value, (int, float)):
        seconds = float(value)
        return timedelta(seconds=seconds) if seconds > 0 else None
    if isinstance(value, str) and value.isdigit():
        seconds = float(value)
        return timedelta(seconds=seconds)
    return None


def retry_after_from_headers(headers: Mapping[str, Any] | None) -> timedelta | None:
    """Extract retry-after from HTTP headers.

    Checks for 'retry-after' header (case-insensitive).

    Args:
        headers: HTTP headers mapping, or None.

    Returns:
        Parsed retry-after duration, or None if not found/invalid.
    """
    if headers is None:
        return None
    # Normalize header keys to lowercase for case-insensitive lookup
    normalized = {str(key).lower(): val for key, val in headers.items()}
    value = normalized.get("retry-after")
    return coerce_retry_after(value)


def retry_after_from_error(error: object) -> timedelta | None:
    """Extract retry-after information from a provider error.

    Searches for retry-after in multiple locations commonly used by
    OpenAI, LiteLLM, and other providers:

    1. error.retry_after attribute
    2. error.headers['retry-after']
    3. error.response['headers']['retry-after']
    4. error.response['retry_after']
    5. error.response.headers['retry-after']

    Args:
        error: An exception object from a provider API.

    Returns:
        Parsed retry-after duration, or None if not found/invalid.
    """
    # Check direct retry_after attribute
    direct = coerce_retry_after(getattr(error, "retry_after", None))
    if direct is not None:
        return direct

    # Check headers attribute
    headers = getattr(error, "headers", None)
    retry_after = retry_after_from_headers(
        cast(Mapping[str, object], headers) if isinstance(headers, Mapping) else None
    )
    if retry_after is not None:
        return retry_after

    # Check response attribute (may be dict or object)
    response = cast(object | None, getattr(error, "response", None))
    if isinstance(response, Mapping):
        response_mapping = cast(Mapping[str, object], response)

        # Check response.headers
        retry_after = retry_after_from_headers(
            cast(Mapping[str, object], response_mapping.get("headers"))
            if isinstance(response_mapping.get("headers"), Mapping)
            else None
        )
        if retry_after is not None:
            return retry_after

        # Check response.retry_after
        retry_after = coerce_retry_after(response_mapping.get("retry_after"))
        if retry_after is not None:
            return retry_after

        response_headers_obj: object | None = response_mapping.get("headers")
    else:
        # Response is an object, try to get headers attribute
        response_headers_obj = (
            getattr(response, "headers", None) if response is not None else None
        )

    return retry_after_from_headers(
        cast(Mapping[str, object], response_headers_obj)
        if isinstance(response_headers_obj, Mapping)
        else None
    )


def extract_error_payload(error: object) -> dict[str, Any] | None:
    """Extract the error payload from a provider error for debugging.

    Checks common locations for error details:
    1. error.response (if mapping)
    2. error.json_body (OpenAI-specific)

    Args:
        error: An exception object from a provider API.

    Returns:
        A dictionary with error details, or None if not found.
    """
    # Check response attribute
    payload_candidate = getattr(error, "response", None)
    if isinstance(payload_candidate, Mapping):
        payload_mapping = cast(Mapping[object, Any], payload_candidate)
        return {str(key): value for key, value in payload_mapping.items()}

    # Check json_body (OpenAI-specific)
    payload_candidate = getattr(error, "json_body", None)
    if isinstance(payload_candidate, Mapping):
        payload_mapping = cast(Mapping[object, Any], payload_candidate)
        return {str(key): value for key, value in payload_mapping.items()}

    return None


__all__ = [
    "coerce_retry_after",
    "extract_error_payload",
    "retry_after_from_error",
    "retry_after_from_headers",
]

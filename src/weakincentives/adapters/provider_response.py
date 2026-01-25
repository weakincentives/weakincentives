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

"""Provider response processing utilities for adapters.

This module provides common utilities for processing responses from LLM provider
APIs (OpenAI, Anthropic, etc.). These functions handle response extraction,
payload normalization, and error handling in a provider-agnostic way.

Key utilities:
- `extract_payload`: Extract dict payloads from SDK response objects
- `first_choice`: Get the first completion choice from a response
- `call_provider_with_normalization`: Wrap API calls with consistent error handling
- `mapping_to_str_dict`: Safely convert mappings to string-keyed dicts
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from .core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PromptEvaluationError,
)
from .throttle import ThrottleError


def mapping_to_str_dict(mapping: Mapping[Any, Any]) -> dict[str, Any] | None:
    """Convert a mapping to a dict with string keys, returning None if any key is not a string.

    This utility validates that all keys in the mapping are strings before
    converting. Used for safely extracting provider payloads where string
    keys are required.

    Args:
        mapping: Any mapping to convert. Keys will be checked for string type.

    Returns:
        A new dict with string keys and the original values if all keys are
        strings, or None if any key is not a string.

    Example:
        >>> mapping_to_str_dict({"a": 1, "b": 2})
        {'a': 1, 'b': 2}
        >>> mapping_to_str_dict({1: "a", 2: "b"})  # Non-string keys
        None
    """
    if any(not isinstance(key, str) for key in mapping):
        return None
    return {cast(str, key): value for key, value in mapping.items()}


def extract_payload(response: object) -> dict[str, Any] | None:
    """Extract a dictionary payload from a provider SDK response object.

    Attempts to extract structured data from provider responses using common
    SDK patterns. First tries calling `model_dump()` (Pydantic-style), then
    falls back to treating the response as a mapping directly.

    This is useful for capturing raw provider response data for debugging,
    logging, or error reporting.

    Args:
        response: A provider SDK response object. May have a `model_dump()`
            method (Pydantic models) or be a mapping directly.

    Returns:
        A dict with string keys containing the response payload, or None if
        the payload cannot be extracted (no model_dump method, non-string
        keys, or extraction fails).

    Example:
        >>> # With a Pydantic-style response
        >>> payload = extract_payload(openai_response)
        >>> if payload:
        ...     print(payload.get("usage"))
    """
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            payload = model_dump()
        except Exception:  # pragma: no cover - defensive
            return None
        if isinstance(payload, Mapping):
            mapping_payload = mapping_to_str_dict(cast(Mapping[Any, Any], payload))
            if mapping_payload is not None:
                return mapping_payload
        return None
    if isinstance(response, Mapping):  # pragma: no cover - defensive
        mapping_payload = mapping_to_str_dict(cast(Mapping[Any, Any], response))
        if mapping_payload is not None:
            return mapping_payload
    return None


def first_choice(response: object, *, prompt_name: str) -> object:
    """Extract the first choice from a provider response.

    Most LLM provider APIs return responses with a `choices` array containing
    one or more completion options. This function extracts the first choice
    and raises a consistent error if no choices are available.

    Args:
        response: A provider SDK response object with a `choices` attribute
            that should be a sequence.
        prompt_name: Name of the prompt being evaluated, used for error
            context if extraction fails.

    Returns:
        The first element from the response's `choices` sequence.

    Raises:
        PromptEvaluationError: If the response has no `choices` attribute,
            `choices` is not a sequence, or `choices` is empty. The error
            includes the prompt name and indicates the response phase.

    Example:
        >>> choice = first_choice(openai_response, prompt_name="my_prompt")
        >>> message = choice.message
    """
    choices = getattr(response, "choices", None)
    if not isinstance(choices, Sequence):
        raise PromptEvaluationError(
            "Provider response did not include any choices.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
        )
    sequence_choices = cast(Sequence[object], choices)
    try:
        return sequence_choices[0]
    except IndexError as error:  # pragma: no cover - defensive
        raise PromptEvaluationError(
            "Provider response did not include any choices.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
        ) from error


def call_provider_with_normalization(
    call_provider: Callable[[], object],
    *,
    prompt_name: str,
    normalize_throttle: Callable[[Exception], ThrottleError | None],
    provider_payload: Callable[[Exception], dict[str, Any] | None],
    request_error_message: str,
) -> object:
    """Invoke a provider callable with standardized error handling.

    Wraps a provider API call to normalize exceptions into consistent error
    types. This ensures that adapter implementations handle errors uniformly
    regardless of the underlying provider SDK.

    Error handling priority:
    1. If `normalize_throttle` identifies a rate-limit error, raises ThrottleError
    2. Otherwise, wraps the exception in PromptEvaluationError with context

    Args:
        call_provider: Zero-argument callable that invokes the provider API.
            Should return the raw provider response object.
        prompt_name: Name of the prompt being evaluated, included in any
            error for debugging and logging.
        normalize_throttle: Function to detect rate-limiting errors. Receives
            the caught exception and returns a ThrottleError if the exception
            indicates throttling, or None otherwise.
        provider_payload: Function to extract diagnostic data from errors.
            Receives the caught exception and returns a dict of provider-specific
            error details, or None if unavailable.
        request_error_message: Human-readable message describing what failed,
            used in the PromptEvaluationError if the call fails.

    Returns:
        The provider response object from `call_provider()` on success.

    Raises:
        ThrottleError: If `normalize_throttle` identifies the error as a
            rate-limit condition. Callers can use this to implement retry logic.
        PromptEvaluationError: For all other provider errors, with phase set
            to REQUEST and including any available provider payload.

    Example:
        >>> response = call_provider_with_normalization(
        ...     lambda: client.chat.completions.create(**params),
        ...     prompt_name="my_prompt",
        ...     normalize_throttle=detect_openai_throttle,
        ...     provider_payload=extract_openai_error_payload,
        ...     request_error_message="Failed to call OpenAI API",
        ... )
    """
    try:
        return call_provider()
    except Exception as error:  # pragma: no cover - network/SDK failure
        throttle_error = normalize_throttle(error)
        if throttle_error is not None:
            raise throttle_error from error
        raise PromptEvaluationError(
            request_error_message,
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload=provider_payload(error),
        ) from error


__all__ = [
    "call_provider_with_normalization",
    "extract_payload",
    "first_choice",
    "mapping_to_str_dict",
]

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

"""Provider response processing utilities for adapters."""

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
    if any(not isinstance(key, str) for key in mapping):
        return None
    return {cast(str, key): value for key, value in mapping.items()}


def extract_payload(response: object) -> dict[str, Any] | None:
    """Return a provider payload from an SDK response when available."""

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
    """Return the first choice in a provider response or raise consistently."""

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


def call_provider_with_normalization(  # noqa: PLR0913
    call_provider: Callable[[], object],
    *,
    prompt_name: str,
    normalize_throttle: Callable[[Exception], ThrottleError | None],
    provider_payload: Callable[[Exception], dict[str, Any] | None],
    request_error_message: str,
    error_factory: Callable[
        [str, str, dict[str, Any] | None, Exception], PromptEvaluationError
    ]
    | None = None,
) -> object:
    """Invoke a provider callable and normalize errors into PromptEvaluationError.

    Args:
        call_provider: Callable that invokes the provider and returns a response.
        prompt_name: Name of the prompt being evaluated.
        normalize_throttle: Function to normalize throttle errors.
        provider_payload: Function to extract provider payload from the error.
        request_error_message: Default error message for request failures.
        error_factory: Optional factory function to create adapter-specific errors.
            Signature: (message, prompt_name, provider_payload, error) -> Exception.
            If not provided, raises generic PromptEvaluationError.
    """

    try:
        return call_provider()
    except Exception as error:  # pragma: no cover - network/SDK failure
        throttle_error = normalize_throttle(error)
        if throttle_error is not None:
            raise throttle_error from error
        payload = provider_payload(error)
        if error_factory is not None:
            raise error_factory(
                str(error) or request_error_message,
                prompt_name,
                payload,
                error,
            ) from error
        raise PromptEvaluationError(
            request_error_message,
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload=payload,
        ) from error


__all__ = [
    "call_provider_with_normalization",
    "extract_payload",
    "first_choice",
    "mapping_to_str_dict",
]

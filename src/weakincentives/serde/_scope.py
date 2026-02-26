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

"""Scoped field visibility for schema generation and parsing."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, get_args, get_origin

from ..dataclasses import FrozenDataclass


class SerdeScope(Enum):
    """Scope for schema generation and parsing operations.

    Controls which fields are visible during serialization/deserialization
    based on the context of the operation.
    """

    DEFAULT = "default"
    """Standard serde - all fields visible."""

    STRUCTURED_OUTPUT = "structured_output"
    """LLM structured output context - hidden fields excluded."""


@FrozenDataclass()
class HiddenInStructuredOutput:
    """Marker to exclude a field from schema/parsing in STRUCTURED_OUTPUT scope.

    Use with ``Annotated`` to mark fields that should not be included in
    JSON schemas sent to LLMs and should be skipped during parsing of LLM
    responses. These fields are typically populated during post-processing
    in the ``finalize()`` hook.

    Example::

        from dataclasses import dataclass
        from typing import Annotated
        from weakincentives.serde import HiddenInStructuredOutput

        @dataclass
        class AnalysisResult:
            summary: str  # LLM generates this
            confidence: float  # LLM generates this

            # Hidden from LLM - populated in finalize()
            processing_time_ms: Annotated[int, HiddenInStructuredOutput()] = 0
            model_version: Annotated[str, HiddenInStructuredOutput()] = ""

    Note:
        Hidden fields MUST have a default value or ``default_factory`` since
        the LLM cannot provide them.
    """


def is_hidden_in_scope(typ: object, scope: SerdeScope) -> bool:
    """Check if a type annotation is hidden in the given scope.

    Args:
        typ: The type annotation to check (may be ``Annotated[T, ...]``).
        scope: The current serialization/deserialization scope.

    Returns:
        True if the field should be hidden (excluded) in this scope.
    """
    if scope == SerdeScope.DEFAULT:
        return False

    # scope == SerdeScope.STRUCTURED_OUTPUT
    return _has_hidden_in_structured_output_marker(typ)


_MIN_ANNOTATED_ARGS = 2  # Annotated[T, ...] has at least base type + one annotation


def _has_hidden_in_structured_output_marker(typ: object) -> bool:
    """Check if type has HiddenInStructuredOutput in its Annotated metadata."""
    if get_origin(typ) is not Annotated:
        return False

    args = get_args(typ)
    if len(args) < _MIN_ANNOTATED_ARGS:
        return False  # pragma: no cover - malformed Annotated

    return any(isinstance(meta, HiddenInStructuredOutput) for meta in args[1:])


__all__ = [
    "HiddenInStructuredOutput",
    "SerdeScope",
    "is_hidden_in_scope",
]

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

"""Dataset loading utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from ..serde import parse
from ._types import Sample


def _coerce[T](value: object, target: type[T]) -> T:
    """Coerce JSON value to target type.

    Primitives (str, int, float, bool) pass through directly.
    Mappings are parsed as dataclasses via serde.parse.
    """
    if target in {str, int, float, bool}:
        if not isinstance(value, target):
            raise TypeError(f"expected {target.__name__}, got {type(value).__name__}")
        return value
    if isinstance(value, Mapping):
        return parse(target, cast(Mapping[str, object], value))
    raise TypeError(f"cannot coerce {type(value).__name__} to {target.__name__}")


def load_jsonl[I, E](
    path: Path,
    input_type: type[I],
    expected_type: type[E],
) -> tuple[Sample[I, E], ...]:
    """Load samples from JSONL file.

    Each line must be a JSON object with "id", "input", and "expected" keys.
    Primitives (str, int, float, bool) are used directly; mappings are
    deserialized into dataclasses via serde.parse.
    """
    samples: list[Sample[I, E]] = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            samples.append(
                Sample(
                    id=obj["id"],
                    input=_coerce(obj["input"], input_type),
                    expected=_coerce(obj["expected"], expected_type),
                )
            )
    return tuple(samples)


__all__ = ["load_jsonl"]

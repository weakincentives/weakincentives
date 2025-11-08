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

"""Backward compatible exports for dataclass serde helpers."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import get_args

from ._utils import (
    _SLOTTED_EXTRAS,
    _UNION_TYPE,
    MISSING_SENTINEL,
    _AnyType,
    _apply_constraints,
    _ExtrasDescriptor,
    _merge_annotated_meta,
    _ordered_values,
    _ParseConfig,
    _set_extras,
)
from .dump import clone, dump
from .parse import _bool_from_str, _coerce_to_type, parse
from .schema import schema

__all__ = [
    "MISSING_SENTINEL",
    "_SLOTTED_EXTRAS",
    "_UNION_TYPE",
    "_AnyType",
    "_ExtrasDescriptor",
    "_ParseConfig",
    "_apply_constraints",
    "_bool_from_str",
    "_coerce_to_type",
    "_merge_annotated_meta",
    "_ordered_values",
    "_set_extras",
    "clone",
    "dump",
    "get_args",
    "parse",
    "schema",
]

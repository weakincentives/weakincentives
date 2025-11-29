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

"""Backwards-compatible exports for dataclass serde helpers."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import get_args

from ._utils import (
    _SLOTTED_EXTRAS,
    MISSING_SENTINEL,
    UNION_TYPE,
    ExtrasDescriptor,
    ParseConfig,
    apply_constraints,
    merge_annotated_meta,
    ordered_values,
    set_extras,
)
from .dump import clone, dump
from .parse import _bool_from_str, _coerce_to_type, parse
from .schema import schema

# ruff: noqa: RUF022
__all__ = [
    "get_args",
    "_SLOTTED_EXTRAS",
    "_bool_from_str",
    "_coerce_to_type",
    "apply_constraints",
    "clone",
    "dump",
    "ExtrasDescriptor",
    "merge_annotated_meta",
    "MISSING_SENTINEL",
    "ordered_values",
    "parse",
    "ParseConfig",
    "schema",
    "set_extras",
    "UNION_TYPE",
]

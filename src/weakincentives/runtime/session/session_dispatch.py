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

"""Session dispatch helpers for slice operations.

This module provides helper functions for applying slice operations.
The core dispatch logic remains in Session to avoid private member access.
"""

from __future__ import annotations

from typing import assert_never

from ...types.dataclass import SupportsDataclass
from .slices import Append, Clear, Extend, Replace, Slice, SliceOp


def apply_slice_op[S: SupportsDataclass](
    op: SliceOp[S],
    slice_instance: Slice[S],
) -> None:
    """Apply slice operation using optimal method.

    Args:
        op: The slice operation to apply (Append, Extend, Replace, or Clear).
        slice_instance: The slice instance to mutate.
    """
    match op:
        case Append(item=item):
            slice_instance.append(item)
        case Extend(items=items):
            slice_instance.extend(items)
        case Replace(items=items):
            slice_instance.replace(items)
        case Clear(predicate=pred):
            slice_instance.clear(pred)
        case _ as unreachable:  # pragma: no cover - exhaustiveness sentinel
            assert_never(unreachable)  # pyright: ignore[reportUnreachable]


__all__ = [
    "apply_slice_op",
]

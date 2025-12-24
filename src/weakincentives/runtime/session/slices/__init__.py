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

"""Slice storage backends for session state."""

from ._config import SliceFactoryConfig, default_slice_config
from ._jsonl import JsonlSlice, JsonlSliceFactory, JsonlSliceView
from ._memory import MemorySlice, MemorySliceFactory, MemorySliceView
from ._ops import Append, Clear, Extend, Replace, SliceOp
from ._protocols import Slice, SliceFactory, SliceView

__all__ = [
    "Append",
    "Clear",
    "Extend",
    "JsonlSlice",
    "JsonlSliceFactory",
    "JsonlSliceView",
    "MemorySlice",
    "MemorySliceFactory",
    "MemorySliceView",
    "Replace",
    "Slice",
    "SliceFactory",
    "SliceFactoryConfig",
    "SliceOp",
    "SliceView",
    "default_slice_config",
]

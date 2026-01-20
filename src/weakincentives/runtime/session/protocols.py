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

"""Protocols describing Session behavior exposed to other modules.

This module re-exports protocols from _protocols for backward compatibility.
For new code, prefer importing directly from _protocols.
"""

from ._protocols import (
    ReadOnlySliceAccessorProtocol,
    ReducerContextProtocol,
    SessionProtocol,
    SessionViewProtocol,
    SliceAccessorProtocol,
    SnapshotProtocol,
    TypedReducerProtocol,
)

__all__ = [
    "ReadOnlySliceAccessorProtocol",
    "ReducerContextProtocol",
    "SessionProtocol",
    "SessionViewProtocol",
    "SliceAccessorProtocol",
    "SnapshotProtocol",
    "TypedReducerProtocol",
]

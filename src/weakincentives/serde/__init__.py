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

"""Stdlib dataclass serde utilities."""

from ._utils import TYPE_REF_KEY, resolve_type_identifier, type_identifier
from .dump import clone, dump
from .parse import parse
from .schema import schema

__all__ = [
    "TYPE_REF_KEY",
    "clone",
    "dump",
    "parse",
    "resolve_type_identifier",
    "schema",
    "type_identifier",
]

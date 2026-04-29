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

"""Provider integrations.

The base install ships :mod:`weakincentives.adapters.noop`, which is enough
to drive an :class:`weakincentives.runtime.AgentLoop` end-to-end in tests
and demos. Real provider adapters live in their own subpackages and arrive
behind their own pip extras as they are rebuilt:

- ``weakincentives.adapters.openai`` (``pip install weakincentives[openai]``)
- ``weakincentives.adapters.claude`` (``pip install weakincentives[claude]``)
- ``weakincentives.adapters.litellm`` (``pip install weakincentives[litellm]``)
- ``weakincentives.adapters.acp`` (``pip install weakincentives[acp]``)
- ``weakincentives.adapters.codex`` (``pip install weakincentives[codex]``)

Every adapter implements :class:`weakincentives.core.ProviderAdapter`.
"""

from __future__ import annotations

from .noop import NoopAdapter, ScriptedResponse

__all__ = [
    "NoopAdapter",
    "ScriptedResponse",
]

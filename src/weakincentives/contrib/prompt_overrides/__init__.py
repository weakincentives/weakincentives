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

"""Redis-backed prompt overrides store.

This module provides a Redis implementation of the ``PromptOverridesStore``
protocol, enabling distributed prompt override storage for multi-worker
deployments.

See ``specs/REDIS_PROMPT_OVERRIDES.md`` for the complete specification.

Example::

    from redis import Redis
    from weakincentives.contrib.prompt_overrides import RedisPromptOverridesStore

    client = Redis(host="localhost", port=6379)
    store = RedisPromptOverridesStore(client=client)

    # Use with Prompt
    prompt = Prompt(
        template,
        overrides_store=store,
        overrides_tag="stable",
    ).bind(params)
"""

from __future__ import annotations

from ._redis import (
    DEFAULT_TTL_SECONDS,
    RedisPromptOverridesStore,
    RedisPromptOverridesStoreFactory,
)

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "RedisPromptOverridesStore",
    "RedisPromptOverridesStoreFactory",
]

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

"""Contributed mailbox implementations.

This package contains mailbox implementations that require optional dependencies.

- ``RedisMailbox``: Redis-backed mailbox (requires ``redis`` package)

Example::

    from redis import Redis
    from weakincentives.contrib.mailbox import RedisMailbox

    client = Redis(host="localhost", port=6379)
    mailbox: RedisMailbox[MyEvent] = RedisMailbox(
        name="events",
        client=client,
    )
"""

from __future__ import annotations

from ._redis import RedisMailbox

__all__ = ["RedisMailbox"]

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

"""Lua scripts for atomic Redis mailbox operations.

These scripts ensure atomicity of multi-step operations in Redis.
Each script operates on keys with a common hash tag {queue:name} to
ensure cluster compatibility.

All scripts use Redis server TIME for visibility calculations to eliminate
client clock skew as a correctness factor.
"""

from __future__ import annotations

# Helper: compute server time as float seconds
LUA_NOW = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
"""

LUA_SEND = """
-- KEYS: [pending, invisible, data, meta]
-- ARGV: [msg_id, payload, enqueued_at, reply_to, max_size, ttl]
-- reply_to may be empty string for no reply
local max_size = tonumber(ARGV[5])
local ttl = tonumber(ARGV[6])
if max_size and max_size > 0 then
    local pending_n = redis.call('LLEN', KEYS[1])
    local invisible_n = redis.call('ZCARD', KEYS[2])
    if (pending_n + invisible_n) >= max_size then
        return 0
    end
end
redis.call('HSET', KEYS[3], ARGV[1], ARGV[2])
redis.call('HSET', KEYS[4], ARGV[1] .. ':enqueued', ARGV[3])
if ARGV[4] ~= '' then
    redis.call('HSET', KEYS[4], ARGV[1] .. ':reply_to', ARGV[4])
end
redis.call('LPUSH', KEYS[1], ARGV[1])
-- Apply TTL to all keys
if ttl and ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('EXPIRE', KEYS[3], ttl)
    redis.call('EXPIRE', KEYS[4], ttl)
end
return 1
"""

LUA_RECEIVE = """
-- KEYS: [pending, invisible, data, meta]
-- ARGV: [visibility_timeout_seconds, receipt_suffix, ttl]
local msg_id = redis.call('RPOP', KEYS[1])
if not msg_id then return nil end
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local expiry = now + tonumber(ARGV[1])
redis.call('ZADD', KEYS[2], expiry, msg_id)
redis.call('HSET', KEYS[4], msg_id .. ':handle', ARGV[2])
local data = redis.call('HGET', KEYS[3], msg_id)
local count = redis.call('HINCRBY', KEYS[4], msg_id .. ':count', 1)
local enqueued = redis.call('HGET', KEYS[4], msg_id .. ':enqueued')
local reply_to = redis.call('HGET', KEYS[4], msg_id .. ':reply_to')
-- Apply TTL to all keys
local ttl = tonumber(ARGV[3])
if ttl and ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('EXPIRE', KEYS[3], ttl)
    redis.call('EXPIRE', KEYS[4], ttl)
end
return {msg_id, data, count, enqueued, reply_to}
"""

LUA_ACKNOWLEDGE = """
-- KEYS: [invisible, data, meta]
-- ARGV: [msg_id, receipt_suffix, ttl]
local expected = redis.call('HGET', KEYS[3], ARGV[1] .. ':handle')
if expected ~= ARGV[2] then return 0 end
local removed = redis.call('ZREM', KEYS[1], ARGV[1])
if removed == 0 then return 0 end
redis.call('HDEL', KEYS[2], ARGV[1])
redis.call('HDEL', KEYS[3], ARGV[1] .. ':count')
redis.call('HDEL', KEYS[3], ARGV[1] .. ':enqueued')
redis.call('HDEL', KEYS[3], ARGV[1] .. ':handle')
redis.call('HDEL', KEYS[3], ARGV[1] .. ':reply_to')
-- Refresh TTL on remaining keys
local ttl = tonumber(ARGV[3])
if ttl and ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('EXPIRE', KEYS[3], ttl)
end
return 1
"""

LUA_NACK = """
-- KEYS: [invisible, pending, meta, data]
-- ARGV: [msg_id, receipt_suffix, visibility_timeout_seconds, ttl]
local expected = redis.call('HGET', KEYS[3], ARGV[1] .. ':handle')
if expected ~= ARGV[2] then return 0 end
local removed = redis.call('ZREM', KEYS[1], ARGV[1])
if removed == 0 then return 0 end
redis.call('HDEL', KEYS[3], ARGV[1] .. ':handle')
local timeout = tonumber(ARGV[3])
if timeout <= 0 then
    redis.call('LPUSH', KEYS[2], ARGV[1])
else
    local t = redis.call('TIME')
    local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
    redis.call('ZADD', KEYS[1], now + timeout, ARGV[1])
end
-- Refresh TTL on all keys including data
local ttl = tonumber(ARGV[4])
if ttl and ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('EXPIRE', KEYS[3], ttl)
    redis.call('EXPIRE', KEYS[4], ttl)
end
return 1
"""

LUA_EXTEND = """
-- KEYS: [invisible, meta, data]
-- ARGV: [msg_id, receipt_suffix, timeout_seconds, ttl]
local expected = redis.call('HGET', KEYS[2], ARGV[1] .. ':handle')
if expected ~= ARGV[2] then return 0 end
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local expiry = now + tonumber(ARGV[3])
redis.call('ZADD', KEYS[1], 'XX', expiry, ARGV[1])
local score = redis.call('ZSCORE', KEYS[1], ARGV[1])
if not score then return 0 end
-- Refresh TTL on all keys including data
local ttl = tonumber(ARGV[4])
if ttl and ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('EXPIRE', KEYS[3], ttl)
end
return 1
"""

LUA_REAP = """
-- KEYS: [invisible, pending, meta, data]
-- ARGV: [ttl] (computes now from server time)
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', now, 'LIMIT', 0, 100)
local count = 0
for i, msg_id in ipairs(expired) do
    redis.call('ZREM', KEYS[1], msg_id)
    redis.call('LPUSH', KEYS[2], msg_id)
    redis.call('HDEL', KEYS[3], msg_id .. ':handle')
    count = count + 1
end
-- Always refresh TTL to keep active queues alive even when no messages expire.
-- This prevents data loss for queues with long visibility timeouts.
local ttl = tonumber(ARGV[1])
if ttl and ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('EXPIRE', KEYS[3], ttl)
    redis.call('EXPIRE', KEYS[4], ttl)
end
return count
"""

LUA_PURGE = """
local pending_count = redis.call('LLEN', KEYS[1])
local invisible_count = redis.call('ZCARD', KEYS[2])
redis.call('DEL', KEYS[1], KEYS[2], KEYS[3], KEYS[4])
return pending_count + invisible_count
"""

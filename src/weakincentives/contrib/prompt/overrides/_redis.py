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

"""Redis-backed prompt overrides store implementation.

This module provides a distributed prompt overrides store using Redis.
It supports both standalone Redis and Redis Cluster deployments.

See ``specs/REDIS_PROMPT_OVERRIDES_STORE.md`` for the complete specification.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportInvalidTypeArguments=false, reportAttributeAccessIssue=false
# pyright: reportUnknownArgumentType=false, reportUnusedCallResult=false
# pyright: reportArgumentType=false, reportGeneralTypeIssues=false
# pyright: reportMissingImports=false, reportImplicitStringConcatenation=false

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

from weakincentives.prompt.overrides.validation import (
    FORMAT_VERSION,
    filter_override_for_descriptor,
    load_sections,
    load_tools,
    seed_sections,
    seed_tools,
    serialize_sections,
    serialize_tools,
    validate_sections_for_write,
    validate_tools_for_write,
)
from weakincentives.prompt.overrides.versioning import (
    HexDigest,
    PromptDescriptor,
    PromptLike,
    PromptOverride,
    PromptOverridesError,
    PromptOverridesStore,
    SectionOverride,
    descriptor_for_prompt,
)
from weakincentives.runtime.logging import StructuredLogger, get_logger
from weakincentives.types import JSONValue

if TYPE_CHECKING:
    from redis import Redis
    from redis.cluster import RedisCluster

_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "redis_prompt_overrides"}
)

# Identifier validation pattern (same as LocalPromptOverridesStore)
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

# Lua script for atomic get-or-create (used by seed)
_LUA_GET_OR_CREATE = """
local key = KEYS[1]
local exists = redis.call('EXISTS', key)
if exists == 1 then
    return redis.call('HGETALL', key)
end
-- Key doesn't exist, return nil to signal creation needed
return nil
"""

# Lua script for atomic read-modify-write (used by set_section_override)
_LUA_UPDATE_SECTION = """
local key = KEYS[1]
local path = ARGV[1]
local section_json = ARGV[2]
local exists = redis.call('EXISTS', key)
if exists == 0 then
    return nil
end
local sections_raw = redis.call('HGET', key, 'sections')
local tools_raw = redis.call('HGET', key, 'tools')
return {sections_raw or '{}', tools_raw or '{}'}
"""


class RedisPromptOverridesError(PromptOverridesError):
    """Raised when Redis operations fail."""


@dataclass(slots=True, frozen=True)
class OverrideMetadata:
    """Lightweight metadata for listing overrides."""

    ns: str
    prompt_key: str
    tag: str
    section_count: int
    tool_count: int


@dataclass(slots=True)
class RedisPromptOverridesStore(PromptOverridesStore):
    """Redis-backed prompt overrides store.

    Supports both standalone Redis and Redis Cluster deployments. Uses hash
    tags for cluster slot locality to ensure all operations for a namespace
    route to the same node.

    Data is stored as Redis HASHes with fields for version, namespace,
    prompt_key, tag, sections (JSON), and tools (JSON).

    Example::

        from redis import Redis
        from weakincentives.contrib.prompt.overrides import RedisPromptOverridesStore

        client = Redis(host="localhost", port=6379)
        store = RedisPromptOverridesStore(client=client)

        # Use with prompts
        prompt = Prompt(
            ns="demo",
            key="greeting",
            ...,
            overrides_store=store,
            overrides_tag="stable",
        )

        # Seed initial override
        override = store.seed(prompt, tag="stable")
    """

    client: Redis[bytes] | RedisCluster[bytes]
    """Redis client instance. Can be standalone Redis or RedisCluster."""

    key_prefix: str = "po"
    """Prefix for all Redis keys. Defaults to 'po' for prompt overrides."""

    _scripts: dict[str, object] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Register Lua scripts with Redis."""
        self._scripts["get_or_create"] = self.client.register_script(
            _LUA_GET_OR_CREATE
        )
        self._scripts["update_section"] = self.client.register_script(
            _LUA_UPDATE_SECTION
        )

    def _override_key(self, ns: str, prompt_key: str, tag: str) -> str:
        """Build Redis key with hash tag for cluster compatibility.

        Uses namespace as the hash tag to ensure all overrides for a namespace
        route to the same slot in Redis Cluster.
        """
        return f"{{{self.key_prefix}:{ns}}}:{prompt_key}:{tag}"

    def _validate_identifier(self, value: str, field_name: str) -> str:
        """Validate and normalize an identifier.

        Args:
            value: The identifier to validate.
            field_name: Name of the field for error messages.

        Returns:
            The normalized (lowercased) identifier.

        Raises:
            RedisPromptOverridesError: If the identifier is invalid.
        """
        normalized = value.lower()
        if not _IDENTIFIER_RE.match(normalized):
            msg = (
                f"Invalid {field_name}: {value!r}. "
                f"Must match pattern {_IDENTIFIER_RE.pattern}"
            )
            raise RedisPromptOverridesError(msg)
        return normalized

    def _validate_namespace(self, ns: str) -> str:
        """Validate namespace, checking each segment."""
        segments = ns.split("/")
        validated = [
            self._validate_identifier(segment, "namespace segment")
            for segment in segments
            if segment
        ]
        return "/".join(validated)

    @override
    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None:
        """Load and validate an override from Redis.

        Returns None if no override exists or all sections/tools are stale.

        Args:
            descriptor: Prompt descriptor containing hash metadata.
            tag: Override variant to resolve.

        Returns:
            The validated override, or None if not found or fully stale.

        Raises:
            RedisPromptOverridesError: On Redis connection errors.
        """
        validated_ns = self._validate_namespace(descriptor.ns)
        validated_key = self._validate_identifier(descriptor.key, "prompt_key")
        validated_tag = self._validate_identifier(tag, "tag")

        redis_key = self._override_key(validated_ns, validated_key, validated_tag)

        try:
            data = self.client.hgetall(redis_key)
        except Exception as e:
            raise RedisPromptOverridesError(
                f"Failed to retrieve override from Redis: {e}"
            ) from e

        if not data:
            _LOGGER.debug(
                "Override not found in Redis.",
                event="redis_override_missing",
                context={
                    "ns": descriptor.ns,
                    "prompt_key": descriptor.key,
                    "tag": validated_tag,
                },
            )
            return None

        # Parse stored data
        payload = self._parse_hash_data(data)
        self._validate_header(payload, descriptor, validated_tag)

        # Load and validate sections/tools
        sections_payload = payload.get("sections")
        sections = load_sections(sections_payload, descriptor)
        tools_payload = payload.get("tools")
        tools = load_tools(tools_payload, descriptor)

        raw_override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=sections,
            tool_overrides=tools,
        )

        # Filter stale overrides
        filtered_sections, filtered_tools = filter_override_for_descriptor(
            descriptor, raw_override
        )

        if not filtered_sections and not filtered_tools:
            _LOGGER.debug(
                "No applicable overrides remain after validation.",
                event="redis_override_empty",
                context={
                    "ns": descriptor.ns,
                    "prompt_key": descriptor.key,
                    "tag": validated_tag,
                },
            )
            return None

        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=filtered_sections,
            tool_overrides=filtered_tools,
        )
        _LOGGER.info(
            "Resolved prompt override from Redis.",
            event="redis_override_resolved",
            context={
                "ns": descriptor.ns,
                "prompt_key": descriptor.key,
                "tag": validated_tag,
                "section_count": len(filtered_sections),
                "tool_count": len(filtered_tools),
            },
        )
        return override

    @override
    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride:
        """Persist an override to Redis.

        Validates sections and tools against the descriptor before writing.

        Args:
            descriptor: Prompt descriptor for validation.
            override: The override to persist.

        Returns:
            The persisted override.

        Raises:
            RedisPromptOverridesError: On validation or Redis errors.
        """
        if override.ns != descriptor.ns or override.prompt_key != descriptor.key:
            raise RedisPromptOverridesError(
                "Override metadata does not match descriptor."
            )

        validated_ns = self._validate_namespace(descriptor.ns)
        validated_key = self._validate_identifier(descriptor.key, "prompt_key")
        validated_tag = self._validate_identifier(override.tag, "tag")

        # Validate sections and tools
        validated_sections = validate_sections_for_write(
            override.sections,
            descriptor,
        )
        validated_tools = validate_tools_for_write(
            override.tool_overrides,
            descriptor,
        )

        redis_key = self._override_key(validated_ns, validated_key, validated_tag)

        # Serialize sections and tools to JSON
        sections_json = json.dumps(serialize_sections(validated_sections))
        tools_json = json.dumps(serialize_tools(validated_tools))

        try:
            self.client.hset(
                redis_key,
                mapping={
                    "version": str(FORMAT_VERSION),
                    "ns": descriptor.ns,
                    "prompt_key": descriptor.key,
                    "tag": validated_tag,
                    "sections": sections_json,
                    "tools": tools_json,
                },
            )
        except Exception as e:
            raise RedisPromptOverridesError(
                f"Failed to persist override to Redis: {e}"
            ) from e

        persisted = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=validated_tag,
            sections=validated_sections,
            tool_overrides=validated_tools,
        )
        _LOGGER.info(
            "Persisted prompt override to Redis.",
            event="redis_override_persisted",
            context={
                "ns": descriptor.ns,
                "prompt_key": descriptor.key,
                "tag": validated_tag,
                "section_count": len(validated_sections),
                "tool_count": len(validated_tools),
            },
        )
        return persisted

    @override
    def delete(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> None:
        """Remove an override from Redis.

        Args:
            ns: Namespace of the override.
            prompt_key: Prompt key of the override.
            tag: Tag of the override.

        Raises:
            RedisPromptOverridesError: On Redis connection errors.
        """
        validated_ns = self._validate_namespace(ns)
        validated_key = self._validate_identifier(prompt_key, "prompt_key")
        validated_tag = self._validate_identifier(tag, "tag")

        redis_key = self._override_key(validated_ns, validated_key, validated_tag)

        try:
            deleted = self.client.delete(redis_key)
        except Exception as e:
            raise RedisPromptOverridesError(
                f"Failed to delete override from Redis: {e}"
            ) from e

        if deleted == 0:
            _LOGGER.debug(
                "No override to delete in Redis.",
                event="redis_override_delete_missing",
                context={
                    "ns": ns,
                    "prompt_key": prompt_key,
                    "tag": validated_tag,
                },
            )

    @override
    def set_section_override(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
        path: tuple[str, ...],
        body: str,
    ) -> PromptOverride:
        """Update a single section override.

        Reads the existing override, updates the section, and writes back
        atomically using optimistic locking.

        Args:
            prompt: The prompt to update.
            tag: Override variant to update.
            path: Section path tuple.
            body: New section body.

        Returns:
            The updated override.

        Raises:
            RedisPromptOverridesError: On Redis or validation errors.
        """
        descriptor = descriptor_for_prompt(prompt)
        validated_tag = self._validate_identifier(tag, "tag")

        existing_override = self.resolve(descriptor=descriptor, tag=validated_tag)
        sections = dict(existing_override.sections) if existing_override else {}
        tools = dict(existing_override.tool_overrides) if existing_override else {}

        expected_hash = self._lookup_section_hash(descriptor, path)
        sections[path] = SectionOverride(expected_hash=expected_hash, body=body)

        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=validated_tag,
            sections=sections,
            tool_overrides=tools,
        )
        return self.upsert(descriptor, override)

    @override
    def seed(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        """Bootstrap an override from the current prompt state.

        If an override already exists for this prompt/tag, returns the existing
        override without modification.

        Args:
            prompt: The prompt to seed from.
            tag: Override variant to create.

        Returns:
            The existing or newly created override.

        Raises:
            RedisPromptOverridesError: On Redis or validation errors.
        """
        descriptor = descriptor_for_prompt(prompt)
        validated_ns = self._validate_namespace(descriptor.ns)
        validated_key = self._validate_identifier(descriptor.key, "prompt_key")
        validated_tag = self._validate_identifier(tag, "tag")

        redis_key = self._override_key(validated_ns, validated_key, validated_tag)

        # Check if override already exists
        try:
            exists = self.client.exists(redis_key)
        except Exception as e:
            raise RedisPromptOverridesError(
                f"Failed to check override existence: {e}"
            ) from e

        if exists:
            existing = self.resolve(descriptor=descriptor, tag=validated_tag)
            if existing is None:
                raise RedisPromptOverridesError(
                    "Override key exists but could not be resolved."
                )
            return existing

        # Create new override from prompt
        sections = seed_sections(prompt, descriptor)
        tools = seed_tools(prompt, descriptor)

        seed_override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=validated_tag,
            sections=sections,
            tool_overrides=tools,
        )
        return self.upsert(descriptor, seed_override)

    def list_overrides(
        self,
        *,
        ns: str | None = None,
        prompt_key: str | None = None,
    ) -> list[OverrideMetadata]:
        """List stored overrides with optional filtering.

        Uses SCAN for iteration, which is safe for large keyspaces.

        Args:
            ns: Filter by namespace (optional).
            prompt_key: Filter by prompt key (optional, requires ns).

        Returns:
            List of override metadata.

        Raises:
            RedisPromptOverridesError: On Redis connection errors.
        """
        if prompt_key is not None and ns is None:
            raise RedisPromptOverridesError(
                "Cannot filter by prompt_key without specifying ns."
            )

        pattern = self._build_scan_pattern(ns, prompt_key)
        return self._scan_for_metadata(pattern)

    def _build_scan_pattern(
        self,
        ns: str | None,
        prompt_key: str | None,
    ) -> str:
        """Build SCAN pattern for listing overrides."""
        if ns is None:
            return f"{{{self.key_prefix}:*"

        validated_ns = self._validate_namespace(ns)
        if prompt_key is not None:
            validated_key = self._validate_identifier(prompt_key, "prompt_key")
            return f"{{{self.key_prefix}:{validated_ns}}}:{validated_key}:*"
        return f"{{{self.key_prefix}:{validated_ns}}}:*"

    def _scan_for_metadata(self, pattern: str) -> list[OverrideMetadata]:
        """Scan Redis keys and load metadata for matching overrides."""
        results: list[OverrideMetadata] = []
        try:
            cursor: int = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                for key in keys:
                    key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                    metadata = self._load_metadata(key_str)
                    if metadata is not None:
                        results.append(metadata)
                if cursor == 0:
                    break
        except Exception as e:
            raise RedisPromptOverridesError(
                f"Failed to list overrides from Redis: {e}"
            ) from e
        return results

    def close(self) -> None:
        """No-op. Does not close the Redis client (caller's responsibility)."""
        pass

    def _parse_hash_data(self, data: dict[bytes, bytes]) -> dict[str, JSONValue]:
        """Parse Redis HASH data into a dictionary."""
        payload: dict[str, JSONValue] = {}
        for key_bytes, value_bytes in data.items():
            key = key_bytes.decode("utf-8")
            value_str = value_bytes.decode("utf-8")

            if key in ("sections", "tools"):
                try:
                    payload[key] = json.loads(value_str)
                except json.JSONDecodeError:
                    payload[key] = {}
            elif key == "version":
                try:
                    payload[key] = int(value_str)
                except ValueError:
                    payload[key] = value_str
            else:
                payload[key] = value_str

        return payload

    def _validate_header(
        self,
        payload: dict[str, JSONValue],
        descriptor: PromptDescriptor,
        tag: str,
    ) -> None:
        """Validate override header matches descriptor."""
        version = payload.get("version")
        if version != FORMAT_VERSION:
            raise RedisPromptOverridesError(
                f"Unsupported override version {version!r}."
            )
        ns = payload.get("ns")
        prompt_key = payload.get("prompt_key")
        tag_value = payload.get("tag")
        if ns != descriptor.ns or prompt_key != descriptor.key or tag_value != tag:
            raise RedisPromptOverridesError(
                "Override metadata does not match descriptor inputs."
            )

    def _lookup_section_hash(
        self, descriptor: PromptDescriptor, path: tuple[str, ...]
    ) -> HexDigest:
        """Find the expected hash for a section path."""
        for candidate in descriptor.sections:
            if candidate.path == path:
                return candidate.content_hash
        raise RedisPromptOverridesError(
            f"Section {path!r} not registered in prompt descriptor; cannot override."
        )

    def _load_metadata(self, key: str) -> OverrideMetadata | None:
        """Load override metadata from a Redis key."""
        try:
            data = self.client.hmget(
                key, "ns", "prompt_key", "tag", "sections", "tools"
            )
        except Exception:
            return None

        if not data or data[0] is None:
            return None

        ns = data[0].decode("utf-8") if isinstance(data[0], bytes) else str(data[0])
        prompt_key = (
            data[1].decode("utf-8") if isinstance(data[1], bytes) else str(data[1])
        )
        tag = data[2].decode("utf-8") if isinstance(data[2], bytes) else str(data[2])

        section_count = 0
        tool_count = 0

        if data[3]:
            sections_str = (
                data[3].decode("utf-8") if isinstance(data[3], bytes) else str(data[3])
            )
            try:
                sections_data = json.loads(sections_str)
                section_count = len(sections_data) if isinstance(sections_data, dict) else 0
            except json.JSONDecodeError:
                pass

        if data[4]:
            tools_str = (
                data[4].decode("utf-8") if isinstance(data[4], bytes) else str(data[4])
            )
            try:
                tools_data = json.loads(tools_str)
                tool_count = len(tools_data) if isinstance(tools_data, dict) else 0
            except json.JSONDecodeError:
                pass

        return OverrideMetadata(
            ns=ns,
            prompt_key=prompt_key,
            tag=tag,
            section_count=section_count,
            tool_count=tool_count,
        )


__all__ = ["OverrideMetadata", "RedisPromptOverridesError", "RedisPromptOverridesStore"]

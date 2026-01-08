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

This module provides a Redis implementation of the ``PromptOverridesStore``
protocol for distributed prompt override storage.

See ``specs/REDIS_PROMPT_OVERRIDES.md`` for the complete specification.
"""

# Pyright suppressions for redis library type stub limitations:
# - Redis[bytes]/RedisCluster[bytes] type args not recognized by stubs
# - get()/set()/delete() and other methods have incomplete type annotations
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportInvalidTypeArguments=false, reportAttributeAccessIssue=false
# pyright: reportUnknownArgumentType=false, reportUnusedCallResult=false

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, override

from weakincentives.prompt.overrides.validation import (
    FORMAT_VERSION,
    filter_override_for_descriptor,
    load_sections,
    load_task_example_overrides,
    load_tools,
    seed_sections,
    seed_tools,
    serialize_sections,
    serialize_task_example_overrides,
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
    TaskExampleOverride,
    ToolOverride,
    descriptor_for_prompt,
)
from weakincentives.runtime.logging import StructuredLogger, get_logger
from weakincentives.types import JSONValue

if TYPE_CHECKING:
    from redis import Redis
    from redis.cluster import RedisCluster

# Default TTL for Redis keys: 30 days in seconds.
# Keys are refreshed on read operations, so active overrides stay alive.
# Set to 0 to disable TTL expiration.
DEFAULT_TTL_SECONDS: int = 2592000  # 30 days = 30 * 24 * 60 * 60

_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "redis_prompt_overrides"}
)

_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"


def _validate_identifier(value: str, label: str) -> str:
    """Validate and normalize an identifier string."""
    stripped = value.strip()
    if not stripped:
        raise PromptOverridesError(f"{label.capitalize()} must be a non-empty string.")
    if not re.fullmatch(_IDENTIFIER_PATTERN, stripped):
        raise PromptOverridesError(
            f"{label.capitalize()} must match pattern {_IDENTIFIER_PATTERN}."
        )
    return stripped


def _split_namespace(ns: str) -> tuple[str, ...]:
    """Split and validate a namespace string into segments."""
    stripped = ns.strip()
    if not stripped:
        raise PromptOverridesError("Namespace must be a non-empty string.")
    segments = tuple(part.strip() for part in stripped.split("/") if part.strip())
    if not segments:
        raise PromptOverridesError("Namespace must contain at least one segment.")
    return tuple(
        _validate_identifier(segment, "namespace segment") for segment in segments
    )


def _lookup_section_hash(
    descriptor: PromptDescriptor, path: tuple[str, ...]
) -> HexDigest:
    """Look up the expected hash for a section path."""
    for candidate in descriptor.sections:
        if candidate.path == path:
            return candidate.content_hash
    raise PromptOverridesError(
        f"Section {path!r} not registered in prompt descriptor; cannot override."
    )


def _lookup_tool_hash(descriptor: PromptDescriptor, name: str) -> HexDigest:
    """Look up the expected hash for a tool name."""
    for candidate in descriptor.tools:
        if candidate.name == name:
            return candidate.contract_hash
    raise PromptOverridesError(
        f"Tool {name!r} not registered in prompt descriptor; cannot override."
    )


def _merge_section_override(
    descriptor: PromptDescriptor,
    sections: dict[tuple[str, ...], SectionOverride],
    override: SectionOverride,
) -> None:
    """Validate and merge a section override into the sections dict."""
    expected_hash = _lookup_section_hash(descriptor, override.path)
    if override.expected_hash != expected_hash:
        msg = (
            f"Hash mismatch for section {override.path!r}: expected "
            f"{expected_hash}, got {override.expected_hash}."
        )
        raise PromptOverridesError(msg)
    sections[override.path] = override


def _merge_tool_override(
    descriptor: PromptDescriptor,
    tools: dict[str, ToolOverride],
    override: ToolOverride,
) -> None:
    """Validate and merge a tool override into the tools dict."""
    expected_hash = _lookup_tool_hash(descriptor, override.name)
    if override.expected_contract_hash != expected_hash:
        msg = (
            f"Hash mismatch for tool {override.name!r}: expected "
            f"{expected_hash}, got {override.expected_contract_hash}."
        )
        raise PromptOverridesError(msg)
    tools[override.name] = override


def _merge_task_example_override(
    task_examples: list[TaskExampleOverride],
    override: TaskExampleOverride,
) -> None:
    """Merge a task example override into the task_examples list."""
    for i, existing in enumerate(task_examples):
        if existing.path == override.path and existing.index == override.index:
            task_examples[i] = override
            return
    task_examples.append(override)


@dataclass(slots=True)
class RedisPromptOverridesStore(PromptOverridesStore):
    """Redis-backed prompt overrides store.

    Implements the ``PromptOverridesStore`` protocol using Redis for storage.
    Supports both standalone Redis and Redis Cluster deployments.

    Keys use hash tags for cluster slot locality::

        {prompt:<ns>:<key>}:<tag>

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

    client: Redis[bytes] | RedisCluster[bytes]
    """Redis client instance. Can be standalone Redis or RedisCluster."""

    default_ttl: int = DEFAULT_TTL_SECONDS
    """Default TTL in seconds for override keys (default: 30 days).
    Set to 0 to disable TTL. Keys are refreshed on read operations."""

    key_prefix: str = "prompt"
    """Prefix for all Redis keys (default: "prompt")."""

    def _get_with_ttl_refresh(self, key: str) -> bytes | None:
        """Get value and refresh TTL if value exists."""
        data = cast(bytes | None, self.client.get(key))
        if data is not None and self.default_ttl > 0:
            self.client.expire(key, self.default_ttl)
        return data

    def _set_with_ttl(self, key: str, value: str) -> None:
        """Set value with TTL."""
        if self.default_ttl > 0:
            self.client.setex(key, self.default_ttl, value)
        else:
            self.client.set(key, value)

    def _make_key(self, ns: str, prompt_key: str, tag: str) -> str:
        """Build Redis key with hash tag for cluster compatibility.

        Format: {prefix:<ns>:<key>}:<tag>

        The hash tag {prefix:<ns>:<key>} ensures all tags for the same
        prompt are co-located on the same Redis node.
        """
        # Validate and normalize inputs
        ns_segments = _split_namespace(ns)
        validated_key = _validate_identifier(prompt_key, "prompt key")
        validated_tag = _validate_identifier(tag, "tag")

        # Build namespaced key with hash tag
        ns_path = "/".join(ns_segments)
        return f"{{{self.key_prefix}:{ns_path}:{validated_key}}}:{validated_tag}"

    @staticmethod
    def _serialize_override(
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> str:
        """Serialize a PromptOverride to JSON string."""
        payload = {
            "version": FORMAT_VERSION,
            "ns": descriptor.ns,
            "prompt_key": descriptor.key,
            "tag": override.tag,
            "sections": serialize_sections(override.sections),
            "tools": serialize_tools(override.tool_overrides),
            "task_example_overrides": serialize_task_example_overrides(
                override.task_example_overrides
            ),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _deserialize_override(
        data: bytes | str,
        descriptor: PromptDescriptor,
        tag: str,
    ) -> PromptOverride | None:
        """Deserialize and validate a PromptOverride from JSON."""
        try:
            json_str = data.decode("utf-8") if isinstance(data, bytes) else data
            payload = cast(dict[str, JSONValue], json.loads(json_str))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise PromptOverridesError(
                f"Failed to parse prompt override JSON: {error}"
            ) from error

        # Validate header
        version = payload.get("version")
        if version != FORMAT_VERSION:
            raise PromptOverridesError(
                f"Unsupported override format version {version!r}."
            )
        ns = payload.get("ns")
        prompt_key = payload.get("prompt_key")
        tag_value = payload.get("tag")
        if ns != descriptor.ns or prompt_key != descriptor.key or tag_value != tag:
            raise PromptOverridesError(
                "Override metadata does not match descriptor inputs."
            )

        # Load and filter overrides
        sections = load_sections(payload.get("sections"), descriptor)
        tools = load_tools(payload.get("tools"), descriptor)
        task_example_overrides = load_task_example_overrides(
            payload.get("task_example_overrides")
        )

        raw_override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=sections,
            tool_overrides=tools,
            task_example_overrides=task_example_overrides,
        )

        filtered_sections, filtered_tools = filter_override_for_descriptor(
            descriptor, raw_override
        )

        if not filtered_sections and not filtered_tools and not task_example_overrides:
            _LOGGER.debug(
                "No applicable overrides remain after validation.",
                event="prompt_override_empty",
                context={
                    "ns": descriptor.ns,
                    "prompt_key": descriptor.key,
                    "tag": tag,
                },
            )
            return None

        return PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=filtered_sections,
            tool_overrides=filtered_tools,
            task_example_overrides=task_example_overrides,
        )

    @override
    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None:
        """Load override from Redis, filtering stale entries.

        Returns None if key doesn't exist or all entries are stale.
        Refreshes TTL on successful read.
        """
        key = self._make_key(descriptor.ns, descriptor.key, tag)

        try:
            result = self._get_with_ttl_refresh(key)
        except Exception as error:  # pragma: no cover
            raise PromptOverridesError(
                f"Failed to read override from Redis: {error}"
            ) from error

        if result is None:
            _LOGGER.debug(
                "Override not found in Redis.",
                event="prompt_override_missing",
                context={
                    "ns": descriptor.ns,
                    "prompt_key": descriptor.key,
                    "tag": tag,
                },
            )
            return None

        override = self._deserialize_override(result, descriptor, tag)
        if override is not None:
            _LOGGER.info(
                "Resolved prompt override from Redis.",
                event="prompt_override_resolved",
                context={
                    "ns": descriptor.ns,
                    "prompt_key": descriptor.key,
                    "tag": tag,
                    "section_count": len(override.sections),
                    "tool_count": len(override.tool_overrides),
                    "task_example_count": len(override.task_example_overrides),
                },
            )
        return override

    @override
    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride:
        """Persist override to Redis with validation.

        Validates all hashes match descriptor before writing.
        Raises PromptOverridesError on hash mismatch.
        """
        if override.ns != descriptor.ns or override.prompt_key != descriptor.key:
            raise PromptOverridesError(
                "Override metadata does not match descriptor.",
            )

        # Validate sections and tools (strict mode)
        validated_sections = validate_sections_for_write(
            override.sections,
            descriptor,
        )
        validated_tools = validate_tools_for_write(
            override.tool_overrides,
            descriptor,
        )

        validated_override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=override.tag,
            sections=validated_sections,
            tool_overrides=validated_tools,
            task_example_overrides=override.task_example_overrides,
        )

        key = self._make_key(descriptor.ns, descriptor.key, override.tag)
        payload = self._serialize_override(descriptor, validated_override)

        try:
            self._set_with_ttl(key, payload)
        except Exception as error:  # pragma: no cover
            raise PromptOverridesError(
                f"Failed to write override to Redis: {error}"
            ) from error

        _LOGGER.info(
            "Persisted prompt override to Redis.",
            event="prompt_override_persisted",
            context={
                "ns": descriptor.ns,
                "prompt_key": descriptor.key,
                "tag": override.tag,
                "section_count": len(validated_sections),
                "tool_count": len(validated_tools),
                "task_example_count": len(override.task_example_overrides),
            },
        )
        return validated_override

    @override
    def delete(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> None:
        """Remove override key from Redis."""
        key = self._make_key(ns, prompt_key, tag)

        try:
            result = self.client.delete(key)
        except Exception as error:  # pragma: no cover
            raise PromptOverridesError(
                f"Failed to delete override from Redis: {error}"
            ) from error

        if result == 0:
            _LOGGER.debug(
                "No override to delete.",
                event="prompt_override_delete_missing",
                context={
                    "ns": ns,
                    "prompt_key": prompt_key,
                    "tag": tag,
                },
            )

    @override
    def store(
        self,
        descriptor: PromptDescriptor,
        override: SectionOverride | ToolOverride | TaskExampleOverride,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        """Store single override using read-modify-write.

        Uses optimistic concurrency (last-write-wins).
        """
        normalized_tag = _validate_identifier(tag, "tag")
        key = self._make_key(descriptor.ns, descriptor.key, normalized_tag)

        # Read existing override (if any)
        try:
            existing_data = self._get_with_ttl_refresh(key)
        except Exception as error:  # pragma: no cover
            raise PromptOverridesError(
                f"Failed to read override from Redis: {error}"
            ) from error

        existing_override: PromptOverride | None = None
        if existing_data is not None:
            existing_override = self._deserialize_override(
                existing_data, descriptor, normalized_tag
            )

        # Merge new override
        sections = dict(existing_override.sections) if existing_override else {}
        tools = dict(existing_override.tool_overrides) if existing_override else {}
        task_examples = (
            list(existing_override.task_example_overrides) if existing_override else []
        )

        if isinstance(override, SectionOverride):
            _merge_section_override(descriptor, sections, override)
        elif isinstance(override, ToolOverride):
            _merge_tool_override(descriptor, tools, override)
        else:
            _merge_task_example_override(task_examples, override)

        prompt_override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=normalized_tag,
            sections=sections,
            tool_overrides=tools,
            task_example_overrides=tuple(task_examples),
        )
        return self.upsert(descriptor, prompt_override)

    @override
    def seed(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        """Bootstrap override from current prompt state.

        Does not overwrite existing overrides. If an override already exists,
        it is resolved and returned instead of creating a new one.
        """
        descriptor = descriptor_for_prompt(prompt)
        normalized_tag = _validate_identifier(tag, "tag")
        key = self._make_key(descriptor.ns, descriptor.key, normalized_tag)

        # Check if override already exists
        try:
            existing_data = self._get_with_ttl_refresh(key)
        except Exception as error:  # pragma: no cover
            raise PromptOverridesError(
                f"Failed to read override from Redis: {error}"
            ) from error

        if existing_data is not None:
            existing = self._deserialize_override(
                existing_data, descriptor, normalized_tag
            )
            if existing is not None:
                return existing
            raise PromptOverridesError("Override exists but could not be resolved.")

        # Create new seeded override
        sections = seed_sections(prompt, descriptor)
        tools = seed_tools(prompt, descriptor)

        seed_override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=normalized_tag,
            sections=sections,
            tool_overrides=tools,
            task_example_overrides=(),
        )
        return self.upsert(descriptor, seed_override)


class RedisPromptOverridesStoreFactory:
    """Factory for creating RedisPromptOverridesStore instances.

    Useful for dependency injection and testing.

    Example::

        factory = RedisPromptOverridesStoreFactory(client=redis_client)
        store = factory.create()
    """

    __slots__ = ("client", "default_ttl", "key_prefix")

    client: Redis[bytes] | RedisCluster[bytes]
    """Redis client to use for created stores."""

    default_ttl: int
    """Default TTL in seconds for override keys."""

    key_prefix: str
    """Prefix for all Redis keys."""

    def __init__(
        self,
        client: Redis[bytes] | RedisCluster[bytes],
        *,
        default_ttl: int = DEFAULT_TTL_SECONDS,
        key_prefix: str = "prompt",
    ) -> None:
        """Initialize factory with shared Redis client.

        Args:
            client: Redis client to use for created stores.
            default_ttl: Default TTL in seconds (default: 30 days).
            key_prefix: Prefix for all Redis keys (default: "prompt").
        """
        super().__init__()
        self.client = client
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix

    def create(self) -> RedisPromptOverridesStore:
        """Create a new RedisPromptOverridesStore instance.

        Returns:
            A new store instance with the configured settings.
        """
        return RedisPromptOverridesStore(
            client=self.client,
            default_ttl=self.default_ttl,
            key_prefix=self.key_prefix,
        )


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "RedisPromptOverridesStore",
    "RedisPromptOverridesStoreFactory",
]

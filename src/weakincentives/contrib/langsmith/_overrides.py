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

"""LangSmith Hub-backed prompt overrides store."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, cast

from ...prompt.overrides.versioning import (
    PromptDescriptor,
    PromptOverride,
    PromptOverridesError,
    SectionOverride,
    descriptor_for_prompt,
)
from ...runtime.logging import StructuredLogger, get_logger
from ._config import LangSmithConfig

if TYPE_CHECKING:
    from ...prompt.overrides.versioning import PromptLike

# Length of SHA-256 hex digest
_SHA256_HEX_LEN = 64


class FallbackStoreProtocol(Protocol):
    """Protocol for fallback override stores."""

    def resolve(
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None: ...

    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride: ...


logger: StructuredLogger = get_logger(
    __name__, context={"component": "langsmith_overrides"}
)


class LangSmithHubClientProtocol(Protocol):
    """Protocol for LangSmith Hub client operations.

    This protocol matches the langsmith SDK's Client API.
    """

    def pull_prompt(
        self,
        prompt_identifier: str,
        *,
        include_model: bool = False,
    ) -> object: ...

    def push_prompt(  # noqa: PLR0913
        self,
        prompt_identifier: str,
        *,
        object: object,  # noqa: A002
        parent_commit_hash: str | None = None,
        is_public: bool = False,
        description: str | None = None,
        readme: str | None = None,
        tags: list[str] | None = None,
    ) -> str: ...


@dataclass(slots=True)
class CacheEntry:
    """Cached override entry with TTL tracking."""

    override: PromptOverride | None
    timestamp: float
    tag: str
    is_versioned: bool


class LangSmithPromptOverridesStore:
    """Fetch and persist prompt overrides via LangSmith Hub.

    This store implements the ``PromptOverridesStore`` protocol and provides
    bidirectional prompt management with LangSmith Hub.

    Features:

    - **Caching**: TTL-based cache with configurable expiration
    - **Tag-aware**: Versioned tags (commit hashes) cache indefinitely
    - **Fail-open**: Network errors fall back to cache or skip overrides
    - **Fallback store**: Optional local store for offline scenarios

    Example::

        config = LangSmithConfig(hub_enabled=True)
        store = LangSmithPromptOverridesStore(config)

        # Resolve overrides from Hub
        override = store.resolve(descriptor, tag="production")

        # Push changes to Hub
        commit_hash = store.push(prompt, tag="staging")
    """

    def __init__(
        self,
        config: LangSmithConfig,
        *,
        client: LangSmithHubClientProtocol | None = None,
        fallback_store: FallbackStoreProtocol | None = None,
    ) -> None:
        """Initialize the Hub-backed store.

        Args:
            config: LangSmith configuration settings.
            client: Optional Hub client for testing.
            fallback_store: Optional local store for fallback resolution.
        """
        self._config = config
        self._client = client
        self._fallback_store = fallback_store
        self._cache: dict[tuple[str, str, str], CacheEntry] = {}

    def _get_client(self) -> LangSmithHubClientProtocol:
        """Get or create the LangSmith Hub client."""
        if self._client is not None:
            return self._client

        try:
            from langsmith import Client  # type: ignore[import-not-found]

            self._client = Client(
                api_key=self._config.resolved_api_key(),
                api_url=self._config.resolved_api_url(),
            )
        except ImportError as error:
            msg = "langsmith package is required for Hub integration"
            raise ImportError(msg) from error

        return self._client

    @staticmethod
    def _cache_key(ns: str, prompt_key: str, tag: str) -> tuple[str, str, str]:
        """Generate cache key for override."""
        return (ns, prompt_key, tag)

    @staticmethod
    def _is_versioned_tag(tag: str) -> bool:
        """Check if tag is a versioned commit hash."""
        # Commit hashes are 64-char hex strings
        if len(tag) != _SHA256_HEX_LEN:
            return False
        try:
            int(tag, 16)
        except ValueError:
            return False
        return True

    def _is_cache_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid."""
        if entry.is_versioned and self._config.cache_versioned_indefinitely:
            return True
        elapsed = time.time() - entry.timestamp
        return elapsed < self._config.cache_ttl_seconds

    def _get_cached(self, ns: str, prompt_key: str, tag: str) -> PromptOverride | None:
        """Get override from cache if valid."""
        key = self._cache_key(ns, prompt_key, tag)
        entry = self._cache.get(key)
        if entry is not None and self._is_cache_valid(entry):
            return entry.override
        return None

    def _set_cached(
        self,
        ns: str,
        prompt_key: str,
        tag: str,
        override: PromptOverride | None,
    ) -> None:
        """Store override in cache."""
        key = self._cache_key(ns, prompt_key, tag)
        self._cache[key] = CacheEntry(
            override=override,
            timestamp=time.time(),
            tag=tag,
            is_versioned=self._is_versioned_tag(tag),
        )

    def _invalidate_cache(self, ns: str, prompt_key: str, tag: str) -> None:
        """Invalidate cache entry."""
        key = self._cache_key(ns, prompt_key, tag)
        self._cache.pop(key, None)

    @staticmethod
    def _hub_prompt_name(ns: str, prompt_key: str) -> str:
        """Generate Hub prompt name from WINK identifiers."""
        # Replace / with - for valid Hub names
        return f"{ns}-{prompt_key}".replace("/", "-")

    def resolve(  # noqa: PLR0911
        self,
        descriptor: PromptDescriptor,
        tag: str = "latest",
    ) -> PromptOverride | None:
        """Fetch override from Hub, with caching.

        Args:
            descriptor: The prompt descriptor to resolve overrides for.
            tag: Version tag (``"latest"``, commit hash, or alias).

        Returns:
            The resolved ``PromptOverride`` or ``None`` if not found.
        """
        if not self._config.hub_enabled:
            if self._fallback_store is not None:
                return self._fallback_store.resolve(descriptor, tag)
            return None

        ns = descriptor.ns
        prompt_key = descriptor.key

        # Check cache first (skip for "latest" to ensure freshness)
        if tag != "latest":
            cached = self._get_cached(ns, prompt_key, tag)
            if cached is not None:
                return cached

        try:
            override = self._fetch_from_hub(descriptor, tag)
            self._set_cached(ns, prompt_key, tag, override)
        except Exception as error:
            logger.warning(
                "Failed to fetch from LangSmith Hub",
                event="langsmith_hub_fetch_failed",
                context={
                    "ns": ns,
                    "prompt_key": prompt_key,
                    "tag": tag,
                    "error": str(error),
                },
            )
            # Try cache even if expired
            cached = self._get_cached(ns, prompt_key, tag)
            if cached is not None:
                return cached
            # Fall back to local store
            if self._fallback_store is not None:
                return self._fallback_store.resolve(descriptor, tag)
            return None
        else:
            return override

    def _fetch_from_hub(
        self,
        descriptor: PromptDescriptor,
        tag: str,
    ) -> PromptOverride | None:
        """Fetch override from LangSmith Hub."""
        client = self._get_client()
        prompt_name = self._hub_prompt_name(descriptor.ns, descriptor.key)

        # Build prompt identifier with tag
        identifier = prompt_name if tag == "latest" else f"{prompt_name}:{tag}"

        try:
            hub_prompt = client.pull_prompt(identifier)
        except Exception as error:
            # Check if it's a 404 (not found) - return None
            error_str = str(error).lower()
            if "not found" in error_str or "404" in error_str:
                return None
            raise

        return self._hub_prompt_to_override(hub_prompt, descriptor, tag)

    @staticmethod
    def _hub_prompt_to_override(
        hub_prompt: object,
        descriptor: PromptDescriptor,
        tag: str,
    ) -> PromptOverride | None:
        """Convert LangSmith Hub prompt to WINK override format."""
        if hub_prompt is None:
            return None

        # Extract template content from Hub prompt
        # Hub prompts can be in various formats - try common patterns
        template_content: str | None = None
        # Cast to Any for dynamic attribute access on unknown hub prompt types
        prompt: Any = cast(Any, hub_prompt)

        if hasattr(prompt, "template"):
            template_content = str(prompt.template)
        elif hasattr(prompt, "messages"):
            # Chat prompt template
            messages = prompt.messages
            if messages:
                parts = [
                    str(msg.content) for msg in messages if hasattr(msg, "content")
                ]
                template_content = "\n".join(parts)
        elif isinstance(hub_prompt, str):
            template_content = hub_prompt
        elif isinstance(hub_prompt, dict):
            hub_dict = cast("dict[str, object]", hub_prompt)
            raw_template = hub_dict.get("template") or hub_dict.get("content")
            if raw_template is not None:
                template_content = str(raw_template)

        if not template_content:
            return None

        # Build section overrides
        sections: dict[tuple[str, ...], SectionOverride] = {}
        for section_desc in descriptor.sections:
            # Use the hub template as override for the first section
            # In a real implementation, you'd parse the hub prompt structure
            sections[section_desc.path] = SectionOverride(
                expected_hash=section_desc.content_hash,
                body=template_content,
            )
            # Only override first section for simple prompts
            break

        return PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
            sections=sections,
            tool_overrides={},
        )

    def upsert(
        self,
        descriptor: PromptDescriptor,
        override: PromptOverride,
    ) -> PromptOverride:
        """Persist override to Hub.

        Args:
            descriptor: The prompt descriptor.
            override: The override to persist.

        Returns:
            The persisted override (may have updated tag/commit hash).
        """
        if not self._config.hub_enabled:
            if self._fallback_store is not None:
                return self._fallback_store.upsert(descriptor, override)
            msg = "Hub is disabled and no fallback store configured"
            raise PromptOverridesError(msg)

        client = self._get_client()
        prompt_name = self._hub_prompt_name(descriptor.ns, descriptor.key)

        # Convert override to Hub format
        hub_prompt = self._override_to_hub_prompt(override, descriptor)

        try:
            commit_hash = client.push_prompt(
                prompt_name,
                object=hub_prompt,
            )
        except Exception as error:
            logger.error(  # noqa: TRY400 - StructuredLogger, re-raising
                "Failed to push to LangSmith Hub",
                event="langsmith_hub_push_failed",
                context={
                    "ns": descriptor.ns,
                    "prompt_key": descriptor.key,
                    "error": str(error),
                },
            )
            raise PromptOverridesError(f"Failed to push to Hub: {error}") from error

        # Invalidate cache
        self._invalidate_cache(descriptor.ns, descriptor.key, override.tag)

        # Return updated override with commit hash
        return replace(override, tag=commit_hash)

    @staticmethod
    def _override_to_hub_prompt(
        override: PromptOverride,
        descriptor: PromptDescriptor,
    ) -> dict[str, object]:
        """Convert WINK override to LangSmith Hub prompt format."""
        del descriptor  # Unused but kept for API consistency
        # Combine section overrides into template
        parts = [
            section_override.body for section_override in override.sections.values()
        ]

        template = "\n\n".join(parts) if parts else ""

        # Return as dict for Hub API
        return {
            "template": template,
            "metadata": {
                "wink_ns": override.ns,
                "wink_prompt_key": override.prompt_key,
                "wink_tag": override.tag,
            },
        }

    def delete(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str,
    ) -> None:
        """Remove override entry.

        Note: LangSmith Hub doesn't support deletion, so this only
        invalidates the local cache.
        """
        self._invalidate_cache(ns, prompt_key, tag)

    def set_section_override(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
        path: tuple[str, ...],
        body: str,
    ) -> PromptOverride:
        """Set single section override.

        Args:
            prompt: The prompt to override.
            tag: Version tag.
            path: Section path to override.
            body: New section body.

        Returns:
            The updated override.
        """
        descriptor = descriptor_for_prompt(prompt)

        # Find matching section
        matching_section = None
        for section_desc in descriptor.sections:
            if section_desc.path == path:
                matching_section = section_desc
                break

        if matching_section is None:
            msg = f"Section not found: {path}"
            raise PromptOverridesError(msg)

        # Get or create override
        existing = self.resolve(descriptor, tag)
        if existing is None:
            existing = PromptOverride(
                ns=descriptor.ns,
                prompt_key=descriptor.key,
                tag=tag,
            )

        # Update section
        new_sections = dict(existing.sections)
        new_sections[path] = SectionOverride(
            expected_hash=matching_section.content_hash,
            body=body,
        )

        updated = replace(existing, sections=new_sections)
        return self.upsert(descriptor, updated)

    def seed(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
    ) -> PromptOverride:
        """Initialize empty override for prompt.

        Args:
            prompt: The prompt to seed.
            tag: Version tag.

        Returns:
            The seeded override.
        """
        descriptor = descriptor_for_prompt(prompt)
        override = PromptOverride(
            ns=descriptor.ns,
            prompt_key=descriptor.key,
            tag=tag,
        )
        return self.upsert(descriptor, override)

    def pull(
        self,
        *,
        ns: str,
        prompt_key: str,
        tag: str = "latest",
    ) -> PromptOverride | None:
        """Pull prompt from Hub without descriptor (for initial sync).

        Args:
            ns: Prompt namespace.
            prompt_key: Prompt key.
            tag: Version tag.

        Returns:
            The pulled override or ``None`` if not found.
        """
        # Create minimal descriptor for lookup
        descriptor = PromptDescriptor(
            ns=ns,
            key=prompt_key,
            sections=[],
            tools=[],
        )
        return self.resolve(descriptor, tag)

    def push(
        self,
        prompt: PromptLike,
        *,
        tag: str = "latest",
        commit_message: str | None = None,
    ) -> str:
        """Push current prompt to Hub, returning commit hash.

        Args:
            prompt: The prompt to push.
            tag: Version tag.
            commit_message: Optional commit message.

        Returns:
            The commit hash of the pushed prompt.
        """
        if not self._config.hub_enabled:
            msg = "Hub is disabled"
            raise PromptOverridesError(msg)

        descriptor = descriptor_for_prompt(prompt)
        client = self._get_client()
        prompt_name = self._hub_prompt_name(descriptor.ns, descriptor.key)

        # Build Hub prompt from live prompt
        hub_prompt = self._prompt_to_hub_prompt(prompt, descriptor)

        try:
            commit_hash = client.push_prompt(
                prompt_name,
                object=hub_prompt,
                description=commit_message,
            )
        except Exception as error:
            logger.error(  # noqa: TRY400 - StructuredLogger, re-raising
                "Failed to push to LangSmith Hub",
                event="langsmith_hub_push_failed",
                context={
                    "ns": descriptor.ns,
                    "prompt_key": descriptor.key,
                    "error": str(error),
                },
            )
            raise PromptOverridesError(f"Failed to push to Hub: {error}") from error

        return commit_hash

    @staticmethod
    def _prompt_to_hub_prompt(
        prompt: PromptLike,
        descriptor: PromptDescriptor,
    ) -> dict[str, object]:
        """Convert WINK prompt to LangSmith Hub format."""
        del descriptor  # Unused but kept for API consistency
        # Collect section templates
        parts: list[str] = []
        for node in prompt.sections:
            template = node.section.original_body_template()
            if template is not None:
                parts.append(template)

        template = "\n\n".join(parts) if parts else ""

        return {
            "template": template,
            "metadata": {
                "wink_ns": prompt.ns,
                "wink_key": prompt.key,
            },
        }


__all__ = ["LangSmithPromptOverridesStore"]

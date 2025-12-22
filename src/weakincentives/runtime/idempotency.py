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

"""Idempotency ledger for exactly-once tool execution semantics.

This module provides an optional runtime layer that enables tools to:

1. Declare an idempotency key strategy (how to compute a unique key for each call)
2. Record tool results in a persistent ledger
3. Reuse cached results on retry, guarding external side effects

Usage::

    from weakincentives.runtime import EffectLedger, IdempotencyConfig
    from weakincentives.prompt import Tool

    # Tool with automatic idempotency (key derived from name + params hash)
    tool = Tool[Params, Result](
        name="my_tool",
        description="Does something",
        handler=my_handler,
        idempotency=IdempotencyConfig(strategy="auto"),
    )

    # Tool with custom key function
    tool = Tool[Params, Result](
        name="my_tool",
        description="Does something",
        handler=my_handler,
        idempotency=IdempotencyConfig(
            strategy="custom",
            key_fn=lambda params: f"order:{params.order_id}",
        ),
    )

    # Create ledger and check for cached results
    ledger = EffectLedger()
    effect = ledger.lookup(idempotency_key)
    if effect is not None:
        return effect.result  # Reuse cached result
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, Literal
from uuid import UUID, uuid4

from ..dataclasses import FrozenDataclass
from ..serde import dump
from ..types.dataclass import SupportsDataclass, is_dataclass_instance

if TYPE_CHECKING:
    from ..prompt.tool_result import ToolResult

_DEFAULT_TTL: Final = timedelta(hours=24)
_HASH_ALGORITHM: Final = "sha256"
_HASH_TRUNCATE_LENGTH: Final = 16


class IdempotencyStrategy(Enum):
    """Strategy for computing idempotency keys.

    AUTO: Key derived from tool name + hash of all serialized params (default)
    PARAMS: Key derived from tool name + hash of specific param fields
    CUSTOM: Key computed by a user-provided function
    NONE: No idempotency (always execute, never cache)
    """

    AUTO = "auto"
    PARAMS = "params"
    CUSTOM = "custom"
    NONE = "none"


@FrozenDataclass()
class IdempotencyConfig:
    """Configuration for tool idempotency behavior.

    Attributes:
        strategy: How to compute the idempotency key.
        param_keys: For PARAMS strategy, which param field names to include in the key.
        key_fn: For CUSTOM strategy, a function that computes the key from params.
        ttl: How long to cache results. None means indefinite.
        scope: Namespace prefix for keys (e.g., "session", "global").
    """

    strategy: IdempotencyStrategy | Literal["auto", "params", "custom", "none"] = (
        IdempotencyStrategy.AUTO
    )
    param_keys: tuple[str, ...] = ()
    key_fn: Callable[[SupportsDataclass | None], str] | None = None
    ttl: timedelta | None = _DEFAULT_TTL
    scope: str = "session"

    @classmethod
    def __pre_init__(
        cls,
        *,
        strategy: IdempotencyStrategy
        | Literal["auto", "params", "custom", "none"] = IdempotencyStrategy.AUTO,
        param_keys: tuple[str, ...] = (),
        key_fn: Callable[[SupportsDataclass | None], str] | None = None,
        ttl: timedelta | None = _DEFAULT_TTL,
        scope: str = "session",
    ) -> Mapping[str, object]:
        """Normalize strategy from string to enum."""
        if isinstance(strategy, str):
            strategy = IdempotencyStrategy(strategy)
        return {
            "strategy": strategy,
            "param_keys": param_keys,
            "key_fn": key_fn,
            "ttl": ttl,
            "scope": scope,
        }

    def __post_init__(self) -> None:
        """Validate configuration."""
        strategy = (
            self.strategy
            if isinstance(self.strategy, IdempotencyStrategy)
            else IdempotencyStrategy(self.strategy)
        )

        if strategy == IdempotencyStrategy.PARAMS and not self.param_keys:
            raise ValueError(
                "IdempotencyConfig with strategy='params' requires param_keys"
            )

        if strategy == IdempotencyStrategy.CUSTOM and self.key_fn is None:
            raise ValueError("IdempotencyConfig with strategy='custom' requires key_fn")


@FrozenDataclass()
class ToolEffect:
    """Record of a tool execution stored in the effect ledger.

    Attributes:
        idempotency_key: The computed key for this execution.
        tool_name: Name of the tool that was executed.
        params_hash: Hash of the serialized parameters.
        result_message: The message from the ToolResult.
        result_value: The serialized value from the ToolResult.
        result_success: Whether the tool execution succeeded.
        created_at: When the effect was recorded.
        expires_at: When the effect expires (None = never).
        effect_id: Unique identifier for this effect.
    """

    idempotency_key: str
    tool_name: str
    params_hash: str
    result_message: str
    result_value: dict[str, Any] | list[Any] | None
    result_success: bool
    created_at: datetime
    expires_at: datetime | None = None
    effect_id: UUID = field(default_factory=uuid4)

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check if this effect has expired."""
        if self.expires_at is None:
            return False
        current = now or datetime.now(UTC)
        return current >= self.expires_at


def compute_params_hash(params: SupportsDataclass | None) -> str:
    """Compute a deterministic hash of tool parameters.

    Args:
        params: The tool parameters dataclass, or None.

    Returns:
        A truncated hex digest of the serialized parameters.
    """
    if params is None:
        content = b""
    elif is_dataclass_instance(params):
        try:
            serialized = dump(params, exclude_none=False)
            # Sort keys for deterministic serialization
            content = json.dumps(serialized, sort_keys=True).encode("utf-8")
        except (TypeError, ValueError):
            # Fallback to repr if serialization fails
            content = repr(params).encode("utf-8")
    else:
        content = repr(params).encode("utf-8")

    digest = hashlib.new(_HASH_ALGORITHM, content).hexdigest()
    return digest[:_HASH_TRUNCATE_LENGTH]


def compute_idempotency_key(
    *,
    tool_name: str,
    params: SupportsDataclass | None,
    config: IdempotencyConfig,
) -> str | None:
    """Compute the idempotency key for a tool call.

    Args:
        tool_name: Name of the tool being called.
        params: The tool parameters dataclass.
        config: The idempotency configuration.

    Returns:
        The computed idempotency key, or None if strategy is NONE.
    """
    strategy = (
        config.strategy
        if isinstance(config.strategy, IdempotencyStrategy)
        else IdempotencyStrategy(config.strategy)
    )

    if strategy == IdempotencyStrategy.NONE:
        return None

    scope_prefix = f"{config.scope}:" if config.scope else ""

    if strategy == IdempotencyStrategy.AUTO:
        params_hash = compute_params_hash(params)
        return f"{scope_prefix}{tool_name}:{params_hash}"

    if strategy == IdempotencyStrategy.PARAMS:
        if params is None:
            selected_params = {}
        elif is_dataclass_instance(params):
            selected_params = {
                key: getattr(params, key, None)
                for key in config.param_keys
                if hasattr(params, key)
            }
        else:
            selected_params = {}
        content = json.dumps(selected_params, sort_keys=True).encode("utf-8")
        params_hash = hashlib.new(_HASH_ALGORITHM, content).hexdigest()[
            :_HASH_TRUNCATE_LENGTH
        ]
        return f"{scope_prefix}{tool_name}:{params_hash}"

    if strategy == IdempotencyStrategy.CUSTOM:
        if config.key_fn is None:  # pragma: no cover - validated in IdempotencyConfig
            raise ValueError("Custom strategy requires key_fn")
        custom_key = config.key_fn(params)
        return f"{scope_prefix}{custom_key}"

    return None  # pragma: no cover - exhaustive match


@dataclass(slots=True)
class EffectLedger:
    """In-memory ledger for tracking tool execution effects.

    The ledger provides exactly-once semantics by:
    1. Recording tool results keyed by their idempotency key
    2. Returning cached results when the same key is seen again
    3. Automatically expiring stale entries based on TTL

    Thread-safety: This implementation is NOT thread-safe. For concurrent
    access, wrap in appropriate locking or use a thread-safe storage backend.

    Usage::

        ledger = EffectLedger()

        # Check for cached result
        effect = ledger.lookup("session:my_tool:abc123")
        if effect is not None:
            return effect.result_message  # Reuse cached

        # Execute tool and record
        result = execute_tool(...)
        ledger.record(
            idempotency_key="session:my_tool:abc123",
            tool_name="my_tool",
            params=params,
            result=result,
            ttl=timedelta(hours=1),
        )
    """

    _effects: dict[str, ToolEffect] = field(default_factory=lambda: {})

    def lookup(
        self,
        idempotency_key: str,
        *,
        now: datetime | None = None,
    ) -> ToolEffect | None:
        """Look up a cached effect by idempotency key.

        Args:
            idempotency_key: The key to look up.
            now: Current time for expiry check (defaults to UTC now).

        Returns:
            The cached ToolEffect if found and not expired, None otherwise.
        """
        effect = self._effects.get(idempotency_key)
        if effect is None:
            return None

        if effect.is_expired(now):
            # Remove expired entry
            del self._effects[idempotency_key]
            return None

        return effect

    def record(  # noqa: PLR0913
        self,
        *,
        idempotency_key: str,
        tool_name: str,
        params: SupportsDataclass | None,
        result: ToolResult[Any],
        ttl: timedelta | None = None,
        now: datetime | None = None,
    ) -> ToolEffect:
        """Record a tool execution in the ledger.

        Args:
            idempotency_key: The computed idempotency key.
            tool_name: Name of the tool.
            params: The tool parameters.
            result: The ToolResult from execution.
            ttl: Time-to-live for the entry (None = indefinite).
            now: Current time (defaults to UTC now).

        Returns:
            The recorded ToolEffect.
        """
        current = now or datetime.now(UTC)
        expires_at = current + ttl if ttl is not None else None

        # Serialize result value
        result_value: dict[str, Any] | list[Any] | None = None
        if result.value is not None:
            if is_dataclass_instance(result.value):
                try:
                    result_value = dump(result.value, exclude_none=False)
                except (TypeError, ValueError):
                    result_value = None
            elif isinstance(result.value, dict):
                raw_value: dict[str, Any] = result.value  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                result_value = raw_value
            elif isinstance(result.value, list):
                raw_value_list: list[Any] = result.value  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                result_value = raw_value_list

        effect = ToolEffect(
            idempotency_key=idempotency_key,
            tool_name=tool_name,
            params_hash=compute_params_hash(params),
            result_message=result.message,
            result_value=result_value,
            result_success=result.success,
            created_at=current,
            expires_at=expires_at,
        )

        self._effects[idempotency_key] = effect
        return effect

    def invalidate(self, idempotency_key: str) -> bool:
        """Remove an effect from the ledger.

        Args:
            idempotency_key: The key to invalidate.

        Returns:
            True if an entry was removed, False otherwise.
        """
        if idempotency_key in self._effects:
            del self._effects[idempotency_key]
            return True
        return False

    def invalidate_by_tool(self, tool_name: str) -> int:
        """Remove all effects for a specific tool.

        Args:
            tool_name: Name of the tool to invalidate.

        Returns:
            Number of entries removed.
        """
        keys_to_remove = [
            key
            for key, effect in self._effects.items()
            if effect.tool_name == tool_name
        ]
        for key in keys_to_remove:
            del self._effects[key]
        return len(keys_to_remove)

    def clear(self) -> int:
        """Remove all effects from the ledger.

        Returns:
            Number of entries removed.
        """
        count = len(self._effects)
        self._effects.clear()
        return count

    def prune_expired(self, now: datetime | None = None) -> int:
        """Remove all expired effects from the ledger.

        Args:
            now: Current time for expiry check (defaults to UTC now).

        Returns:
            Number of entries removed.
        """
        current = now or datetime.now(UTC)
        keys_to_remove = [
            key for key, effect in self._effects.items() if effect.is_expired(current)
        ]
        for key in keys_to_remove:
            del self._effects[key]
        return len(keys_to_remove)

    def __len__(self) -> int:
        """Return the number of effects in the ledger."""
        return len(self._effects)

    def __contains__(self, idempotency_key: str) -> bool:
        """Check if an idempotency key exists in the ledger."""
        return idempotency_key in self._effects


__all__ = [
    "EffectLedger",
    "IdempotencyConfig",
    "IdempotencyStrategy",
    "ToolEffect",
    "compute_idempotency_key",
    "compute_params_hash",
]

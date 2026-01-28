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

"""Thread-safe slice storage with policy-based factory selection.

SliceStore is a pure state container that manages typed slice storage
with support for different slice policies (STATE, LOG). It provides
thread-safe access to slices and handles slice creation using
configurable factories.
"""

from __future__ import annotations

from collections.abc import Iterator
from threading import RLock
from typing import Any, cast

from ...types.dataclass import SupportsDataclass
from ._slice_types import SessionSlice, SessionSliceType
from .slice_policy import SlicePolicy
from .slices import Slice, SliceFactoryConfig, default_slice_config


class SliceStore:
    """Thread-safe slice storage with policy-based factory selection.

    SliceStore manages typed slice instances, creating them on-demand using
    the appropriate factory based on slice policy. All operations are
    thread-safe via an internal lock.

    Example::

        config = SliceFactoryConfig(
            state_factory=MemorySliceFactory(),
            log_factory=JsonlSliceFactory(base_dir=Path("./logs")),
        )
        store = SliceStore(config)

        # Get or create a slice
        slice = store.get_or_create(MyDataClass)
        items = store.select_all(MyDataClass)

    """

    __slots__ = ("_config", "_lock", "_slice_policies", "_slices")

    def __init__(
        self,
        config: SliceFactoryConfig | None = None,
        *,
        initial_policies: dict[SessionSliceType, SlicePolicy] | None = None,
    ) -> None:
        """Initialize the slice store.

        Args:
            config: Factory configuration for creating slices. Uses default
                memory-backed factories if not provided.
            initial_policies: Optional initial slice policies to set.
        """
        super().__init__()
        self._config = config if config is not None else default_slice_config()
        self._slices: dict[SessionSliceType, Slice[Any]] = {}
        self._slice_policies: dict[SessionSliceType, SlicePolicy] = (
            dict(initial_policies) if initial_policies is not None else {}
        )
        self._lock = RLock()

    @property
    def config(self) -> SliceFactoryConfig:
        """Return the slice factory configuration."""
        return self._config

    def get_or_create[T: SupportsDataclass](self, slice_type: type[T]) -> Slice[T]:
        """Get existing slice or create one using the appropriate factory.

        Thread-safe operation that ensures exactly one slice exists per type.

        Args:
            slice_type: The dataclass type for the slice.

        Returns:
            The slice instance for the given type.
        """
        with self._lock:
            if slice_type not in self._slices:
                policy = self._slice_policies.get(slice_type, SlicePolicy.STATE)
                factory = self._config.factory_for_policy(policy)
                self._slices[slice_type] = factory.create(slice_type)
            return cast(Slice[T], self._slices[slice_type])

    def select_all[T: SupportsDataclass](self, slice_type: type[T]) -> tuple[T, ...]:
        """Return all items in the slice for the given type.

        Thread-safe operation that returns the current slice contents.

        Args:
            slice_type: The dataclass type to query.

        Returns:
            Tuple of all items in the slice.
        """
        with self._lock:
            slice_instance = self.get_or_create(slice_type)
            return slice_instance.all()

    def set_policy(self, slice_type: SessionSliceType, policy: SlicePolicy) -> None:
        """Set the policy for a slice type.

        Args:
            slice_type: The slice type to configure.
            policy: The policy to apply.
        """
        with self._lock:
            self._slice_policies[slice_type] = policy

    def get_policy(self, slice_type: SessionSliceType) -> SlicePolicy:
        """Get the policy for a slice type.

        Args:
            slice_type: The slice type to query.

        Returns:
            The policy for the type, defaulting to STATE.
        """
        with self._lock:
            return self._slice_policies.get(slice_type, SlicePolicy.STATE)

    def ensure_policy(
        self, slice_type: SessionSliceType, policy: SlicePolicy | None
    ) -> None:
        """Set policy if provided, or ensure default exists.

        Args:
            slice_type: The slice type to configure.
            policy: The policy to set, or None to ensure default.
        """
        with self._lock:
            if policy is not None:
                self._slice_policies[slice_type] = policy
            else:
                _ = self._slice_policies.setdefault(slice_type, SlicePolicy.STATE)

    def all_slice_types(self) -> set[SessionSliceType]:
        """Return all registered slice types.

        Returns:
            Set of all slice types with existing slices.
        """
        with self._lock:
            return set(self._slices)

    def iter_slices(self) -> Iterator[tuple[SessionSliceType, Slice[Any]]]:
        """Iterate over all slices.

        Thread-safe snapshot iteration. Creates a copy of the slice mapping
        to allow safe iteration outside the lock.

        Yields:
            Tuples of (slice_type, slice_instance).
        """
        with self._lock:
            items = list(self._slices.items())
        yield from items

    def snapshot_slices(self) -> dict[SessionSliceType, SessionSlice]:
        """Create a snapshot of all slice contents.

        Returns:
            Dict mapping slice types to their snapshot tuples.
        """
        with self._lock:
            return {
                slice_type: slice_instance.snapshot()
                for slice_type, slice_instance in self._slices.items()
            }

    def snapshot_policies(
        self, registered_types: set[SessionSliceType]
    ) -> dict[SessionSliceType, SlicePolicy]:
        """Create a snapshot of slice policies.

        Args:
            registered_types: Set of all registered slice types.

        Returns:
            Dict mapping slice types to their policies.
        """
        with self._lock:
            return {
                slice_type: self._slice_policies.get(slice_type, SlicePolicy.STATE)
                for slice_type in registered_types
            }

    def apply_policies(self, policies: dict[SessionSliceType, SlicePolicy]) -> None:
        """Apply a policy snapshot.

        Args:
            policies: Dict mapping slice types to policies.
        """
        with self._lock:
            self._slice_policies = dict(policies)

    def clear_all(self, slice_types: set[SessionSliceType]) -> None:
        """Clear all slices for the given types.

        Args:
            slice_types: Set of slice types to clear.
        """
        with self._lock:
            for slice_type in slice_types:
                slice_instance = self.get_or_create(slice_type)
                slice_instance.clear()


__all__ = ["SliceStore"]

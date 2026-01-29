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

"""Slice storage with policy-based factory selection.

SliceStore is a pure state container that manages typed slice storage
with support for different slice policies (STATE, LOG). It handles slice
creation using configurable factories.

Note: SliceStore is NOT thread-safe on its own. Thread safety is provided
by Session's lock. All access to SliceStore must be made while holding
Session's lock.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

from ...types.dataclass import SupportsDataclass
from ._slice_types import SessionSlice, SessionSliceType
from .slice_policy import SlicePolicy
from .slices import Slice, SliceFactoryConfig, default_slice_config


class SliceStore:
    """Slice storage with policy-based factory selection.

    SliceStore manages typed slice instances, creating them on-demand using
    the appropriate factory based on slice policy.

    Note: This class is NOT thread-safe. Callers must hold Session's lock
    before calling any methods.

    Example::

        config = SliceFactoryConfig(
            state_factory=MemorySliceFactory(),
            log_factory=JsonlSliceFactory(base_dir=Path("./logs")),
        )
        store = SliceStore(config)

        # Get or create a slice (caller holds lock)
        slice = store.get_or_create(MyDataClass)
        items = store.select_all(MyDataClass)

    """

    __slots__ = ("_config", "_slice_policies", "_slices")

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

    @property
    def config(self) -> SliceFactoryConfig:
        """Return the slice factory configuration."""
        return self._config

    def get_or_create[T: SupportsDataclass](self, slice_type: type[T]) -> Slice[T]:
        """Get existing slice or create one using the appropriate factory.

        Ensures exactly one slice exists per type. Caller must hold lock.

        Args:
            slice_type: The dataclass type for the slice.

        Returns:
            The slice instance for the given type.
        """
        if slice_type not in self._slices:
            policy = self._slice_policies.get(slice_type, SlicePolicy.STATE)
            factory = self._config.factory_for_policy(policy)
            self._slices[slice_type] = factory.create(slice_type)
        return cast(Slice[T], self._slices[slice_type])

    def select_all[T: SupportsDataclass](self, slice_type: type[T]) -> tuple[T, ...]:
        """Return all items in the slice for the given type.

        Returns the current slice contents. Caller must hold lock.

        Args:
            slice_type: The dataclass type to query.

        Returns:
            Tuple of all items in the slice.
        """
        slice_instance = self.get_or_create(slice_type)
        return slice_instance.all()

    def set_policy(self, slice_type: SessionSliceType, policy: SlicePolicy) -> None:
        """Set the policy for a slice type. Caller must hold lock.

        Args:
            slice_type: The slice type to configure.
            policy: The policy to apply.
        """
        self._slice_policies[slice_type] = policy

    def get_policy(self, slice_type: SessionSliceType) -> SlicePolicy:
        """Get the policy for a slice type. Caller must hold lock.

        Args:
            slice_type: The slice type to query.

        Returns:
            The policy for the type, defaulting to STATE.
        """
        return self._slice_policies.get(slice_type, SlicePolicy.STATE)

    def ensure_policy(
        self, slice_type: SessionSliceType, policy: SlicePolicy | None
    ) -> None:
        """Set policy if provided, or ensure default exists. Caller must hold lock.

        Args:
            slice_type: The slice type to configure.
            policy: The policy to set, or None to ensure default.
        """
        if policy is not None:
            self._slice_policies[slice_type] = policy
        else:
            _ = self._slice_policies.setdefault(slice_type, SlicePolicy.STATE)

    def all_slice_types(self) -> set[SessionSliceType]:
        """Return all registered slice types. Caller must hold lock.

        Returns:
            Set of all slice types with existing slices.
        """
        return set(self._slices)

    def iter_slices(self) -> Iterator[tuple[SessionSliceType, Slice[Any]]]:
        """Iterate over all slices. Caller must hold lock.

        Yields:
            Tuples of (slice_type, slice_instance).
        """
        yield from self._slices.items()

    def snapshot_slices(self) -> dict[SessionSliceType, SessionSlice]:
        """Create a snapshot of all slice contents. Caller must hold lock.

        Returns:
            Dict mapping slice types to their snapshot tuples.
        """
        return {
            slice_type: slice_instance.snapshot()
            for slice_type, slice_instance in self._slices.items()
        }

    def snapshot_policies(
        self, registered_types: set[SessionSliceType]
    ) -> dict[SessionSliceType, SlicePolicy]:
        """Create a snapshot of slice policies. Caller must hold lock.

        Args:
            registered_types: Set of all registered slice types.

        Returns:
            Dict mapping slice types to their policies.
        """
        return {
            slice_type: self._slice_policies.get(slice_type, SlicePolicy.STATE)
            for slice_type in registered_types
        }

    def apply_policies(self, policies: dict[SessionSliceType, SlicePolicy]) -> None:
        """Apply a policy snapshot. Caller must hold lock.

        Args:
            policies: Dict mapping slice types to policies.
        """
        self._slice_policies = dict(policies)

    def clear_all(self, slice_types: set[SessionSliceType]) -> None:
        """Clear all slices for the given types. Caller must hold lock.

        Args:
            slice_types: Set of slice types to clear.
        """
        for slice_type in slice_types:
            slice_instance = self.get_or_create(slice_type)
            slice_instance.clear()


__all__ = ["SliceStore"]

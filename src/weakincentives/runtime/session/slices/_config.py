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

"""Slice factory configuration for session state backends."""

from __future__ import annotations

from ....dataclasses import FrozenDataclass
from ..slice_policy import SlicePolicy
from ._memory import MemorySliceFactory
from ._protocols import SliceFactory


@FrozenDataclass()
class SliceFactoryConfig:
    """Configuration mapping slice policies to factories.

    Allows different storage backends for STATE vs LOG slices:
    - STATE: Working state rolled back on failure (default: memory)
    - LOG: Append-only records preserved during restore (default: memory)

    Example::

        config = SliceFactoryConfig(
            state_factory=MemorySliceFactory(),
            log_factory=JsonlSliceFactory(base_dir=Path("./logs")),
        )
        session = Session(slice_config=config)

    """

    state_factory: SliceFactory
    log_factory: SliceFactory

    def factory_for_policy(self, policy: SlicePolicy) -> SliceFactory:
        """Return the factory for the given policy.

        Args:
            policy: The slice policy to get a factory for.

        Returns:
            The factory configured for the given policy.
        """
        if policy == SlicePolicy.LOG:
            return self.log_factory
        return self.state_factory


def default_slice_config() -> SliceFactoryConfig:
    """Create the default slice configuration.

    Returns:
        A SliceFactoryConfig with in-memory factories for both policies.
    """
    memory_factory = MemorySliceFactory()
    return SliceFactoryConfig(
        state_factory=memory_factory,
        log_factory=memory_factory,
    )


__all__ = [
    "SliceFactoryConfig",
    "default_slice_config",
]

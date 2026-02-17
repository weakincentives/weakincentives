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

"""Experiment configuration for A/B testing and prompt optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field, replace
from typing import Annotated, TypeVar, overload

from .dataclasses import FrozenDataclass

T = TypeVar("T")


@FrozenDataclass()
class Experiment:
    """Named configuration variant for systematic evaluation.

    Bundles a prompt overrides tag with feature flags, enabling coordinated
    changes to prompt content and runtime behavior for A/B testing.

    Attributes:
        name: Unique identifier for this experiment (e.g., "baseline",
            "v2-concise-prompts", "aggressive-tool-use").
        overrides_tag: Tag for prompt overrides resolution. Maps to files in
            ``.weakincentives/prompts/overrides/{ns}/{key}/{tag}.json``.
            Defaults to "latest" if not specified.
        flags: Feature flags controlling runtime behavior. Keys are flag names,
            values are flag settings. Agent implementations check these flags
            to conditionally enable features.
        owner: Optional owner identifier (e.g., email, username) for tracking
            who created or is responsible for this experiment.
        description: Optional human-readable description of what this experiment
            tests or changes.

    Example:
        >>> experiment = Experiment(
        ...     name="v2-concise-prompts",
        ...     overrides_tag="v2",
        ...     flags={"verbose_logging": True, "max_retries": 5},
        ...     owner="alice@example.com",
        ...     description="Test shorter, more direct prompt phrasing",
        ... )
        >>> experiment.get_flag("max_retries", 3)
        5
    """

    name: str
    overrides_tag: str = "latest"
    flags: Annotated[Mapping[str, object], {"untyped": True}] = field(
        default_factory=lambda: {}
    )
    owner: str | None = None
    description: str | None = None

    def with_flag(self, key: str, value: object) -> Experiment:
        """Return new experiment with flag added/updated.

        Args:
            key: The flag name.
            value: The flag value.

        Returns:
            New experiment with the flag set.

        Example:
            >>> exp = Experiment(name="test")
            >>> exp2 = exp.with_flag("debug", True)
            >>> exp2.get_flag("debug")
            True
        """
        return replace(self, flags={**self.flags, key: value})

    def with_tag(self, tag: str) -> Experiment:
        """Return new experiment with different overrides tag.

        Args:
            tag: The new overrides tag.

        Returns:
            New experiment with the specified tag.

        Example:
            >>> exp = Experiment(name="test")
            >>> exp2 = exp.with_tag("v2")
            >>> exp2.overrides_tag
            'v2'
        """
        return replace(self, overrides_tag=tag)

    @overload
    def get_flag(self, key: str) -> object: ...

    @overload
    def get_flag(self, key: str, default: T) -> T | object: ...

    def get_flag(self, key: str, default: T | None = None) -> T | object | None:
        """Get flag value with optional default.

        Args:
            key: The flag name to look up.
            default: Value to return if flag is not set. Defaults to None.

        Returns:
            The flag value if set, otherwise the default.

        Example:
            >>> exp = Experiment(name="test", flags={"retries": 5})
            >>> exp.get_flag("retries", 3)
            5
            >>> exp.get_flag("timeout", 30)
            30
        """
        return self.flags.get(key, default)

    def has_flag(self, key: str) -> bool:
        """Check if flag is set (any value including False/None).

        Args:
            key: The flag name to check.

        Returns:
            True if the flag exists, False otherwise.

        Example:
            >>> exp = Experiment(name="test", flags={"debug": False})
            >>> exp.has_flag("debug")
            True
            >>> exp.has_flag("verbose")
            False
        """
        return key in self.flags


# Sentinel experiments
BASELINE = Experiment(name="baseline", overrides_tag="latest")
"""Baseline experiment with no overrides or flags.

Use as the control group in A/B tests.
"""

CONTROL = Experiment(name="control", overrides_tag="latest")
"""Control experiment for A/B tests (explicit name for clarity).

Semantically identical to BASELINE but with a more explicit name.
"""


__all__ = [
    "BASELINE",
    "CONTROL",
    "Experiment",
]

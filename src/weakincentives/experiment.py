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

"""Experiment configuration for A/B testing and prompt optimization.

This module provides the `Experiment` class for defining named configuration
variants that bundle prompt overrides with feature flags. Use experiments to:

- Run A/B tests comparing different prompt phrasings or behaviors
- Systematically evaluate prompt changes with coordinated flag settings
- Track experiment ownership and purpose through metadata

Typical usage::

    from weakincentives.experiment import Experiment, BASELINE

    # Define a treatment variant
    treatment = Experiment(
        name="concise-v2",
        overrides_tag="v2",
        flags={"max_tokens": 500},
        description="Test shorter responses",
    )

    # Use BASELINE as the control group
    control = BASELINE

See Also:
    - Prompt overrides are resolved from ``.weakincentives/prompts/overrides/``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field, replace
from typing import TypeVar, overload

from .dataclasses import FrozenDataclass

T = TypeVar("T")


@FrozenDataclass()
class Experiment:
    """Named configuration variant for systematic evaluation.

    Bundles a prompt overrides tag with feature flags, enabling coordinated
    changes to prompt content and runtime behavior for A/B testing. Experiments
    are immutable; use ``with_flag()`` or ``with_tag()`` to create modified copies.

    Attributes:
        name: Unique identifier for this experiment (e.g., "baseline",
            "v2-concise-prompts", "aggressive-tool-use"). Should be descriptive
            and follow a consistent naming convention across your project.
        overrides_tag: Tag for prompt overrides resolution. Maps to files in
            ``.weakincentives/prompts/overrides/{ns}/{key}/{tag}.json``.
            Defaults to "latest" if not specified.
        flags: Feature flags controlling runtime behavior. Keys are flag names,
            values are flag settings (any type). Agent implementations check
            these flags to conditionally enable features or adjust parameters.
        owner: Optional owner identifier (e.g., email, username) for tracking
            who created or is responsible for this experiment. Useful for
            follow-up questions about experiment rationale.
        description: Optional human-readable description of what this experiment
            tests or changes. Document your hypothesis here.

    Example:
        Create an experiment with custom overrides and flags::

            >>> experiment = Experiment(
            ...     name="v2-concise-prompts",
            ...     overrides_tag="v2",
            ...     flags={"verbose_logging": True, "max_retries": 5},
            ...     owner="alice@example.com",
            ...     description="Test shorter, more direct prompt phrasing",
            ... )
            >>> experiment.get_flag("max_retries", 3)
            5

        Use method chaining to build experiments incrementally::

            >>> base = Experiment(name="test")
            >>> configured = base.with_tag("v2").with_flag("debug", True)
            >>> configured.overrides_tag
            'v2'

    Note:
        Use the pre-defined ``BASELINE`` or ``CONTROL`` constants as control
        groups in A/B tests rather than creating new baseline experiments.
    """

    name: str
    overrides_tag: str = "latest"
    flags: Mapping[str, object] = field(default_factory=lambda: {})
    owner: str | None = None
    description: str | None = None

    def with_flag(self, key: str, value: object) -> Experiment:
        """Return a new experiment with a flag added or updated.

        Creates a copy of this experiment with the specified flag set. If the
        flag already exists, its value is replaced. The original experiment
        is not modified.

        Args:
            key: The flag name. Use descriptive names like "max_retries" or
                "enable_caching" rather than abbreviations.
            value: The flag value. Can be any type (bool, int, str, etc.).

        Returns:
            A new ``Experiment`` instance with the flag set.

        Example:
            >>> exp = Experiment(name="test")
            >>> exp2 = exp.with_flag("debug", True)
            >>> exp2.get_flag("debug")
            True
            >>> exp.has_flag("debug")  # Original unchanged
            False
        """
        return replace(self, flags={**self.flags, key: value})

    def with_tag(self, tag: str) -> Experiment:
        """Return a new experiment with a different overrides tag.

        Creates a copy of this experiment pointing to different prompt
        overrides. The tag determines which override files are loaded from
        ``.weakincentives/prompts/overrides/{ns}/{key}/{tag}.json``.

        Args:
            tag: The new overrides tag (e.g., "v2", "concise", "latest").

        Returns:
            A new ``Experiment`` instance with the specified tag.

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
        """Retrieve a flag value, returning a default if not set.

        Use this method to read experiment configuration in your agent or
        tool implementations. Always provide a sensible default to ensure
        your code works with experiments that don't define the flag.

        Args:
            key: The flag name to look up.
            default: Value to return if the flag is not set. Defaults to None.

        Returns:
            The flag value if set, otherwise the default. Note that flag
            values are untyped (``object``), so you may need to cast or
            validate the returned value.

        Example:
            >>> exp = Experiment(name="test", flags={"retries": 5})
            >>> exp.get_flag("retries", 3)
            5
            >>> exp.get_flag("timeout", 30)
            30
            >>> exp.get_flag("missing")  # Returns None
        """
        return self.flags.get(key, default)

    def has_flag(self, key: str) -> bool:
        """Check whether a flag is explicitly set.

        Unlike ``get_flag()``, this method distinguishes between a flag that
        is not set and a flag explicitly set to ``False`` or ``None``. Use
        this when the presence of a flag (regardless of value) has meaning.

        Args:
            key: The flag name to check.

        Returns:
            ``True`` if the flag exists in the experiment, ``False`` otherwise.

        Example:
            >>> exp = Experiment(name="test", flags={"debug": False})
            >>> exp.has_flag("debug")  # Flag exists (even though False)
            True
            >>> exp.has_flag("verbose")  # Flag not set
            False
        """
        return key in self.flags


# Sentinel experiments
BASELINE = Experiment(name="baseline", overrides_tag="latest")
"""Default experiment with no custom flags, using "latest" overrides.

Use ``BASELINE`` as the control group in A/B tests to compare against
treatment variants. This experiment uses the default prompt overrides
(tag "latest") and has no feature flags set.

Example:
    Compare a treatment against the baseline::

        experiments = [BASELINE, Experiment(name="treatment", ...)]
        for exp in experiments:
            results[exp.name] = run_evaluation(exp)
"""

CONTROL = Experiment(name="control", overrides_tag="latest")
"""Alias for a control experiment with an explicit name.

Functionally equivalent to ``BASELINE`` but uses the name "control" instead
of "baseline". Choose whichever name fits your project's terminology.

Use ``CONTROL`` when:
- Your team prefers "control" vs "treatment" terminology
- You want the experiment name to be self-documenting in results
"""


__all__ = [
    "BASELINE",
    "CONTROL",
    "Experiment",
]

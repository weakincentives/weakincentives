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

"""Tests for the HostedTool abstraction."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.prompt import HostedTool
from weakincentives.prompt.errors import PromptValidationError


@dataclass(slots=True, frozen=True)
class MockConfig:
    """Mock configuration for testing."""

    option: str = "default"


class TestHostedToolValidation:
    """Tests for HostedTool validation."""

    def test_valid_hosted_tool(self) -> None:
        """Valid hosted tool passes validation."""
        tool = HostedTool(
            kind="web_search",
            name="my_search",
            description="Search the web for information.",
            config=MockConfig(),
        )
        assert tool.kind == "web_search"
        assert tool.name == "my_search"
        assert tool.description == "Search the web for information."
        assert isinstance(tool.config, MockConfig)

    def test_empty_name_raises(self) -> None:
        """Empty name raises PromptValidationError."""
        with pytest.raises(PromptValidationError, match="HostedTool name must match"):
            HostedTool(
                kind="web_search",
                name="",
                description="Valid description.",
                config=MockConfig(),
            )

    def test_whitespace_only_name_raises(self) -> None:
        """Whitespace-only name raises PromptValidationError."""
        with pytest.raises(
            PromptValidationError, match="HostedTool name must not contain"
        ):
            HostedTool(
                kind="web_search",
                name="   ",
                description="Valid description.",
                config=MockConfig(),
            )

    def test_name_with_surrounding_whitespace_raises(self) -> None:
        """Name with surrounding whitespace raises PromptValidationError."""
        with pytest.raises(
            PromptValidationError, match="HostedTool name must not contain"
        ):
            HostedTool(
                kind="web_search",
                name=" my_search ",
                description="Valid description.",
                config=MockConfig(),
            )

    def test_name_too_long_raises(self) -> None:
        """Name exceeding 64 characters raises PromptValidationError."""
        long_name = "a" * 65
        with pytest.raises(PromptValidationError, match="HostedTool name must match"):
            HostedTool(
                kind="web_search",
                name=long_name,
                description="Valid description.",
                config=MockConfig(),
            )

    def test_name_with_uppercase_raises(self) -> None:
        """Name with uppercase characters raises PromptValidationError."""
        with pytest.raises(PromptValidationError, match="HostedTool name must match"):
            HostedTool(
                kind="web_search",
                name="MySearch",
                description="Valid description.",
                config=MockConfig(),
            )

    def test_name_with_invalid_characters_raises(self) -> None:
        """Name with invalid characters raises PromptValidationError."""
        with pytest.raises(PromptValidationError, match="HostedTool name must match"):
            HostedTool(
                kind="web_search",
                name="my.search",
                description="Valid description.",
                config=MockConfig(),
            )

    def test_name_with_hyphens_allowed(self) -> None:
        """Name with hyphens is allowed."""
        tool = HostedTool(
            kind="web_search",
            name="my-search",
            description="Valid description here.",
            config=MockConfig(),
        )
        assert tool.name == "my-search"

    def test_name_with_underscores_allowed(self) -> None:
        """Name with underscores is allowed."""
        tool = HostedTool(
            kind="web_search",
            name="my_search",
            description="Valid description here.",
            config=MockConfig(),
        )
        assert tool.name == "my_search"

    def test_name_with_digits_allowed(self) -> None:
        """Name with digits is allowed."""
        tool = HostedTool(
            kind="web_search",
            name="search123",
            description="Valid description here.",
            config=MockConfig(),
        )
        assert tool.name == "search123"

    def test_empty_description_raises(self) -> None:
        """Empty description raises PromptValidationError."""
        with pytest.raises(
            PromptValidationError, match="HostedTool description must be"
        ):
            HostedTool(
                kind="web_search",
                name="my_search",
                description="",
                config=MockConfig(),
            )

    def test_whitespace_only_description_raises(self) -> None:
        """Whitespace-only description raises PromptValidationError."""
        with pytest.raises(
            PromptValidationError, match="HostedTool description must be"
        ):
            HostedTool(
                kind="web_search",
                name="my_search",
                description="   ",
                config=MockConfig(),
            )

    def test_description_too_long_raises(self) -> None:
        """Description exceeding 200 characters raises PromptValidationError."""
        long_desc = "a" * 201
        with pytest.raises(
            PromptValidationError, match="HostedTool description must be"
        ):
            HostedTool(
                kind="web_search",
                name="my_search",
                description=long_desc,
                config=MockConfig(),
            )

    def test_description_exactly_200_chars_allowed(self) -> None:
        """Description at exactly 200 characters is allowed."""
        desc = "a" * 200
        tool = HostedTool(
            kind="web_search",
            name="my_search",
            description=desc,
            config=MockConfig(),
        )
        assert len(tool.description) == 200

    def test_non_ascii_description_raises(self) -> None:
        """Description with non-ASCII characters raises PromptValidationError."""
        with pytest.raises(
            PromptValidationError, match="HostedTool description must be ASCII"
        ):
            HostedTool(
                kind="web_search",
                name="my_search",
                description="Search with émoji 🔍",
                config=MockConfig(),
            )

    def test_hosted_tool_is_frozen(self) -> None:
        """HostedTool is immutable."""
        tool = HostedTool(
            kind="web_search",
            name="my_search",
            description="Valid description here.",
            config=MockConfig(),
        )
        with pytest.raises(AttributeError):
            tool.name = "new_name"  # type: ignore[misc]

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

"""Shared fixtures and helpers for local prompt overrides store tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from weakincentives.prompt import MarkdownSection, PromptTemplate, Tool
from weakincentives.prompt.overrides import (
    HexDigest,
    PromptDescriptor,
    PromptOverride,
    SectionOverride,
    ToolOverride,
)


@dataclass
class GreetingParams:
    subject: str


@dataclass
class ToolParams:
    query: str = field(metadata={"description": "User provided keywords."})


@dataclass
class ToolResult:
    result: str


VALID_DIGEST = HexDigest("a" * 64)
OTHER_DIGEST = HexDigest("b" * 64)


def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        ns="tests.versioning",
        key="versioned-greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                template="Greet ${subject} warmly.",
                key="greeting",
            )
        ],
    )


def build_prompt_with_tool() -> PromptTemplate:
    tool = Tool[ToolParams, ToolResult](
        name="search",
        description="Search stored notes.",
        handler=None,
    )
    return PromptTemplate(
        ns="tests.versioning",
        key="versioned-greeting-tools",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                template="Greet ${subject} warmly.",
                key="greeting",
                tools=[tool],
            )
        ],
    )


def override_path(
    tmp_path: Path, descriptor: PromptDescriptor, tag: str = "latest"
) -> Path:
    """Build expected override path when root_path is explicitly provided.

    When root_path is explicit, it is used directly as the overrides directory
    (no .weakincentives/prompts/overrides prefix).
    """
    override_dir = tmp_path
    for segment in descriptor.ns.split("/"):
        override_dir /= segment
    override_dir /= descriptor.key
    return override_dir / f"{tag}.json"


def auto_discovery_override_path(
    repo_root: Path, descriptor: PromptDescriptor, tag: str = "latest"
) -> Path:
    """Build expected override path when using automatic discovery.

    When using automatic discovery (no explicit root_path), the path
    includes the .weakincentives/prompts/overrides prefix.
    """
    override_dir = repo_root / ".weakincentives" / "prompts" / "overrides"
    for segment in descriptor.ns.split("/"):
        override_dir /= segment
    override_dir /= descriptor.key
    return override_dir / f"{tag}.json"


def make_section_override(
    section: object,
    *,
    path: tuple[str, ...] | None = None,
    expected_hash: str | None = None,
    body: object = "Body",
) -> dict[tuple[str, ...], SectionOverride]:
    """Build a single-entry sections dict for testing overrides."""
    return {
        path or section.path: SectionOverride(  # type: ignore[union-attr]
            path=path or section.path,  # type: ignore[union-attr]
            expected_hash=expected_hash or section.content_hash,  # type: ignore[union-attr]
            body=body,  # type: ignore[arg-type]
        )
    }


def make_override(
    descriptor: PromptDescriptor,
    sections: dict[tuple[str, ...], SectionOverride],
    tool_overrides: dict[str, ToolOverride] | None = None,
) -> PromptOverride:
    """Build a PromptOverride for validation testing."""
    kwargs: dict[str, object] = {
        "ns": descriptor.ns,
        "prompt_key": descriptor.key,
        "tag": "latest",
        "sections": sections,
    }
    if tool_overrides is not None:
        kwargs["tool_overrides"] = tool_overrides
    return PromptOverride(**kwargs)  # type: ignore[arg-type]

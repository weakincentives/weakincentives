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

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.prompt import MarkdownSection, PromptTemplate
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptOverridesError,
)
from weakincentives.prompt.overrides.versioning import PromptDescriptor
from weakincentives.runtime.session import Session
from weakincentives.tools.digests import (
    WorkspaceDigest,
    WorkspaceDigestSection,
)


def _build_prompt() -> PromptTemplate[str]:
    sections = (
        MarkdownSection(title="One", template="first", key="one"),
        MarkdownSection(title="Two", template="second", key="two"),
    )
    return PromptTemplate(ns="test", key="prompt", sections=sections)


def test_prompt_find_section_by_key_and_candidates() -> None:
    prompt = _build_prompt()

    assert isinstance(prompt.find_section(MarkdownSection), MarkdownSection)
    assert isinstance(
        prompt.find_section((MarkdownSection,)),
        MarkdownSection,
    )
    assert prompt.find_section((MarkdownSection, WorkspaceDigestSection)).key == "one"
    with pytest.raises(TypeError):
        prompt.find_section(())
    with pytest.raises(KeyError):
        prompt.find_section((WorkspaceDigestSection,))


def test_set_section_override_requires_registered_path(tmp_path: Path) -> None:
    prompt = _build_prompt()
    store = LocalPromptOverridesStore(root_path=tmp_path)

    descriptor = PromptDescriptor.from_prompt(prompt)
    assert any(section.path == ("one",) for section in descriptor.sections)

    with pytest.raises(PromptOverridesError):
        store.set_section_override(prompt, path=("missing",), body="X")


def test_workspace_digest_section_in_descriptor() -> None:
    section = WorkspaceDigestSection(session=Session())
    descriptor = PromptDescriptor.from_prompt(
        PromptTemplate(ns="digest", key="prompt", sections=(section,))
    )

    assert section.original_body_template() == section._placeholder
    assert ("workspace-digest",) in {entry.path for entry in descriptor.sections}
    assert WorkspaceDigest(section_key="a", body="b").section_key == "a"
    with pytest.raises(TypeError):
        section.clone()
    clone = section.clone(session=Session())
    assert isinstance(clone, WorkspaceDigestSection)
    heading = section._render_block("", depth=0, number="1.")
    assert heading.startswith("## 1. Workspace Digest")


"""Coverage tests for prompt utilities and workspace digest plumbing."""

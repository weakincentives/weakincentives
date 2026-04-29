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

"""Tests for ``weakincentives.overrides``."""

from __future__ import annotations

from weakincentives.core import MarkdownSection, Prompt
from weakincentives.overrides import (
    OverrideStore,
    SectionOverride,
    apply_overrides,
    describe_prompt,
)


def _prompt(template: str = "original body") -> Prompt:
    section = MarkdownSection[None](
        title="Body",
        key="body",
        template=template,
        children=(MarkdownSection[None](title="Leaf", key="leaf", template="child"),),
    )
    return Prompt(ns="ns", key="k", sections=(section,))


def test_describe_prompt_emits_per_section_hashes() -> None:
    descriptor = describe_prompt(_prompt())
    assert descriptor.ns == "ns"
    assert descriptor.key == "k"
    keys = [s.key for s in descriptor.sections]
    assert keys == ["body", "leaf"]


def test_describe_prompt_handles_non_markdown_section() -> None:
    """Cover the empty-body branch in _describe_section."""
    from weakincentives.core import Section

    plain = Section[None](title="Plain", key="plain")
    prompt = Prompt(ns="ns", key="k", sections=(plain,))
    descriptor = describe_prompt(prompt)
    assert descriptor.sections[0].key == "plain"
    assert len(descriptor.sections[0].content_hash) == 64


def test_descriptor_get_returns_matching_section() -> None:
    descriptor = describe_prompt(_prompt())
    assert descriptor.get("body") is not None
    assert descriptor.get("missing") is None


def test_apply_overrides_swaps_template_when_hash_matches() -> None:
    prompt = _prompt()
    descriptor = describe_prompt(prompt)
    section_hash = descriptor.get("body")
    assert section_hash is not None
    store = OverrideStore()
    store.upsert(
        SectionOverride(
            section_key="body",
            template="overridden body",
            expected_hash=section_hash.content_hash,
        )
    )
    rebuilt = apply_overrides(prompt, store)
    text = rebuilt.render().text
    assert "overridden body" in text
    assert "original body" not in text


def test_apply_overrides_skips_when_hash_drifts() -> None:
    prompt = _prompt()
    store = OverrideStore()
    store.upsert(
        SectionOverride(
            section_key="body",
            template="overridden",
            expected_hash="stale-hash",
        )
    )
    rebuilt = apply_overrides(prompt, store)
    assert rebuilt is prompt
    assert "original body" in rebuilt.render().text


def test_apply_overrides_returns_original_when_store_empty() -> None:
    prompt = _prompt()
    rebuilt = apply_overrides(prompt, OverrideStore())
    assert rebuilt is prompt


def test_store_remove_drops_override() -> None:
    descriptor = describe_prompt(_prompt())
    section_hash = descriptor.get("body")
    assert section_hash is not None
    store = OverrideStore()
    override = SectionOverride(
        section_key="body",
        template="x",
        expected_hash=section_hash.content_hash,
    )
    store.upsert(override)
    assert store.all() == (override,)
    store.remove("body")
    assert store.all() == ()
    store.remove("body")  # idempotent


def test_apply_overrides_walks_into_children() -> None:
    """When a deep child is overridden, the parent must be rebuilt too."""
    import hashlib

    leaf = MarkdownSection[None](title="Leaf", key="leaf", template="orig")
    parent = MarkdownSection[None](
        title="Parent", key="parent", template="parent body", children=(leaf,)
    )
    prompt = Prompt(ns="ns", key="k", sections=(parent,))
    leaf_hash = hashlib.sha256(b"Leaf\norig").hexdigest()
    store = OverrideStore()
    store.upsert(
        SectionOverride(
            section_key="leaf",
            template="rewritten",
            expected_hash=leaf_hash,
        )
    )
    rebuilt = apply_overrides(prompt, store)
    text = rebuilt.render().text
    assert "rewritten" in text
    # parent must have been rebuilt to carry the new child
    assert rebuilt.sections[0] is not parent

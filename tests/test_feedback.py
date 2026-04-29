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

"""Tests for ``weakincentives.feedback``."""

from __future__ import annotations

from dataclasses import dataclass

from weakincentives.core import MarkdownSection, Prompt, Session
from weakincentives.feedback import (
    FeedbackProvider,
    compose_with_feedback,
    make_feedback_section,
)


@dataclass(frozen=True)
class Note:
    text: str


def _make_prompt() -> Prompt:
    section = MarkdownSection[None](title="System", key="system", template="hi")
    return Prompt(ns="ns", key="k", sections=(section,))


def test_callable_feedback_satisfies_protocol() -> None:
    provider = make_feedback_section(
        title="Notes",
        key="notes",
        builder=lambda session: f"{len(session[Note].all())} notes",
    )
    assert isinstance(provider, FeedbackProvider)


def test_compose_appends_section_when_builder_returns_text() -> None:
    session = Session()
    session[Note].seed(Note(text="first"))
    provider = make_feedback_section(
        title="Notes",
        key="notes",
        builder=lambda s: f"{len(s[Note].all())} note(s)",
    )
    composed = compose_with_feedback(_make_prompt(), session, (provider,))
    text = composed.render().text
    assert "## 1. System" in text
    assert "## 2. Notes" in text
    assert "1 note(s)" in text


def test_compose_skips_section_when_builder_returns_none() -> None:
    session = Session()
    provider = make_feedback_section(title="Notes", key="notes", builder=lambda _: None)
    composed = compose_with_feedback(_make_prompt(), session, (provider,))
    assert composed is _make_prompt() or "Notes" not in composed.render().text


def test_compose_skips_section_when_builder_returns_blank() -> None:
    session = Session()
    provider = make_feedback_section(
        title="Notes", key="notes", builder=lambda _: "   "
    )
    composed = compose_with_feedback(_make_prompt(), session, (provider,))
    assert "Notes" not in composed.render().text


def test_compose_returns_original_when_no_feedback() -> None:
    base = _make_prompt()
    composed = compose_with_feedback(base, Session(), ())
    assert composed is base


def test_compose_preserves_output_type() -> None:
    @dataclass(frozen=True)
    class Answer:
        value: int

    prompt = Prompt(
        ns="ns",
        key="k",
        sections=(MarkdownSection[None](title="S", key="s", template="hi"),),
        output_type=Answer,
    )
    provider = make_feedback_section(
        title="F", key="f", builder=lambda _: "feedback body"
    )
    composed = compose_with_feedback(prompt, Session(), (provider,))
    assert composed.output_type is Answer

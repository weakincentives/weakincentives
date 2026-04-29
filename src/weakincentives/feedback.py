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

"""Feedback providers: session-aware prompt augmentation.

A :class:`FeedbackProvider` inspects the current :class:`Session` and
returns a :class:`Section[None]` that the runtime injects into a prompt
before each iteration. Use it for things like "remind the model what
tool calls just failed", "list previously seen filenames", or "show the
running token budget."

The spine's :class:`weakincentives.core.AgentLoop` has no built-in
feedback support; the helper here composes feedback-aware prompts on
demand so feedback stays an opt-in concern of the runtime layer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, final, runtime_checkable

from weakincentives.core import (
    MarkdownSection,
    Prompt,
    Section,
    Session,
)

__all__ = [
    "FeedbackProvider",
    "compose_with_feedback",
    "make_feedback_section",
]


@runtime_checkable
class FeedbackProvider(Protocol):
    """Builds a feedback :class:`Section` from a :class:`Session`.

    Returning ``None`` means "no feedback this iteration"; the composed
    prompt then drops the placeholder section entirely.
    """

    def render(self, session: Session) -> Section[None] | None: ...


@final
@dataclass(frozen=True, slots=True)
class _CallableFeedback:
    title: str
    key: str
    builder: Callable[[Session], str | None]

    def render(self, session: Session) -> Section[None] | None:
        body = self.builder(session)
        if body is None or not body.strip():
            return None
        return MarkdownSection[None](title=self.title, key=self.key, template=body)


def make_feedback_section(
    *, title: str, key: str, builder: Callable[[Session], str | None]
) -> FeedbackProvider:
    """Wrap a ``Session -> str | None`` callable as a :class:`FeedbackProvider`.

    The builder returns the markdown body to render, or ``None`` to skip
    the section for this iteration.
    """
    return _CallableFeedback(title=title, key=key, builder=builder)


def compose_with_feedback(
    prompt: Prompt,
    session: Session,
    providers: Sequence[FeedbackProvider],
) -> Prompt:
    """Return a new :class:`Prompt` with feedback sections appended.

    Each provider is called once; ``None`` returns are skipped. The
    appended sections appear after the prompt's existing sections so the
    model sees the system prompt first, then the running feedback.
    """
    feedback_sections: list[Section[None]] = []
    for provider in providers:
        section = provider.render(session)
        if section is not None:
            feedback_sections.append(section)
    if not feedback_sections:
        return prompt
    return Prompt(
        ns=prompt.ns,
        key=prompt.key,
        sections=(*prompt.sections, *feedback_sections),
        params=prompt.params,
        output_type=prompt.output_type,
    )

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

"""Adapter-specific ACK scenarios for skill installation."""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.prompt import Prompt
from weakincentives.runtime.session import Session

from ..adapters import AdapterFixture
from . import (
    SKILL_SECRET_ANSWER,
    SkillQuestionParams,
    build_skill_prompt,
    create_test_skill,
    make_adapter_ns,
)

pytestmark = pytest.mark.ack_capability("skill_installation")


def test_skill_knowledge_available(
    adapter_fixture: AdapterFixture,
    session: Session,
    tmp_path: Path,
) -> None:
    """Skills attached to prompt sections are installed and accessible to the agent."""
    skill_dir = create_test_skill(tmp_path)

    adapter = adapter_fixture.create_adapter(tmp_path)
    prompt = Prompt(
        build_skill_prompt(make_adapter_ns(adapter_fixture.adapter_name), skill_dir)
    ).bind(SkillQuestionParams(question="What is the secret codeword?"))

    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    assert SKILL_SECRET_ANSWER in response.text, (
        f"Expected secret codeword '{SKILL_SECRET_ANSWER}' in response, "
        f"got: {response.text[:200]}"
    )

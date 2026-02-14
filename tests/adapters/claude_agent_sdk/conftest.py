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

"""Shared fixtures for Claude Agent SDK hook tests."""

from __future__ import annotations

from typing import cast

import pytest

from weakincentives.adapters.claude_agent_sdk._hooks import HookContext
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session

from ._hook_helpers import _make_prompt


@pytest.fixture
def session() -> Session:
    dispatcher = InProcessDispatcher()
    return Session(dispatcher=dispatcher)


@pytest.fixture
def hook_context(session: Session) -> HookContext:
    return HookContext(
        session=session,
        prompt=cast("PromptProtocol[object]", _make_prompt()),
        adapter_name="claude_agent_sdk",
        prompt_name="test_prompt",
    )

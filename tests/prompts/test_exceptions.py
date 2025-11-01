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

from dataclasses import dataclass

import pytest

from weakincentives.prompt import PromptRenderError, PromptValidationError


@dataclass
class FakeParams:
    value: str = "example"


@pytest.mark.parametrize("exception_cls", [PromptValidationError, PromptRenderError])
def test_prompt_exceptions_capture_context(
    exception_cls: type[PromptValidationError | PromptRenderError],
) -> None:
    context = {
        "section_path": ("root", "child"),
        "dataclass_type": FakeParams,
        "placeholder": "value",
    }

    exc = exception_cls("boom", **context)

    assert isinstance(exc, Exception)
    assert exc.message == "boom"
    assert exc.args[0] == "boom"
    assert exc.section_path == context["section_path"]
    assert exc.dataclass_type is FakeParams
    assert exc.placeholder == "value"


def test_prompt_exceptions_default_context_optional() -> None:
    exc = PromptValidationError(
        "missing", section_path=None, dataclass_type=None, placeholder=None
    )

    assert exc.section_path == ()
    assert exc.dataclass_type is None
    assert exc.placeholder is None

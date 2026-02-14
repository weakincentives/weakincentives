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

"""Tier 1 ACK scenarios for structured output parsing."""

from __future__ import annotations

import pytest

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.prompt import Prompt
from weakincentives.runtime.session import Session

from ..adapters import AdapterFixture
from . import ReviewAnalysis, ReviewParams, build_structured_prompt, make_adapter_ns

pytestmark = pytest.mark.ack_capability("structured_output")


_SAMPLE_REVIEW = (
    "The release improves onboarding with clearer copy and fewer clicks. "
    "Some users still report minor setup friction."
)


def test_structured_output_parsing(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """ACK adapters parse structured output into the requested dataclass."""
    prompt = Prompt(
        build_structured_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(ReviewParams(text=_SAMPLE_REVIEW))

    response = adapter.evaluate(prompt, session=session)

    assert isinstance(response.output, ReviewAnalysis)
    assert response.output.summary.strip()
    assert response.output.sentiment.strip()


def test_structured_output_type_fidelity(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """Structured output preserves the exact declared output type."""
    prompt = Prompt(
        build_structured_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(ReviewParams(text=_SAMPLE_REVIEW))

    response = adapter.evaluate(prompt, session=session)

    assert type(response.output) is ReviewAnalysis

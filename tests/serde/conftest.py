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

"""Shared test data types for loop serde tests."""

from __future__ import annotations

from dataclasses import dataclass

# =============================================================================
# Test data types for generic parameters
# =============================================================================


@dataclass(slots=True, frozen=True)
class QuestionInput:
    """Sample input for QA-style evaluation."""

    question: str
    context: str | None = None


@dataclass(slots=True, frozen=True)
class AnswerExpected:
    """Expected answer for QA-style evaluation."""

    answer: str
    keywords: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class TaskRequest:
    """User request for AgentLoop."""

    task: str
    priority: int = 1


@dataclass(slots=True, frozen=True)
class TaskOutput:
    """Output from AgentLoop processing."""

    result: str
    success: bool = True

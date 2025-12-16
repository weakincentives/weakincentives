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

"""Tests for dataset loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from weakincentives.evals import Sample, load_jsonl

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(slots=True, frozen=True)
class _MathProblem:
    """Test dataclass for math problems."""

    a: int
    b: int
    operator: str


@pytest.fixture
def simple_jsonl(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a simple JSONL file for testing."""
    file_path = tmp_path / "simple.jsonl"
    content = """\
{"id": "1", "input": "What is 2+2?", "expected": "4"}
{"id": "2", "input": "Capital of France?", "expected": "Paris"}
"""
    file_path.write_text(content)
    yield file_path


@pytest.fixture
def int_jsonl(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a JSONL file with int types."""
    file_path = tmp_path / "int.jsonl"
    content = """\
{"id": "1", "input": 5, "expected": 10}
{"id": "2", "input": 3, "expected": 6}
"""
    file_path.write_text(content)
    yield file_path


@pytest.fixture
def dataclass_jsonl(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a JSONL file with dataclass types."""
    file_path = tmp_path / "dataclass.jsonl"
    content = """\
{"id": "1", "input": {"a": 2, "b": 3, "operator": "+"}, "expected": 5}
{"id": "2", "input": {"a": 10, "b": 5, "operator": "-"}, "expected": 5}
"""
    file_path.write_text(content)
    yield file_path


def test_load_jsonl_string_types(simple_jsonl: Path) -> None:
    """load_jsonl handles string types."""
    dataset = load_jsonl(simple_jsonl, str, str)
    assert len(dataset) == 2
    assert isinstance(dataset, tuple)  # Immutable

    sample1 = dataset[0]
    assert sample1.id == "1"
    assert sample1.input == "What is 2+2?"
    assert sample1.expected == "4"

    sample2 = dataset[1]
    assert sample2.id == "2"
    assert sample2.input == "Capital of France?"
    assert sample2.expected == "Paris"


def test_load_jsonl_int_types(int_jsonl: Path) -> None:
    """load_jsonl handles int types."""
    dataset = load_jsonl(int_jsonl, int, int)
    assert len(dataset) == 2

    assert dataset[0].input == 5
    assert dataset[0].expected == 10


def test_load_jsonl_dataclass_input(dataclass_jsonl: Path) -> None:
    """load_jsonl handles dataclass input types."""
    dataset = load_jsonl(dataclass_jsonl, _MathProblem, int)
    assert len(dataset) == 2

    sample1 = dataset[0]
    assert isinstance(sample1.input, _MathProblem)
    assert sample1.input.a == 2
    assert sample1.input.b == 3
    assert sample1.input.operator == "+"
    assert sample1.expected == 5


def test_load_jsonl_returns_tuple(simple_jsonl: Path) -> None:
    """load_jsonl returns a tuple (immutable)."""
    dataset = load_jsonl(simple_jsonl, str, str)
    assert isinstance(dataset, tuple)


def test_load_jsonl_type_mismatch_raises(simple_jsonl: Path) -> None:
    """load_jsonl raises on type mismatch."""
    with pytest.raises(TypeError, match="expected int"):
        load_jsonl(simple_jsonl, int, str)


def test_load_jsonl_empty_file(tmp_path: Path) -> None:
    """load_jsonl handles empty file."""
    file_path = tmp_path / "empty.jsonl"
    file_path.write_text("")
    dataset = load_jsonl(file_path, str, str)
    assert dataset == ()


def test_load_jsonl_float_type(tmp_path: Path) -> None:
    """load_jsonl handles float types."""
    file_path = tmp_path / "float.jsonl"
    file_path.write_text('{"id": "1", "input": 3.14, "expected": 6.28}\n')
    dataset = load_jsonl(file_path, float, float)
    assert dataset[0].input == 3.14
    assert dataset[0].expected == 6.28


def test_load_jsonl_bool_type(tmp_path: Path) -> None:
    """load_jsonl handles bool types."""
    file_path = tmp_path / "bool.jsonl"
    file_path.write_text('{"id": "1", "input": true, "expected": false}\n')
    dataset = load_jsonl(file_path, bool, bool)
    assert dataset[0].input is True
    assert dataset[0].expected is False


# =============================================================================
# Sample Construction Tests
# =============================================================================


def test_sample_generic_types() -> None:
    """Sample supports generic types."""
    sample: Sample[str, int] = Sample(id="1", input="five", expected=5)
    assert sample.input == "five"
    assert sample.expected == 5


def test_sample_dataclass_input() -> None:
    """Sample supports dataclass input."""
    problem = _MathProblem(a=1, b=2, operator="+")
    sample: Sample[_MathProblem, int] = Sample(id="1", input=problem, expected=3)
    assert sample.input == problem
    assert sample.expected == 3


def test_programmatic_dataset_creation() -> None:
    """Dataset can be created programmatically."""
    dataset = tuple(
        Sample(
            id=str(i),
            input=f"What is {a} + {b}?",
            expected=str(a + b),
        )
        for i, (a, b) in enumerate([(1, 1), (2, 3), (10, 20)])
    )
    assert len(dataset) == 3
    assert dataset[0].input == "What is 1 + 1?"
    assert dataset[0].expected == "2"
    assert dataset[2].expected == "30"


def test_load_jsonl_unsupported_type_raises(tmp_path: Path) -> None:
    """load_jsonl raises on unsupported type (non-primitive, non-mapping)."""
    file_path = tmp_path / "list.jsonl"
    file_path.write_text('{"id": "1", "input": [1, 2, 3], "expected": "sum"}\n')
    with pytest.raises(TypeError, match="cannot coerce list"):
        load_jsonl(file_path, list, str)

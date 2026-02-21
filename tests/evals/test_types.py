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

"""Tests for evals core types."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

import weakincentives.evals as evals_module
from weakincentives.evals import (
    Dataset,
    EvalResult,
    Sample,
    Score,
)
from weakincentives.evals._types import _coerce

# =============================================================================
# Module Tests
# =============================================================================


def test_module_dir() -> None:
    """Module __dir__ includes all exports."""
    module_dir = dir(evals_module)
    # Should include all __all__ exports
    assert "Sample" in module_dir
    assert "Dataset" in module_dir
    assert "Score" in module_dir
    assert "EvalResult" in module_dir
    assert "EvalReport" in module_dir
    assert "EvalLoop" in module_dir
    assert "exact_match" in module_dir
    assert "contains" in module_dir
    assert "all_of" in module_dir
    assert "any_of" in module_dir
    assert "llm_judge" in module_dir
    # Should also include module attributes
    assert "__all__" in module_dir


# =============================================================================
# Sample Tests
# =============================================================================


def test_sample_creation() -> None:
    """Sample can be created with typed fields."""
    sample = Sample(id="1", input="What is 2+2?", expected="4")
    assert sample.id == "1"
    assert sample.input == "What is 2+2?"
    assert sample.expected == "4"


def test_sample_is_frozen() -> None:
    """Sample is immutable."""
    sample = Sample(id="1", input="hello", expected="world")
    with pytest.raises(AttributeError):
        sample.id = "2"  # type: ignore[misc]


def test_sample_with_complex_types() -> None:
    """Sample supports complex generic types."""

    @dataclass(slots=True, frozen=True)
    class MathProblem:
        a: int
        b: int
        operation: str

    sample = Sample(
        id="1",
        input=MathProblem(a=2, b=3, operation="+"),
        expected=5,
    )
    assert sample.input.a == 2
    assert sample.input.b == 3
    assert sample.expected == 5


# =============================================================================
# Score Tests
# =============================================================================


def test_score_creation() -> None:
    """Score can be created with default reason."""
    score = Score(value=0.8, passed=True)
    assert score.value == 0.8
    assert score.passed is True
    assert score.reason == ""


def test_score_with_reason() -> None:
    """Score can include an explanation."""
    score = Score(value=0.5, passed=False, reason="Missing key details")
    assert score.reason == "Missing key details"


def test_score_is_frozen() -> None:
    """Score is immutable."""
    score = Score(value=1.0, passed=True)
    with pytest.raises(AttributeError):
        score.value = 0.5  # type: ignore[misc]


# =============================================================================
# Dataset Tests
# =============================================================================


def test_dataset_creation() -> None:
    """Dataset can be created with samples tuple."""
    samples = (
        Sample(id="1", input="a", expected="b"),
        Sample(id="2", input="c", expected="d"),
    )
    dataset = Dataset(samples=samples)
    assert len(dataset) == 2


def test_dataset_iteration() -> None:
    """Dataset supports iteration."""
    samples = (
        Sample(id="1", input="a", expected="b"),
        Sample(id="2", input="c", expected="d"),
    )
    dataset = Dataset(samples=samples)
    ids = [s.id for s in dataset]
    assert ids == ["1", "2"]


def test_dataset_indexing() -> None:
    """Dataset supports index access."""
    samples = (
        Sample(id="1", input="a", expected="b"),
        Sample(id="2", input="c", expected="d"),
    )
    dataset = Dataset(samples=samples)
    assert dataset[0].id == "1"
    assert dataset[1].id == "2"


def test_dataset_is_frozen() -> None:
    """Dataset is immutable."""
    samples = (Sample(id="1", input="a", expected="b"),)
    dataset = Dataset(samples=samples)
    with pytest.raises(AttributeError):
        dataset.samples = ()  # type: ignore[misc]


def test_dataset_load_from_jsonl() -> None:
    """Dataset.load reads JSONL files."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "1", "input": "What is 2+2?", "expected": "4"}\n')
        f.write('{"id": "2", "input": "Capital of France?", "expected": "Paris"}\n')
        path = Path(f.name)

    try:
        dataset = Dataset.load(path, str, str)
        assert len(dataset) == 2
        assert dataset[0].id == "1"
        assert dataset[0].input == "What is 2+2?"
        assert dataset[0].expected == "4"
        assert dataset[1].id == "2"
    finally:
        path.unlink()


def test_dataset_load_with_complex_types() -> None:
    """Dataset.load deserializes complex types."""

    @dataclass(slots=True, frozen=True)
    class Problem:
        question: str
        difficulty: int

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(
            json.dumps(
                {
                    "id": "1",
                    "input": {"question": "What is 2+2?", "difficulty": 1},
                    "expected": "4",
                }
            )
            + "\n"
        )
        path = Path(f.name)

    try:
        dataset = Dataset.load(path, Problem, str)
        assert len(dataset) == 1
        assert dataset[0].input.question == "What is 2+2?"
        assert dataset[0].input.difficulty == 1
    finally:
        path.unlink()


# =============================================================================
# _coerce Tests
# =============================================================================


def test_coerce_string() -> None:
    """_coerce handles string primitives."""
    result = _coerce("hello", str)
    assert result == "hello"


def test_coerce_int() -> None:
    """_coerce handles int primitives."""
    result = _coerce(42, int)
    assert result == 42


def test_coerce_float() -> None:
    """_coerce handles float primitives."""
    result = _coerce(3.14, float)
    assert result == 3.14


def test_coerce_bool() -> None:
    """_coerce handles bool primitives."""
    result = _coerce(True, bool)
    assert result is True


def test_coerce_type_mismatch_raises() -> None:
    """_coerce raises TypeError on type mismatch."""
    with pytest.raises(TypeError, match="expected str, got int"):
        _coerce(42, str)


def test_coerce_mapping_to_dataclass() -> None:
    """_coerce parses mappings into dataclasses."""

    @dataclass(slots=True, frozen=True)
    class Point:
        x: int
        y: int

    result = _coerce({"x": 1, "y": 2}, Point)
    assert result == Point(x=1, y=2)


def test_coerce_invalid_value_type_raises() -> None:
    """_coerce raises TypeError for non-primitive, non-mapping values."""

    @dataclass(slots=True, frozen=True)
    class Point:
        x: int
        y: int

    with pytest.raises(TypeError, match="cannot coerce list to Point"):
        _coerce([1, 2], Point)


# =============================================================================
# EvalResult Tests
# =============================================================================


def test_eval_result_creation() -> None:
    """EvalResult can be created with required fields."""
    result = EvalResult(
        sample_id="1",
        experiment_name="baseline",
        score=Score(value=1.0, passed=True),
        latency_ms=150,
    )
    assert result.sample_id == "1"
    assert result.experiment_name == "baseline"
    assert result.score.value == 1.0
    assert result.latency_ms == 150
    assert result.error is None
    assert result.success is True


def test_eval_result_with_error() -> None:
    """EvalResult with error is not successful."""
    result = EvalResult(
        sample_id="1",
        experiment_name="baseline",
        score=Score(value=0.0, passed=False),
        latency_ms=0,
        error="Execution failed",
    )
    assert result.success is False
    assert result.error == "Execution failed"


def test_eval_result_is_frozen() -> None:
    """EvalResult is immutable."""
    result = EvalResult(
        sample_id="1",
        experiment_name="baseline",
        score=Score(value=1.0, passed=True),
        latency_ms=100,
    )
    with pytest.raises(AttributeError):
        result.sample_id = "2"  # type: ignore[misc]

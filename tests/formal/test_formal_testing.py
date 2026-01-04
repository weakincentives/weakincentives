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

"""Tests for formal specification testing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from weakincentives.formal import Action, Invariant, StateVar, formal_spec
from weakincentives.formal.testing import (
    ModelCheckError,
    extract_and_verify,
    extract_spec,
    write_spec,
)


@formal_spec(
    module="TestCounter",
    constants={"MaxValue": 10},
    state_vars=[StateVar("count", "Int", "Counter value")],
    actions=[
        Action(
            name="Increment",
            preconditions=("count < MaxValue",),
            updates={"count": "count + 1"},
        )
    ],
    invariants=[
        Invariant(
            id="INV-1",
            name="CountInRange",
            predicate="count >= 0 /\\ count <= MaxValue",
            description="Counter stays in valid range",
        )
    ],
)
class TestCounter:
    """Test counter class with formal spec."""

    pass


def test_extract_spec() -> None:
    """Test extracting formal spec from decorated class."""
    spec = extract_spec(TestCounter)

    assert spec.module == "TestCounter"
    assert spec.constants == {"MaxValue": 10}
    assert len(spec.state_vars) == 1
    assert spec.state_vars[0].name == "count"
    assert len(spec.actions) == 1
    assert spec.actions[0].name == "Increment"
    assert len(spec.invariants) == 1
    assert spec.invariants[0].name == "CountInRange"


def test_extract_spec_no_decorator() -> None:
    """Test extracting spec from class without decorator raises error."""

    class NoSpec:
        pass

    with pytest.raises(ValueError, match="does not have @formal_spec decorator"):
        extract_spec(NoSpec)


def test_write_spec(tmp_path: Path) -> None:
    """Test writing spec to files."""
    spec = extract_spec(TestCounter)

    tla_file, cfg_file = write_spec(spec, tmp_path)

    assert tla_file.exists()
    assert cfg_file.exists()
    assert tla_file.name == "TestCounter.tla"
    assert cfg_file.name == "TestCounter.cfg"

    tla_content = tla_file.read_text()
    assert "MODULE TestCounter" in tla_content
    assert "count >= 0" in tla_content

    cfg_content = cfg_file.read_text()
    assert "MaxValue = 10" in cfg_content
    assert "CountInRange" in cfg_content


def test_extract_and_verify_no_tlc(tmp_path: Path) -> None:
    """Test extract_and_verify without running TLC."""
    spec, tla_file, cfg_file, result = extract_and_verify(
        TestCounter, output_dir=tmp_path, model_check_enabled=False
    )

    assert spec.module == "TestCounter"
    assert tla_file.exists()
    assert cfg_file.exists()
    assert result is None  # No model checking was done


def test_extract_and_verify_creates_directory(tmp_path: Path) -> None:
    """Test that extract_and_verify creates output directory."""
    nested_dir = tmp_path / "nested" / "dir"

    _spec, tla_file, _cfg_file, _result = extract_and_verify(
        TestCounter, output_dir=nested_dir, model_check_enabled=False
    )

    assert nested_dir.exists()
    assert tla_file.parent == nested_dir


def test_model_check_error_tlc_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test model_check raises error when TLC not found."""
    from unittest.mock import MagicMock

    from weakincentives.formal.testing import model_check

    # Mock subprocess.run to raise FileNotFoundError
    def mock_run(*args: Any, **kwargs: Any) -> MagicMock:  # noqa: ANN401
        raise FileNotFoundError("tlc not found")

    import subprocess

    monkeypatch.setattr(subprocess, "run", mock_run)

    spec = extract_spec(TestCounter)

    with pytest.raises(ModelCheckError, match="TLC not found"):
        model_check(spec)


def test_model_check_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test model_check with successful TLC run."""
    from unittest.mock import MagicMock

    from weakincentives.formal.testing import model_check

    call_count = 0

    def mock_run(*args: Any, **kwargs: Any) -> MagicMock:  # noqa: ANN401
        nonlocal call_count
        call_count += 1

        result = MagicMock()

        if call_count == 1:
            # First call is TLC availability check
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        # Second call is actual model checking
        result.returncode = 0
        result.stdout = "TLC finished checking.\n12345 states generated.\n"
        result.stderr = ""
        return result

    import subprocess

    monkeypatch.setattr(subprocess, "run", mock_run)

    spec = extract_spec(TestCounter)
    result = model_check(spec, tlc_config={"workers": "4", "cleanup": True})

    assert result.passed
    assert result.states_generated == 12345
    assert result.returncode == 0


def test_model_check_invariant_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test model_check with invariant violation."""
    from unittest.mock import MagicMock

    from weakincentives.formal.testing import model_check

    call_count = 0

    def mock_run(*args: Any, **kwargs: Any) -> MagicMock:  # noqa: ANN401
        nonlocal call_count
        call_count += 1

        result = MagicMock()

        if call_count == 1:
            # First call is TLC availability check
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        # Second call is actual model checking - invariant violated
        result.returncode = 1
        result.stdout = (
            "Error: Invariant CountInRange violated.\n500 states generated.\n"
        )
        result.stderr = ""
        return result

    import subprocess

    monkeypatch.setattr(subprocess, "run", mock_run)

    spec = extract_spec(TestCounter)
    result = model_check(spec, tlc_config={"workers": "auto", "cleanup": False})

    assert not result.passed
    assert result.states_generated == 500
    assert result.returncode == 1


def test_extract_and_verify_with_model_check_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test extract_and_verify raises error when model checking fails."""
    from unittest.mock import MagicMock

    call_count = 0

    def mock_run(*args: Any, **kwargs: Any) -> MagicMock:  # noqa: ANN401
        nonlocal call_count
        call_count += 1

        result = MagicMock()

        if call_count == 1:
            # First call is TLC availability check
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        # Second call is actual model checking - fails
        result.returncode = 1
        result.stdout = "Error: Invariant violated.\n"
        result.stderr = "Model checking failed"
        return result

    import subprocess

    monkeypatch.setattr(subprocess, "run", mock_run)

    with pytest.raises(ModelCheckError, match="Model checking failed for TestCounter"):
        extract_and_verify(
            TestCounter,
            output_dir=tmp_path,
            model_check_enabled=True,
            tlc_config={"workers": "auto"},
        )


def test_extract_and_verify_with_model_check_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test extract_and_verify with successful model checking."""
    from unittest.mock import MagicMock

    call_count = 0

    def mock_run(*args: Any, **kwargs: Any) -> MagicMock:  # noqa: ANN401
        nonlocal call_count
        call_count += 1

        result = MagicMock()

        if call_count == 1:
            # First call is TLC availability check
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        # Second call is actual model checking - succeeds
        result.returncode = 0
        result.stdout = "Model checking completed.\n1000 states generated.\n"
        result.stderr = ""
        return result

    import subprocess

    monkeypatch.setattr(subprocess, "run", mock_run)

    _spec, _tla_file, _cfg_file, result = extract_and_verify(
        TestCounter,
        output_dir=tmp_path,
        model_check_enabled=True,
        tlc_config={"workers": "auto"},
    )

    assert result is not None
    assert result.passed
    assert result.states_generated == 1000


def test_model_check_no_state_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test model_check when TLC output doesn't contain state count."""
    from unittest.mock import MagicMock

    from weakincentives.formal.testing import model_check

    call_count = 0

    def mock_run(*args: Any, **kwargs: Any) -> MagicMock:  # noqa: ANN401
        nonlocal call_count
        call_count += 1

        result = MagicMock()

        if call_count == 1:
            # First call is TLC availability check
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        # Second call is actual model checking - no state count in output
        result.returncode = 0
        result.stdout = "Model checking completed successfully.\n"
        result.stderr = ""
        return result

    import subprocess

    monkeypatch.setattr(subprocess, "run", mock_run)

    spec = extract_spec(TestCounter)
    result = model_check(spec)

    assert result.passed
    assert result.states_generated == 0  # Default when not found


def test_model_check_non_digit_in_state_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test model_check when state line has non-digit words before the number."""
    from unittest.mock import MagicMock

    from weakincentives.formal.testing import model_check

    call_count = 0

    def mock_run(*args: Any, **kwargs: Any) -> MagicMock:  # noqa: ANN401
        nonlocal call_count
        call_count += 1

        result = MagicMock()

        if call_count == 1:
            # First call is TLC availability check
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        # Second call with state line that has non-digits before the number
        result.returncode = 0
        result.stdout = (
            "Model checking completed.\nTLC: 789 states generated successfully.\n"
        )
        result.stderr = ""
        return result

    import subprocess

    monkeypatch.setattr(subprocess, "run", mock_run)

    spec = extract_spec(TestCounter)
    result = model_check(spec)

    assert result.passed
    assert result.states_generated == 789

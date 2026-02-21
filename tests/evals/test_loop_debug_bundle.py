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

"""Tests for EvalLoop debug bundle functionality."""

from __future__ import annotations

from pathlib import Path

from tests.evals.conftest import (
    NoneOutputAdapter,
    NoneOutputLoop,
    Output,
    create_test_loop,
    output_to_str,
    session_aware_evaluator,
)
from weakincentives.evals import (
    BASELINE,
    EvalLoop,
    EvalRequest,
    EvalResult,
    Sample,
)
from weakincentives.runtime import InMemoryMailbox
from weakincentives.runtime.agent_loop import AgentLoopRequest, AgentLoopResult

# =============================================================================
# EvalLoop Debug Bundle Tests
# =============================================================================


def test_eval_loop_creates_debug_bundle(tmp_path: Path) -> None:
    """EvalLoop creates debug bundle when debug_bundle is configured."""
    from weakincentives.debug import DebugBundle
    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = create_test_loop(result="correct")
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-1", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        # Check result
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.sample_id == "sample-1"
        assert result.score.passed is True
        assert result.bundle_path is not None

        # Verify bundle exists and is valid
        assert result.bundle_path.exists()
        bundle = DebugBundle.load(result.bundle_path)

        # Verify bundle contains expected artifacts
        files = bundle.list_files()
        assert "manifest.json" in files
        assert "request/input.json" in files
        assert "request/output.json" in files
        assert "logs/app.jsonl" in files
        assert "metrics.json" in files
        assert "eval.json" in files
        # session/after.jsonl is only written if session has slices

        # Verify eval.json content
        eval_data = bundle.eval
        assert eval_data is not None
        assert eval_data["sample_id"] == "sample-1"
        assert eval_data["experiment_name"] == "baseline"
        assert eval_data["score"]["passed"] is True
        assert eval_data["score"]["value"] == 1.0
        assert eval_data["latency_ms"] >= 0

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_contains_request_id_directory(tmp_path: Path) -> None:
    """EvalLoop creates bundle in request-specific directory."""
    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = create_test_loop(result="correct")
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-1", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.bundle_path is not None

        # Bundle should be in a request_id subdirectory
        assert result.bundle_path.parent.parent == tmp_path
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_captures_failed_evaluation(tmp_path: Path) -> None:
    """EvalLoop bundle captures evaluation failures."""
    from weakincentives.debug import DebugBundle
    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        # Create loop that returns "wrong" - will fail exact_match
        agent_loop = create_test_loop(result="wrong")
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-fail", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.score.passed is False
        assert result.bundle_path is not None

        # Verify bundle captures the failed evaluation
        bundle = DebugBundle.load(result.bundle_path)
        eval_data = bundle.eval
        assert eval_data is not None
        assert eval_data["score"]["passed"] is False
        assert eval_data["score"]["value"] == 0.0

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_captures_none_output(tmp_path: Path) -> None:
    """EvalLoop bundle captures None output scenario."""
    from weakincentives.debug import DebugBundle
    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        # Use the NoneOutputAdapter
        adapter = NoneOutputAdapter()
        dummy_requests: InMemoryMailbox[
            AgentLoopRequest[str], AgentLoopResult[Output]
        ] = InMemoryMailbox(name="dummy-requests")
        agent_loop = NoneOutputLoop(adapter=adapter, requests=dummy_requests)
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-none", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.score.passed is False
        assert result.error == "No output from AgentLoop"
        assert result.bundle_path is not None

        # Verify bundle captures the error
        bundle = DebugBundle.load(result.bundle_path)
        eval_data = bundle.eval
        assert eval_data is not None
        assert eval_data["error"] == "No output from AgentLoop"
        assert eval_data["score"]["reason"] == "No output from AgentLoop"

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
        dummy_requests.close()


def test_eval_loop_no_bundle_without_config() -> None:
    """EvalLoop does not create bundle when debug_bundle is not set."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = create_test_loop(result="correct")
        # No debug_bundle configured
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample, experiment=BASELINE), reply_to=results)

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.bundle_path is None
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_with_session_aware_evaluator(tmp_path: Path) -> None:
    """EvalLoop bundle works with session-aware evaluators."""
    from weakincentives.debug import DebugBundle
    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = create_test_loop(result="correct")
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=session_aware_evaluator,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-session", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.score.passed is True
        assert result.bundle_path is not None

        # Verify bundle is valid
        bundle = DebugBundle.load(result.bundle_path)
        eval_data = bundle.eval
        assert eval_data is not None
        assert eval_data["score"]["passed"] is True

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_fallback_on_error(tmp_path: Path) -> None:
    """EvalLoop falls back to non-bundled path when bundle creation fails."""
    from unittest.mock import patch

    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = create_test_loop(result="correct")
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-fail-bundle", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        # Mock BundleWriter to raise an exception
        with patch(
            "weakincentives.debug._bundle_writer.BundleWriter.__enter__",
            side_effect=RuntimeError("Simulated bundle creation failure"),
        ):
            eval_loop.run(max_iterations=1)

        # Result should still be successful (just without bundle)
        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.sample_id == "sample-fail-bundle"
        assert result.score.passed is True
        assert result.bundle_path is None  # Bundle creation failed
        assert result.error is None  # But evaluation succeeded

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_no_reexecution_on_finalization_error(
    tmp_path: Path,
) -> None:
    """EvalLoop does NOT re-execute when bundle finalization fails after execution.

    This test verifies the fix for the data consistency bug where a post-execution
    failure (e.g., in bundle writing) would cause the sample to be re-executed,
    potentially returning different results and creating inconsistency.
    """
    import contextlib
    from collections.abc import Iterator
    from unittest.mock import patch

    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    execution_count = 0

    try:
        agent_loop = create_test_loop(result="correct")
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(
            id="sample-finalize-fail", input="test input", expected="correct"
        )
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        # Mock the context manager's __exit__ to raise after execution completes
        original_execute_with_bundle = agent_loop.execute_with_bundle

        @contextlib.contextmanager
        def failing_bundle_context(*args: object, **kwargs: object) -> Iterator[object]:
            nonlocal execution_count
            with original_execute_with_bundle(*args, **kwargs) as ctx:  # type: ignore[arg-type]
                execution_count += 1
                yield ctx
            # Raise during finalization (after yield returns)
            raise RuntimeError("Simulated finalization failure")

        with patch.object(agent_loop, "execute_with_bundle", failing_bundle_context):
            eval_loop.run(max_iterations=1)

        # Verify execution happened exactly once (no re-execution)
        assert execution_count == 1, f"Expected 1 execution, got {execution_count}"

        # Result should still be successful
        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.sample_id == "sample-finalize-fail"
        assert result.score.passed is True
        assert result.error is None  # Evaluation succeeded

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_invokes_storage_handler(tmp_path: Path) -> None:
    """EvalLoop invokes storage_handler when bundle is finalized."""
    from weakincentives.debug.bundle import BundleConfig, BundleManifest
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    # Track storage handler invocations
    stored_bundles: list[tuple[Path, BundleManifest]] = []

    class TestStorageHandler:
        """Test storage handler that records invocations."""

        def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
            stored_bundles.append((bundle_path, manifest))

    try:
        agent_loop = create_test_loop(result="correct")
        storage_handler = TestStorageHandler()
        config = EvalLoopConfig(
            debug_bundle=BundleConfig(target=tmp_path, storage_handler=storage_handler)
        )
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-upload", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        # Check result
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.sample_id == "sample-upload"
        assert result.score.passed is True
        assert result.bundle_path is not None

        # Verify storage handler was invoked
        assert len(stored_bundles) == 1
        stored_path, stored_manifest = stored_bundles[0]
        assert stored_path == result.bundle_path
        assert stored_manifest.bundle_id is not None
        assert stored_manifest.request.status == "success"

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_storage_handler_error_does_not_fail_eval(
    tmp_path: Path,
) -> None:
    """EvalLoop evaluation succeeds even if storage handler fails."""
    from weakincentives.debug.bundle import BundleConfig, BundleManifest
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    class FailingStorageHandler:
        """Storage handler that always fails."""

        def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
            raise RuntimeError("Simulated upload failure")

    try:
        agent_loop = create_test_loop(result="correct")
        config = EvalLoopConfig(
            debug_bundle=BundleConfig(
                target=tmp_path, storage_handler=FailingStorageHandler()
            )
        )
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-fail-upload", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        # Evaluation should still succeed
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.sample_id == "sample-fail-upload"
        assert result.score.passed is True
        assert result.error is None  # No eval error despite storage failure
        assert result.bundle_path is not None  # Bundle was still created locally

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()

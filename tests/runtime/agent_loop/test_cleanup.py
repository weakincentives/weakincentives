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

"""Tests for AgentLoop prompt cleanup lifecycle."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from weakincentives.prompt import Prompt
from weakincentives.runtime.agent_loop import (
    AgentLoopConfig,
    AgentLoopRequest,
    AgentLoopResult,
)
from weakincentives.runtime.mailbox import InMemoryMailbox

from .conftest import (
    MockAdapter,
    SampleLoop,
    SampleOutput,
    SampleRequest,
)


def test_loop_calls_prompt_cleanup_on_success() -> None:
    """AgentLoop calls prompt.cleanup() after successful execution."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
        requests.send(request, reply_to=results)

        with patch.object(Prompt, "cleanup") as mock_cleanup:
            loop.run(max_iterations=1, wait_time_seconds=0)
            assert mock_cleanup.call_count == 1

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_calls_prompt_cleanup_on_adapter_failure() -> None:
    """AgentLoop calls prompt.cleanup() even when adapter raises."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter(error=RuntimeError("boom"))
        loop = SampleLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
        requests.send(request, reply_to=results)

        with patch.object(Prompt, "cleanup") as mock_cleanup:
            loop.run(max_iterations=1, wait_time_seconds=0)
            assert mock_cleanup.call_count == 1

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is False
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_calls_prompt_cleanup_with_bundle(tmp_path: Path) -> None:
    """AgentLoop calls prompt.cleanup() after bundle artifacts are written."""
    from weakincentives.debug.bundle import BundleConfig

    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(
            request=SampleRequest(message="hello bundle cleanup")
        )
        requests.send(request, reply_to=results)

        with patch.object(Prompt, "cleanup") as mock_cleanup:
            loop.run(max_iterations=1, wait_time_seconds=0)
            assert mock_cleanup.call_count == 1

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        assert msgs[0].body.bundle_path is not None
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_calls_prompt_cleanup_with_bundle_on_failure(tmp_path: Path) -> None:
    """AgentLoop calls prompt.cleanup() in bundle error path."""
    from weakincentives.debug.bundle import BundleConfig

    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter(error=RuntimeError("execution failed"))
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=SampleRequest(message="hello bundle fail"))
        requests.send(request, reply_to=results)

        with patch.object(Prompt, "cleanup") as mock_cleanup:
            loop.run(max_iterations=1, wait_time_seconds=0)
            assert mock_cleanup.call_count == 1

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is False
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_bundle_reply_failure_does_not_double_cleanup(tmp_path: Path) -> None:
    """Bundle path failures after cleanup should not trigger cleanup twice."""
    from weakincentives.debug.bundle import BundleConfig

    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(
            request=SampleRequest(message="hello bundle reply fail")
        )
        requests.send(request, reply_to=results)

        with (
            patch(
                "weakincentives.runtime.agent_loop.reply_and_ack"
            ) as mock_reply_and_ack,
            patch.object(Prompt, "cleanup") as mock_cleanup,
        ):
            mock_reply_and_ack.side_effect = RuntimeError("reply failed")
            loop.run(max_iterations=1, wait_time_seconds=0)
            assert mock_cleanup.call_count == 1

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is False
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_execute_calls_prompt_cleanup() -> None:
    """AgentLoop.execute() calls prompt.cleanup()."""
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests)

        with patch.object(Prompt, "cleanup") as mock_cleanup:
            loop.execute(SampleRequest(message="direct"))
            assert mock_cleanup.call_count == 1
    finally:
        requests.close()


def test_execute_with_bundle_calls_prompt_cleanup(tmp_path: Path) -> None:
    """AgentLoop.execute_with_bundle() calls prompt.cleanup()."""
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests)

        with patch.object(Prompt, "cleanup") as mock_cleanup:
            with loop.execute_with_bundle(
                SampleRequest(message="bundle direct"),
                bundle_target=tmp_path,
            ) as ctx:
                assert ctx.response is not None
            assert mock_cleanup.call_count == 1
    finally:
        requests.close()

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

"""Tests for AgentLoop debug bundle lifecycle (creation, failure, filesystem)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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


def test_loop_with_debug_bundle_enabled(tmp_path: Path) -> None:
    """AgentLoop creates debug bundle when enabled in config."""
    from weakincentives.debug import DebugBundle
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

        request = AgentLoopRequest(request=SampleRequest(message="hello with bundle"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True

        # Bundle path should be set
        assert result.bundle_path is not None
        assert isinstance(result.bundle_path, Path)
        assert result.bundle_path.exists()

        # Bundle should be loadable and contain request data
        bundle = DebugBundle.load(result.bundle_path)
        assert bundle.manifest is not None
        assert bundle.request_input is not None
        assert bundle.request_output is not None
        assert bundle.run_context is not None

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_with_debug_bundle_includes_filesystem(tmp_path: Path) -> None:
    """AgentLoop includes filesystem snapshot in debug bundle when provided."""
    from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
    from weakincentives.debug import DebugBundle
    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.filesystem import Filesystem

    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        # Create filesystem with some files
        fs = InMemoryFilesystem()
        _ = fs.write("/test.txt", "Hello, World!")
        _ = fs.write("/subdir/nested.txt", "Nested content")

        adapter = MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(
            debug_bundle=bundle_config,
            resources={Filesystem: fs},
        )
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=SampleRequest(message="hello with fs"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle should contain filesystem files
        bundle = DebugBundle.load(result.bundle_path)
        files = bundle.list_files()
        filesystem_files = [f for f in files if f.startswith("filesystem/")]
        assert len(filesystem_files) > 0
        assert any("test.txt" in f for f in filesystem_files)
        assert any("nested.txt" in f for f in filesystem_files)

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_with_debug_bundle_no_filesystem_in_resources(tmp_path: Path) -> None:
    """AgentLoop handles debug bundle when resources exist but no Filesystem."""
    from weakincentives.debug import DebugBundle
    from weakincentives.debug.bundle import BundleConfig

    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        # Provide resources but without Filesystem
        class DummyResource:
            pass

        adapter = MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(
            debug_bundle=bundle_config,
            resources={DummyResource: DummyResource()},  # Resources but no Filesystem
        )
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=SampleRequest(message="hello without fs"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle should NOT contain filesystem files
        bundle = DebugBundle.load(result.bundle_path)
        files = bundle.list_files()
        filesystem_files = [f for f in files if f.startswith("filesystem/")]
        assert len(filesystem_files) == 0

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_without_debug_bundle() -> None:
    """AgentLoop works normally without debug bundle."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        # No debug_bundle in config
        config = AgentLoopConfig()
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(
            request=SampleRequest(message="hello without bundle")
        )
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        # Bundle path should be None when not enabled
        assert result.bundle_path is None

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_with_bundle_config_no_target() -> None:
    """AgentLoop falls back to unbundled when bundle target is None."""
    from weakincentives.debug.bundle import BundleConfig

    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        # BundleConfig with target=None
        bundle_config = BundleConfig(target=None)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(
            request=SampleRequest(message="hello with no target")
        )
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        # Bundle path should be None when target is not set
        assert result.bundle_path is None

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_with_bundle_failure_uses_handle_failure(tmp_path: Path) -> None:
    """AgentLoop uses handle_failure when bundle creation fails."""
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

        request = AgentLoopRequest(request=SampleRequest(message="hello bundle fail"))
        requests.send(request, reply_to=results)

        # Make BundleWriter raise an exception on enter
        def failing_enter(self: object) -> object:
            raise RuntimeError("Bundle creation failed")

        with patch(
            "weakincentives.debug._bundle_writer.BundleWriter.__enter__",
            failing_enter,
        ):
            loop.run(max_iterations=1, wait_time_seconds=0)

        # The error should have been handled via handle_failure path
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        # Result should indicate failure
        assert result.success is False
        assert result.error is not None
        assert "Bundle creation failed" in result.error

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_with_execution_failure_but_bundle_created(tmp_path: Path) -> None:
    """AgentLoop includes bundle_path in error response when execution fails."""
    from weakincentives.debug import DebugBundle
    from weakincentives.debug.bundle import BundleConfig

    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        # Adapter that raises an error during evaluate (after bundle is entered)
        adapter = MockAdapter(error=RuntimeError("Execution failed"))
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(
            request=SampleRequest(message="hello execution fail")
        )
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # The error should have been handled but bundle_path should still be set
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        # Result should indicate failure
        assert result.success is False
        assert result.error is not None
        assert "Execution failed" in result.error

        # Bundle path should be set even though execution failed
        assert result.bundle_path is not None
        assert isinstance(result.bundle_path, Path)
        assert result.bundle_path.exists()

        # Bundle should be loadable and contain error info
        bundle = DebugBundle.load(result.bundle_path)
        assert bundle.manifest is not None
        assert bundle.manifest.request.status == "error"
        assert bundle.request_input is not None

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_debug_bundle_calls_session_methods(tmp_path: Path) -> None:
    """AgentLoop attempts to write session snapshots to debug bundle."""
    from weakincentives.debug import DebugBundle
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
            request=SampleRequest(message="hello session methods")
        )
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle should be loadable and have standard files
        # Note: session files may be empty for fresh sessions with no state,
        # so we just verify the bundle was created successfully
        bundle = DebugBundle.load(result.bundle_path)
        files = bundle.list_files()
        # At minimum, the bundle should contain manifest, readme, and request files
        assert any("manifest.json" in f for f in files)
        assert any("request/input.json" in f for f in files)

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()

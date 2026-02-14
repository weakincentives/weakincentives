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

"""Tests for AgentLoop debug bundle content (metrics, prompt info, overrides)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from weakincentives.budget import Budget
from weakincentives.prompt import (
    SectionPath,
    SectionVisibility,
)
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


def test_loop_debug_bundle_includes_metrics(tmp_path: Path) -> None:
    """AgentLoop writes metrics.json to debug bundle with timing and token info."""
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

        request = AgentLoopRequest(request=SampleRequest(message="hello metrics"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle should contain metrics
        bundle = DebugBundle.load(result.bundle_path)
        assert bundle.metrics is not None
        assert "timing" in bundle.metrics
        assert "token_usage" in bundle.metrics
        assert "duration_ms" in bundle.metrics["timing"]

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_debug_bundle_metrics_include_budget(tmp_path: Path) -> None:
    """AgentLoop writes budget info to metrics.json when budget tracking is enabled."""
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
        budget = Budget(max_total_tokens=1000)
        config = AgentLoopConfig(debug_bundle=bundle_config, budget=budget)
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(
            request=SampleRequest(message="hello budget metrics")
        )
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle metrics should include budget info
        bundle = DebugBundle.load(result.bundle_path)
        assert bundle.metrics is not None
        assert "budget" in bundle.metrics
        assert "consumed" in bundle.metrics["budget"]
        assert "limits" in bundle.metrics["budget"]

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_debug_bundle_includes_prompt_info(tmp_path: Path) -> None:
    """AgentLoop writes prompt info (ns, key, adapter) to bundle manifest."""
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

        request = AgentLoopRequest(request=SampleRequest(message="hello prompt info"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle manifest should include prompt info
        bundle = DebugBundle.load(result.bundle_path)
        assert bundle.manifest is not None
        assert bundle.manifest.prompt.ns == "test"
        assert bundle.manifest.prompt.key == "test-prompt"
        # Adapter name is from the mock adapter class name since it's not a known adapter
        assert bundle.manifest.prompt.adapter is not None

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_get_adapter_name_uses_codex_canonical_name() -> None:
    """AgentLoop maps Codex adapter instances to canonical adapter name."""
    from weakincentives.adapters.codex_app_server import (
        CODEX_APP_SERVER_ADAPTER_NAME,
        CodexAppServerAdapter,
    )

    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        loop = SampleLoop(adapter=CodexAppServerAdapter(), requests=requests)
        assert loop._get_adapter_name() == CODEX_APP_SERVER_ADAPTER_NAME
    finally:
        requests.close()


def test_loop_debug_bundle_run_context_has_session_id(tmp_path: Path) -> None:
    """AgentLoop writes run_context.json with session_id after execution."""
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

        request = AgentLoopRequest(request=SampleRequest(message="hello run context"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle run_context should have session_id
        bundle = DebugBundle.load(result.bundle_path)
        assert bundle.run_context is not None
        assert "session_id" in bundle.run_context
        # session_id should match result
        assert bundle.run_context["session_id"] == str(result.session_id)

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_debug_bundle_includes_prompt_overrides(tmp_path: Path) -> None:
    """AgentLoop writes prompt_overrides.json when visibility overrides are set."""
    from weakincentives.debug import DebugBundle
    from weakincentives.debug.bundle import BundleConfig

    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        # Request visibility expansion to trigger visibility overrides
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
        ]
        adapter = MockAdapter(visibility_requests=visibility_requests)
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=SampleRequest(message="hello overrides"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle should contain prompt overrides
        bundle = DebugBundle.load(result.bundle_path)
        # prompt_overrides should have overrides dict with section key
        # Note: the file list should include prompt_overrides.json
        files = bundle.list_files()
        assert any("prompt_overrides.json" in f for f in files)
        assert bundle.prompt_overrides is not None
        assert "overrides" in bundle.prompt_overrides
        assert "section1" in bundle.prompt_overrides["overrides"]

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_debug_bundle_per_request_override(tmp_path: Path) -> None:
    """AgentLoop uses per-request debug_bundle config override."""
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
        # No debug_bundle in config
        config = AgentLoopConfig()
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        # But include debug_bundle in request
        bundle_config = BundleConfig(target=tmp_path)
        request = AgentLoopRequest(
            request=SampleRequest(message="hello request bundle"),
            debug_bundle=bundle_config,
        )
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True

        # Bundle should be created even though config didn't have it
        assert result.bundle_path is not None
        assert result.bundle_path.exists()

        bundle = DebugBundle.load(result.bundle_path)
        # Trigger should be "request" since it came from per-request config
        assert bundle.manifest.capture.trigger == "request"

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_handles_visibility_expansion_with_bundle(tmp_path: Path) -> None:
    """AgentLoop handles visibility expansion correctly with bundling enabled."""
    from weakincentives.debug import DebugBundle
    from weakincentives.debug.bundle import BundleConfig

    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
        ]
        adapter = MockAdapter(visibility_requests=visibility_requests)
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=SampleRequest(message="hello expansion"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Should succeed after visibility expansion
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        assert adapter._call_count == 2  # 1 visibility expansion + 1 success
        assert msgs[0].body.bundle_path is not None

        # Bundle should have visibility overrides recorded in prompt_overrides.json
        bundle = DebugBundle.load(msgs[0].body.bundle_path)
        files = bundle.list_files()
        assert any("prompt_overrides.json" in f for f in files)
        assert bundle.prompt_overrides is not None

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_with_debug_bundle_includes_environment(tmp_path: Path) -> None:
    """AgentLoop includes environment capture in debug bundle."""
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

        request = AgentLoopRequest(request=SampleRequest(message="hello environment"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        assert msgs[0].body.bundle_path is not None

        # Bundle should have environment files
        bundle = DebugBundle.load(msgs[0].body.bundle_path)
        files = bundle.list_files()
        assert any("environment/system.json" in f for f in files)
        assert any("environment/python.json" in f for f in files)
        assert any("environment/env_vars.json" in f for f in files)
        assert any("environment/command.txt" in f for f in files)

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()

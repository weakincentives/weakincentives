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

"""Tests for AgentLoop resource injection."""

from __future__ import annotations

from collections.abc import Mapping

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
    CustomResource,
    MockAdapter,
    SampleLoop,
    SampleOutput,
    SampleRequest,
)


def test_loop_passes_resources_from_config() -> None:
    """AgentLoop binds config resources to prompt."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        resource = CustomResource(name="config-resource")
        config = AgentLoopConfig(resources={CustomResource: resource})
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Resources are now bound to prompt and accessible via prompt.resources
        # (captured during evaluate while context is active)
        assert adapter._last_custom_resource is resource
    finally:
        requests.close()
        results.close()


def test_loop_request_resources_overrides_config() -> None:
    """AgentLoop request resources override config resources."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        config_resource = CustomResource(name="config-resource")
        config = AgentLoopConfig(resources={CustomResource: config_resource})
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        override_resource = CustomResource(name="override-resource")
        request = AgentLoopRequest(
            request=SampleRequest(message="hello"),
            resources={CustomResource: override_resource},
        )
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Override resources are bound to prompt, overriding config resources
        # (captured during evaluate while context is active)
        assert adapter._last_custom_resource is override_resource
    finally:
        requests.close()
        results.close()


def test_same_resources_used_across_visibility_retries() -> None:
    """Same resources are used across visibility expansion retries."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        resource = CustomResource(name="persistent-resource")
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
            {("section2",): SectionVisibility.FULL},
        ]
        adapter = MockAdapter(visibility_requests=visibility_requests)
        loop = SampleLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(
            request=SampleRequest(message="hello"),
            resources={CustomResource: resource},
        )
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Called 3 times: 2 visibility expansions + 1 success
        assert adapter._call_count == 3
        # Same resource should be used for all calls
        # (captured during evaluate while context is active)
        assert len(adapter._custom_resources) == 3
        assert all(r is resource for r in adapter._custom_resources)
    finally:
        requests.close()
        results.close()


def test_no_resources_when_not_set() -> None:
    """Prompt has empty resource context when no resources configured."""
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

        loop.run(max_iterations=1, wait_time_seconds=0)

        # When no resources configured, the custom resource is None
        # (captured during evaluate while context is active)
        assert adapter._last_custom_resource is None
    finally:
        requests.close()
        results.close()

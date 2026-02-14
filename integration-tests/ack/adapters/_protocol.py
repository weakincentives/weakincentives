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

"""Protocols and capability declarations for ACK adapter fixtures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.runtime.session import Session


@dataclass(slots=True, frozen=True)
class AdapterCapabilities:
    """Declares which ACK capabilities an adapter supports."""

    # Tier 1: Basic evaluation
    text_response: bool = True
    tool_invocation: bool = True
    structured_output: bool = True

    # Tier 2: Observability
    event_emission: bool = True
    transcript: bool = True
    rendered_tools_event: bool = False

    # Tier 3: Advanced behavior
    progressive_disclosure: bool = True
    transactional_tools: bool = True
    deadline_enforcement: bool = False
    budget_enforcement: bool = False

    # Adapter-specific scenarios
    native_tools: bool = False
    workspace_isolation: bool = False
    custom_env_forwarding: bool = False
    network_policy: bool = False
    sandbox_policy: bool = False
    skill_installation: bool = False


@runtime_checkable
class AdapterFixture(Protocol):
    """Fixture protocol implemented by each adapter integration harness."""

    @property
    def adapter_name(self) -> str:
        """Canonical adapter name."""
        ...

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Capabilities supported by this adapter in ACK."""
        ...

    def is_available(self) -> bool:
        """Return True when provider credentials/binary are available."""
        ...

    def create_adapter(self, tmp_path: Path) -> ProviderAdapter[object]:
        """Create a configured adapter rooted at ``tmp_path``."""
        ...

    def create_adapter_with_sandbox(
        self,
        tmp_path: Path,
        *,
        sandbox_mode: str,
    ) -> ProviderAdapter[object]:
        """Create an adapter with a specific sandbox mode.

        Args:
            tmp_path: Workspace root directory.
            sandbox_mode: Sandbox mode string. Adapters map this to their
                native sandbox configuration:
                - ``"read-only"``: No writes outside sandbox defaults.
                - ``"workspace-write"``: Writes allowed under ``tmp_path``.
        """
        ...

    def create_adapter_with_env(
        self,
        tmp_path: Path,
        *,
        env: Mapping[str, str],
    ) -> ProviderAdapter[object]:
        """Create an adapter that forwards extra environment variables.

        Args:
            tmp_path: Workspace root directory.
            env: Additional environment variables to forward.
        """
        ...

    def create_session(self) -> Session:
        """Create a fresh integration-test session."""
        ...

    def get_model(self) -> str:
        """Return the model identifier to use for this adapter."""
        ...

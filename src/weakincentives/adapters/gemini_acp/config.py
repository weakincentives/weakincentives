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

"""Configuration dataclasses for the Gemini CLI ACP adapter."""

from __future__ import annotations

from dataclasses import dataclass

from ..acp.config import ACPAdapterConfig, ACPClientConfig

__all__ = ["GeminiACPAdapterConfig", "GeminiACPClientConfig"]


@dataclass(slots=True, frozen=True)
class GeminiACPClientConfig(ACPClientConfig):
    """Gemini CLI-specific client configuration.

    Overrides defaults from :class:`ACPClientConfig` for the Gemini CLI
    binary (``gemini --experimental-acp``).
    """

    agent_bin: str = "gemini"
    agent_args: tuple[str, ...] = ("--experimental-acp",)
    startup_timeout_s: float = 15.0


@dataclass(slots=True, frozen=True)
class GeminiACPAdapterConfig(ACPAdapterConfig):
    """Gemini CLI-specific adapter configuration.

    Extends :class:`ACPAdapterConfig` with Gemini defaults and CLI flag
    fields for ``--approval-mode`` and ``--sandbox``.

    Attributes:
        sandbox: Enable OS-level sandboxing (``--sandbox`` flag).
            On macOS this uses seatbelt; on Linux it uses Docker/Podman.
            **Note:** ``--sandbox`` is incompatible with ``--experimental-acp``
            in Gemini v0.29.5 — the sandbox re-launches gemini via
            ``sandbox-exec``, breaking ACP's stdio pipe protocol.
        sandbox_profile: Seatbelt profile name (macOS only).  Passed via
            the ``SEATBELT_PROFILE`` environment variable.  Common values:
            ``"permissive-open"`` (default), ``"restrictive-open"``.
    """

    model_id: str | None = "gemini-2.5-flash"
    quiet_period_ms: int = 200
    emit_thought_chunks: bool = True
    approval_mode: str | None = None
    sandbox: bool = False
    sandbox_profile: str | None = None

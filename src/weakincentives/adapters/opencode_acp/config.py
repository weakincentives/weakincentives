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

"""Configuration dataclasses for the OpenCode ACP adapter."""

from __future__ import annotations

from dataclasses import dataclass

from ..acp.config import ACPAdapterConfig, ACPClientConfig

__all__ = ["OpenCodeACPAdapterConfig", "OpenCodeACPClientConfig"]


@dataclass(slots=True, frozen=True)
class OpenCodeACPClientConfig(ACPClientConfig):
    """OpenCode-specific client configuration.

    Inherits all fields from :class:`ACPClientConfig` with defaults
    already suited for the OpenCode ACP agent. Exists as a distinct
    type for isinstance dispatch.
    """


@dataclass(slots=True, frozen=True)
class OpenCodeACPAdapterConfig(ACPAdapterConfig):
    """OpenCode-specific adapter configuration.

    Inherits all fields from :class:`ACPAdapterConfig` with defaults
    already suited for the OpenCode ACP agent. Exists as a distinct
    type for isinstance dispatch.
    """

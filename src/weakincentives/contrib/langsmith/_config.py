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

"""Configuration dataclass for LangSmith integration."""

from __future__ import annotations

import os

from ...dataclasses import FrozenDataclass

_DEFAULT_API_URL = "https://api.smith.langchain.com"
_DEFAULT_PROJECT = "default"
_DEFAULT_UPLOAD_BATCH_SIZE = 100
_DEFAULT_UPLOAD_INTERVAL_SECONDS = 1.0
_DEFAULT_MAX_QUEUE_SIZE = 10000
_DEFAULT_CACHE_TTL_SECONDS = 300.0


@FrozenDataclass()
class LangSmithConfig:
    """Configuration for LangSmith integration.

    Settings can be provided directly or fall back to environment variables:

    - ``api_key``: Falls back to ``LANGCHAIN_API_KEY``
    - ``project``: Falls back to ``LANGCHAIN_PROJECT`` (default: ``"default"``)
    - ``api_url``: Falls back to ``LANGCHAIN_ENDPOINT``

    Example::

        config = LangSmithConfig(
            project="my-agent",
            tracing_enabled=True,
            trace_sample_rate=0.5,  # 50% sampling for high-volume
        )
    """

    # API settings
    api_key: str | None = None
    api_url: str = _DEFAULT_API_URL
    project: str | None = None

    # Telemetry settings
    tracing_enabled: bool = True
    trace_sample_rate: float = 1.0
    async_upload: bool = True
    upload_batch_size: int = _DEFAULT_UPLOAD_BATCH_SIZE
    upload_interval_seconds: float = _DEFAULT_UPLOAD_INTERVAL_SECONDS
    max_queue_size: int = _DEFAULT_MAX_QUEUE_SIZE
    flush_on_exit: bool = True

    # Tool tracing for Claude Agent SDK
    trace_native_tools: bool = True

    # Hub settings
    hub_enabled: bool = True
    cache_ttl_seconds: float = _DEFAULT_CACHE_TTL_SECONDS
    cache_versioned_indefinitely: bool = True

    def resolved_api_key(self) -> str | None:
        """Return API key, falling back to environment variable."""
        return self.api_key or os.getenv("LANGCHAIN_API_KEY")

    def resolved_project(self) -> str:
        """Return project name, falling back to environment variable."""
        return self.project or os.getenv("LANGCHAIN_PROJECT") or _DEFAULT_PROJECT

    def resolved_api_url(self) -> str:
        """Return API URL, falling back to environment variable."""
        return os.getenv("LANGCHAIN_ENDPOINT") or self.api_url

    def is_tracing_enabled(self) -> bool:
        """Check if tracing is enabled based on config and environment.

        Tracing is enabled when:
        - ``tracing_enabled`` is ``True`` AND
        - ``LANGCHAIN_TRACING_V2`` is not explicitly set to ``"false"``
        """
        if not self.tracing_enabled:
            return False
        env_tracing = os.getenv("LANGCHAIN_TRACING_V2", "").lower()
        return env_tracing != "false"


__all__ = ["LangSmithConfig"]

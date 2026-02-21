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

"""Environment variable helpers for the ACP adapter."""

from __future__ import annotations

__all__ = ["build_env"]


def build_env(env_config: dict[str, str] | None) -> dict[str, str] | None:
    """Build merged environment variables.

    When ``env_config`` is set, the full ``os.environ`` is forwarded with
    config entries taking precedence.  This mirrors stdlib
    ``subprocess.Popen`` behaviour where ``env=None`` inherits the parent
    environment.  Returning ``None`` (no config env) lets the subprocess
    inherit the parent env via the stdlib default.
    """
    if not env_config:
        return None
    import os

    return {**os.environ, **env_config}

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

"""Debug bundle utilities for capturing and inspecting execution state.

The primary API consists of:
- ``BundleConfig``: Configure automatic bundle creation in MainLoop
- ``BundleWriter``: Create debug bundles programmatically
- ``DebugBundle``: Load and inspect existing bundles

Example - MainLoop integration::

    from weakincentives.debug import BundleConfig
    from weakincentives.runtime import MainLoop, MainLoopConfig

    config = MainLoopConfig(
        debug_bundle=BundleConfig(
            target="./debug_bundles/",
        ),
    )
    loop = MyLoop(adapter=adapter, requests=requests, config=config)

Example - Manual bundle creation::

    from weakincentives.debug import BundleWriter

    with BundleWriter(target="./debug/", bundle_id=run_id) as writer:
        writer.write_request_input(request)
        with writer.capture_logs():
            response = adapter.evaluate(prompt, session=session)
        writer.write_request_output(response)
        writer.write_session_after(session)

Example - Inspect existing bundle::

    from weakincentives.debug import DebugBundle

    bundle = DebugBundle.load("./debug/abc123.zip")
    print(bundle.manifest.request.status)
    print(bundle.request_input)
    print(bundle.logs)
"""

from __future__ import annotations

from .bundle import (
    BundleConfig,
    BundleError,
    BundleManifest,
    BundleValidationError,
    BundleWriter,
    DebugBundle,
)

__all__ = [
    "BundleConfig",
    "BundleError",
    "BundleManifest",
    "BundleValidationError",
    "BundleWriter",
    "DebugBundle",
]

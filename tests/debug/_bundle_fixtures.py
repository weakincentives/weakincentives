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

"""Shared fixtures and helpers for debug bundle tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from weakincentives.debug.bundle import BundleConfig, BundleWriter, DebugBundle

WriteFn = Callable[[BundleWriter], None]


def create_bundle(
    tmp_path: Path,
    *,
    config: BundleConfig | None = None,
    write_fn: WriteFn | None = None,
) -> DebugBundle:
    """Create a bundle and return the loaded DebugBundle."""
    with BundleWriter(tmp_path, config=config) as writer:
        if write_fn is not None:
            write_fn(writer)

    assert writer.path is not None
    return DebugBundle.load(writer.path)


def create_bundle_path(
    tmp_path: Path,
    *,
    config: BundleConfig | None = None,
    write_fn: WriteFn | None = None,
) -> Path:
    """Create a bundle and return the bundle path."""
    with BundleWriter(tmp_path, config=config) as writer:
        if write_fn is not None:
            write_fn(writer)

    assert writer.path is not None
    return writer.path

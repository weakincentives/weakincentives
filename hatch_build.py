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

"""Hatch build hook to synchronize documentation files into the package."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class DocsSyncHook(BuildHookInterface):
    """Synchronize documentation files into package before build."""

    PLUGIN_NAME = "docs-sync"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Copy documentation files into the package directory."""
        root = Path(self.root)
        docs_dir = root / "src" / "weakincentives" / "docs"
        specs_dir = docs_dir / "specs"

        # Create directories
        docs_dir.mkdir(parents=True, exist_ok=True)
        specs_dir.mkdir(exist_ok=True)

        # Copy documentation files
        shutil.copy(root / "llms.md", docs_dir / "llms.md")
        shutil.copy(root / "WINK_GUIDE.md", docs_dir / "WINK_GUIDE.md")

        # Copy all spec files
        for spec_file in (root / "specs").glob("*.md"):
            shutil.copy(spec_file, specs_dir / spec_file.name)

        # Ensure __init__.py exists
        init_file = docs_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()

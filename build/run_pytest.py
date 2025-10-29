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

"""Pytest runner that emits output only for errors and warnings."""

from __future__ import annotations

import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import pytest


def main() -> int:
    args = sys.argv[1:]
    buffer = StringIO()

    with redirect_stdout(buffer), redirect_stderr(buffer):
        code = pytest.main(args)

    output = buffer.getvalue()
    if code != 0 or "warning" in output.lower():
        print(output, end="")

    return code


if __name__ == "__main__":  # pragma: no cover - exercised via Makefile target
    sys.exit(main())

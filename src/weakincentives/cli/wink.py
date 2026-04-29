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

"""``wink`` CLI entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO, cast

__all__ = ["main"]


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """Run the ``wink`` CLI.

    Returns the exit code; callers can hand stdout to a writer for tests.
    """
    out = stdout if stdout is not None else sys.stdout
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "debug":
        return _run_debug(Path(cast("str", args.path)), out=out)
    parser.print_help(out)  # pragma: no cover - argparse forbids missing
    return 2  # pragma: no cover - argparse exits earlier


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wink",
        description="Inspect debug bundles produced by weakincentives.debug.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    debug = sub.add_parser("debug", help="Pretty-print a debug bundle JSON file.")
    _ = debug.add_argument("path", help="Path to the debug bundle JSON.")
    return parser


def _run_debug(path: Path, *, out: TextIO) -> int:
    if not path.is_file():
        print(f"wink: debug bundle not found: {path}", file=out)
        return 1
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        print(f"wink: invalid JSON in {path}: {error}", file=out)
        return 1
    if not isinstance(payload, dict):
        print(f"wink: bundle payload must be an object: {path}", file=out)
        return 1
    bundle = cast("dict[str, Any]", payload)
    _print_bundle(bundle, out=out)
    return 0


def _print_bundle(bundle: dict[str, Any], *, out: TextIO) -> None:
    print(f"schema_version: {bundle.get('schema_version', '?')}", file=out)
    print(f"created_at:     {bundle.get('created_at', '?')}", file=out)
    metadata = cast("dict[str, Any]", bundle.get("metadata") or {})
    if metadata:
        print("metadata:", file=out)
        for key in sorted(metadata):
            print(f"  {key} = {metadata[key]!r}", file=out)
    entries = cast("list[dict[str, Any]]", bundle.get("entries") or [])
    print(f"entries: {len(entries)}", file=out)
    for entry in entries:
        timestamp = entry.get("timestamp", "?")
        kind = entry.get("kind", "?")
        print(f"  [{timestamp}] {kind}", file=out)
    snapshot = cast("dict[str, Any]", bundle.get("snapshot") or {})
    slices = cast("list[dict[str, Any]]", snapshot.get("slices") or [])
    print(f"slices: {len(slices)}", file=out)
    for slice_data in slices:
        type_id = slice_data.get("type", "?")
        items = cast("list[Any]", slice_data.get("items") or [])
        print(f"  {type_id} ({len(items)} items)", file=out)


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())

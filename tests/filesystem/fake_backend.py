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

"""Toy ``FilesystemBackend`` proving the narrow waist stays narrow.

The whole backend — including snapshots — fits in under 100 lines (see
``test_fake_backend_is_under_100_lines``). The full validation suites run
against it through the ``Filesystem`` facade.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import accumulate
from uuid import uuid4

from weakincentives.errors import SnapshotRestoreError
from weakincentives.filesystem import (
    FileEntry,
    FileStat,
    GlobMatch,
    GrepMatch,
    SnapshotRef,
    WriteMode,
    glob_match,
    is_path_under,
    now,
)


class FakeBackend:
    """Minimal in-memory filesystem backend (files dict + dirs set)."""

    def __init__(self, *, read_only: bool = False) -> None:
        self._files: dict[str, bytes] = {}
        self._dirs, self._read_only = {""}, read_only
        self._snapshots: dict[str, tuple[dict[str, bytes], set[str]]] = {}

    @property
    def root(self) -> str:
        return "/"

    @property
    def read_only(self) -> bool:
        return self._read_only

    def stat(self, path: str) -> FileStat:
        if path in self._dirs:
            return FileStat(path or "/", False, True, 0)
        if path not in self._files:
            raise FileNotFoundError(path)
        return FileStat(path, True, False, len(self._files[path]), now(), now())

    def list(self, path: str) -> Sequence[FileEntry]:
        prefix = f"{path}/" if path else ""
        names = {
            c[len(prefix) :].split("/")[0]
            for c in [*self._files, *self._dirs]
            if c != path and is_path_under(c, path)
        }
        return [
            FileEntry(n, (p := prefix + n), p in self._files, p not in self._files)
            for n in sorted(names)
        ]

    def glob(self, pattern: str, *, path: str) -> Sequence[GlobMatch]:
        found = (p for p in self._files if glob_match(p, pattern, path))
        return sorted((GlobMatch(p, True) for p in found), key=lambda m: m.path)

    def grep(
        self, pattern: str, *, path: str, glob: str | None, max_matches: int
    ) -> Sequence[GrepMatch]:
        regex, matches = re.compile(pattern), list[GrepMatch]()
        for file_path in sorted(self._files):
            if not glob_match(file_path, glob or "*", path):
                continue
            try:
                lines = self._files[file_path].decode("utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            matches += [
                GrepMatch(file_path, n, line, m.start(), m.end())
                for n, line in enumerate(lines, 1)
                if (m := regex.search(line))
            ]
            if len(matches) >= max_matches:
                break
        return matches[:max_matches]

    def read_range(self, path: str, *, offset: int, length: int | None) -> bytes:
        if path in self._dirs:
            raise IsADirectoryError(f"Is a directory: {path}")
        if path not in self._files:
            raise FileNotFoundError(path)
        return self._files[path][offset : None if length is None else offset + length]

    def write(self, path: str, data: bytes, *, mode: WriteMode) -> None:
        if mode == "create" and path in self._files:
            raise FileExistsError(f"File already exists: {path}")
        data = self._files.get(path, b"") + data if mode == "append" else data
        self.mkdir(path.rsplit("/", 1)[0] if "/" in path else "")
        self._files[path] = data

    def delete(self, path: str, *, recursive: bool) -> None:
        if path in self._files:
            del self._files[path]
            return
        if path not in self._dirs:
            raise FileNotFoundError(path)
        nested = [p for p in [*self._files, *self._dirs] if is_path_under(p, path)]
        if len(nested) > 1 and not recursive:
            raise IsADirectoryError(f"Directory not empty: {path}")
        for nested_path in nested:
            _ = self._files.pop(nested_path, None)
            self._dirs.discard(nested_path)

    def mkdir(self, path: str) -> None:
        self._dirs.update(accumulate(path.split("/"), lambda a, b: f"{a}/{b}"))

    def rename(self, src: str, dst: str) -> None:
        if dst in self._files or dst in self._dirs:
            raise FileExistsError(f"File already exists: {dst}")
        moved = [p for p in [*self._files, *self._dirs] if is_path_under(p, src)]
        if not moved:
            raise FileNotFoundError(src)
        self.mkdir(dst.rsplit("/", 1)[0] if "/" in dst else "")
        for path in moved:
            target = dst + path[len(src) :]
            if path in self._files:
                self._files[target] = self._files.pop(path)
            else:
                self._dirs.discard(path)
                self._dirs.add(target)

    def snapshot(self, *, tag: str | None = None) -> SnapshotRef:
        token = f"fake-{len(self._snapshots)}"
        self._snapshots[token] = (dict(self._files), set(self._dirs))
        return SnapshotRef(uuid4(), now(), token, tag)

    def restore(self, ref: SnapshotRef) -> None:
        if ref.token not in self._snapshots:
            raise SnapshotRestoreError(f"Unknown snapshot: {ref.token}")
        files, dirs = self._snapshots[ref.token]
        self._files, self._dirs = dict(files), set(dirs)


__all__ = ["FakeBackend"]

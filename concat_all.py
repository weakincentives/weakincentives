#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def iter_project_files(target: Path) -> list[Path]:
    if not target.exists():
        return []
    return [path for path in sorted(target.rglob("*")) if path.is_file()]


def emit_file(path: Path, project_root: Path) -> None:
    relative_path = path.relative_to(project_root).as_posix()
    sys.stdout.write(f"{relative_path}\n")
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        contents = stream.read()
    sys.stdout.write(contents)
    if not contents.endswith("\n"):
        sys.stdout.write("\n")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    prelude_files = [project_root / "README.md", project_root / "ROADMAP.md"]
    postlude_files = [
        project_root / "pyproject.toml",
        project_root / "openai_example.py",
    ]
    targets = [project_root / "specs", project_root / "src"]

    for path in prelude_files:
        if path.exists():
            emit_file(path, project_root)

    for target in targets:
        for path in iter_project_files(target):
            emit_file(path, project_root)

    for path in postlude_files:
        if path.exists():
            emit_file(path, project_root)


if __name__ == "__main__":
    main()

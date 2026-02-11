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

"""Git repository state capture for debug bundles.

Captures git metadata (commit, branch, remotes, dirty status) and uncommitted
changes (tracked diffs plus untracked file contents) needed to reproduce the
exact working-tree state at bundle-creation time.

All credential material in remote URLs is automatically redacted, and files
matching sensitive patterns (keys, tokens, .env) are excluded from untracked
file capture.
"""

from __future__ import annotations

import logging
import re
import subprocess  # nosec B404 - needed for git introspection
from collections.abc import Mapping
from dataclasses import field
from pathlib import Path

from ..dataclasses import FrozenDataclass

_logger = logging.getLogger(__name__)

# Maximum git diff size in bytes (100KB)
_MAX_GIT_DIFF_SIZE = 100_000

# Minimum expected parts in git remote output lines
_MIN_REMOTE_PARTS = 2

# Maximum size per untracked file in bytes (10KB)
_MAX_UNTRACKED_FILE_SIZE = 10_000

# Patterns for sensitive files that should be excluded from untracked capture
# Patterns use (^|/) to match at start of path or after directory separator
_SENSITIVE_FILE_PATTERNS = (
    re.compile(r"(^|/)\.env($|\.)", re.IGNORECASE),  # .env, .env.local, .env.production
    re.compile(r"(^|/)credentials?\.json$", re.IGNORECASE),
    re.compile(r"(^|/)secrets?\.json$", re.IGNORECASE),
    re.compile(r"(^|/)secrets?\.ya?ml$", re.IGNORECASE),
    re.compile(r"\.pem$", re.IGNORECASE),
    re.compile(r"\.key$", re.IGNORECASE),
    re.compile(r"\.p12$", re.IGNORECASE),
    re.compile(r"\.pfx$", re.IGNORECASE),
    re.compile(r"(^|/)id_rsa", re.IGNORECASE),
    re.compile(r"(^|/)id_ed25519", re.IGNORECASE),
    re.compile(r"(^|/)\.ssh/", re.IGNORECASE),
    re.compile(r"(^|/)\.netrc$", re.IGNORECASE),
    re.compile(r"(^|/)\.npmrc$", re.IGNORECASE),
    re.compile(r"(^|/)\.pypirc$", re.IGNORECASE),
    re.compile(r"(^|/)token", re.IGNORECASE),
)


@FrozenDataclass()
class GitInfo:
    """Git repository state.

    Attributes:
        repo_root: Path to the repository root.
        commit_sha: Current HEAD commit SHA.
        commit_short: Short commit SHA (first 8 characters).
        branch: Current branch name, or None if detached HEAD.
        is_dirty: True if working tree has uncommitted changes.
        remotes: Mapping of remote names to URLs.
        tags: List of tags pointing to HEAD.
    """

    repo_root: str = ""
    commit_sha: str = ""
    commit_short: str = ""
    branch: str | None = None
    is_dirty: bool = False
    remotes: Mapping[str, str] = field(default_factory=lambda: {})
    tags: tuple[str, ...] = ()


def _run_git_command(*args: str, cwd: Path | None = None) -> str | None:
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(  # nosec B603 B607 - trusted git command
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def _redact_url_credentials(url: str) -> str:
    """Redact credentials from a URL (e.g., https://user:token@host/...).

    Returns the URL with any userinfo (username:password) replaced with [REDACTED].
    """
    # Match URLs with credentials: scheme://user:pass@host or scheme://user@host
    # Pattern captures: (scheme://)(userinfo@)(rest)
    pattern = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*://)[^@/]+@(.+)$")
    match = pattern.match(url)
    if match:
        return f"{match.group(1)}[REDACTED]@{match.group(2)}"
    return url


def _is_sensitive_file(filename: str) -> bool:
    """Check if a filename matches sensitive file patterns."""
    return any(pattern.search(filename) for pattern in _SENSITIVE_FILE_PATTERNS)


def _read_file_with_limit(file_path: Path) -> tuple[str, bool]:
    """Read file content with size limit, streaming only the needed bytes.

    Returns (content, truncated) tuple.
    """
    file_size = file_path.stat().st_size
    if file_size > _MAX_UNTRACKED_FILE_SIZE:
        # Stream only the first N bytes to avoid loading large files into memory
        with file_path.open("rb") as f:
            content = f.read(_MAX_UNTRACKED_FILE_SIZE).decode(errors="replace")
        return content, True
    return file_path.read_text(errors="replace"), False


def _format_untracked_file_diff(filename: str, content: str, truncated: bool) -> str:
    """Format a single untracked file as unified diff."""
    parts = [
        f"\ndiff --git a/{filename} b/{filename}",
        "new file mode 100644",
        "--- /dev/null",
        f"+++ b/{filename}",
    ]
    lines = content.splitlines(keepends=True)
    if lines:
        parts.append(f"@@ -0,0 +1,{len(lines)} @@")
        parts.extend(f"+{line.rstrip()}" for line in lines)
    if truncated:
        parts.append(f"+[TRUNCATED: file exceeded {_MAX_UNTRACKED_FILE_SIZE}B]")
    return "\n".join(parts)


def _capture_untracked_files(working_dir: Path) -> str:
    """Capture untracked file names and their contents.

    Returns a unified-diff-style representation of untracked files.
    Sensitive files (credentials, keys, etc.) are excluded.
    Large files are truncated to _MAX_UNTRACKED_FILE_SIZE.
    """
    untracked_output = _run_git_command(
        "ls-files", "--others", "--exclude-standard", cwd=working_dir
    )
    if not untracked_output:
        return ""

    untracked_files = [f.strip() for f in untracked_output.splitlines() if f.strip()]
    if not untracked_files:
        return ""

    parts: list[str] = ["\n# Untracked files:"]
    for filename in untracked_files:
        if _is_sensitive_file(filename):
            parts.append(f"\n# {filename}: [excluded - sensitive file]")
            continue

        file_path = working_dir / filename
        try:
            if not file_path.is_file():
                continue
            content, truncated = _read_file_with_limit(file_path)
            parts.append(_format_untracked_file_diff(filename, content, truncated))
        except (OSError, UnicodeDecodeError):
            parts.append(f"\n# {filename}: [unable to read]")

    return "\n".join(parts)


def capture_git_info(working_dir: Path | None = None) -> GitInfo | None:
    """Capture git repository state."""
    cwd = working_dir or Path.cwd()

    # Check if we're in a git repo
    repo_root = _run_git_command("rev-parse", "--show-toplevel", cwd=cwd)
    if repo_root is None:
        return None

    commit_sha = _run_git_command("rev-parse", "HEAD", cwd=cwd) or ""
    commit_short = commit_sha[:8] if commit_sha else ""

    # Get branch name (may be None for detached HEAD)
    branch = _run_git_command("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
    if branch == "HEAD":
        branch = None

    # Check if dirty
    status = _run_git_command("status", "--porcelain", cwd=cwd)
    is_dirty = bool(status)

    # Get remotes (with credential redaction)
    remotes: dict[str, str] = {}
    remote_output = _run_git_command("remote", "-v", cwd=cwd)
    if remote_output:
        for line in remote_output.splitlines():
            parts = line.split()
            if len(parts) >= _MIN_REMOTE_PARTS and "(fetch)" in line:
                remotes[parts[0]] = _redact_url_credentials(parts[1])

    # Get tags pointing to HEAD
    tags: list[str] = []
    tag_output = _run_git_command("tag", "--points-at", "HEAD", cwd=cwd)
    if tag_output:
        tags = [t.strip() for t in tag_output.splitlines() if t.strip()]

    return GitInfo(
        repo_root=repo_root,
        commit_sha=commit_sha,
        commit_short=commit_short,
        branch=branch,
        is_dirty=is_dirty,
        remotes=remotes,
        tags=tuple(tags),
    )


def capture_git_diff(working_dir: Path | None = None) -> str | None:
    """Capture uncommitted git changes including untracked files.

    Captures both tracked file changes (staged and unstaged) and untracked files.
    Output is capped at _MAX_GIT_DIFF_SIZE bytes.
    """
    cwd = working_dir or Path.cwd()

    # Get staged and unstaged changes to tracked files
    diff = _run_git_command("diff", "HEAD", cwd=cwd)
    if diff is None:
        return None

    # Capture untracked files
    untracked = _capture_untracked_files(cwd)
    combined = diff + untracked if untracked else diff

    if len(combined) > _MAX_GIT_DIFF_SIZE:
        truncation_msg = f"\n\n[TRUNCATED: diff exceeded {_MAX_GIT_DIFF_SIZE} bytes]"
        return combined[: _MAX_GIT_DIFF_SIZE - len(truncation_msg)] + truncation_msg

    return combined if combined else None

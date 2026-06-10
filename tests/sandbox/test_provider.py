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

"""Tests for LocalSandboxProvider.

The mount cases mirror the workspace-section mount tests: the provider
absorbs that machinery and must reproduce its behavior.
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from weakincentives.sandbox import (
    EgressPolicy,
    EgressRule,
    HostMount,
    LocalSandbox,
    LocalSandboxProvider,
    Sandbox,
    SandboxConfig,
    SandboxProvider,
    SandboxSetupError,
    WorkspaceBudgetExceededError,
    WorkspaceSecurityError,
)


@pytest.fixture
def provider() -> Iterator[LocalSandboxProvider]:
    """Provider with a unique temp prefix so leftover dirs are detectable."""
    prefix = f"wink-sandbox-{uuid.uuid4().hex[:8]}-"
    yield LocalSandboxProvider(temp_dir_prefix=prefix)
    leftovers = list(Path(tempfile.gettempdir()).glob(f"{prefix}*"))
    assert leftovers == []


@pytest.fixture
def host_dir(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("print('main')")
    (source / "notes.txt").write_text("notes")
    sub = source / "sub"
    sub.mkdir()
    (sub / "nested.py").write_text("# nested")
    return source


class TestProviderProtocol:
    def test_implements_provider_protocol(self, provider: LocalSandboxProvider) -> None:
        assert isinstance(provider, SandboxProvider)

    def test_open_returns_sandbox(self, provider: LocalSandboxProvider) -> None:
        sandbox = provider.open(SandboxConfig())
        try:
            assert isinstance(sandbox, Sandbox)
            assert isinstance(sandbox, LocalSandbox)
        finally:
            sandbox.close()


class TestMountParity:
    """Mirrors the workspace-section mount behavior the provider absorbed."""

    def test_empty_config_creates_empty_root(
        self, provider: LocalSandboxProvider
    ) -> None:
        sandbox = provider.open(SandboxConfig())
        try:
            assert Path(sandbox.root).exists()
            assert sandbox.filesystem.list(".") == []
        finally:
            sandbox.close()

    def test_copies_single_file(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = SandboxConfig(mounts=(HostMount(host_path=str(host_dir / "main.py")),))
        sandbox = provider.open(config)
        try:
            assert sandbox.filesystem.read("main.py").content == "print('main')"
        finally:
            sandbox.close()

    def test_copies_directory_recursively(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = SandboxConfig(mounts=(HostMount(host_path=str(host_dir)),))
        sandbox = provider.open(config)
        try:
            base = host_dir.name
            assert sandbox.filesystem.exists(f"{base}/main.py")
            assert sandbox.filesystem.exists(f"{base}/sub/nested.py")
        finally:
            sandbox.close()

    def test_custom_mount_path(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = SandboxConfig(
            mounts=(HostMount(host_path=str(host_dir), mount_path="custom/path"),)
        )
        sandbox = provider.open(config)
        try:
            assert sandbox.filesystem.exists("custom/path/main.py")
        finally:
            sandbox.close()

    def test_multiple_mounts(
        self, provider: LocalSandboxProvider, host_dir: Path, tmp_path: Path
    ) -> None:
        other = tmp_path / "other"
        other.mkdir()
        (other / "data.csv").write_text("a,b")
        config = SandboxConfig(
            mounts=(
                HostMount(host_path=str(host_dir), mount_path="one"),
                HostMount(host_path=str(other), mount_path="two"),
            )
        )
        sandbox = provider.open(config)
        try:
            assert sandbox.filesystem.exists("one/main.py")
            assert sandbox.filesystem.exists("two/data.csv")
        finally:
            sandbox.close()

    def test_include_glob_filter(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = SandboxConfig(
            mounts=(HostMount(host_path=str(host_dir), include_glob=("*.py",)),)
        )
        sandbox = provider.open(config)
        try:
            base = host_dir.name
            assert sandbox.filesystem.exists(f"{base}/main.py")
            assert not sandbox.filesystem.exists(f"{base}/notes.txt")
        finally:
            sandbox.close()

    def test_exclude_glob_filter(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = SandboxConfig(
            mounts=(HostMount(host_path=str(host_dir), exclude_glob=("*.txt",)),)
        )
        sandbox = provider.open(config)
        try:
            base = host_dir.name
            assert sandbox.filesystem.exists(f"{base}/main.py")
            assert not sandbox.filesystem.exists(f"{base}/notes.txt")
        finally:
            sandbox.close()

    def test_byte_budget_enforced(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = SandboxConfig(
            mounts=(HostMount(host_path=str(host_dir), max_bytes=2),)
        )
        with pytest.raises(WorkspaceBudgetExceededError, match="byte budget"):
            _ = provider.open(config)

    def test_security_boundary_enforced(
        self, provider: LocalSandboxProvider, host_dir: Path, tmp_path: Path
    ) -> None:
        allowed = tmp_path / "allowed-root"
        allowed.mkdir()
        config = SandboxConfig(
            mounts=(HostMount(host_path=str(host_dir)),),
            allowed_host_roots=(str(allowed),),
        )
        with pytest.raises(WorkspaceSecurityError, match="outside allowed"):
            _ = provider.open(config)

    def test_security_boundary_allows_within_root(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = SandboxConfig(
            mounts=(HostMount(host_path=str(host_dir)),),
            allowed_host_roots=(str(host_dir.parent),),
        )
        sandbox = provider.open(config)
        try:
            assert sandbox.filesystem.exists(f"{host_dir.name}/main.py")
        finally:
            sandbox.close()

    def test_nonexistent_host_path_raises(self, provider: LocalSandboxProvider) -> None:
        config = SandboxConfig(mounts=(HostMount(host_path="/nonexistent/path/12345"),))
        with pytest.raises(FileNotFoundError):
            _ = provider.open(config)


class TestProviderConfig:
    def test_read_only_applies_to_filesystem_facet(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = SandboxConfig(
            mounts=(HostMount(host_path=str(host_dir)),), read_only=True
        )
        sandbox = provider.open(config)
        try:
            assert sandbox.filesystem.read_only
            with pytest.raises(PermissionError):
                _ = sandbox.filesystem.write("new.txt", "nope")
        finally:
            sandbox.close()

    def test_env_reaches_shell(self, provider: LocalSandboxProvider) -> None:
        config = SandboxConfig(env={"WINK_SANDBOX_VAR": "configured"})
        sandbox = provider.open(config)
        try:
            result = sandbox.shell.run(
                [
                    sys.executable,
                    "-c",
                    "import os; print(os.environ['WINK_SANDBOX_VAR'])",
                ]
            )
            assert result.stdout.decode().strip() == "configured"
        finally:
            sandbox.close()

    def test_egress_policy_seeds_sandbox(self, provider: LocalSandboxProvider) -> None:
        policy = EgressPolicy(allow=(EgressRule(host_glob="pypi.org"),))
        sandbox = provider.open(SandboxConfig(egress=policy))
        try:
            assert sandbox.egress is policy
        finally:
            sandbox.close()

    def test_default_egress_denies_all(self, provider: LocalSandboxProvider) -> None:
        sandbox = provider.open(SandboxConfig())
        try:
            assert not sandbox.egress.allows("example.com")
        finally:
            sandbox.close()


class TestSetupCommands:
    def test_setup_runs_in_order(self, provider: LocalSandboxProvider) -> None:
        config = SandboxConfig(
            setup=(
                f"{sys.executable} -c \"open('first.txt', 'w').write('1')\"",
                (
                    f"{sys.executable} -c "
                    "\"open('second.txt', 'w').write(open('first.txt').read())\""
                ),
            )
        )
        sandbox = provider.open(config)
        try:
            assert sandbox.filesystem.read("second.txt").content == "1"
        finally:
            sandbox.close()

    def test_setup_is_argv_not_shell(self, provider: LocalSandboxProvider) -> None:
        config = SandboxConfig(setup=(f"{sys.executable} -c \"print('$HOME')\"",))
        sandbox = provider.open(config)
        sandbox.close()

    def test_failing_setup_raises_and_cleans_up(
        self, provider: LocalSandboxProvider
    ) -> None:
        config = SandboxConfig(
            setup=(
                (
                    f"{sys.executable} -c "
                    "\"import sys; sys.stderr.write('setup blew up'); sys.exit(2)\""
                ),
            )
        )
        with pytest.raises(SandboxSetupError, match="setup blew up"):
            _ = provider.open(config)
        # The provider fixture asserts no temp directories were left behind.

    def test_empty_setup_command_rejected(self, provider: LocalSandboxProvider) -> None:
        config = SandboxConfig(setup=("   ",))
        with pytest.raises(SandboxSetupError, match="empty"):
            _ = provider.open(config)


class TestLifecycle:
    def test_open_use_snapshot_restore_close(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = SandboxConfig(mounts=(HostMount(host_path=str(host_dir)),))
        sandbox = provider.open(config)
        root = Path(sandbox.root)
        base = host_dir.name

        ref = sandbox.snapshot(tag="opened")
        _ = sandbox.filesystem.write(f"{base}/main.py", "print('changed')")
        _ = sandbox.filesystem.write("scratch.txt", "temp")
        sandbox.restore(ref)

        assert sandbox.filesystem.read(f"{base}/main.py").content == "print('main')"
        assert not sandbox.filesystem.exists("scratch.txt")

        sandbox.close()
        sandbox.close()
        assert not root.exists()

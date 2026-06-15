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

"""Tests for LocalSandboxProvider and RemoteSandboxProvider.

The mount cases mirror the workspace-section mount tests: the provider
absorbs that machinery and must reproduce its behavior. The remote
provider materializes the same intent through a loopback transport.
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from weakincentives.filesystem import WriteMode
from weakincentives.sandbox import (
    EgressPolicy,
    EgressRule,
    HostMount,
    LocalSandbox,
    LocalSandboxProvider,
    LoopbackTransport,
    RemoteSandbox,
    RemoteSandboxProvider,
    Sandbox,
    SandboxProvider,
    SandboxSetupError,
    TransportFault,
    WorkspaceBudgetExceededError,
    WorkspaceConfig,
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
        sandbox = provider.open(WorkspaceConfig())
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
        sandbox = provider.open(WorkspaceConfig())
        try:
            assert Path(sandbox.root).exists()
            assert sandbox.filesystem.list(".") == []
        finally:
            sandbox.close()

    def test_copies_single_file(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = WorkspaceConfig(
            mounts=(HostMount(host_path=str(host_dir / "main.py")),)
        )
        sandbox = provider.open(config)
        try:
            assert sandbox.filesystem.read("main.py").content == "print('main')"
        finally:
            sandbox.close()

    def test_copies_directory_recursively(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = WorkspaceConfig(mounts=(HostMount(host_path=str(host_dir)),))
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
        config = WorkspaceConfig(
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
        config = WorkspaceConfig(
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
        config = WorkspaceConfig(
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
        config = WorkspaceConfig(
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
        config = WorkspaceConfig(
            mounts=(HostMount(host_path=str(host_dir), max_bytes=2),)
        )
        with pytest.raises(WorkspaceBudgetExceededError, match="byte budget"):
            _ = provider.open(config)

    def test_security_boundary_enforced(
        self, provider: LocalSandboxProvider, host_dir: Path, tmp_path: Path
    ) -> None:
        allowed = tmp_path / "allowed-root"
        allowed.mkdir()
        config = WorkspaceConfig(
            mounts=(HostMount(host_path=str(host_dir)),),
            allowed_host_roots=(str(allowed),),
        )
        with pytest.raises(WorkspaceSecurityError, match="outside allowed"):
            _ = provider.open(config)

    def test_security_boundary_allows_within_root(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = WorkspaceConfig(
            mounts=(HostMount(host_path=str(host_dir)),),
            allowed_host_roots=(str(host_dir.parent),),
        )
        sandbox = provider.open(config)
        try:
            assert sandbox.filesystem.exists(f"{host_dir.name}/main.py")
        finally:
            sandbox.close()

    def test_nonexistent_host_path_raises(self, provider: LocalSandboxProvider) -> None:
        config = WorkspaceConfig(
            mounts=(HostMount(host_path="/nonexistent/path/12345"),)
        )
        with pytest.raises(FileNotFoundError):
            _ = provider.open(config)


class TestProviderConfig:
    def test_read_only_applies_to_filesystem_facet(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = WorkspaceConfig(
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
        config = WorkspaceConfig(env={"WINK_SANDBOX_VAR": "configured"})
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
        sandbox = provider.open(WorkspaceConfig(egress=policy))
        try:
            assert sandbox.egress is policy
        finally:
            sandbox.close()

    def test_default_egress_denies_all(self, provider: LocalSandboxProvider) -> None:
        sandbox = provider.open(WorkspaceConfig())
        try:
            assert not sandbox.egress.allows("example.com")
        finally:
            sandbox.close()


class TestSetupCommands:
    def test_setup_runs_in_order(self, provider: LocalSandboxProvider) -> None:
        config = WorkspaceConfig(
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
        config = WorkspaceConfig(setup=(f"{sys.executable} -c \"print('$HOME')\"",))
        sandbox = provider.open(config)
        sandbox.close()

    def test_failing_setup_raises_and_cleans_up(
        self, provider: LocalSandboxProvider
    ) -> None:
        config = WorkspaceConfig(
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
        config = WorkspaceConfig(setup=("   ",))
        with pytest.raises(SandboxSetupError, match="empty"):
            _ = provider.open(config)


class TestLifecycle:
    def test_open_use_snapshot_restore_close(
        self, provider: LocalSandboxProvider, host_dir: Path
    ) -> None:
        config = WorkspaceConfig(mounts=(HostMount(host_path=str(host_dir)),))
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


class _RemoteHarness:
    """Connect factory tracking the transports it hands out."""

    def __init__(self, env_base: Path) -> None:
        self.env_base = env_base
        self.transports: list[LoopbackTransport] = []

    def connect(self) -> LoopbackTransport:
        env_root = self.env_base / f"env-{len(self.transports)}"
        env_root.mkdir(parents=True)
        transport = LoopbackTransport(env_root, owns_root=True)
        self.transports.append(transport)
        return transport


@pytest.fixture
def remote_harness(tmp_path: Path) -> _RemoteHarness:
    return _RemoteHarness(tmp_path / "remote-envs")


@pytest.fixture
def remote_provider(
    remote_harness: _RemoteHarness,
) -> Iterator[RemoteSandboxProvider]:
    """Remote provider with a unique staging prefix so leftovers are detectable."""
    prefix = f"wink-staging-{uuid.uuid4().hex[:8]}-"
    yield RemoteSandboxProvider(remote_harness.connect, temp_dir_prefix=prefix)
    leftovers = list(Path(tempfile.gettempdir()).glob(f"{prefix}*"))
    assert leftovers == []


class TestRemoteSandboxProvider:
    def test_implements_provider_protocol(
        self, remote_provider: RemoteSandboxProvider
    ) -> None:
        assert isinstance(remote_provider, SandboxProvider)

    def test_open_returns_remote_sandbox(
        self, remote_provider: RemoteSandboxProvider, remote_harness: _RemoteHarness
    ) -> None:
        sandbox = remote_provider.open(WorkspaceConfig())
        try:
            assert isinstance(sandbox, Sandbox)
            assert isinstance(sandbox, RemoteSandbox)
            assert sandbox.root == remote_harness.transports[0].root
            assert sandbox.filesystem.list(".") == []
        finally:
            sandbox.close()

    def test_mounts_upload_through_transport(
        self,
        remote_provider: RemoteSandboxProvider,
        host_dir: Path,
    ) -> None:
        config = WorkspaceConfig(
            mounts=(HostMount(host_path=str(host_dir), mount_path="proj"),)
        )
        sandbox = remote_provider.open(config)
        try:
            assert sandbox.filesystem.read("proj/main.py").content == "print('main')"
            assert sandbox.filesystem.exists("proj/sub/nested.py")
            result = sandbox.shell.run(["cat", "proj/notes.txt"])
            assert result.stdout == b"notes"
        finally:
            sandbox.close()

    def test_egress_seeds_transport_and_sandbox(
        self, remote_provider: RemoteSandboxProvider, remote_harness: _RemoteHarness
    ) -> None:
        policy = EgressPolicy(allow=(EgressRule(host_glob="pypi.org"),))
        sandbox = remote_provider.open(WorkspaceConfig(egress=policy))
        try:
            assert sandbox.egress is policy
            assert remote_harness.transports[0].egress is policy
        finally:
            sandbox.close()

    def test_read_only_applies_to_filesystem_facet(
        self, remote_provider: RemoteSandboxProvider, host_dir: Path
    ) -> None:
        config = WorkspaceConfig(
            mounts=(HostMount(host_path=str(host_dir)),), read_only=True
        )
        sandbox = remote_provider.open(config)
        try:
            assert sandbox.filesystem.read_only
            assert sandbox.filesystem.exists(f"{host_dir.name}/main.py")
            with pytest.raises(PermissionError):
                _ = sandbox.filesystem.write("new.txt", "nope")
        finally:
            sandbox.close()

    def test_env_reaches_remote_shell(
        self, remote_provider: RemoteSandboxProvider
    ) -> None:
        config = WorkspaceConfig(env={"WINK_REMOTE_VAR": "configured"})
        sandbox = remote_provider.open(config)
        try:
            result = sandbox.shell.run(
                [
                    sys.executable,
                    "-c",
                    "import os; print(os.environ['WINK_REMOTE_VAR'])",
                ]
            )
            assert result.stdout.decode().strip() == "configured"
        finally:
            sandbox.close()

    def test_setup_runs_through_remote_shell(
        self, remote_provider: RemoteSandboxProvider
    ) -> None:
        config = WorkspaceConfig(
            setup=(f"{sys.executable} -c \"open('made.txt', 'w').write('yes')\"",)
        )
        sandbox = remote_provider.open(config)
        try:
            assert sandbox.filesystem.read("made.txt").content == "yes"
        finally:
            sandbox.close()

    def test_failing_setup_closes_transport(
        self, remote_provider: RemoteSandboxProvider, remote_harness: _RemoteHarness
    ) -> None:
        config = WorkspaceConfig(
            setup=(
                (
                    f"{sys.executable} -c "
                    "\"import sys; sys.stderr.write('remote setup blew up'); "
                    'sys.exit(2)"'
                ),
            )
        )
        with pytest.raises(SandboxSetupError, match="remote setup blew up"):
            _ = remote_provider.open(config)
        transport = remote_harness.transports[0]
        assert transport.closed
        assert not Path(transport.root).exists()

    def test_empty_setup_command_closes_transport(
        self, remote_provider: RemoteSandboxProvider, remote_harness: _RemoteHarness
    ) -> None:
        with pytest.raises(SandboxSetupError, match="empty"):
            _ = remote_provider.open(WorkspaceConfig(setup=("   ",)))
        assert remote_harness.transports[0].closed

    def test_connect_failure_cleans_staging(self, host_dir: Path) -> None:
        prefix = f"wink-staging-{uuid.uuid4().hex[:8]}-"

        def refuse() -> LoopbackTransport:
            raise ConnectionError("no route to environment")

        provider = RemoteSandboxProvider(refuse, temp_dir_prefix=prefix)
        config = WorkspaceConfig(mounts=(HostMount(host_path=str(host_dir)),))
        with pytest.raises(ConnectionError, match="no route"):
            _ = provider.open(config)
        assert list(Path(tempfile.gettempdir()).glob(f"{prefix}*")) == []

    def test_upload_failure_closes_transport(
        self, remote_harness: _RemoteHarness, host_dir: Path
    ) -> None:
        class WriteRefusingTransport(LoopbackTransport):
            def write(self, path: str, data: bytes, *, mode: WriteMode) -> None:
                raise TransportFault("io", "remote disk full")

        env_root = remote_harness.env_base / "refusing"
        env_root.mkdir(parents=True)
        transport = WriteRefusingTransport(env_root)
        provider = RemoteSandboxProvider(lambda: transport)
        config = WorkspaceConfig(mounts=(HostMount(host_path=str(host_dir)),))
        with pytest.raises(TransportFault, match="remote disk full"):
            _ = provider.open(config)
        assert transport.closed

    def test_mount_guards_apply_before_connecting(
        self, remote_harness: _RemoteHarness, host_dir: Path, tmp_path: Path
    ) -> None:
        allowed = tmp_path / "allowed-root"
        allowed.mkdir()
        provider = RemoteSandboxProvider(remote_harness.connect)
        config = WorkspaceConfig(
            mounts=(HostMount(host_path=str(host_dir)),),
            allowed_host_roots=(str(allowed),),
        )
        with pytest.raises(WorkspaceSecurityError, match="outside allowed"):
            _ = provider.open(config)
        assert remote_harness.transports == []

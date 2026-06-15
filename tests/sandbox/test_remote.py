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

"""Tests for the remote sandbox facets over the loopback transport.

The full filesystem and shell contract suites run against the remote
facets here — `Filesystem(RemoteBackend)` and `RemoteShell` over an
in-process `LoopbackTransport` — exercising every fault translation
round-trip without sockets. Fault→exception mapping is additionally
unit-tested against a transport stub that fails on demand.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from tests.helpers.filesystem import FilesystemValidationSuite
from tests.helpers.shell import ShellHarness, ShellValidationSuite, python_argv
from weakincentives.errors import SnapshotRestoreError
from weakincentives.filesystem import (
    FileEntry,
    FileStat,
    Filesystem,
    FilesystemBackend,
    GlobMatch,
    GrepMatch,
    SnapshotRef,
    WriteMode,
)
from weakincentives.sandbox import (
    DEFAULT_SHELL_TIMEOUT_S,
    MAX_COMMAND_OUTPUT_BYTES,
    CommandResult,
    CredentialBinding,
    EgressPolicy,
    EgressRule,
    LoopbackTransport,
    RemoteBackend,
    RemoteSandbox,
    RemoteShell,
    Sandbox,
    SandboxClosedError,
    Shell,
    TransportFault,
    TransportFaultCode,
)


def make_remote_sandbox(root: Path) -> tuple[RemoteSandbox, LoopbackTransport]:
    """Build a RemoteSandbox over a loopback transport rooted at ``root``."""
    transport = LoopbackTransport(root)
    sandbox = RemoteSandbox(
        transport=transport,
        filesystem=Filesystem(RemoteBackend(transport)),
        shell=RemoteShell(transport),
    )
    return sandbox, transport


class RaisingTransport:
    """SandboxTransport stub whose effect methods raise a configured fault."""

    def __init__(self, fault: TransportFault) -> None:
        self.fault = fault
        self.closed = False

    @property
    def root(self) -> str:
        return "/remote/env"

    def stat(self, path: str) -> FileStat:
        raise self.fault

    def list(self, path: str) -> Sequence[FileEntry]:
        raise self.fault

    def read_range(self, path: str, *, offset: int, length: int | None) -> bytes:
        raise self.fault

    def write(self, path: str, data: bytes, *, mode: WriteMode) -> None:
        raise self.fault

    def glob(self, pattern: str, *, path: str) -> Sequence[GlobMatch]:
        raise self.fault

    def grep(
        self,
        pattern: str,
        *,
        path: str,
        glob: str | None,
        max_matches: int,
    ) -> Sequence[GrepMatch]:
        raise self.fault

    def delete(self, path: str, *, recursive: bool) -> None:
        raise self.fault

    def mkdir(self, path: str) -> None:
        raise self.fault

    def rename(self, src: str, dst: str) -> None:
        raise self.fault

    def snapshot(self, *, tag: str | None) -> SnapshotRef:
        raise self.fault

    def restore(self, ref: SnapshotRef) -> None:
        raise self.fault

    def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        stdin: bytes | None,
        timeout_s: float,
    ) -> CommandResult:
        raise self.fault

    def configure_egress(self, policy: EgressPolicy) -> None:
        raise self.fault

    def configure_credentials(self, bindings: Sequence[CredentialBinding]) -> None:
        raise self.fault

    def close(self) -> None:
        self.closed = True


def _dummy_ref() -> SnapshotRef:
    return SnapshotRef(
        snapshot_id=uuid4(),
        created_at=datetime.now(tz=UTC),
        token="opaque",
    )


class TestRemoteFilesystemContract(FilesystemValidationSuite):
    """The full filesystem validation suite over the loopback transport."""

    @pytest.fixture
    def fs(self, tmp_path: Path) -> Filesystem:
        return Filesystem(RemoteBackend(LoopbackTransport(tmp_path)))


class TestRemoteShellContract(ShellValidationSuite):
    """The generic Shell contract over RemoteShell + loopback."""

    @pytest.fixture
    def harness(self, tmp_path: Path) -> ShellHarness:
        def make(
            *,
            env: Mapping[str, str] | None = None,
            default_timeout_s: float | None = None,
            max_output_bytes: int | None = None,
        ) -> Shell:
            transport = LoopbackTransport(
                tmp_path,
                max_output_bytes=(
                    max_output_bytes
                    if max_output_bytes is not None
                    else MAX_COMMAND_OUTPUT_BYTES
                ),
            )
            return RemoteShell(
                transport,
                env=env,
                default_timeout_s=(
                    default_timeout_s
                    if default_timeout_s is not None
                    else DEFAULT_SHELL_TIMEOUT_S
                ),
            )

        return ShellHarness(root=tmp_path, make=make)


class TestRemoteBackend:
    def test_implements_backend_protocol(self, tmp_path: Path) -> None:
        backend = RemoteBackend(LoopbackTransport(tmp_path))
        assert isinstance(backend, FilesystemBackend)

    def test_root_reported_by_transport(self, tmp_path: Path) -> None:
        backend = RemoteBackend(LoopbackTransport(tmp_path))
        assert backend.root == str(tmp_path.resolve())

    def test_read_only_flag(self, tmp_path: Path) -> None:
        transport = LoopbackTransport(tmp_path)
        assert RemoteBackend(transport).read_only is False
        assert RemoteBackend(transport, read_only=True).read_only is True

    def test_read_only_enforced_by_facade(self, tmp_path: Path) -> None:
        transport = LoopbackTransport(tmp_path)
        fs = Filesystem(RemoteBackend(transport, read_only=True))
        with pytest.raises(PermissionError, match="read-only"):
            fs.write("a.txt", "nope")

    @pytest.mark.parametrize(
        "operation",
        [
            lambda b: b.stat("a"),
            lambda b: b.list("a"),
            lambda b: b.glob("*", path=""),
            lambda b: b.grep("x", path="", glob=None, max_matches=1),
            lambda b: b.read_range("a", offset=0, length=None),
            lambda b: b.write("a", b"data", mode="overwrite"),
            lambda b: b.delete("a", recursive=False),
            lambda b: b.mkdir("a"),
            lambda b: b.rename("a", "b"),
            lambda b: b.snapshot(tag=None),
            lambda b: b.restore(_dummy_ref()),
        ],
    )
    def test_connectivity_fault_maps_to_runtime_error(
        self, operation: Callable[[RemoteBackend], object]
    ) -> None:
        backend = RemoteBackend(
            RaisingTransport(TransportFault("connectivity", "link down"))
        )
        with pytest.raises(RuntimeError, match="link down"):
            operation(backend)

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("not-found", FileNotFoundError),
            ("permission", PermissionError),
            ("io", OSError),
        ],
    )
    def test_fault_codes_map_to_contract_exceptions(
        self, code: TransportFaultCode, expected: type[Exception]
    ) -> None:
        backend = RemoteBackend(RaisingTransport(TransportFault(code, "boom")))
        with pytest.raises(expected, match="boom"):
            backend.stat("a")

    def test_snapshot_fault_maps_to_restore_error(self) -> None:
        backend = RemoteBackend(
            RaisingTransport(TransportFault("snapshot", "unknown ref"))
        )
        with pytest.raises(SnapshotRestoreError, match="unknown ref"):
            backend.restore(_dummy_ref())

    def test_fault_chains_to_original(self) -> None:
        fault = TransportFault("not-found", "missing")
        backend = RemoteBackend(RaisingTransport(fault))
        with pytest.raises(FileNotFoundError) as excinfo:
            backend.stat("a")
        assert excinfo.value.__cause__ is fault


class TestRemoteShell:
    def test_implements_shell_protocol(self, tmp_path: Path) -> None:
        assert isinstance(RemoteShell(LoopbackTransport(tmp_path)), Shell)

    def test_root_reported_by_transport(self, tmp_path: Path) -> None:
        shell = RemoteShell(LoopbackTransport(tmp_path))
        assert shell.root == str(tmp_path.resolve())

    def test_exec_fault_maps_to_contract_exception(self) -> None:
        shell = RemoteShell(
            RaisingTransport(TransportFault("permission", "cwd escapes"))
        )
        with pytest.raises(PermissionError, match="cwd escapes"):
            shell.run(["true"])

    def test_connectivity_fault_maps_to_runtime_error(self) -> None:
        shell = RemoteShell(
            RaisingTransport(TransportFault("connectivity", "link down"))
        )
        with pytest.raises(RuntimeError, match="link down"):
            shell.run(["true"])

    def test_validation_happens_before_transport(self) -> None:
        shell = RemoteShell(
            RaisingTransport(TransportFault("connectivity", "must not be reached"))
        )
        with pytest.raises(ValueError, match="argv must not be empty"):
            shell.run([])
        with pytest.raises(ValueError, match="timeout_s must be positive"):
            shell.run(["true"], timeout_s=-1)
        with pytest.raises(ValueError, match="must be relative"):
            shell.run(["true"], cwd="/absolute")


class TestRemoteSandbox:
    def test_satisfies_sandbox_protocol(self, tmp_path: Path) -> None:
        sandbox, _ = make_remote_sandbox(tmp_path)
        assert isinstance(sandbox, Sandbox)
        sandbox.close()

    def test_one_environment_for_both_facets(self, tmp_path: Path) -> None:
        sandbox, _ = make_remote_sandbox(tmp_path)
        try:
            assert sandbox.root == sandbox.filesystem.root == sandbox.shell.root
            sandbox.filesystem.write("from-fs.txt", "via filesystem")
            result = sandbox.shell.run(["cat", "from-fs.txt"])
            assert result.stdout == b"via filesystem"
            result = sandbox.shell.run(
                python_argv("open('from-shell.txt', 'w').write('via shell')")
            )
            assert result.ok
            assert sandbox.filesystem.read("from-shell.txt").content == "via shell"
        finally:
            sandbox.close()

    def test_snapshot_restore_round_trip(self, tmp_path: Path) -> None:
        sandbox, _ = make_remote_sandbox(tmp_path)
        try:
            sandbox.filesystem.write("state.txt", "before")
            ref = sandbox.snapshot(tag="checkpoint")
            assert ref.tag == "checkpoint"
            sandbox.filesystem.write("state.txt", "after")
            sandbox.restore(ref)
            assert sandbox.filesystem.read("state.txt").content == "before"
        finally:
            sandbox.close()

    def test_configure_egress_forwards_to_transport(self, tmp_path: Path) -> None:
        sandbox, transport = make_remote_sandbox(tmp_path)
        try:
            policy = EgressPolicy(allow=(EgressRule(host_glob="api.example.com"),))
            sandbox.configure_egress(policy)
            assert sandbox.egress == policy
            assert transport.egress == policy
        finally:
            sandbox.close()

    def test_configure_credentials_keeps_names_only(self, tmp_path: Path) -> None:
        sandbox, transport = make_remote_sandbox(tmp_path)
        try:
            sandbox.configure_credentials(
                [CredentialBinding(name="api", secret="s3cret-material")]
            )
            assert sandbox.credential_names == frozenset({"api"})
            assert transport.credential_names == frozenset({"api"})
            assert "s3cret-material" not in repr(sandbox)
        finally:
            sandbox.close()

    def test_duplicate_credentials_rejected_before_transport(
        self, tmp_path: Path
    ) -> None:
        sandbox, transport = make_remote_sandbox(tmp_path)
        try:
            bindings = [
                CredentialBinding(name="api", secret="one"),
                CredentialBinding(name="api", secret="two"),
            ]
            with pytest.raises(ValueError, match="duplicate credential binding"):
                sandbox.configure_credentials(bindings)
            assert transport.credential_names == frozenset()
        finally:
            sandbox.close()

    def test_control_plane_fault_maps_to_runtime_error(self) -> None:
        transport = RaisingTransport(TransportFault("connectivity", "link down"))
        sandbox = RemoteSandbox(
            transport=transport,
            filesystem=Filesystem(RemoteBackend(transport)),
            shell=RemoteShell(transport),
        )
        with pytest.raises(RuntimeError, match="link down"):
            sandbox.configure_egress(EgressPolicy())
        with pytest.raises(RuntimeError, match="link down"):
            sandbox.configure_credentials([CredentialBinding(name="a", secret="b")])

    def test_close_tears_down_transport_and_is_idempotent(self, tmp_path: Path) -> None:
        sandbox, transport = make_remote_sandbox(tmp_path)
        sandbox.close()
        assert transport.closed
        sandbox.close()

    def test_closed_sandbox_rejects_use(self, tmp_path: Path) -> None:
        sandbox, _ = make_remote_sandbox(tmp_path)
        sandbox.filesystem.write("a.txt", "x")
        ref = sandbox.snapshot()
        sandbox.configure_credentials([CredentialBinding(name="api", secret="s")])
        sandbox.close()
        with pytest.raises(SandboxClosedError):
            _ = sandbox.filesystem
        with pytest.raises(SandboxClosedError):
            _ = sandbox.shell
        with pytest.raises(SandboxClosedError):
            sandbox.configure_egress(EgressPolicy())
        with pytest.raises(SandboxClosedError):
            sandbox.configure_credentials([])
        with pytest.raises(SandboxClosedError):
            sandbox.snapshot()
        with pytest.raises(SandboxClosedError):
            sandbox.restore(ref)

    def test_identity_readable_after_close(self, tmp_path: Path) -> None:
        sandbox, _ = make_remote_sandbox(tmp_path)
        sandbox.configure_credentials([CredentialBinding(name="api", secret="s")])
        sandbox.close()
        assert sandbox.id
        assert sandbox.root == str(tmp_path.resolve())
        assert sandbox.egress == EgressPolicy()
        assert sandbox.credential_names == frozenset()

    def test_custom_sandbox_id_and_repr(self, tmp_path: Path) -> None:
        transport = LoopbackTransport(tmp_path)
        sandbox = RemoteSandbox(
            transport=transport,
            filesystem=Filesystem(RemoteBackend(transport)),
            shell=RemoteShell(transport),
            sandbox_id="sbx-42",
        )
        assert sandbox.id == "sbx-42"
        assert "sbx-42" in repr(sandbox)
        assert "closed=False" in repr(sandbox)
        sandbox.close()
        assert "closed=True" in repr(sandbox)

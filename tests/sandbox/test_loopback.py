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

"""Tests for the in-process loopback sandbox transport."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from weakincentives.sandbox import (
    CredentialBinding,
    EgressPolicy,
    EgressRule,
    LoopbackTransport,
    TransportFault,
)


@pytest.fixture
def transport(tmp_path: Path) -> LoopbackTransport:
    return LoopbackTransport(tmp_path)


class TestLoopbackPrimitives:
    def test_root_resolved(self, tmp_path: Path) -> None:
        assert LoopbackTransport(tmp_path).root == str(tmp_path.resolve())

    def test_write_stat_read_round_trip(self, transport: LoopbackTransport) -> None:
        transport.write("a.txt", b"hello", mode="overwrite")
        stat = transport.stat("a.txt")
        assert stat.is_file
        assert stat.size_bytes == 5
        assert transport.read_range("a.txt", offset=1, length=3) == b"ell"

    def test_list_mkdir_delete(self, transport: LoopbackTransport) -> None:
        transport.mkdir("d/e")
        transport.write("d/f.txt", b"x", mode="create")
        names = [entry.name for entry in transport.list("d")]
        assert names == ["e", "f.txt"]
        transport.delete("d", recursive=True)
        with pytest.raises(TransportFault) as excinfo:
            transport.stat("d")
        assert excinfo.value.code == "not-found"

    def test_glob_and_grep_run_against_backend(
        self, transport: LoopbackTransport
    ) -> None:
        transport.write("src/a.py", b"needle = 1", mode="overwrite")
        transport.write("src/b.txt", b"no", mode="overwrite")
        matches = transport.glob("*.py", path="src")
        assert [m.path for m in matches] == ["src/a.py"]
        hits = transport.grep("needle", path="", glob=None, max_matches=10)
        assert [(h.path, h.line_number) for h in hits] == [("src/a.py", 1)]

    def test_native_errors_become_faults(self, transport: LoopbackTransport) -> None:
        with pytest.raises(TransportFault) as excinfo:
            transport.read_range("missing.txt", offset=0, length=None)
        assert excinfo.value.code == "not-found"

    def test_exec_runs_commands(self, transport: LoopbackTransport) -> None:
        result = transport.exec(
            [sys.executable, "-c", "print('hi')"],
            cwd=None,
            env=None,
            stdin=None,
            timeout_s=10.0,
        )
        assert result.ok
        assert result.stdout == b"hi\n"

    def test_exec_validation_becomes_fault(self, transport: LoopbackTransport) -> None:
        with pytest.raises(TransportFault) as excinfo:
            transport.exec([], cwd=None, env=None, stdin=None, timeout_s=10.0)
        assert excinfo.value.code == "invalid"

    def test_snapshot_restore_round_trip(self, transport: LoopbackTransport) -> None:
        transport.write("keep.txt", b"original", mode="overwrite")
        ref = transport.snapshot(tag="before")
        transport.write("keep.txt", b"mutated", mode="overwrite")
        transport.write("extra.txt", b"new", mode="overwrite")
        transport.restore(ref)
        assert transport.read_range("keep.txt", offset=0, length=None) == b"original"
        with pytest.raises(TransportFault) as excinfo:
            transport.stat("extra.txt")
        assert excinfo.value.code == "not-found"


class TestLoopbackControlPlane:
    def test_records_egress_policy(self, transport: LoopbackTransport) -> None:
        assert transport.egress == EgressPolicy()
        policy = EgressPolicy(allow=(EgressRule(host_glob="*.pypi.org"),))
        transport.configure_egress(policy)
        assert transport.egress == policy

    def test_records_credential_names_only(self, transport: LoopbackTransport) -> None:
        transport.configure_credentials(
            [CredentialBinding(name="api", secret="s3cret")]
        )
        assert transport.credential_names == frozenset({"api"})

    def test_duplicate_credentials_fault(self, transport: LoopbackTransport) -> None:
        bindings = [
            CredentialBinding(name="api", secret="one"),
            CredentialBinding(name="api", secret="two"),
        ]
        with pytest.raises(TransportFault) as excinfo:
            transport.configure_credentials(bindings)
        assert excinfo.value.code == "invalid"


class TestLoopbackLifecycle:
    def test_close_idempotent(self, transport: LoopbackTransport) -> None:
        assert not transport.closed
        transport.close()
        transport.close()
        assert transport.closed

    def test_close_clears_credentials(self, transport: LoopbackTransport) -> None:
        transport.configure_credentials(
            [CredentialBinding(name="api", secret="s3cret")]
        )
        transport.close()
        assert transport.credential_names == frozenset()

    def test_owns_root_removes_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "env"
        root.mkdir()
        transport = LoopbackTransport(root, owns_root=True)
        transport.write("a.txt", b"x", mode="overwrite")
        transport.close()
        assert not root.exists()

    def test_unowned_root_survives_close(self, tmp_path: Path) -> None:
        transport = LoopbackTransport(tmp_path)
        transport.write("a.txt", b"x", mode="overwrite")
        transport.close()
        assert (tmp_path / "a.txt").exists()

    @pytest.mark.parametrize(
        "operation",
        [
            lambda t: t.stat(""),
            lambda t: t.list(""),
            lambda t: t.read_range("a", offset=0, length=None),
            lambda t: t.write("a", b"x", mode="overwrite"),
            lambda t: t.glob("*", path=""),
            lambda t: t.grep("x", path="", glob=None, max_matches=1),
            lambda t: t.delete("a", recursive=False),
            lambda t: t.mkdir("a"),
            lambda t: t.rename("a", "b"),
            lambda t: t.snapshot(tag=None),
            lambda t: t.exec(["true"], cwd=None, env=None, stdin=None, timeout_s=1.0),
            lambda t: t.configure_egress(EgressPolicy()),
            lambda t: t.configure_credentials([]),
        ],
    )
    def test_closed_transport_raises_connectivity(
        self,
        transport: LoopbackTransport,
        operation: Callable[[LoopbackTransport], object],
    ) -> None:
        transport.close()
        with pytest.raises(TransportFault) as excinfo:
            operation(transport)
        assert excinfo.value.code == "connectivity"

    def test_closed_restore_raises_connectivity(
        self, transport: LoopbackTransport
    ) -> None:
        ref = transport.snapshot(tag=None)
        transport.close()
        with pytest.raises(TransportFault) as excinfo:
            transport.restore(ref)
        assert excinfo.value.code == "connectivity"

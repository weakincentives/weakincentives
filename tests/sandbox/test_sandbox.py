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

"""Tests for the Sandbox aggregate and credential control plane."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import pytest

from weakincentives.filesystem import Filesystem, HostBackend, SnapshotRef
from weakincentives.sandbox import (
    CredentialBinding,
    EgressPolicy,
    EgressRule,
    LocalSandbox,
    LocalShell,
    Sandbox,
    SandboxClosedError,
)

SECRET = "s3cr3t-material-xyz"  # deliberately fake material for tests


@pytest.fixture
def sandbox() -> LocalSandbox:
    root = Path(tempfile.mkdtemp(prefix="wink-sandbox-test-"))
    return LocalSandbox(
        root=root,
        filesystem=Filesystem.host(root),
        shell=LocalShell(root),
    )


class TestAggregate:
    def test_implements_sandbox_protocol(self, sandbox: LocalSandbox) -> None:
        try:
            assert isinstance(sandbox, Sandbox)
        finally:
            sandbox.close()

    def test_root_matches_filesystem_root(self, sandbox: LocalSandbox) -> None:
        try:
            assert sandbox.root == sandbox.filesystem.root
            assert (
                sandbox.root
                == sandbox.shell.run(
                    [sys.executable, "-c", "import os; print(os.getcwd())"]
                )
                .stdout.decode()
                .strip()
            )
        finally:
            sandbox.close()

    def test_generated_ids_are_unique(self, sandbox: LocalSandbox) -> None:
        root = Path(tempfile.mkdtemp(prefix="wink-sandbox-test-"))
        other = LocalSandbox(
            root=root, filesystem=Filesystem.host(root), shell=LocalShell(root)
        )
        try:
            assert sandbox.id != other.id
        finally:
            sandbox.close()
            other.close()

    def test_custom_id_respected(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="wink-sandbox-test-"))
        box = LocalSandbox(
            root=root,
            filesystem=Filesystem.host(root),
            shell=LocalShell(root),
            sandbox_id="my-sandbox",
        )
        try:
            assert box.id == "my-sandbox"
        finally:
            box.close()

    def test_default_egress_denies_all(self, sandbox: LocalSandbox) -> None:
        try:
            assert sandbox.egress == EgressPolicy.deny_all()
        finally:
            sandbox.close()

    def test_repr_shows_identity_not_credentials(self, sandbox: LocalSandbox) -> None:
        try:
            sandbox.configure_credentials(
                [CredentialBinding(name="token", secret=SECRET)]
            )
            rendered = repr(sandbox)
            assert sandbox.id in rendered
            assert sandbox.root in rendered
            assert SECRET not in rendered
        finally:
            sandbox.close()


class TestSnapshotRestore:
    def test_round_trip_via_filesystem_backend(self, sandbox: LocalSandbox) -> None:
        try:
            _ = sandbox.filesystem.write("kept.txt", "original")
            ref = sandbox.snapshot(tag="before")
            assert isinstance(ref, SnapshotRef)
            assert ref.tag == "before"

            _ = sandbox.filesystem.write("kept.txt", "changed")
            _ = sandbox.filesystem.write("scratch.txt", "new")
            sandbox.restore(ref)

            assert sandbox.filesystem.read("kept.txt").content == "original"
            assert not sandbox.filesystem.exists("scratch.txt")
        finally:
            sandbox.close()


class TestConfigureEgress:
    def test_replaces_policy_live(self, sandbox: LocalSandbox) -> None:
        try:
            opened = EgressPolicy(allow=(EgressRule(host_glob="pypi.org"),))
            sandbox.configure_egress(opened)
            assert sandbox.egress is opened
            assert sandbox.egress.allows("pypi.org")

            sandbox.configure_egress(EgressPolicy.deny_all())
            assert not sandbox.egress.allows("pypi.org")
        finally:
            sandbox.close()


class TestConfigureCredentials:
    def test_binds_and_rebinds_names(self, sandbox: LocalSandbox) -> None:
        try:
            sandbox.configure_credentials(
                [CredentialBinding(name="alpha", secret=SECRET)]
            )
            assert sandbox.credential_names == frozenset({"alpha"})

            sandbox.configure_credentials(
                [CredentialBinding(name="beta", secret=SECRET)]
            )
            assert sandbox.credential_names == frozenset({"beta"})

            sandbox.configure_credentials([])
            assert sandbox.credential_names == frozenset()
        finally:
            sandbox.close()

    def test_duplicate_names_rejected(self, sandbox: LocalSandbox) -> None:
        try:
            with pytest.raises(ValueError, match="duplicate credential"):
                sandbox.configure_credentials(
                    [
                        CredentialBinding(name="alpha", secret=SECRET),
                        CredentialBinding(name="alpha", secret="other"),
                    ]
                )
        finally:
            sandbox.close()

    def test_binding_validates_name_and_secret(self) -> None:
        with pytest.raises(ValueError, match="credential name"):
            CredentialBinding(name="", secret=SECRET)
        with pytest.raises(ValueError, match="non-empty secret"):
            CredentialBinding(name="alpha", secret="")

    def test_binding_repr_redacts_secret(self) -> None:
        binding = CredentialBinding(name="alpha", secret=SECRET)
        assert SECRET not in repr(binding)
        assert "alpha" in repr(binding)

    def test_secret_never_reaches_sandbox_env(self, sandbox: LocalSandbox) -> None:
        try:
            sandbox.configure_credentials(
                [CredentialBinding(name="token", secret=SECRET)]
            )
            result = sandbox.shell.run(
                [
                    sys.executable,
                    "-c",
                    "import os, json; print(json.dumps(dict(os.environ)))",
                ]
            )
            env = json.loads(result.stdout.decode())
            assert SECRET not in json.dumps(env)
        finally:
            sandbox.close()

    def test_secret_never_reaches_sandbox_filesystem(
        self, sandbox: LocalSandbox
    ) -> None:
        try:
            _ = sandbox.filesystem.write("app.py", "print('hi')")
            sandbox.configure_credentials(
                [CredentialBinding(name="token", secret=SECRET)]
            )
            root = Path(sandbox.root)
            contents = [p.read_bytes() for p in root.rglob("*") if p.is_file()]
            assert all(SECRET.encode() not in blob for blob in contents)
        finally:
            sandbox.close()

    def test_secret_never_logged(
        self, sandbox: LocalSandbox, caplog: pytest.LogCaptureFixture
    ) -> None:
        try:
            with caplog.at_level(logging.DEBUG):
                sandbox.configure_credentials(
                    [CredentialBinding(name="token", secret=SECRET)]
                )
                sandbox.configure_egress(EgressPolicy.deny_all())
            assert SECRET not in caplog.text
        finally:
            sandbox.close()


class TestClose:
    def test_close_removes_root(self, sandbox: LocalSandbox) -> None:
        root = Path(sandbox.root)
        assert root.exists()
        sandbox.close()
        assert not root.exists()

    def test_close_is_idempotent(self, sandbox: LocalSandbox) -> None:
        sandbox.close()
        sandbox.close()

    def test_close_removes_snapshot_storage(self, sandbox: LocalSandbox) -> None:
        _ = sandbox.filesystem.write("file.txt", "content")
        _ = sandbox.snapshot(tag="snap")

        backend = sandbox.filesystem.backend
        assert isinstance(backend, HostBackend)
        git_dir = backend.git_dir
        assert git_dir is not None
        assert Path(git_dir).exists()

        sandbox.close()
        assert not Path(git_dir).exists()

    def test_close_clears_credentials(self, sandbox: LocalSandbox) -> None:
        sandbox.configure_credentials([CredentialBinding(name="token", secret=SECRET)])
        sandbox.close()
        assert sandbox.credential_names == frozenset()

    def test_close_handles_already_removed_root(self, sandbox: LocalSandbox) -> None:
        import shutil

        shutil.rmtree(sandbox.root)
        sandbox.close()

    def test_close_with_non_host_backend(self, tmp_path: Path) -> None:
        box = LocalSandbox(
            root=tmp_path,
            filesystem=Filesystem.in_memory(),
            shell=LocalShell(tmp_path),
        )
        box.close()
        assert not tmp_path.exists()

    def test_facets_raise_after_close(self, sandbox: LocalSandbox) -> None:
        sandbox.close()
        with pytest.raises(SandboxClosedError):
            _ = sandbox.filesystem
        with pytest.raises(SandboxClosedError):
            _ = sandbox.shell

    def test_control_plane_raises_after_close(self, sandbox: LocalSandbox) -> None:
        ref = sandbox.snapshot()
        sandbox.close()
        with pytest.raises(SandboxClosedError):
            sandbox.configure_egress(EgressPolicy.deny_all())
        with pytest.raises(SandboxClosedError):
            sandbox.configure_credentials([])
        with pytest.raises(SandboxClosedError):
            _ = sandbox.snapshot()
        with pytest.raises(SandboxClosedError):
            sandbox.restore(ref)

    def test_identity_and_policy_readable_after_close(
        self, sandbox: LocalSandbox
    ) -> None:
        sandbox_id, root = sandbox.id, sandbox.root
        sandbox.close()
        assert sandbox.id == sandbox_id
        assert sandbox.root == root
        assert sandbox.egress == EgressPolicy.deny_all()
        assert "closed=True" in repr(sandbox)

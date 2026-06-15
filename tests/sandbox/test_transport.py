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

"""Tests for the SandboxTransport fault model.

The fault→exception mapping is the remote facets' exception contract;
the exception→fault mapping is what transport implementations use
server-side. Both directions are exercised exhaustively, plus the
round-trip law: native exception → fault → exception preserves type
and message.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from weakincentives.errors import SnapshotRestoreError, WinkError
from weakincentives.sandbox import (
    TRANSPORT_FAULT_CODES,
    LoopbackTransport,
    SandboxTransport,
    TransportFault,
    TransportFaultCode,
    exception_for_fault,
    fault_for_exception,
)

_CODE_TO_EXCEPTION: dict[TransportFaultCode, type[Exception]] = {
    "not-found": FileNotFoundError,
    "is-a-directory": IsADirectoryError,
    "not-a-directory": NotADirectoryError,
    "exists": FileExistsError,
    "permission": PermissionError,
    "invalid": ValueError,
    "io": OSError,
    "snapshot": SnapshotRestoreError,
    "connectivity": RuntimeError,
}


class TestTransportFault:
    def test_is_wink_and_runtime_error(self) -> None:
        fault = TransportFault("io", "disk failed")
        assert isinstance(fault, WinkError)
        assert isinstance(fault, RuntimeError)

    def test_carries_code_and_message(self) -> None:
        fault = TransportFault("not-found", "no such file: a.txt")
        assert fault.code == "not-found"
        assert fault.message == "no such file: a.txt"
        assert str(fault) == "no such file: a.txt"

    def test_codes_constant_is_exhaustive(self) -> None:
        assert set(TRANSPORT_FAULT_CODES) == set(_CODE_TO_EXCEPTION)


class TestExceptionForFault:
    @pytest.mark.parametrize("code", TRANSPORT_FAULT_CODES)
    def test_every_code_maps_to_contract_exception(
        self, code: TransportFaultCode
    ) -> None:
        error = exception_for_fault(TransportFault(code, "the message"))
        assert type(error) is _CODE_TO_EXCEPTION[code]
        assert str(error) == "the message"

    def test_connectivity_maps_to_plain_runtime_error(self) -> None:
        error = exception_for_fault(TransportFault("connectivity", "link down"))
        assert type(error) is RuntimeError
        assert not isinstance(error, WinkError)


class TestFaultForException:
    @pytest.mark.parametrize(
        ("error", "code"),
        [
            (FileNotFoundError("missing.txt"), "not-found"),
            (IsADirectoryError("Is a directory: d"), "is-a-directory"),
            (NotADirectoryError("Not a directory: f"), "not-a-directory"),
            (FileExistsError("File already exists: f"), "exists"),
            (PermissionError("Path escapes root directory: x"), "permission"),
            (ConnectionResetError("peer reset"), "connectivity"),
            (TimeoutError("deadline"), "connectivity"),
            (SnapshotRestoreError("unknown ref"), "snapshot"),
            (OSError("disk on fire"), "io"),
            (ValueError("Invalid regex pattern"), "invalid"),
            (KeyError("anything else"), "invalid"),
        ],
    )
    def test_native_errors_map_to_codes(
        self, error: BaseException, code: TransportFaultCode
    ) -> None:
        fault = fault_for_exception(error)
        assert fault.code == code
        assert fault.message == str(error)

    def test_existing_fault_passes_through(self) -> None:
        fault = TransportFault("snapshot", "kept as-is")
        assert fault_for_exception(fault) is fault

    @pytest.mark.parametrize(
        "error",
        [
            FileNotFoundError("missing.txt"),
            IsADirectoryError("Is a directory: d"),
            NotADirectoryError("Not a directory: f"),
            FileExistsError("File already exists: f"),
            PermissionError("Path escapes root directory: x"),
            SnapshotRestoreError("unknown ref"),
            ValueError("Invalid regex pattern"),
        ],
    )
    def test_round_trip_preserves_type_and_message(self, error: Exception) -> None:
        restored = exception_for_fault(fault_for_exception(error))
        assert type(restored) is type(error)
        assert str(restored) == str(error)


class TestSandboxTransportProtocol:
    def test_loopback_satisfies_protocol(self, tmp_path: Path) -> None:
        transport = LoopbackTransport(tmp_path)
        assert isinstance(transport, SandboxTransport)
        transport.close()

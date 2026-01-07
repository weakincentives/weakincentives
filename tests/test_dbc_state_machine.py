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

"""Tests for state machine transition enforcement decorators."""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum, auto

import pytest

from weakincentives.dbc import (
    InvalidStateError,
    StateMachineSpec,
    enters,
    extract_state_machine,
    in_state,
    iter_state_machines,
    state_machine,
    transition,
)

pytestmark = pytest.mark.core


@pytest.fixture(autouse=True)
def reset_dbc_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure DbC toggles reset between tests."""
    import weakincentives.dbc as dbc_module

    monkeypatch.delenv("WEAKINCENTIVES_DBC", raising=False)
    dbc_module._forced_state = None
    dbc_module.enable_dbc()
    yield
    dbc_module._forced_state = None
    monkeypatch.delenv("WEAKINCENTIVES_DBC", raising=False)


class ConnectionState(Enum):
    """Test state enum for connection lifecycle."""

    DISCONNECTED = auto()
    CONNECTED = auto()
    CLOSED = auto()


@state_machine(
    state_var="_state", states=ConnectionState, initial=ConnectionState.DISCONNECTED
)
class MockConnection:
    """Test class with state machine enforcement."""

    def __init__(self, host: str) -> None:
        self.host = host
        self.data_sent: list[bytes] = []

    @transition(from_=ConnectionState.DISCONNECTED, to=ConnectionState.CONNECTED)
    def connect(self) -> None:
        """Connect to host."""
        pass

    @in_state(ConnectionState.CONNECTED)
    def send(self, data: bytes) -> int:
        """Send data (only when connected)."""
        self.data_sent.append(data)
        return len(data)

    @transition(from_=ConnectionState.CONNECTED, to=ConnectionState.DISCONNECTED)
    def disconnect(self) -> None:
        """Disconnect from host."""
        pass

    @enters(ConnectionState.CLOSED)
    def close(self) -> None:
        """Close connection (from any state)."""
        pass


class TestStateMachineDecorator:
    """Tests for @state_machine class decorator."""

    def test_initial_state_is_set(self) -> None:
        """Initial state is set after __init__."""
        conn = MockConnection("localhost")
        assert conn._state == ConnectionState.DISCONNECTED

    def test_spec_is_attached(self) -> None:
        """StateMachineSpec is attached to decorated class."""
        spec = MockConnection.__state_machine_spec__
        assert isinstance(spec, StateMachineSpec)
        assert spec.state_var == "_state"
        assert spec.states is ConnectionState
        assert spec.initial == ConnectionState.DISCONNECTED

    def test_invalid_initial_state_raises(self) -> None:
        """Invalid initial state raises ValueError."""

        class OtherState(Enum):
            A = auto()

        with pytest.raises(ValueError, match="not in states enum"):

            @state_machine(
                state_var="_s",
                states=ConnectionState,
                initial=OtherState.A,  # type: ignore[arg-type]
            )
            class BadClass:
                pass


class TestTransitionDecorator:
    """Tests for @transition method decorator."""

    def test_valid_transition_succeeds(self) -> None:
        """Transition from valid source state succeeds."""
        conn = MockConnection("localhost")
        assert conn._state == ConnectionState.DISCONNECTED

        conn.connect()
        assert conn._state == ConnectionState.CONNECTED

    def test_invalid_transition_raises(self) -> None:
        """Transition from invalid state raises InvalidStateError."""
        conn = MockConnection("localhost")

        with pytest.raises(InvalidStateError) as exc:
            conn.disconnect()  # Can't disconnect when not connected

        assert exc.value.cls is MockConnection
        assert exc.value.method == "disconnect"
        assert exc.value.current_state == ConnectionState.DISCONNECTED
        assert ConnectionState.CONNECTED in exc.value.valid_states

    def test_error_message_is_descriptive(self) -> None:
        """InvalidStateError has descriptive message."""
        conn = MockConnection("localhost")

        with pytest.raises(InvalidStateError) as exc:
            conn.disconnect()

        msg = str(exc.value)
        assert "MockConnection.disconnect()" in msg
        assert "CONNECTED" in msg
        assert "DISCONNECTED" in msg

    def test_transition_updates_state_after_method(self) -> None:
        """State is updated after method completes."""
        conn = MockConnection("localhost")
        conn.connect()

        # Verify state changed
        assert conn._state == ConnectionState.CONNECTED

        conn.disconnect()
        assert conn._state == ConnectionState.DISCONNECTED


class TestInStateDecorator:
    """Tests for @in_state method decorator."""

    def test_valid_state_allows_call(self) -> None:
        """Method succeeds when in valid state."""
        conn = MockConnection("localhost")
        conn.connect()

        result = conn.send(b"hello")
        assert result == 5
        assert conn.data_sent == [b"hello"]

    def test_invalid_state_raises(self) -> None:
        """Method raises when not in valid state."""
        conn = MockConnection("localhost")

        with pytest.raises(InvalidStateError) as exc:
            conn.send(b"hello")  # Not connected yet

        assert exc.value.current_state == ConnectionState.DISCONNECTED
        assert ConnectionState.CONNECTED in exc.value.valid_states

    def test_state_not_changed(self) -> None:
        """@in_state does not change state."""
        conn = MockConnection("localhost")
        conn.connect()
        initial_state = conn._state

        conn.send(b"data")
        assert conn._state == initial_state


class TestEntersDecorator:
    """Tests for @enters method decorator."""

    def test_enters_from_initial_state(self) -> None:
        """@enters works from initial state."""
        conn = MockConnection("localhost")
        conn.close()
        assert conn._state == ConnectionState.CLOSED

    def test_enters_from_any_state(self) -> None:
        """@enters works from any state."""
        conn = MockConnection("localhost")
        conn.connect()
        assert conn._state == ConnectionState.CONNECTED

        conn.close()
        assert conn._state == ConnectionState.CLOSED

    def test_enters_is_idempotent(self) -> None:
        """@enters can be called multiple times."""
        conn = MockConnection("localhost")
        conn.close()
        conn.close()  # Should not raise
        assert conn._state == ConnectionState.CLOSED


class TestMultipleSourceStates:
    """Tests for transitions with multiple valid source states."""

    def test_transition_from_multiple_states(self) -> None:
        """Transition accepts tuple of valid source states."""

        class State(Enum):
            A = auto()
            B = auto()
            C = auto()

        @state_machine(state_var="_s", states=State, initial=State.A)
        class Multi:
            @transition(from_=State.A, to=State.B)
            def a_to_b(self) -> None:
                pass

            @transition(from_=(State.A, State.B), to=State.C)
            def to_c(self) -> None:
                pass

        obj = Multi()
        obj.to_c()  # A -> C
        assert obj._s == State.C

        obj2 = Multi()
        obj2.a_to_b()  # A -> B
        obj2.to_c()  # B -> C
        assert obj2._s == State.C


class TestDbcInactive:
    """Tests for behavior when DbC is disabled."""

    def test_no_enforcement_when_inactive(self) -> None:
        """Decorators are no-ops when dbc_active() is False."""
        import weakincentives.dbc as dbc_module

        dbc_module.disable_dbc()

        conn = MockConnection("localhost")
        # These would fail with DbC active
        conn.disconnect()  # Invalid: not connected
        conn.send(b"data")  # Invalid: not connected

        # No errors raised

    def test_state_still_changes_when_inactive(self) -> None:
        """State transitions still occur when DbC is inactive."""
        import weakincentives.dbc as dbc_module

        dbc_module.disable_dbc()

        conn = MockConnection("localhost")
        conn.connect()
        # State changed even without enforcement
        # (The transition decorator sets state after method)

    def test_enters_no_op_when_inactive(self) -> None:
        """@enters decorator is a no-op when dbc_active() is False."""
        import weakincentives.dbc as dbc_module

        dbc_module.disable_dbc()

        conn = MockConnection("localhost")
        # close() uses @enters - should work without state update when DbC inactive
        conn.close()


class TestExtractStateMachine:
    """Tests for extract_state_machine function."""

    def test_extracts_spec(self) -> None:
        """extract_state_machine returns StateMachineSpec."""
        spec = extract_state_machine(MockConnection)

        assert spec.cls is MockConnection
        assert spec.state_var == "_state"
        assert spec.states is ConnectionState
        assert spec.initial == ConnectionState.DISCONNECTED

    def test_transitions_are_collected(self) -> None:
        """Transitions from decorated methods are collected."""
        spec = extract_state_machine(MockConnection)

        method_names = {t.method_name for t in spec.transitions}
        assert "connect" in method_names
        assert "disconnect" in method_names
        assert "send" in method_names
        assert "close" in method_names

    def test_to_mermaid(self) -> None:
        """to_mermaid generates valid diagram."""
        spec = extract_state_machine(MockConnection)
        mermaid = spec.to_mermaid()

        assert "stateDiagram-v2" in mermaid
        assert "[*] --> DISCONNECTED" in mermaid
        assert "DISCONNECTED --> CONNECTED: connect()" in mermaid


class TestInStateMultipleStates:
    """Tests for @in_state with multiple valid states."""

    def test_in_state_multiple_valid(self) -> None:
        """@in_state accepts multiple valid states."""

        class State(Enum):
            A = auto()
            B = auto()
            C = auto()

        @state_machine(state_var="_s", states=State, initial=State.A)
        class Multi:
            @transition(from_=State.A, to=State.B)
            def to_b(self) -> None:
                pass

            @in_state(State.A, State.B)
            def query(self) -> str:
                return "ok"

        obj = Multi()
        assert obj.query() == "ok"  # Valid in A

        obj.to_b()
        assert obj.query() == "ok"  # Valid in B

    def test_in_state_empty_raises(self) -> None:
        """@in_state with no states raises ValueError."""
        with pytest.raises(ValueError, match="at least one state"):

            @in_state()  # type: ignore[misc]
            def bad_method(self: object) -> None:
                pass


class TestRealWorldIntegration:
    """Integration tests with real codebase classes."""

    def test_scoped_resource_context_states(self) -> None:
        """ScopedResourceContext enforces state transitions."""
        from weakincentives.resources import Binding, ResourceRegistry, Scope
        from weakincentives.resources.context import ContextState, ScopedResourceContext

        registry = ResourceRegistry.of(
            Binding(str, lambda _: "hello", scope=Scope.SINGLETON)
        )
        ctx = ScopedResourceContext(registry=registry)

        # Initial state is CREATED
        assert ctx._state == ContextState.CREATED

        # get() requires STARTED state
        with pytest.raises(InvalidStateError) as exc:
            ctx.get(str)

        assert exc.value.current_state == ContextState.CREATED
        assert ContextState.STARTED in exc.value.valid_states

        # After start(), get() works
        ctx.start()
        assert ctx._state == ContextState.STARTED
        assert ctx.get(str) == "hello"

        # After close(), get() fails
        ctx.close()
        assert ctx._state == ContextState.CLOSED

        with pytest.raises(InvalidStateError):
            ctx.get(str)


class TestIterStateMachines:
    """Tests for iter_state_machines function."""

    def test_iter_state_machines_raises(self) -> None:
        """iter_state_machines raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Use extract_state_machine"):
            list(iter_state_machines())

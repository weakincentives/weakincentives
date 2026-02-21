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

"""Tests for ShutdownCoordinator and signal handling."""

from __future__ import annotations

import signal
from unittest.mock import MagicMock, patch

from weakincentives.runtime import ShutdownCoordinator

# =============================================================================
# ShutdownCoordinator Tests
# =============================================================================


def test_coordinator_install_returns_singleton(reset_coordinator: None) -> None:
    """ShutdownCoordinator.install() returns the same instance."""
    _ = reset_coordinator
    coordinator1 = ShutdownCoordinator.install()
    coordinator2 = ShutdownCoordinator.install()
    assert coordinator1 is coordinator2


def test_coordinator_get_returns_none_before_install(reset_coordinator: None) -> None:
    """ShutdownCoordinator.get() returns None before install."""
    _ = reset_coordinator
    assert ShutdownCoordinator.get() is None


def test_coordinator_get_returns_instance_after_install(
    reset_coordinator: None,
) -> None:
    """ShutdownCoordinator.get() returns instance after install."""
    _ = reset_coordinator
    installed = ShutdownCoordinator.install()
    assert ShutdownCoordinator.get() is installed


def test_coordinator_register_adds_callback(reset_coordinator: None) -> None:
    """ShutdownCoordinator.register() adds callback to list."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    callback = MagicMock()

    coordinator.register(callback)
    coordinator.trigger()

    callback.assert_called_once()


def test_coordinator_trigger_invokes_all_callbacks(reset_coordinator: None) -> None:
    """ShutdownCoordinator.trigger() invokes all registered callbacks."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    callbacks = [MagicMock() for _ in range(3)]

    for cb in callbacks:
        coordinator.register(cb)

    coordinator.trigger()

    for cb in callbacks:
        cb.assert_called_once()


def test_coordinator_unregister_removes_callback(reset_coordinator: None) -> None:
    """ShutdownCoordinator.unregister() removes callback from list."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    callback = MagicMock()

    coordinator.register(callback)
    coordinator.unregister(callback)
    coordinator.trigger()

    callback.assert_not_called()


def test_coordinator_unregister_nonexistent_callback_is_safe(
    reset_coordinator: None,
) -> None:
    """ShutdownCoordinator.unregister() is safe for unregistered callbacks."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    callback = MagicMock()

    # Should not raise
    coordinator.unregister(callback)


def test_coordinator_triggered_property(reset_coordinator: None) -> None:
    """ShutdownCoordinator.triggered property reflects state."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()

    assert coordinator.triggered is False
    coordinator.trigger()
    assert coordinator.triggered is True


def test_coordinator_late_register_invokes_immediately(reset_coordinator: None) -> None:
    """Callback registered after trigger is invoked immediately."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    coordinator.trigger()

    callback = MagicMock()
    coordinator.register(callback)

    callback.assert_called_once()


def test_coordinator_reset_clears_state(reset_coordinator: None) -> None:
    """ShutdownCoordinator.reset() clears singleton and state."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    callback = MagicMock()
    coordinator.register(callback)
    coordinator.trigger()

    ShutdownCoordinator.reset()

    assert ShutdownCoordinator.get() is None

    # New coordinator should be fresh
    new_coordinator = ShutdownCoordinator.install()
    assert new_coordinator is not coordinator
    assert new_coordinator.triggered is False


def test_coordinator_signal_handler(reset_coordinator: None) -> None:
    """ShutdownCoordinator installs signal handlers."""
    _ = reset_coordinator
    with patch("signal.signal") as mock_signal:
        ShutdownCoordinator.install(signals=(signal.SIGTERM,))
        mock_signal.assert_called()


# =============================================================================
# Signal Handler Tests
# =============================================================================


def test_coordinator_handle_signal_triggers_shutdown(reset_coordinator: None) -> None:
    """ShutdownCoordinator._handle_signal triggers shutdown."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    callback = MagicMock()
    coordinator.register(callback)

    # Directly call _handle_signal to simulate signal receipt
    coordinator._handle_signal(15, None)  # 15 = SIGTERM

    assert coordinator.triggered
    callback.assert_called_once()

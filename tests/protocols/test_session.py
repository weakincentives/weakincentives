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

"""Tests for weakincentives.protocols.session module."""

from __future__ import annotations

from uuid import uuid4

from weakincentives.protocols.session import Subscription


class TestSubscription:
    """Tests for the Subscription class."""

    def test_unsubscribe_calls_callback(self) -> None:
        """Unsubscribe should call the unsubscribe function."""
        called = []

        def unsubscribe_fn() -> None:
            called.append(True)

        sub_id = uuid4()
        sub = Subscription(unsubscribe_fn=unsubscribe_fn, subscription_id=sub_id)
        assert sub.subscription_id == sub_id

        result = sub.unsubscribe()

        assert result is True
        assert called == [True]

    def test_unsubscribe_returns_false_when_already_unsubscribed(self) -> None:
        """Unsubscribe should return False if already unsubscribed."""
        sub = Subscription(unsubscribe_fn=lambda: None, subscription_id=uuid4())

        # First unsubscribe succeeds
        assert sub.unsubscribe() is True
        # Second unsubscribe returns False
        assert sub.unsubscribe() is False

    def test_repr(self) -> None:
        """Repr should include the subscription_id."""
        sub_id = uuid4()
        sub = Subscription(unsubscribe_fn=lambda: None, subscription_id=sub_id)

        repr_str = repr(sub)

        assert "Subscription" in repr_str
        assert str(sub_id) in repr_str

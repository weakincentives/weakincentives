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

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from ._types import SupportsDataclass

if TYPE_CHECKING:
    from ..runtime.session._protocols import SessionProtocol

EnabledPredicate = Callable[[SupportsDataclass], bool] | Callable[[], bool]

# Normalized callable signature that accepts params and session keyword argument
NormalizedEnabledPredicate = Callable[
    [SupportsDataclass | None, "SessionProtocol | None"], bool
]


def callable_requires_positional_argument(callback: EnabledPredicate) -> bool:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if (
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and parameter.default is inspect.Signature.empty
        ):
            return True
    return False


def callable_accepts_session_kwarg(callback: Callable[..., object]) -> bool:
    """Check if the callable accepts a 'session' keyword-only argument."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    param = signature.parameters.get("session")
    if param is None:
        return False
    # Only accept keyword-only parameters for session
    return param.kind == inspect.Parameter.KEYWORD_ONLY


def _normalize_zero_arg_predicate(
    enabled: Callable[..., bool], accepts_session: bool
) -> NormalizedEnabledPredicate:
    """Normalize a zero-argument enabled predicate."""
    if accepts_session:

        def _without_params_with_session(
            _: SupportsDataclass | None,
            session: SessionProtocol | None,
        ) -> bool:
            return bool(enabled(session=session))

        return _without_params_with_session

    zero_arg = cast(Callable[[], bool], enabled)

    def _without_params(
        _: SupportsDataclass | None,
        session: SessionProtocol | None,
    ) -> bool:
        del session
        return bool(zero_arg())

    return _without_params


def _normalize_params_predicate(
    enabled: Callable[..., bool], accepts_session: bool
) -> NormalizedEnabledPredicate:
    """Normalize a predicate that takes params."""
    if accepts_session:

        def _with_params_and_session(
            value: SupportsDataclass | None,
            session: SessionProtocol | None,
        ) -> bool:
            return bool(enabled(cast(SupportsDataclass, value), session=session))

        return _with_params_and_session

    coerced = cast(Callable[[SupportsDataclass], bool], enabled)

    def _with_params(
        value: SupportsDataclass | None,
        session: SessionProtocol | None,
    ) -> bool:
        del session
        return bool(coerced(cast(SupportsDataclass, value)))

    return _with_params


def normalize_enabled_predicate(
    enabled: EnabledPredicate | None,
    params_type: type[SupportsDataclass] | None,
) -> NormalizedEnabledPredicate | None:
    """Normalize enabled predicate to a callable accepting (params, *, session).

    The returned callable always accepts params and session arguments.
    The session argument must be keyword-only in the original callable.
    If the original callable does not accept a session keyword argument,
    the session is not passed to it.

    Supported signatures:
        - () -> bool
        - (*, session) -> bool
        - (params) -> bool
        - (params, *, session) -> bool
    """
    if enabled is None:
        return None

    accepts_session = callable_accepts_session_kwarg(enabled)
    requires_positional = callable_requires_positional_argument(enabled)

    if params_type is None and not requires_positional:
        return _normalize_zero_arg_predicate(enabled, accepts_session)
    return _normalize_params_predicate(enabled, accepts_session)


__all__ = [
    "EnabledPredicate",
    "NormalizedEnabledPredicate",
    "callable_accepts_session_kwarg",
    "callable_requires_positional_argument",
    "normalize_enabled_predicate",
]

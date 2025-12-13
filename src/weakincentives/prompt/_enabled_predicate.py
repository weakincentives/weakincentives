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
    from ..runtime.session.protocols import SessionProtocol

EnabledPredicate = Callable[[SupportsDataclass], bool] | Callable[[], bool]

# Normalized callable signature that accepts both params and optional session
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
    """Check if the callable accepts a 'session' keyword argument."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    param = signature.parameters.get("session")
    if param is None:
        return False
    return param.kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def normalize_enabled_predicate(
    enabled: EnabledPredicate | None,
    params_type: type[SupportsDataclass] | None,
) -> NormalizedEnabledPredicate | None:
    """Normalize enabled predicate to a callable accepting (params, session).

    The returned callable always accepts both params and session arguments.
    If the original callable does not accept a session keyword argument,
    the session is not passed to it.
    """
    if enabled is None:
        return None

    accepts_session = callable_accepts_session_kwarg(enabled)

    if params_type is None and not callable_requires_positional_argument(enabled):
        if accepts_session:
            # Zero-arg + session callable
            zero_arg_with_session = cast(Callable[..., bool], enabled)

            def _without_params_with_session(
                _: SupportsDataclass | None,
                session: SessionProtocol | None,
            ) -> bool:
                return bool(zero_arg_with_session(session=session))

            return _without_params_with_session

        # Zero-arg callable without session
        zero_arg = cast(Callable[[], bool], enabled)

        def _without_params(
            _: SupportsDataclass | None,
            session: SessionProtocol | None,
        ) -> bool:
            del session
            return bool(zero_arg())

        return _without_params

    if accepts_session:
        # Params + session callable
        coerced_with_session = cast(Callable[..., bool], enabled)

        def _with_params_and_session(
            value: SupportsDataclass | None,
            session: SessionProtocol | None,
        ) -> bool:
            return bool(
                coerced_with_session(cast(SupportsDataclass, value), session=session)
            )

        return _with_params_and_session

    # Params only callable
    coerced = cast(Callable[[SupportsDataclass], bool], enabled)

    def _with_params(
        value: SupportsDataclass | None,
        session: SessionProtocol | None,
    ) -> bool:
        del session
        return bool(coerced(cast(SupportsDataclass, value)))

    return _with_params


__all__ = [
    "EnabledPredicate",
    "NormalizedEnabledPredicate",
    "callable_accepts_session_kwarg",
    "callable_requires_positional_argument",
    "normalize_enabled_predicate",
]

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


def _accepts_session_kwarg(callback: Callable[..., bool]) -> bool:
    """Check if callback accepts a 'session' keyword argument."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.name == "session" and param.kind in {
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            return True
    return False


def normalize_enabled_predicate(
    enabled: EnabledPredicate | None,
    params_type: type[SupportsDataclass] | None,
) -> Callable[[SupportsDataclass | None, SessionProtocol | None], bool] | None:
    if enabled is None:
        return None

    wants_session = _accepts_session_kwarg(enabled)
    needs_params = callable_requires_positional_argument(enabled)

    if params_type is None and not needs_params:
        if wants_session:
            # Cast to Any callable since we know it accepts session as kwarg
            zero_arg_with_session = cast(Callable[..., bool], enabled)

            def _without_params_with_session(
                _: SupportsDataclass | None, session: SessionProtocol | None
            ) -> bool:
                return bool(zero_arg_with_session(session=session))

            return _without_params_with_session

        zero_arg = cast(Callable[[], bool], enabled)

        def _without_params(
            _: SupportsDataclass | None, __: SessionProtocol | None
        ) -> bool:
            return bool(zero_arg())

        return _without_params

    if wants_session:
        # Cast to Any callable since we know it accepts params and session as kwarg
        coerced_with_session = cast(Callable[..., bool], enabled)

        def _with_params_and_session(
            value: SupportsDataclass | None, session: SessionProtocol | None
        ) -> bool:
            return bool(
                coerced_with_session(cast(SupportsDataclass, value), session=session)
            )

        return _with_params_and_session

    coerced = cast(Callable[[SupportsDataclass], bool], enabled)

    def _with_params(
        value: SupportsDataclass | None, _: SessionProtocol | None
    ) -> bool:
        return bool(coerced(cast(SupportsDataclass, value)))

    return _with_params


__all__ = [
    "EnabledPredicate",
    "callable_requires_positional_argument",
    "normalize_enabled_predicate",
]

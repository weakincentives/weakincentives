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

"""Shared helpers used to normalize enabled predicates."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar, cast

from ._types import SupportsDataclass

ParamsT = TypeVar("ParamsT", bound=SupportsDataclass)


def callable_requires_positional_argument(callback: Callable[..., Any]) -> bool:
    """Return ``True`` when ``callback`` needs a positional argument."""

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


def normalize_enabled_predicate(
    enabled: Callable[[ParamsT], bool] | Callable[[], bool] | None,
    params_type: type[ParamsT] | type[SupportsDataclass] | None,
) -> Callable[[ParamsT | None], bool] | None:
    """Return a predicate compatible with ``Section``/``Chapter`` call sites."""

    if enabled is None:
        return None
    if params_type is None and not callable_requires_positional_argument(enabled):
        zero_arg = cast(Callable[[], bool], enabled)

        def _without_params(_: ParamsT | None) -> bool:
            return bool(zero_arg())

        return _without_params

    coerced = cast(Callable[[ParamsT], bool], enabled)

    def _with_params(value: ParamsT | None) -> bool:
        return bool(coerced(cast(ParamsT, value)))

    return _with_params


__all__ = [
    "callable_requires_positional_argument",
    "normalize_enabled_predicate",
]

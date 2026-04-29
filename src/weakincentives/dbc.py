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

"""Design-by-contract decorators.

These are runtime contracts; the spine itself never depends on them. Use
``@require`` for preconditions, ``@ensure`` for postconditions, and
``@invariant`` to enforce class invariants on every public method.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable
from typing import Any, ParamSpec, TypeVar, cast

__all__ = [
    "ContractError",
    "PostconditionError",
    "PreconditionError",
    "ensure",
    "invariant",
    "require",
]


class ContractError(AssertionError):
    """Base for contract violations."""


class PreconditionError(ContractError):
    """A ``@require`` predicate returned False."""


class PostconditionError(ContractError):
    """An ``@ensure`` predicate returned False."""


P = ParamSpec("P")
R = TypeVar("R")
ContractResult = bool | tuple[bool, str]


def _evaluate(
    predicate: Callable[..., ContractResult],
    *args: object,
    **kwargs: object,
) -> tuple[bool, str | None]:
    outcome = predicate(*args, **kwargs)
    if isinstance(outcome, tuple):
        ok, message = outcome
        return ok, message
    return bool(outcome), None


def require(
    predicate: Callable[..., ContractResult],
    *,
    message: str = "precondition failed",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Validate a predicate against the call's arguments before invocation."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            ok, detail = _evaluate(predicate, **bound.arguments)
            if not ok:
                raise PreconditionError(detail or message)
            return fn(*args, **kwargs)

        return wrapper

    return decorate


def ensure(
    predicate: Callable[..., ContractResult],
    *,
    message: str = "postcondition failed",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Validate a predicate against the return value (and arguments).

    The predicate receives every argument by name plus a ``result`` keyword
    holding the return value.
    """

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            result = fn(*args, **kwargs)
            payload = {**bound.arguments, "result": result}
            ok, detail = _evaluate(predicate, **payload)
            if not ok:
                raise PostconditionError(detail or message)
            return result

        return wrapper

    return decorate


def invariant(
    predicate: Callable[[Any], ContractResult],
    *,
    message: str = "invariant violated",
) -> Callable[[type[Any]], type[Any]]:
    """Assert ``predicate(self)`` after every public method call."""

    def decorate(cls: type[Any]) -> type[Any]:
        for name, member in _public_methods(cls):
            wrapped = _wrap_with_invariant(member, predicate, message)
            setattr(cls, name, wrapped)
        return cls

    return decorate


def _public_methods(cls: type[Any]) -> Iterable[tuple[str, Callable[..., Any]]]:
    for name in vars(cls):
        if name.startswith("_"):
            continue
        candidate = vars(cls)[name]
        if not callable(candidate):
            continue
        yield name, candidate


def _wrap_with_invariant(
    fn: Callable[..., Any],
    predicate: Callable[[Any], ContractResult],
    message: str,
) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(self: object, *args: object, **kwargs: object) -> object:
        result = fn(self, *args, **kwargs)
        ok, detail = _evaluate(cast("Callable[..., ContractResult]", predicate), self)
        if not ok:
            raise ContractError(detail or message)
        return result

    return wrapper

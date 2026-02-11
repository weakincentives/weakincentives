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

"""Design by Contract (DbC) utilities for :mod:`weakincentives`.

This module provides decorators and utilities for implementing Design by Contract,
a software correctness methodology pioneered by Bertrand Meyer. DbC allows you to
specify formal, precise, and verifiable interface specifications for software
components through preconditions, postconditions, and invariants.

Overview
--------

Design by Contract treats software components as having "contracts" that define:

- **Preconditions**: What must be true before a function executes (caller's obligation)
- **Postconditions**: What must be true after a function executes (callee's guarantee)
- **Invariants**: What must always be true for an object (class-level consistency)
When a contract is violated, an ``AssertionError`` is raised with a descriptive
message indicating which contract failed and the relevant context.

Decorators
----------

@require
^^^^^^^^

Validates preconditions before invoking the wrapped callable. Use this to
specify what callers must guarantee before calling your function.

Predicate functions receive the same arguments as the decorated function and
should return:

- A boolean (``True`` = pass, ``False`` = fail)
- A tuple ``(bool, str)`` where the string provides a custom error message
- Any truthy/falsy value

Example::

    from weakincentives.dbc import require

    def positive(x: int) -> bool:
        return x > 0

    def non_empty(items: list) -> tuple[bool, str]:
        if not items:
            return False, "items list cannot be empty"
        return True, ""

    @require(positive)
    def square_root(x: int) -> float:
        return x ** 0.5

    @require(non_empty)
    def first_item(items: list) -> object:
        return items[0]

    # Multiple preconditions can be specified:
    @require(positive, lambda x: x < 100)
    def bounded_sqrt(x: int) -> float:
        return x ** 0.5

@ensure
^^^^^^^

Validates postconditions once the callable returns or raises. The predicate
receives all original arguments plus a ``result`` keyword argument containing
the return value (or ``exception`` if the function raised).

Example::

    from weakincentives.dbc import ensure

    def result_positive(x: int, *, result: float) -> bool:
        return result >= 0

    def result_greater_than_input(x: int, *, result: int) -> bool:
        return result > x

    @ensure(result_positive)
    def absolute_value(x: int) -> float:
        return abs(x)

    @ensure(result_greater_than_input)
    def increment(x: int) -> int:
        return x + 1

    # Combining require and ensure:
    @require(lambda x: x >= 0)
    @ensure(lambda x, result: result * result <= x < (result + 1) ** 2)
    def integer_sqrt(x: int) -> int:
        return int(x ** 0.5)

@invariant
^^^^^^^^^^

Enforces class invariants before and after public method calls. The predicate
receives the instance (``self``) and should return a boolean or tuple.

Invariants are checked:

- After ``__init__`` completes
- Before and after every public method call (methods not starting with ``_``)

Use ``@skip_invariant`` to exclude specific methods from invariant checking.

Example::

    from weakincentives.dbc import invariant, skip_invariant

    def balance_non_negative(self) -> bool:
        return self.balance >= 0

    def items_consistent(self) -> tuple[bool, str]:
        if len(self._items) != self._count:
            return False, f"count mismatch: {len(self._items)} != {self._count}"
        return True, ""

    @invariant(balance_non_negative)
    class BankAccount:
        def __init__(self, initial_balance: float) -> None:
            self.balance = initial_balance

        def deposit(self, amount: float) -> None:
            self.balance += amount

        def withdraw(self, amount: float) -> None:
            self.balance -= amount

        @skip_invariant
        def _internal_audit(self) -> dict:
            # This method won't trigger invariant checks
            return {"balance": self.balance}

    @invariant(items_consistent)
    class ItemCollection:
        def __init__(self) -> None:
            self._items: list = []
            self._count = 0

        def add(self, item: object) -> None:
            self._items.append(item)
            self._count += 1

Runtime Behavior
----------------

Contract Checking
^^^^^^^^^^^^^^^^^

By default, all contracts are checked at runtime. When a contract fails,
an ``AssertionError`` is raised with details about:

- Which contract type failed (require/ensure/invariant)
- The function/method involved
- The predicate that failed
- The arguments that were passed
- Any custom message from the predicate

Suspending Checks
^^^^^^^^^^^^^^^^^

For performance-sensitive code paths, use :func:`dbc_suspended` to temporarily
disable contract checking::

    from weakincentives.dbc import dbc_suspended

    with dbc_suspended():
        # No contract checks in this block
        result = expensive_hot_path(data)

You can check if contracts are currently active with :func:`dbc_active`::

    from weakincentives.dbc import dbc_active

    if dbc_active():
        print("Contracts are being checked")

Contract Predicates
^^^^^^^^^^^^^^^^^^^

Predicates can return several types of values:

- ``bool``: ``True`` means pass, ``False`` means fail
- ``tuple[bool, str]``: Boolean result with custom error message
- Any object: Evaluated as truthy/falsy
- ``None``: Treated as ``False`` (contract fails)

Example with custom messages::

    def valid_age(age: int) -> tuple[bool, str]:
        if age < 0:
            return False, f"age cannot be negative, got {age}"
        if age > 150:
            return False, f"age seems unrealistic: {age}"
        return True, ""

    @require(valid_age)
    def create_person(age: int) -> dict:
        return {"age": age}

Thread and Async Safety
^^^^^^^^^^^^^^^^^^^^^^^

The DbC system uses ``ContextVar`` for tracking state, making it safe for use
in both multithreaded and async code. Each thread/task has its own independent
contract checking state.

Public API
----------

Decorators:
    - :func:`require` - Precondition decorator
    - :func:`ensure` - Postcondition decorator
    - :func:`invariant` - Class invariant decorator
    - :func:`skip_invariant` - Exclude method from invariant checks

Functions:
    - :func:`dbc_active` - Check if contracts are currently enabled
    - :func:`dbc_suspended` - Context manager to temporarily disable checks
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import ParamSpec, Protocol, TypeVar, cast

from ..types import ContractResult

P = ParamSpec("P")
Q = ParamSpec("Q")
R = TypeVar("R")
S = TypeVar("S")
T = TypeVar("T", bound=object)

type ContractCallable = Callable[..., ContractResult | object]

# Use a ContextVar for thread-safe and async-safe suspension tracking.
# The default is True (DbC always enabled). Only the context manager can
# temporarily suspend checks for performance-sensitive code paths.
_DBC_ACTIVE: ContextVar[bool] = ContextVar("weakincentives_dbc_active", default=True)


def dbc_active() -> bool:
    """Return ``True`` when DbC checks should run.

    DbC is always enabled by default and cannot be globally disabled.
    Use :func:`dbc_suspended` to temporarily disable checks in
    performance-sensitive code paths.
    """
    return _DBC_ACTIVE.get()


def _qualname(target: object) -> str:
    return getattr(target, "__qualname__", repr(target))


@contextmanager
def dbc_suspended() -> Iterator[None]:
    """Temporarily suspend DbC checks inside a ``with`` block.

    Use this context manager sparingly in performance-sensitive code paths
    where contract checking overhead is unacceptable. DbC checks are
    automatically restored when the block exits.

    Example::

        with dbc_suspended():
            # No contract checks in this block
            result = expensive_hot_path(data)
    """
    token = _DBC_ACTIVE.set(False)
    try:
        yield
    finally:
        _DBC_ACTIVE.reset(token)


def _normalize_contract_result(
    result: ContractResult | object,
) -> tuple[bool, str | None]:
    if isinstance(result, tuple):
        sequence_result = cast(Sequence[object], result)
        if not sequence_result:
            msg = "Contract callables must not return empty tuples"
            raise TypeError(msg)
        outcome = bool(sequence_result[0])
        message = None if len(sequence_result) == 1 else str(sequence_result[1])
        return outcome, message
    if isinstance(result, bool):
        return result, None
    if result is None:
        return False, None
    return bool(result), None


def _contract_failure_message(
    *,
    kind: str,
    func: Callable[..., object],
    predicate: ContractCallable,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    detail: str | None,
) -> str:
    predicate_name = getattr(predicate, "__name__", repr(predicate))
    base = (
        f"{kind} contract for {_qualname(func)} failed via {predicate_name}."
        f" Args={args!r} Kwargs={kwargs!r}"
    )
    if detail:
        return f"{base} Details: {detail}"
    return base


def _evaluate_contract(
    *,
    kind: str,
    func: Callable[..., object],
    predicate: ContractCallable,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> None:
    try:
        result = predicate(*args, **kwargs)
    except AssertionError:
        raise
    except Exception as exc:  # pragma: no cover - diagnostics are important
        msg = (
            f"{kind} contract for {_qualname(func)} raised {type(exc).__name__}: {exc}"
        )
        raise AssertionError(msg) from exc
    outcome, detail = _normalize_contract_result(result)
    if not outcome:
        raise AssertionError(
            _contract_failure_message(
                kind=kind,
                func=func,
                predicate=predicate,
                args=args,
                kwargs=kwargs,
                detail=detail,
            )
        )


def require(
    *predicates: ContractCallable,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Validate preconditions before invoking the wrapped callable."""

    if not predicates:
        msg = "@require expects at least one predicate"
        raise ValueError(msg)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            if dbc_active():
                for predicate in predicates:
                    _evaluate_contract(
                        kind="require",
                        func=func,
                        predicate=predicate,
                        args=tuple(args),
                        kwargs=dict(kwargs),
                    )
            return func(*args, **kwargs)

        return wrapped

    return decorator


def ensure(
    *predicates: ContractCallable,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Validate postconditions once the callable returns or raises."""

    if not predicates:
        msg = "@ensure expects at least one predicate"
        raise ValueError(msg)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            if not dbc_active():
                return func(*args, **kwargs)

            try:
                result = func(*args, **kwargs)
            except BaseException as exc:
                for predicate in predicates:
                    _evaluate_contract(
                        kind="ensure",
                        func=func,
                        predicate=predicate,
                        args=tuple(args),
                        kwargs={**kwargs, "exception": exc},
                    )
                raise

            for predicate in predicates:
                _evaluate_contract(
                    kind="ensure",
                    func=func,
                    predicate=predicate,
                    args=tuple(args),
                    kwargs={**kwargs, "result": result},
                )
            return result

        return wrapped

    return decorator


def skip_invariant(func: Callable[Q, S]) -> Callable[Q, S]:  # noqa: UP047
    """Mark a method so invariants are not evaluated around it."""

    class _InvariantSkippable(Protocol):
        __dbc_skip_invariant__: bool

    skippable_func = cast(_InvariantSkippable, func)
    skippable_func.__dbc_skip_invariant__ = True
    return func


def _check_invariants(
    predicates: tuple[ContractCallable, ...],
    *,
    instance: object,
    func: Callable[..., object],
) -> None:
    for predicate in predicates:
        _evaluate_contract(
            kind="invariant",
            func=func,
            predicate=predicate,
            args=(instance,),
            kwargs={},
        )


def _validate_invariant_predicates(
    predicates: tuple[ContractCallable, ...],
) -> tuple[ContractCallable, ...]:
    if not predicates:
        msg = "@invariant expects at least one predicate"
        raise ValueError(msg)
    return predicates


def _wrap_init_with_invariants(
    cls: type[object],
    *,
    predicates: tuple[ContractCallable, ...],
) -> None:
    original_init = cls.__init__

    @wraps(original_init)
    def init_wrapper(self: object, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)
        if dbc_active():
            _check_invariants(predicates, instance=self, func=original_init)

    type.__setattr__(cls, "__init__", init_wrapper)


def _should_wrap_invariants(attribute_name: str, attribute: object) -> bool:
    if attribute_name.startswith("_"):
        return False
    if getattr(attribute, "__dbc_skip_invariant__", False):
        return False
    if isinstance(attribute, (staticmethod, classmethod)):
        return False
    return callable(attribute)


def _wrap_method_with_invariants(
    method: Callable[..., object],
    *,
    predicates: tuple[ContractCallable, ...],
) -> Callable[..., object]:
    @wraps(method)
    def wrapper(self: object, *args: object, **kwargs: object) -> object:
        if not dbc_active():
            return method(self, *args, **kwargs)
        _check_invariants(predicates, instance=self, func=method)
        try:
            return method(self, *args, **kwargs)
        finally:
            _check_invariants(predicates, instance=self, func=method)

    return wrapper


def _wrap_methods_with_invariants(
    cls: type[object],
    *,
    predicates: tuple[ContractCallable, ...],
) -> None:
    for attribute_name, attribute in list(cls.__dict__.items()):
        if not _should_wrap_invariants(attribute_name, attribute):
            continue
        setattr(
            cls,
            attribute_name,
            _wrap_method_with_invariants(attribute, predicates=predicates),
        )


def invariant(*predicates: ContractCallable) -> Callable[[type[T]], type[T]]:
    """Enforce invariants before and after public method calls."""

    predicate_tuple = _validate_invariant_predicates(tuple(predicates))

    def decorator(cls: type[T]) -> type[T]:
        _wrap_init_with_invariants(cls, predicates=predicate_tuple)
        _wrap_methods_with_invariants(cls, predicates=predicate_tuple)
        return cls

    return decorator


__all__ = [
    "dbc_active",
    "dbc_suspended",
    "ensure",
    "invariant",
    "require",
    "skip_invariant",
]

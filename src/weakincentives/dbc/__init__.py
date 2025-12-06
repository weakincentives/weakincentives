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

"""Design by contract utilities for :mod:`weakincentives`."""

from __future__ import annotations

import builtins
import copy
import logging
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import ParamSpec, Protocol, TypeVar, cast

from ..types import ContractResult

P = ParamSpec("P")
Q = ParamSpec("Q")
R = TypeVar("R")
S = TypeVar("S")
T = TypeVar("T", bound=object)

ContractCallable = Callable[..., ContractResult | object]

_ENV_FLAG = "WEAKINCENTIVES_DBC"
_forced_state: bool | None = None


def _coerce_flag(value: str | None) -> bool:
    if value is None:
        return False
    lowered = value.strip().lower()
    return lowered not in {"", "0", "false", "off", "no"}


def dbc_active() -> bool:
    """Return ``True`` when DbC checks should run."""

    if _forced_state is not None:
        return _forced_state
    return _coerce_flag(os.getenv(_ENV_FLAG))


def _qualname(target: object) -> str:
    return getattr(target, "__qualname__", repr(target))


def enable_dbc() -> None:
    """Force DbC enforcement on."""

    global _forced_state
    _forced_state = True


def disable_dbc() -> None:
    """Force DbC enforcement off."""

    global _forced_state
    _forced_state = False


@contextmanager
def dbc_enabled(*, active: bool = True) -> Iterator[None]:
    """Temporarily set the DbC flag inside a ``with`` block."""

    global _forced_state
    previous = _forced_state
    _forced_state = active
    try:
        yield
    finally:
        _forced_state = previous


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


def ensure(*predicates: ContractCallable) -> Callable[[Callable[P, R]], Callable[P, R]]:
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


_SNAPSHOT_SENTINEL = object()


def _snapshot(value: object) -> object:
    try:
        return copy.deepcopy(value)
    except Exception:
        return _SNAPSHOT_SENTINEL


def _compare_snapshots(
    *,
    func: Callable[..., object],
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    snap_args: tuple[object, ...],
    snap_kwargs: Mapping[str, object],
) -> None:
    for index, (original, snapshot) in enumerate(zip(args, snap_args, strict=False)):
        if snapshot is _SNAPSHOT_SENTINEL:
            continue
        if original != snapshot:
            msg = (
                f"pure contract for {_qualname(func)} detected mutation of "
                f"positional argument {index}"
            )
            raise AssertionError(msg)

    for key, snapshot in snap_kwargs.items():
        if snapshot is _SNAPSHOT_SENTINEL:
            continue
        if kwargs[key] != snapshot:
            msg = (
                f"pure contract for {_qualname(func)} detected mutation of "
                f"keyword argument '{key}'"
            )
            raise AssertionError(msg)


def _pure_violation(func: Callable[..., object], target: str) -> Callable[..., object]:
    def raiser(*args: object, **kwargs: object) -> object:  # pragma: no cover - trivial
        msg = f"pure contract for {_qualname(func)} forbids calling {target}"
        raise AssertionError(msg)

    return raiser


@contextmanager
def _pure_environment(func: Callable[..., object]) -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(
            _patch(builtins, "open", _pure_violation(func, "builtins.open"))
        )
        stack.enter_context(
            _patch(Path, "write_text", _pure_violation(func, "Path.write_text"))
        )
        stack.enter_context(
            _patch(Path, "write_bytes", _pure_violation(func, "Path.write_bytes"))
        )
        stack.enter_context(
            _patch(logging.Logger, "_log", _pure_violation(func, "logging"))
        )
        yield


@contextmanager
def _patch(
    obj: object, attribute: str, replacement: Callable[..., object]
) -> Iterator[None]:
    original = getattr(obj, attribute)
    setattr(obj, attribute, replacement)
    try:
        yield
    finally:
        setattr(obj, attribute, original)


def pure(func: Callable[P, R]) -> Callable[P, R]:  # noqa: UP047
    """Validate that the wrapped callable behaves like a pure function."""

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        if not dbc_active():
            return func(*args, **kwargs)

        snapshot_args = tuple(_snapshot(arg) for arg in args)
        snapshot_kwargs: dict[str, object] = {
            key: _snapshot(value) for key, value in kwargs.items()
        }

        with _pure_environment(func):
            result = func(*args, **kwargs)

        _compare_snapshots(
            func=func,
            args=tuple(args),
            kwargs=dict(kwargs),
            snap_args=snapshot_args,
            snap_kwargs=snapshot_kwargs,
        )
        return result

    return wrapped


__all__ = [
    "dbc_active",
    "dbc_enabled",
    "disable_dbc",
    "enable_dbc",
    "ensure",
    "invariant",
    "pure",
    "require",
    "skip_invariant",
]

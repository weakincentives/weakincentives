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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import ParamSpec, Protocol, TypeVar, cast

from ..types import ContractResult

P = ParamSpec("P")
Q = ParamSpec("Q")
R = TypeVar("R")
S = TypeVar("S")
T = TypeVar("T", bound=object)

ContractCallable = Callable[..., ContractResult | object]

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


_SNAPSHOT_SENTINEL = object()
_PURE_STACK_EMPTY: tuple[Callable[..., object], ...] = ()
_PURE_STACK: ContextVar[tuple[Callable[..., object], ...]] = ContextVar[
    tuple[Callable[..., object], ...]
]("weakincentives_pure_stack", default=_PURE_STACK_EMPTY)
_PURE_PATCH_LOCK = RLock()
_pure_patch_depth = 0
_pure_patch_original_open: object | None = None
_pure_patch_original_write_text: object | None = None
_pure_patch_original_write_bytes: object | None = None
_pure_patch_original_log: object | None = None


def _pure_guard(target: str) -> None:
    stack = _PURE_STACK.get()
    func = stack[-1] if stack else None
    msg = f"pure contract for {_qualname(func)} forbids calling {target}"
    raise AssertionError(msg)


def _pure_open_guard(*args: object, **kwargs: object) -> object:
    if _PURE_STACK.get():
        _pure_guard("builtins.open")
    if _pure_patch_original_open is None:  # pragma: no cover
        raise RuntimeError("Pure patch state invalid: builtins.open is missing.")
    return cast(Callable[..., object], _pure_patch_original_open)(*args, **kwargs)


def _pure_write_text_guard(*args: object, **kwargs: object) -> object:
    if _PURE_STACK.get():
        _pure_guard("Path.write_text")
    if _pure_patch_original_write_text is None:  # pragma: no cover
        raise RuntimeError("Pure patch state invalid: Path.write_text is missing.")
    return cast(Callable[..., object], _pure_patch_original_write_text)(*args, **kwargs)


def _pure_write_bytes_guard(*args: object, **kwargs: object) -> object:
    if _PURE_STACK.get():
        _pure_guard("Path.write_bytes")
    if _pure_patch_original_write_bytes is None:  # pragma: no cover
        raise RuntimeError("Pure patch state invalid: Path.write_bytes is missing.")
    return cast(Callable[..., object], _pure_patch_original_write_bytes)(
        *args, **kwargs
    )


def _pure_log_guard(*args: object, **kwargs: object) -> object:
    if _PURE_STACK.get():
        _pure_guard("logging")
    if _pure_patch_original_log is None:  # pragma: no cover
        raise RuntimeError("Pure patch state invalid: logging.Logger._log is missing.")
    return cast(Callable[..., object], _pure_patch_original_log)(*args, **kwargs)


def _activate_pure_patches() -> None:
    global _pure_patch_depth
    global _pure_patch_original_log
    global _pure_patch_original_open
    global _pure_patch_original_write_bytes
    global _pure_patch_original_write_text

    with _PURE_PATCH_LOCK:
        if _pure_patch_depth == 0:
            _pure_patch_original_open = getattr(builtins, "open")  # noqa: B009
            _pure_patch_original_write_text = getattr(Path, "write_text")  # noqa: B009
            _pure_patch_original_write_bytes = getattr(Path, "write_bytes")  # noqa: B009
            _pure_patch_original_log = getattr(logging.Logger, "_log")  # noqa: B009

            setattr(builtins, "open", _pure_open_guard)  # noqa: B010
            setattr(Path, "write_text", _pure_write_text_guard)  # noqa: B010
            setattr(Path, "write_bytes", _pure_write_bytes_guard)  # noqa: B010
            setattr(logging.Logger, "_log", _pure_log_guard)  # noqa: B010

        _pure_patch_depth += 1


def _deactivate_pure_patches() -> None:
    global _pure_patch_depth

    with _PURE_PATCH_LOCK:
        if _pure_patch_depth <= 0:  # pragma: no cover
            raise RuntimeError("pure patch depth underflow")
        _pure_patch_depth -= 1
        if _pure_patch_depth != 0:
            return

        setattr(builtins, "open", _pure_patch_original_open)  # noqa: B010
        setattr(Path, "write_text", _pure_patch_original_write_text)  # noqa: B010
        setattr(Path, "write_bytes", _pure_patch_original_write_bytes)  # noqa: B010
        setattr(logging.Logger, "_log", _pure_patch_original_log)  # noqa: B010


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


@contextmanager
def _pure_environment(func: Callable[..., object]) -> Iterator[None]:
    token = _PURE_STACK.set((*_PURE_STACK.get(), func))
    _activate_pure_patches()
    try:
        yield
    finally:
        _PURE_STACK.reset(token)
        _deactivate_pure_patches()


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
    "dbc_suspended",
    "ensure",
    "invariant",
    "pure",
    "require",
    "skip_invariant",
]

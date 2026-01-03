# Embedding TLA+ Specifications in Python Code

## Purpose

This document explores approaches for co-locating TLA+ formal specifications
with Python implementation code. The goal is to make formal verification a
first-class citizen in the development workflow by embedding specs directly in
the codebase alongside their implementations.

## Motivation

Currently, TLA+ specs live in `specs/tla/` while implementations are in
`src/weakincentives/`. This separation has drawbacks:

- **Context switching**: Developers must navigate between spec and implementation
- **Drift risk**: Specs and code can diverge when changes miss one or the other
- **Discoverability**: New contributors may not know specs exist
- **Maintenance burden**: Keeping specs synchronized requires discipline

Co-locating specs with code could:

- Make formal verification more visible and approachable
- Reduce drift by making specs harder to ignore
- Enable tooling to extract, validate, and check specs automatically
- Create a single source of truth for behavior

## Design Principles

1. **Non-invasive**: Embedding should not affect runtime behavior or performance
2. **Opt-in extraction**: Specs are inert unless explicitly extracted by tooling
3. **Readable**: Embedded specs should be understandable to humans
4. **Standard Python**: Use only Python syntax (decorators, strings, comments)
5. **Tool-friendly**: Support pytest plugin extraction and TLC validation
6. **Gradual adoption**: Allow incremental migration of existing specs

## Approach 1: Docstring-Based Embedding

### Concept

Embed TLA+ specs in docstrings using a special marker format. The pytest plugin
parses docstrings, extracts TLA+ blocks, and validates them.

### Example

```python
from weakincentives.dbc import require, ensure

class RedisMailbox:
    """Redis-backed message queue with SQS-compatible semantics.

    TLA+ Specification:
    -------------------

    ```tla
    ------------------------ MODULE RedisMailboxReceive ----------------------
    (* Receive operation: atomically move message from pending to invisible *)

    EXTENDS Integers, Sequences

    CONSTANTS MaxMessages, VisibilityTimeout
    NULL == 0

    VARIABLES pending, invisible, handles, deliveryCounts

    Receive(consumer) ==
        /\ Len(pending) > 0
        /\ LET msgId == Head(pending)
               newHandle == NextUUID()
               newExpiry == Now() + VisibilityTimeout
               newCount == deliveryCounts[msgId] + 1
           IN /\ pending' = Tail(pending)
              /\ invisible' = invisible @@ (msgId :> [
                    expiresAt |-> newExpiry,
                    handle |-> newHandle
                 ])
              /\ handles' = handles @@ (msgId :> newHandle)
              /\ deliveryCounts' = [deliveryCounts EXCEPT ![msgId] = newCount]

    ReceiveInvariant ==
        /\ \A msgId \in DOMAIN invisible:
            msgId \notin pending
        /\ \A msgId \in DOMAIN deliveryCounts:
            deliveryCounts[msgId] > 0 => msgId \in DOMAIN invisible
    ==========================================================================
    ```

    Invariants verified:
    - Message state exclusivity (INV-1)
    - Receipt handle freshness (INV-2)
    - Delivery count monotonicity (INV-4)
    """

    @require(lambda self, visibility_timeout: visibility_timeout > 0)
    @ensure(lambda result: all(m.delivery_count > 0 for m in result))
    def receive(
        self,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
        max_messages: int = 1,
    ) -> list[Message]:
        """Receive messages from the queue.

        TLA+ Action: Receive(consumer)
        """
        # Implementation...
```

### Pros

- Uses standard Python docstrings
- Easy to read alongside implementation
- No new syntax to learn
- Tools can extract with regex or AST parsing

### Cons

- Docstrings are meant for documentation, mixing concerns
- Large specs make docstrings unwieldy
- No syntax highlighting for TLA+ in most editors
- Difficult to reference shared TLA+ definitions across modules

### Implementation

```python
# tests/plugins/tla_extraction.py

import ast
import re
from pathlib import Path
from typing import Iterator

def extract_tla_from_docstring(docstring: str) -> str | None:
    """Extract TLA+ spec from docstring."""
    match = re.search(
        r'```tla\s+(.*?)\s+```',
        docstring,
        re.DOTALL | re.MULTILINE
    )
    return match.group(1) if match else None

def extract_tla_from_module(module_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (name, tla_spec) pairs from a Python module."""
    source = module_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            docstring = ast.get_docstring(node)
            if docstring:
                tla_spec = extract_tla_from_docstring(docstring)
                if tla_spec:
                    yield (node.name, tla_spec)

def pytest_configure(config):
    """Extract TLA+ specs and validate with TLC."""
    if not config.getoption("--check-tla"):
        return

    specs_dir = Path("specs/tla/extracted")
    specs_dir.mkdir(exist_ok=True)

    # Extract from source
    for py_file in Path("src").rglob("*.py"):
        for name, tla_spec in extract_tla_from_module(py_file):
            spec_file = specs_dir / f"{py_file.stem}_{name}.tla"
            spec_file.write_text(tla_spec)

    # Run TLC model checker
    # ... (invoke TLC on extracted specs)
```

## Approach 2: Decorator-Based TLA+ Annotations

### Concept

Extend the DbC system with `@tla_spec` and `@tla_invariant` decorators that
attach TLA+ specifications as metadata. The pytest plugin extracts these and
assembles them into complete TLA+ modules.

### Example

```python
from weakincentives.dbc import require, ensure
from weakincentives.tla import tla_spec, tla_invariant, tla_action

@tla_spec(
    module_name="RedisMailboxReceive",
    constants={"MaxMessages": 100, "VisibilityTimeout": 30},
    variables=["pending", "invisible", "handles", "deliveryCounts"],
)
class RedisMailbox:
    """Redis-backed message queue."""

    @tla_action("""
        /\ Len(pending) > 0
        /\ LET msgId == Head(pending)
               newHandle == NextUUID()
               newExpiry == Now() + VisibilityTimeout
               newCount == deliveryCounts[msgId] + 1
           IN /\ pending' = Tail(pending)
              /\ invisible' = invisible @@ (msgId :> [
                    expiresAt |-> newExpiry,
                    handle |-> newHandle
                 ])
              /\ handles' = handles @@ (msgId :> newHandle)
              /\ deliveryCounts' = [deliveryCounts EXCEPT ![msgId] = newCount]
    """)
    @tla_invariant("""
        /\ \\A msgId \\in DOMAIN invisible: msgId \\notin pending
        /\ \\A msgId \\in DOMAIN deliveryCounts:
            deliveryCounts[msgId] > 0 => msgId \\in DOMAIN invisible
    """)
    @require(lambda self, visibility_timeout: visibility_timeout > 0)
    @ensure(lambda result: all(m.delivery_count > 0 for m in result))
    def receive(
        self,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
        max_messages: int = 1,
    ) -> list[Message]:
        """Receive messages from the queue."""
        # Implementation...
```

### Pros

- Clean separation from docstrings
- Decorators are Python-native and well-understood
- Can attach metadata (module name, constants, variables)
- Easy to extract programmatically via `__tla_spec__` attribute
- Composes with existing DbC decorators

### Cons

- Large TLA+ specs as string literals can be hard to read
- No syntax highlighting for embedded TLA+
- Requires new decorators in `weakincentives.tla` module
- Still separates spec from implementation logic

### Implementation

```python
# src/weakincentives/tla/__init__.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")

@dataclass
class TLASpec:
    """Metadata for TLA+ specification."""
    module_name: str
    constants: dict[str, Any] = field(default_factory=dict)
    variables: list[str] = field(default_factory=list)
    actions: dict[str, str] = field(default_factory=dict)
    invariants: dict[str, str] = field(default_factory=dict)

def tla_spec(
    module_name: str,
    constants: dict[str, Any] | None = None,
    variables: list[str] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Attach TLA+ specification metadata to a class."""
    def decorator(cls: type[T]) -> type[T]:
        spec = TLASpec(
            module_name=module_name,
            constants=constants or {},
            variables=variables or [],
        )
        setattr(cls, "__tla_spec__", spec)
        return cls
    return decorator

def tla_action(action_spec: str) -> Callable[[Callable], Callable]:
    """Attach TLA+ action specification to a method."""
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, "__tla_actions__"):
            setattr(func, "__tla_actions__", [])
        func.__tla_actions__.append(action_spec)

        # Also register with class spec
        return func
    return decorator

def tla_invariant(invariant_spec: str) -> Callable[[Callable], Callable]:
    """Attach TLA+ invariant to a method."""
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, "__tla_invariants__"):
            setattr(func, "__tla_invariants__", [])
        func.__tla_invariants__.append(invariant_spec)
        return func
    return decorator
```

```python
# tests/plugins/tla_extraction.py

import inspect
from pathlib import Path

def extract_tla_specs(module) -> dict[str, str]:
    """Extract all TLA+ specs from a module."""
    specs = {}

    for name, obj in inspect.getmembers(module):
        if not inspect.isclass(obj):
            continue

        tla_spec = getattr(obj, "__tla_spec__", None)
        if not tla_spec:
            continue

        # Build complete TLA+ module
        tla_module = build_tla_module(obj, tla_spec)
        specs[tla_spec.module_name] = tla_module

    return specs

def build_tla_module(cls, spec: TLASpec) -> str:
    """Assemble complete TLA+ module from class metadata."""
    lines = [
        f"------------------------ MODULE {spec.module_name} ------------------------",
        "",
        "EXTENDS Integers, Sequences, FiniteSets",
        "",
    ]

    # Constants
    if spec.constants:
        lines.append("CONSTANTS")
        for name, value in spec.constants.items():
            lines.append(f"    {name}")
        lines.append("")

    # Variables
    if spec.variables:
        lines.append("VARIABLES")
        for var in spec.variables:
            lines.append(f"    {var}")
        lines.append("")
        lines.append(f"vars == <<{', '.join(spec.variables)}>>")
        lines.append("")

    # Actions from methods
    for method_name, method in inspect.getmembers(cls, inspect.isfunction):
        actions = getattr(method, "__tla_actions__", [])
        for action in actions:
            lines.append(f"{method_name.title()} ==")
            lines.append(f"    {action}")
            lines.append("")

        invariants = getattr(method, "__tla_invariants__", [])
        for idx, inv in enumerate(invariants):
            lines.append(f"{method_name.title()}Invariant{idx} ==")
            lines.append(f"    {inv}")
            lines.append("")

    lines.append("=" * 77)
    return "\n".join(lines)
```

## Approach 3: Python DSL for TLA+

### Concept

Create a Python DSL that looks like TLA+ but is valid Python. Use operator
overloading and context managers to build TLA+ AST, then render to TLA+ syntax.

### Example

```python
from weakincentives.tla.dsl import (
    TLAModule, CONSTANTS, VARIABLES, Action, Invariant,
    Let, Head, Tail, Len, DOMAIN,
)
from weakincentives.dbc import require, ensure

# Define TLA+ spec in Python
mailbox_spec = TLAModule("RedisMailboxReceive")

with mailbox_spec:
    CONSTANTS(MaxMessages=100, VisibilityTimeout=30)
    VARIABLES("pending", "invisible", "handles", "deliveryCounts")

    @Action
    def Receive(consumer, pending, invisible, handles, deliveryCounts):
        msgId = Head(pending)
        newHandle = "NextUUID()"
        newExpiry = "Now() + VisibilityTimeout"
        newCount = deliveryCounts[msgId] + 1

        return {
            "pending": Tail(pending),
            "invisible": invisible | {msgId: {"expiresAt": newExpiry, "handle": newHandle}},
            "handles": handles | {msgId: newHandle},
            "deliveryCounts": deliveryCounts.update(msgId, newCount),
        }

    @Invariant
    def ReceiveInvariant(pending, invisible, deliveryCounts):
        return (
            all(msgId not in pending for msgId in DOMAIN(invisible))
            and all(
                deliveryCounts[msgId] > 0 implies msgId in DOMAIN(invisible)
                for msgId in DOMAIN(deliveryCounts)
            )
        )

class RedisMailbox:
    """Redis-backed message queue."""

    __tla_spec__ = mailbox_spec

    @require(lambda self, visibility_timeout: visibility_timeout > 0)
    @ensure(lambda result: all(m.delivery_count > 0 for m in result))
    def receive(
        self,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
        max_messages: int = 1,
    ) -> list[Message]:
        """Receive messages from the queue."""
        # Implementation...
```

### Pros

- Full Python syntax with IDE support (autocomplete, type checking)
- Can validate DSL structure at import time
- Pythonic and familiar to developers
- Enables programmatic spec generation

### Cons

- Significant implementation effort for DSL
- May not support all TLA+ features
- Python syntax constraints limit expressiveness
- Not "real" TLA+, requires translation layer
- Learning curve for DSL API

### Implementation Sketch

```python
# src/weakincentives/tla/dsl.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class TLAExpr:
    """Base class for TLA+ expressions."""
    def render(self) -> str:
        raise NotImplementedError

@dataclass
class TLAVariable(TLAExpr):
    name: str

    def render(self) -> str:
        return self.name

@dataclass
class TLAFunctionCall(TLAExpr):
    func: str
    args: list[TLAExpr]

    def render(self) -> str:
        arg_strs = [arg.render() for arg in self.args]
        return f"{self.func}({', '.join(arg_strs)})"

def Head(seq: TLAExpr) -> TLAExpr:
    return TLAFunctionCall("Head", [seq])

def Tail(seq: TLAExpr) -> TLAExpr:
    return TLAFunctionCall("Tail", [seq])

def Len(seq: TLAExpr) -> TLAExpr:
    return TLAFunctionCall("Len", [seq])

def DOMAIN(func: TLAExpr) -> TLAExpr:
    return TLAFunctionCall("DOMAIN", [func])

class TLAModule:
    """TLA+ module builder."""

    def __init__(self, name: str):
        self.name = name
        self.constants: dict[str, Any] = {}
        self.variables: list[str] = []
        self.actions: dict[str, Callable] = {}
        self.invariants: dict[str, Callable] = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def render(self) -> str:
        """Render to TLA+ syntax."""
        lines = [
            f"------------------------ MODULE {self.name} ------------------------",
            "",
            "EXTENDS Integers, Sequences, FiniteSets",
            "",
        ]

        # Constants
        if self.constants:
            lines.append("CONSTANTS")
            for name in self.constants:
                lines.append(f"    {name}")
            lines.append("")

        # Variables
        if self.variables:
            lines.append("VARIABLES")
            for var in self.variables:
                lines.append(f"    {var}")
            lines.append("")

        # Actions
        for name, action_func in self.actions.items():
            # Would need to translate Python AST to TLA+
            lines.append(f"{name} ==")
            lines.append("    (* Generated from Python *)")
            lines.append("")

        lines.append("=" * 77)
        return "\n".join(lines)

def CONSTANTS(**kwargs):
    """Declare TLA+ constants."""
    # Implementation would capture in module context
    pass

def VARIABLES(*names):
    """Declare TLA+ variables."""
    pass

def Action(func: Callable) -> Callable:
    """Mark function as TLA+ action."""
    func.__tla_action__ = True
    return func

def Invariant(func: Callable) -> Callable:
    """Mark function as TLA+ invariant."""
    func.__tla_invariant__ = True
    return func
```

## Approach 4: Hybrid DbC + TLA+ Fragment Decorators

### Concept

Extend existing DbC decorators to optionally include TLA+ fragments. The
`@require`/`@ensure`/`@invariant` decorators gain a `tla=` parameter that
provides the formal TLA+ equivalent of the runtime check.

### Example

```python
from weakincentives.dbc import require, ensure, invariant

class RedisMailbox:
    """Redis-backed message queue."""

    @require(
        lambda self, visibility_timeout: visibility_timeout > 0,
        tla="visibility_timeout > 0"
    )
    @ensure(
        lambda result: all(m.delivery_count > 0 for m in result),
        tla=r"\A m \in result: m.delivery_count > 0"
    )
    def receive(
        self,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
        max_messages: int = 1,
    ) -> list[Message]:
        """Receive messages from the queue.

        TLA+ Preconditions:
        - visibility_timeout > 0

        TLA+ Postconditions:
        - All returned messages have delivery_count > 0
        """
        # Implementation...

@invariant(
    lambda self: self.balance >= 0,
    tla=r"balance >= 0",
    tla_id="INV-AccountBalance"
)
class Account:
    """Bank account with non-negative balance invariant."""

    def __init__(self, initial_balance: int = 0):
        self.balance = initial_balance
```

### Pros

- Minimal extension to existing DbC system
- Runtime checks and formal specs stay synchronized
- Gradual migration: add `tla=` parameters over time
- Developers already familiar with DbC decorators

### Cons

- Duplicates logic (once in Python lambda, once in TLA+)
- TLA+ fragments, not complete specifications
- Difficult to express complex TLA+ invariants as Python predicates
- Can't capture full state machine semantics

### Implementation

```python
# src/weakincentives/dbc/__init__.py (modifications)

from __future__ import annotations

from typing import TypedDict

class TLAMetadata(TypedDict, total=False):
    """TLA+ metadata for DbC decorators."""
    fragment: str
    id: str | None

def require(
    *predicates: ContractCallable,
    tla: str | None = None,
    tla_id: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Validate preconditions before invoking the wrapped callable.

    Args:
        *predicates: Runtime predicate callables
        tla: Optional TLA+ fragment equivalent of precondition
        tla_id: Optional identifier for cross-referencing (e.g., "PRE-1")
    """
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

        # Attach TLA+ metadata
        if tla:
            if not hasattr(wrapped, "__tla_preconditions__"):
                setattr(wrapped, "__tla_preconditions__", [])
            wrapped.__tla_preconditions__.append({
                "fragment": tla,
                "id": tla_id,
            })

        return wrapped

    return decorator
```

```python
# tests/plugins/tla_extraction.py

def extract_dbc_tla_fragments(module) -> dict[str, list[dict]]:
    """Extract TLA+ fragments from DbC-decorated functions."""
    fragments = {}

    for name, obj in inspect.getmembers(module):
        if not callable(obj):
            continue

        preconditions = getattr(obj, "__tla_preconditions__", [])
        postconditions = getattr(obj, "__tla_postconditions__", [])
        invariants = getattr(obj, "__tla_invariants__", [])

        if preconditions or postconditions or invariants:
            fragments[name] = {
                "preconditions": preconditions,
                "postconditions": postconditions,
                "invariants": invariants,
            }

    return fragments
```

## Approach 5: Structured Metadata with `@formal_spec` Decorator

### Concept

Create a dedicated `@formal_spec` decorator that captures structured metadata
about TLA+ state machines, actions, and invariants. The decorator doesn't
contain TLA+ syntax directly but provides a Python-friendly schema that can
be mechanically translated to TLA+.

### Example

```python
from weakincentives.formal import formal_spec, Action, Invariant, StateVar
from weakincentives.dbc import require, ensure

@formal_spec(
    module="RedisMailbox",
    state_vars=[
        StateVar("pending", type="Sequence", description="Pending message IDs"),
        StateVar("invisible", type="Function", description="msg_id -> {expiresAt, handle}"),
        StateVar("handles", type="Function", description="msg_id -> current valid handle"),
        StateVar("deliveryCounts", type="Function", description="msg_id -> count"),
    ],
    actions=[
        Action(
            name="Receive",
            parameters=["consumer", "visibility_timeout"],
            preconditions=[
                "Len(pending) > 0",
                "visibility_timeout > 0",
            ],
            updates={
                "pending": "Tail(pending)",
                "invisible": "invisible @@ (Head(pending) :> NewInvisibleEntry)",
                "handles": "handles @@ (Head(pending) :> NewHandle)",
                "deliveryCounts": "IncrementCount(Head(pending))",
            },
        ),
    ],
    invariants=[
        Invariant(
            id="INV-1",
            name="MessageStateExclusive",
            predicate=r"\A msgId: (msgId \in pending) XOR (msgId \in DOMAIN invisible)",
        ),
        Invariant(
            id="INV-4",
            name="DeliveryCountMonotonic",
            predicate=r"\A msgId: deliveryCounts[msgId] >= 0",
        ),
    ],
)
class RedisMailbox:
    """Redis-backed message queue."""

    @require(lambda self, visibility_timeout: visibility_timeout > 0)
    @ensure(lambda result: all(m.delivery_count > 0 for m in result))
    def receive(
        self,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
        max_messages: int = 1,
    ) -> list[Message]:
        """Receive messages from the queue."""
        # Implementation...
```

### Pros

- Structured data is easier to process than raw TLA+ strings
- Can validate metadata schema at import time
- Generates complete, well-formed TLA+ modules
- Clear separation between spec and implementation
- Supports gradual spec construction

### Cons

- Requires comprehensive metadata schema design
- Translation from metadata to TLA+ adds complexity
- May not support all TLA+ expressiveness
- Another API surface to learn and maintain

### Implementation

```python
# src/weakincentives/formal/__init__.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class StateVar:
    """TLA+ state variable metadata."""
    name: str
    type: str
    description: str = ""

@dataclass
class Action:
    """TLA+ action metadata."""
    name: str
    parameters: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    updates: dict[str, str] = field(default_factory=dict)

@dataclass
class Invariant:
    """TLA+ invariant metadata."""
    id: str
    name: str
    predicate: str

@dataclass
class FormalSpec:
    """Complete formal specification metadata."""
    module: str
    state_vars: list[StateVar] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    invariants: list[Invariant] = field(default_factory=list)

    def to_tla(self) -> str:
        """Generate TLA+ module from metadata."""
        lines = [
            f"------------------------ MODULE {self.module} ------------------------",
            "",
            "EXTENDS Integers, Sequences, FiniteSets",
            "",
        ]

        # State variables
        if self.state_vars:
            lines.append("VARIABLES")
            for var in self.state_vars:
                lines.append(f"    {var.name}  \\* {var.description}")
            lines.append("")

        # Actions
        for action in self.actions:
            lines.append(f"{action.name}({', '.join(action.parameters)}) ==")

            # Preconditions
            for precond in action.preconditions:
                lines.append(f"    /\\ {precond}")

            # Updates
            if action.updates:
                lines.append("    /\\ LET")
                for var, expr in action.updates.items():
                    lines.append(f"           {var}' = {expr}")
                lines.append("       IN TRUE")

            lines.append("")

        # Invariants
        for inv in self.invariants:
            lines.append(f"\\* {inv.id}")
            lines.append(f"{inv.name} ==")
            lines.append(f"    {inv.predicate}")
            lines.append("")

        lines.append("=" * 77)
        return "\n".join(lines)

def formal_spec(
    module: str,
    state_vars: list[StateVar] | None = None,
    actions: list[Action] | None = None,
    invariants: list[Invariant] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Attach formal specification metadata to a class."""
    def decorator(cls: type[T]) -> type[T]:
        spec = FormalSpec(
            module=module,
            state_vars=state_vars or [],
            actions=actions or [],
            invariants=invariants or [],
        )
        setattr(cls, "__formal_spec__", spec)
        return cls
    return decorator
```

## Pytest Plugin Design

### Configuration

```python
# pyproject.toml

[tool.pytest.ini_options]
tla_extraction_enabled = true
tla_output_dir = "specs/tla/extracted"
tla_run_model_checker = true
tla_model_checker = "tlc"  # or "apalache"
```

### Plugin Implementation

```python
# tests/plugins/pytest_tla.py

import subprocess
from pathlib import Path
from typing import Any

import pytest

def pytest_addoption(parser):
    """Add command-line options for TLA+ checking."""
    parser.addoption(
        "--check-tla",
        action="store_true",
        default=False,
        help="Extract and validate TLA+ specs from code",
    )
    parser.addoption(
        "--tla-only",
        action="store_true",
        default=False,
        help="Only run TLA+ extraction/validation, skip tests",
    )

def pytest_configure(config):
    """Extract TLA+ specs if requested."""
    if not config.getoption("--check-tla"):
        return

    output_dir = Path(config.getini("tla_output_dir") or "specs/tla/extracted")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract specs using appropriate extractor
    extractor = get_extractor(config)
    specs = extractor.extract_all()

    # Write to files
    for name, tla_content in specs.items():
        spec_file = output_dir / f"{name}.tla"
        spec_file.write_text(tla_content)

    # Run model checker if enabled
    if config.getini("tla_run_model_checker"):
        run_model_checker(output_dir, config)

def get_extractor(config) -> TLAExtractor:
    """Factory for TLA+ extractors based on embedding approach."""
    approach = config.getini("tla_embedding_approach") or "decorator"

    if approach == "docstring":
        return DocstringExtractor()
    elif approach == "decorator":
        return DecoratorExtractor()
    elif approach == "formal_spec":
        return FormalSpecExtractor()
    else:
        raise ValueError(f"Unknown TLA embedding approach: {approach}")

class TLAExtractor:
    """Base class for TLA+ spec extractors."""

    def extract_all(self) -> dict[str, str]:
        """Extract all TLA+ specs from the codebase."""
        raise NotImplementedError

class DecoratorExtractor(TLAExtractor):
    """Extract TLA+ specs from @tla_spec decorators."""

    def extract_all(self) -> dict[str, str]:
        import importlib
        import sys

        specs = {}

        # Import all modules
        for py_file in Path("src/weakincentives").rglob("*.py"):
            module_name = py_file.relative_to("src").with_suffix("").as_posix().replace("/", ".")
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue

            # Extract specs
            for name, obj in inspect.getmembers(module):
                spec = getattr(obj, "__formal_spec__", None)
                if spec:
                    specs[spec.module] = spec.to_tla()

        return specs

def run_model_checker(specs_dir: Path, config) -> None:
    """Run TLC model checker on extracted specs."""
    checker = config.getini("tla_model_checker") or "tlc"

    for spec_file in specs_dir.glob("*.tla"):
        cfg_file = spec_file.with_suffix(".cfg")

        # Generate default config if missing
        if not cfg_file.exists():
            generate_default_config(spec_file, cfg_file)

        # Run checker
        cmd = [checker, str(spec_file), "-config", str(cfg_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            pytest.fail(
                f"TLC model checking failed for {spec_file.name}:\n{result.stdout}\n{result.stderr}"
            )

def generate_default_config(spec_file: Path, cfg_file: Path) -> None:
    """Generate default TLC configuration."""
    # Read spec to extract invariants
    spec_content = spec_file.read_text()

    # Extract invariant names (simple regex)
    import re
    invariants = re.findall(r'^(\w+Invariant)\s+==', spec_content, re.MULTILINE)

    cfg_content = [
        "SPECIFICATION Spec",
        "",
        "INVARIANTS",
    ]
    for inv in invariants:
        cfg_content.append(f"    {inv}")

    cfg_file.write_text("\n".join(cfg_content))
```

### Usage

```bash
# Extract and validate TLA+ specs from code
pytest --check-tla

# Only run TLA+ extraction, skip Python tests
pytest --tla-only

# Extract specs without running model checker
pytest --check-tla -o tla_run_model_checker=false
```

## Recommended Approach

After evaluating all approaches, **Approach 5 (Structured Metadata)** combined
with **Approach 4 (Hybrid DbC)** provides the best balance:

### Hybrid Recommendation

1. **For simple properties**: Extend DbC decorators with `tla=` parameter
   ```python
   @require(
       lambda x: x > 0,
       tla="x > 0",
       tla_id="PRE-PositiveInput"
   )
   ```

2. **For complex state machines**: Use `@formal_spec` with structured metadata
   ```python
   @formal_spec(
       module="RedisMailbox",
       state_vars=[...],
       actions=[...],
       invariants=[...],
   )
   class RedisMailbox:
       ...
   ```

3. **For supplementary documentation**: Use docstring TLA+ blocks
   ```python
   def receive(self):
       """Receive messages.

       ```tla
       Receive == ...
       ```
       """
   ```

### Implementation Priority

1. **Phase 1**: Implement `@formal_spec` decorator and pytest plugin
2. **Phase 2**: Add `tla=` parameter to existing DbC decorators
3. **Phase 3**: Docstring extraction as fallback/supplement

### Next Steps

1. Create `src/weakincentives/formal/` module with metadata classes
2. Implement pytest plugin in `tests/plugins/pytest_tla.py`
3. Migrate one existing spec (RedisMailbox) as proof-of-concept
4. Document usage in `specs/FORMAL_VERIFICATION.md`
5. Add to `make check` as optional target (`make verify-formal`)

## Open Questions

1. **Spec coverage**: Should we require TLA+ specs for all critical paths or allow gradual adoption?
2. **CI integration**: Run TLC on every commit or only on formal verification changes?
3. **Model bounds**: How to specify TLC model-checking bounds (constants) in Python decorators?
4. **Multi-module specs**: How to handle TLA+ specs that span multiple Python modules?
5. **Property tests**: Should we auto-generate Hypothesis tests from TLA+ invariants?

## References

- `specs/VERIFICATION.md` - Current formal verification approach
- `specs/DBC.md` - Design-by-contract specification
- `specs/tla/RedisMailbox.tla` - Existing TLA+ spec to migrate
- [TLA+ Home](https://lamport.azurewebsites.net/tla/tla.html)
- [PlusCal Algorithm Language](https://lamport.azurewebsites.net/tla/pluscal.html) - Alternative to raw TLA+

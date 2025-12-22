# User Identity Primitives Specification

## Purpose

User identity primitives associate a user with a prompt evaluation and ensure
identity is accessible during tool execution. This enables audit trails,
authorization, and multi-tenant isolation.

## Guiding Principles

- **Immutable**: Identity is frozen at evaluation start.
- **Type-safe**: Access via typed protocols, not string lookups.
- **Opt-in**: Identity is optional; tools must handle its absence.
- **Separation of concerns**: Primitives carry metadata; tools enforce
  authorization.

```mermaid
flowchart LR
    Build["UserIdentity"] --> Resources["ResourceRegistry"]
    Resources --> Adapter["adapter.evaluate()"]
    Adapter --> ToolCtx["ToolContext"]
    ToolCtx --> Handler["context.user_identity"]
```

## Core Components

### UserIdentity

```python
@dataclass(slots=True, frozen=True)
class UserIdentity:
    user_id: str                              # Required, non-empty, max 255 chars
    display_name: str | None = None           # Max 500 chars
    email: str | None = None                  # Valid email format
    roles: tuple[str, ...] = ()               # Each max 100 chars
    permissions: tuple[str, ...] = ()         # Each max 100 chars
    attributes: Mapping[str, str] = field(default_factory=dict)
    tenant_id: str | None = None              # Max 255 chars
    authenticated_at: datetime | None = None  # Must be timezone-aware
```

### UserIdentityProtocol

For custom identity implementations:

```python
@runtime_checkable
class UserIdentityProtocol(Protocol):
    @property
    def user_id(self) -> str: ...

    @property
    def tenant_id(self) -> str | None: ...
```

## Integration

### Injecting Identity

```python
identity = UserIdentity(user_id="user-123", roles=("admin",))

resources = ResourceRegistry.build({UserIdentity: identity})

response = adapter.evaluate(
    prompt,
    bus=bus,
    session=session,
    resources=resources,
)
```

### Accessing in Tools

`ToolContext` provides a convenience property:

```python
@property
def user_identity(self) -> UserIdentityProtocol | None:
    return self.resources.get(UserIdentity)
```

Tool handlers access identity via `context.user_identity`:

```python
def my_handler(params: Params, *, context: ToolContext) -> ToolResult[Result]:
    identity = context.user_identity

    if identity is None:
        return ToolResult(message="Authentication required", value=None, success=False)

    # Protocol fields
    user_id = identity.user_id
    tenant_id = identity.tenant_id

    # Full UserIdentity fields require narrowing
    if isinstance(identity, UserIdentity):
        if "admin" not in identity.roles:
            return ToolResult(message="Admin required", value=None, success=False)

    return ToolResult(message="OK", value=result)
```

### Session Storage

For audit trails or cross-prompt persistence:

```python
session[UserIdentity].seed(identity)
stored = context.session[UserIdentity].latest()
```

Prefer `context.user_identity` for read-only checks; use session state when
identity-related data must persist.

## Limitations

- Identity is frozen at injection; changes require a new evaluation
- Tools must implement their own authorization logic
- Custom identity attributes require type narrowing via `isinstance()`
- Child sessions do not inherit parent identity
